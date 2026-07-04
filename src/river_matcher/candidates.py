from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from river_matcher.models import JunctionGraph

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type CandidateSets = dict[int, list[int]]
type PreparedTargetEdges = tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray]
_SEGMENT_SQUARED_TOLERANCE = 1e-24


def _float64_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _int64_array(value: Any) -> IntArray:
    return cast(IntArray, np.asarray(value, dtype=np.int64))


def _contiguous_float64(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _contiguous_int64(value: Any) -> IntArray:
    return cast(IntArray, np.asarray(value, dtype=np.int64))


def _normalize_parameters(rho: float, top_k: int) -> tuple[float, int]:
    radius = float(rho)
    limit = int(top_k)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"Candidate radius must be finite and nonnegative, got {rho!r}")
    if limit < 1:
        raise ValueError(f"Candidate limit must be at least 1 1, got {top_k!r}")
    return radius, limit


def _prepare_target_edges(target: JunctionGraph) -> PreparedTargetEdges:
    """
    Pack target-edge geometry for repeated source-point queries.
    Edge order is preserved because equal-distance candidate ties inherit the target graph's edge order and then each edge's ``u, v`` endpoint order.
    """
    endpoints: list[tuple[int, int]] = []
    bboxes: list[tuple[float, float, float, float]] = []
    segment_starts: list[FloatArray] = []
    segment_vectors: list[FloatArray] = []
    segment_squared_lengths: list[FloatArray] = []
    edge_offsets = [0]
    segment_count = 0

    for edge in target.edges:
        points = edge.polyline
        starts = _contiguous_float64(points[:-1])
        vectors = _contiguous_float64(np.diff(points, axis=0))
        squared_lengths = _float64_array(np.einsum("ij,ij->i", vectors, vectors))
        valid = squared_lengths > _SEGMENT_SQUARED_TOLERANCE

        starts = _contiguous_float64(starts[valid])
        vectors = _contiguous_float64(vectors[valid])
        squared_lengths = _contiguous_float64(squared_lengths[valid])

        if len(squared_lengths) == 0:
            raise ValueError(f"Target edge e{edge.id} has no usable geometric segments.")
        endpoints.append((edge.u, edge.v))
        bboxes.append((float(np.min(points[:, 0])), float(np.min(points[:, 1])), float(np.max(points[:, 0])), float(np.max(points[:, 1]))))
        segment_starts.append(starts)
        segment_vectors.append(vectors)
        segment_squared_lengths.append(squared_lengths)

        segment_count += len(squared_lengths)
        edge_offsets.append(segment_count)
    if segment_starts:
        packed_starts = _contiguous_float64(np.vstack(segment_starts))
        packed_vectors = _contiguous_float64(np.vstack(segment_vectors))
        packed_lengths = _contiguous_float64(np.concatenate(segment_squared_lengths))
    else:
        packed_starts = _float64_array(np.empty((0, 2), dtype=np.float64))
        packed_vectors = _float64_array(np.empty((0, 2), dtype=np.float64))
        packed_lengths = _float64_array(np.empty(0, dtype=np.float64))
    return (
        _contiguous_int64(np.asarray(endpoints, dtype=np.int64).reshape((-1, 2))),
        _contiguous_float64(np.asarray(bboxes, dtype=np.float64).reshape((-1, 4))),
        packed_starts,
        packed_vectors,
        packed_lengths,
        _contiguous_int64(edge_offsets),
    )


@njit(cache=True, parallel=True, fastmath=False)
def _candidate_edge_distances_numba(
    source_points: FloatArray, bboxes: FloatArray, segment_starts: FloatArray, segment_vectors: FloatArray, segment_squared_lengths: FloatArray, edge_offsets: IntArray, rho: float,
) -> FloatArray:
    """Compute source-point distances to every target edge. Bounding-box rejection only skips edges whose exact distance must exceed rho; all retained distances use
    point-to-segment projection."""
    source_count = source_points.shape[0]
    edge_count = bboxes.shape[0]
    distances = np.full((source_count, edge_count), np.inf, dtype=np.float64)
    for source_index in prange(source_count):
        px, py = source_points[source_index, 0], source_points[source_index, 1]
        for edge_index in range(edge_count):
            min_x, min_y = bboxes[edge_index, 0], bboxes[edge_index, 1]
            max_x, max_y = bboxes[edge_index, 2], bboxes[edge_index, 3]
            dx, dy = max(min_x - px, 0.0, px - max_x), max(min_y - py, 0.0, py - max_y)
            if math.sqrt(dx * dx + dy * dy) > rho:
                continue
            best_squared = math.inf
            start, stop = edge_offsets[edge_index], edge_offsets[edge_index + 1]
            for segment_index in range(start, stop):
                sx, sy = segment_starts[segment_index, 0], segment_starts[segment_index, 1]
                vx, vy = segment_vectors[segment_index, 0], segment_vectors[segment_index, 1]
                squared_length = segment_squared_lengths[segment_index]

                projection = ((px - sx) * vx + (py - sy) * vy) / squared_length
                projection = min(max(projection, 0.0), 1.0)

                offset_x = sx + projection * vx - px
                offset_y = sy + projection * vy - py
                squared_distance = offset_x**2 + offset_y**2
                if squared_distance < best_squared:
                    best_squared = squared_distance
            if math.isfinite(best_squared):
                distances[source_index, edge_index] = math.sqrt(best_squared)
    return distances


def _candidate_sets_from_distances(source_ids: IntArray, endpoints: IntArray, distances: FloatArray, *, rho: float, top_k: int) -> CandidateSets:
    """Apply stable distance ordering and endpoint deduplication."""
    candidate_sets: CandidateSets = {}
    for source_index, raw_vertex in enumerate(source_ids):
        vertex = int(raw_vertex)
        hits: list[tuple[float, int]] = []

        for edge_index in range(len(endpoints)):
            distance = float(distances[source_index, edge_index])
            if distance <= rho:
                hits.append((distance, int(endpoints[edge_index, 0])))
                hits.append((distance, int(endpoints[edge_index, 1])))

        # Equal-distance entries retain target-edge order and endpoint order.
        hits.sort(key=lambda item: item[0])
        output: list[int] = []
        seen: set[int] = set()

        for _, target_vertex in hits:
            if target_vertex in seen:
                continue
            seen.add(target_vertex)
            output.append(target_vertex)

            if len(output) >= top_k:
                break
        candidate_sets[vertex] = output
    return candidate_sets


def compute_candidate_sets(source: JunctionGraph, target: JunctionGraph, *, rho: float = 10.0, top_k: int = 25) -> CandidateSets:
    """Generate target-vertex candidates for every source vertex.
    A target edge contributes both endpoints when the source vertex lies at most ``rho`` from that edge's polyline. Candidate vertices are ordered by the corresponding edge
    distance, deduplicated, and capped at ``top_k``."""
    radius, limit = _normalize_parameters(rho, top_k)
    source_ids = _contiguous_int64(np.asarray(sorted(source.vertices), dtype=np.int64))
    source_points = _contiguous_float64(np.asarray([source.coordinates[int(vertex)] for vertex in source_ids], dtype=np.float64))
    endpoints, bboxes, segment_starts, segment_vectors, segment_squared_lengths, edge_offsets = _prepare_target_edges(target)
    if len(endpoints) == 0:
        return {int(vertex): [] for vertex in source_ids}
    distances = _candidate_edge_distances_numba(source_points, bboxes, segment_starts, segment_vectors, segment_squared_lengths, edge_offsets, radius)
    return _candidate_sets_from_distances(source_ids, endpoints, distances, rho=radius, top_k=limit)


def _bbox_point_lower_bound(point: tuple[float, float], bbox: FloatArray) -> float:
    px, py = point
    min_x, min_y, max_x, max_y = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    dx = max(min_x - px, 0.0, px - max_x)
    dy = max(min_y - py, 0.0, py - max_y)
    return math.hypot(dx, dy)


def _point_to_packed_edge_distance(
    point: tuple[float, float], edge_index: int, segment_starts: FloatArray, segment_vectors: FloatArray, segment_squared_lengths: FloatArray, edge_offsets: IntArray,
) -> float:
    """Reference point-to-edge distance using the packed segment arrays."""
    px, py = point
    best_squared = math.inf
    start, stop = int(edge_offsets[edge_index]), int(edge_offsets[edge_index + 1])
    for segment_index in range(start, stop):
        sx, sy = float(segment_starts[segment_index, 0]), float(segment_starts[segment_index, 1])
        vx, vy = segment_vectors[segment_index, 0], segment_vectors[segment_index, 1]
        squared_length = float(segment_squared_lengths[segment_index])
        projection = ((px - sx) * vx + (py - sy) * vy) / squared_length
        offset_x, offset_y = sx + projection * vx - px, sy + projection * vy - py
        squared_distance = offset_x * offset_x + offset_y * offset_y
        best_squared = min(best_squared, squared_distance)
    return math.sqrt(best_squared)


def compute_candidate_sets_reference(source: JunctionGraph, target: JunctionGraph, *, rho: float = 10.0, top_k: int = 25) -> CandidateSets:
    """Generate candidate sets without Numba.
    This intentionally mirrors ``compute_candidate_sets`` and exists as an independent verification path rather than as the production implementation."""
    radius, limit = _normalize_parameters(rho, top_k)
    endpoints, bboxes, segment_starts, segment_vectors, segment_squared_lengths, edge_offsets = _prepare_target_edges(target)
    candidate_sets: CandidateSets = {}

    for source_vertex in sorted(source.vertices):
        point = source.coordinates[source_vertex]
        hits: list[tuple[float, int]] = []
        for edge_index in range(len(endpoints)):
            if _bbox_point_lower_bound(point, bboxes[edge_index]) > radius:
                continue
            distance = _point_to_packed_edge_distance(point, edge_index, segment_starts, segment_vectors, segment_squared_lengths, edge_offsets)
            if distance <= radius:
                hits.append((distance, int(endpoints[edge_index, 0])))
                hits.append((distance, int(endpoints[edge_index, 1])))
        hits.sort(key=lambda x: x[0])
        output: list[int] = []
        seen: set[int] = set()
        for _, target_vertex in hits:
            if target_vertex in seen:
                continue
            seen.add(target_vertex)
            output.append(target_vertex)

            if len(output) >= limit:
                break
        candidate_sets[source_vertex] = output
    return candidate_sets
