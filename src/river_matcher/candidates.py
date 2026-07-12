from __future__ import annotations

import math
from collections import defaultdict
from enum import StrEnum
from typing import Any, cast

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from river_matcher.models import JunctionEdge, JunctionGraph

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type CandidateSets = dict[int, list[int]]
type PreparedTargetEdges = tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray]
_SEGMENT_SQUARED_TOLERANCE = 1e-24


class CandidateMode(StrEnum):
    TARGET_JUNCTIONS = "target_junctions"
    ORIGINAL_TARGET_VERTICES = "original_target_vertices"
    UNIFORM_TARGET_SUBDIVISION = "uniform_target_subdivision"
    ADAPTIVE_CLOSEST_POINTS = "adaptive_closest_points"

    @property
    def display_name(self) -> str:
        return {CandidateMode.TARGET_JUNCTIONS: "Target junctions", CandidateMode.ORIGINAL_TARGET_VERTICES: "Original target vertices",
            CandidateMode.UNIFORM_TARGET_SUBDIVISION: "Uniform target-edge subdivision", CandidateMode.ADAPTIVE_CLOSEST_POINTS: "Adaptive closest points", }[self]


def _polyline_lengths(polyline: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    segment_lengths = _float64_array(np.linalg.norm(np.diff(polyline, axis=0), axis=1))
    cumulative = _float64_array(np.concatenate((np.asarray([0.0]), np.cumsum(segment_lengths))))
    return segment_lengths, cumulative, float(cumulative[-1])


def point_at_fraction(polyline: FloatArray, fraction: float) -> FloatArray:
    """Return the arc-length position at ``fraction`` along a polyline."""
    points = _contiguous_float64(polyline)
    value = float(fraction)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Polyline fraction must be finite and in [0, 1], got {fraction!r}")
    segment_lengths, cumulative, total = _polyline_lengths(points)
    distance = value * total
    if distance <= 0.0:
        return points[0].copy()
    if distance >= total:
        return points[-1].copy()
    segment = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(segment_lengths) - 1)
    local = (distance - float(cumulative[segment])) / float(segment_lengths[segment])
    return _float64_array(points[segment] + local * (points[segment + 1] - points[segment]))


def subpolyline_between_fractions(polyline: FloatArray, start_fraction: float, end_fraction: float) -> FloatArray:
    """Slice a polyline by arc-length fractions while retaining interior bends."""
    points = _contiguous_float64(polyline)
    start = float(start_fraction)
    end = float(end_fraction)
    if not math.isfinite(start) or not math.isfinite(end) or not 0.0 <= start < end <= 1.0:
        raise ValueError(f"Polyline interval must satisfy 0 <= start < end <= 1, got ({start_fraction!r}, {end_fraction!r})")
    _, cumulative, total = _polyline_lengths(points)
    start_distance = start * total
    end_distance = end * total
    selected = [point_at_fraction(points, start)]
    selected.extend(points[index].copy() for index in range(1, len(points) - 1) if start_distance < float(cumulative[index]) < end_distance)
    selected.append(point_at_fraction(points, end))
    return _contiguous_float64(np.vstack(selected))


def _subdivide_graph(graph: JunctionGraph, fractions_by_edge: dict[int, list[float]], *, name: str) -> JunctionGraph:
    coordinates = dict(graph.coordinates)
    next_vertex_id = max(graph.vertices) + 1
    next_edge_id = 0
    split_edges: list[JunctionEdge] = []

    for edge in graph.edges:
        fractions = [0.0, *fractions_by_edge.get(edge.id, []), 1.0]
        vertices = [edge.u]
        for fraction in fractions[1:-1]:
            point = point_at_fraction(edge.polyline, fraction)
            coordinates[next_vertex_id] = (float(point[0]), float(point[1]))
            vertices.append(next_vertex_id)
            next_vertex_id += 1
        vertices.append(edge.v)

        for index, (start, end) in enumerate(zip(fractions, fractions[1:], strict=False)):
            split_edges.append(JunctionEdge(id=next_edge_id, u=vertices[index], v=vertices[index + 1], polyline=subpolyline_between_fractions(edge.polyline, start, end)))
            next_edge_id += 1
    return JunctionGraph(name=name, coordinates=coordinates, edges=tuple(split_edges))


def subdivide_graph_uniform(graph: JunctionGraph, *, samples_per_edge: int = 2) -> JunctionGraph:
    """Insert an equal number of arc-length-spaced vertices into every target edge."""
    samples = int(samples_per_edge)
    if samples < 0:
        raise ValueError(f"Subdivision points per edge must be nonnegative, got {samples_per_edge!r}")
    if samples == 0 or not graph.edges:
        return graph
    fractions = [index / (samples + 1) for index in range(1, samples + 1)]
    return _subdivide_graph(graph, {edge.id: fractions for edge in graph.edges}, name=f"{graph.name}_uniform_{samples}")


def _closest_polyline_fraction(point: tuple[float, float], polyline: FloatArray) -> tuple[float, float]:
    points = _contiguous_float64(polyline)
    segment_lengths, cumulative, total = _polyline_lengths(points)
    query = _float64_array(point)
    best_squared = math.inf
    best_distance = 0.0
    for index, segment_length in enumerate(segment_lengths):
        vector = points[index + 1] - points[index]
        squared_length = float(segment_length * segment_length)
        projection = float(np.dot(query - points[index], vector) / squared_length)
        projection = min(max(projection, 0.0), 1.0)
        closest = points[index] + projection * vector
        squared_distance = float(np.dot(query - closest, query - closest))
        distance_along = float(cumulative[index]) + projection * float(segment_length)
        if squared_distance < best_squared or (squared_distance == best_squared and distance_along < best_distance):
            best_squared = squared_distance
            best_distance = distance_along
    return math.sqrt(best_squared), best_distance / total


def subdivide_graph_adaptive_closest_points(source: JunctionGraph, target: JunctionGraph, *, rho: float, max_points_per_source: int = 8,
        min_separation: float = 1.0, ) -> JunctionGraph:
    """Split target edges at nearby closest points for each source vertex."""
    radius = float(rho)
    limit = int(max_points_per_source)
    separation = float(min_separation)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"Candidate radius must be finite and nonnegative, got {rho!r}")
    if limit < 1:
        raise ValueError(f"Maximum adaptive points per source must be at least 1, got {max_points_per_source!r}")
    if not math.isfinite(separation) or separation < 0.0:
        raise ValueError(f"Adaptive minimum separation must be finite and nonnegative, got {min_separation!r}")

    split_distances: dict[int, list[float]] = defaultdict(list)
    for source_vertex in sorted(source.vertices):
        point = source.coordinates[source_vertex]
        nearby: list[tuple[float, int, float]] = []
        for edge in target.edges:
            minimum = np.min(edge.polyline, axis=0)
            maximum = np.max(edge.polyline, axis=0)
            bbox = _float64_array((minimum[0], minimum[1], maximum[0], maximum[1]))
            if _bbox_point_lower_bound(point, bbox) > radius:
                continue
            distance, fraction = _closest_polyline_fraction(point, edge.polyline)
            if distance <= radius:
                nearby.append((distance, edge.id, fraction))
        nearby.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, edge_id, fraction in nearby[:limit]:
            edge = target.edge_by_id[edge_id]
            distance_along = fraction * edge.length
            if separation > 0.0 and (distance_along < separation or edge.length - distance_along < separation):
                continue
            split_distances[edge_id].append(distance_along)

    fractions_by_edge: dict[int, list[float]] = {}
    for edge in target.edges:
        accepted: list[float] = []
        tolerance = max(1e-12, edge.length * 1e-12)
        for distance in sorted(split_distances.get(edge.id, [])):
            if distance <= tolerance or edge.length - distance <= tolerance:
                continue
            required_gap = max(separation, tolerance)
            if not accepted or distance - accepted[-1] >= required_gap:
                accepted.append(distance)
        if accepted:
            fractions_by_edge[edge.id] = [distance / edge.length for distance in accepted]
    if not fractions_by_edge:
        return target
    return _subdivide_graph(target, fractions_by_edge, name=f"{target.name}_adaptive")


def prepare_candidate_target(source: JunctionGraph, target: JunctionGraph, *, candidate_mode: CandidateMode | str, rho: float, subdivision_points: int = 2,
        adaptive_max_points_per_source: int = 8, adaptive_min_separation: float = 1.0, ) -> JunctionGraph:
    """Build the exact target graph whose vertices form the candidate universe."""
    mode = CandidateMode(candidate_mode)
    if mode is CandidateMode.UNIFORM_TARGET_SUBDIVISION:
        return subdivide_graph_uniform(target, samples_per_edge=subdivision_points)
    if mode is CandidateMode.ADAPTIVE_CLOSEST_POINTS:
        return subdivide_graph_adaptive_closest_points(source, target, rho=rho, max_points_per_source=adaptive_max_points_per_source, min_separation=adaptive_min_separation, )
    return target


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
    return (_contiguous_int64(np.asarray(endpoints, dtype=np.int64).reshape((-1, 2))), _contiguous_float64(np.asarray(bboxes, dtype=np.float64).reshape((-1, 4))), packed_starts,
            packed_vectors, packed_lengths, _contiguous_int64(edge_offsets),)


@njit(cache=True, parallel=True, fastmath=False)
def _candidate_edge_distances_numba(source_points: FloatArray, bboxes: FloatArray, segment_starts: FloatArray, segment_vectors: FloatArray, segment_squared_lengths: FloatArray,
        edge_offsets: IntArray, rho: float, ) -> FloatArray:
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
                squared_distance = offset_x ** 2 + offset_y ** 2
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


def compute_vertex_candidate_sets(source: JunctionGraph, target: JunctionGraph, *, rho: float = 10.0, top_k: int = 25) -> CandidateSets:
    """Rank existing target vertices by point-to-point distance."""
    radius, limit = _normalize_parameters(rho, top_k)
    target_ids = np.asarray(sorted(target.vertices), dtype=np.int64)
    target_points = np.asarray([target.coordinates[int(vertex)] for vertex in target_ids], dtype=np.float64)
    candidate_sets: CandidateSets = {}

    for source_vertex in sorted(source.vertices):
        point = np.asarray(source.coordinates[source_vertex], dtype=np.float64)
        distances = np.linalg.norm(target_points - point, axis=1)
        ordered = sorted(((float(distances[index]), int(target_ids[index])) for index in range(len(target_ids)) if float(distances[index]) <= radius),
            key=lambda item: (item[0], item[1]), )
        candidate_sets[source_vertex] = [vertex for _, vertex in ordered[:limit]]
    return candidate_sets


def _bbox_point_lower_bound(point: tuple[float, float], bbox: FloatArray) -> float:
    px, py = point
    min_x, min_y, max_x, max_y = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    dx = max(min_x - px, 0.0, px - max_x)
    dy = max(min_y - py, 0.0, py - max_y)
    return math.hypot(dx, dy)


def _point_to_packed_edge_distance(point: tuple[float, float], edge_index: int, segment_starts: FloatArray, segment_vectors: FloatArray, segment_squared_lengths: FloatArray,
        edge_offsets: IntArray, ) -> float:
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
