from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from networkx.algorithms.bipartite import projection
from numba import njit, prange
from numpy import dtype
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
        bboxes.append((float(np.min(points[:, 0])), float(np.min(points[:, 1]), float(np.max(points[:, 0])), float(np.max(points[:, 1])))))
        segment_starts.append(starts)
        segment_vectors.append(vectors)
        segment_squared_lengths.append(squared_lengths)

        segment_count += len(squared_lengths)
        edge_offsets.append(segment_count)
    if segment_starts:
        packed_starts = _contiguous_float64(np.vstack(segment_starts))
        packed_vectors = _contiguous_float64(np.vstack(segment_vectors))
        packed_lengths = _contiguous_float64(np.vstack(segment_squared_lengths))
    else:
        packed_starts = _float64_array(np.empty((0, 2), dtype=np.float64))
        packed_vectors = _float64_array(np.empty((0, 2), dtype=np.float64))
        packed_lengths = _float64_array(np.empty(0, dtype=np.float64))
    return (_contiguous_int64(np.asarray(endpoints, dtype=np.int64).reshape((-1, 2))), _contiguous_float64(np.asarray(bboxes, dtype=np.float64).reshape((-1, 4))), packed_starts,
            packed_vectors, packed_lengths, _contiguous_int64(edge_offsets))


@njit(cache=True, parallel=True, fastmath=False)
def _candidate_edge_distances_numba(source_points: FloatArray, bboxes: FloatArray, segment_starts: FloatArray, segment_vectors: FloatArray, segment_squared_lengths: FloatArray,
                                    edge_offsets: IntArray, rho: float) -> float:
    """ Compute source-point distances to every target edge. Bounding-box rejection only skips edges whose exact distance must exceed rho; all retained distances use
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

                projection = ((px - sx) * vx + (py - sx) * vy) / squared_length
                projection = min(max(projection, 0.0), 1.0)
