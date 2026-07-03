from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type XYArray = NDArray[np.float64]
type PreparedPolyline = tuple[XYArray, XYArray, FloatArray]

_DUPLICATE_SQUARED_TOLERANCE = 1e-24
_SEGMENT_SQUARED_TOLERANCE = 1e-12


def _float64_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _contiguous_float64(value: Any) -> FloatArray:
    return cast(FloatArray, np.ascontiguousarray(value, dtype=np.float64))


def as_xy_array(polyline: Any) -> XYArray | None:
    if polyline is None:
        return None
    try:
        points = np.asarray(polyline, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        return None
    points = points[:, :2]
    if not np.all(np.isfinite(points)):
        return None

    squared_steps = np.einsum("ij,ij->i", np.diff(points, axis=0), np.diff(points, axis=0))
    if np.all(squared_steps > _DUPLICATE_SQUARED_TOLERANCE):
        return np.ascontiguousarray(points, dtype=np.float64)

    keep = [0]
    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]
        if float(np.dot(delta, delta)) > _DUPLICATE_SQUARED_TOLERANCE:
            keep.append(index)
    if len(keep) < 2:
        return None
    return _contiguous_float64(points[keep])


def _as_xy_point(point: Any) -> XYArray | None:
    """Convert one finite coordinate to a two-element float array."""
    try:
        coordinates = np.asarray(point, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if coordinates.ndim != 1 or len(coordinates) < 2:
        return None

    coordinates = _contiguous_float64(coordinates[:2])

    if not np.all(np.isfinite(coordinates)):
        return None

    return coordinates


def orient_polyline(polyline: Any, start_xy: Any, end_xy: Any, ) -> XYArray | None:
    """Orient a polyline from start_xy toward end_xy."""
    points = as_xy_array(polyline)
    start = _as_xy_point(start_xy)
    end = _as_xy_point(end_xy)

    if points is None or start is None or end is None:
        return None

    forward_error = float(np.linalg.norm(points[0] - start) + np.linalg.norm(points[-1] - end))
    reverse_error = float(np.linalg.norm(points[-1] - start) + np.linalg.norm(points[0] - end))

    if reverse_error < forward_error:
        return _contiguous_float64(points[::-1])

    return points


def polyline_length(polyline: Any) -> float:
    points = as_xy_array(polyline)
    if points is None:
        return float("inf")

    vectors = _float64_array(np.diff(points, axis=0))
    lengths = _float64_array(np.linalg.norm(vectors, axis=1))
    return float(np.sum(lengths))


def sample_polyline_by_arclength(polyline: Any, samples: int) -> XYArray | None:
    points = as_xy_point(polyline)
    if points is None:
        return None
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12

    vectors = vectors[valid]
    lengths = lengths[valid]
    starts = points[:-1][valid]

    total = float(np.sum(lengths))
    if len(lengths) == 0 or not math.isfinite(total) or total <= 1e-12:
        return None

    sample_count = max(2, int(samples))
    requested = np.linspace(0.0, total, sample_count, dtype=np.float64)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(lengths)))
    segment_indices = (np.searchsorted(cumulative, requested, side='right') - 1)
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)
    fractions = (requested - cumulative[segment_indices]) / lengths[segment_indices]
    sampled = (starts[segment_indices] + fractions[:, None] * vectors[segment_indices])
    return np.ascontiguousarray(sampled, dtype=np.float64)


def sample_polyline_with_tangents(polyline: Any, samples: int) -> tuple[XYArray | None, XYArray | None]:
    points = _as_xy_point(polyline)
    if points is None:
        return None, None
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12

    vectors = vectors[valid]
    lengths = lengths[valid]
    starts = points[:-1][valid]
    total = float(np.sum(lengths))
    if len(lengths) == 0 or not math.isfinite(total) or total <= 1e-12:
        return None, None
    sample_count = max(2, int(samples))
    requested = np.linspace(0.0, total, sample_count, dtype=np.float64)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(lengths)))
    segment_indices = (np.searchsorted(cumulative, requested, side='right') - 1)
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)
    fractions = (requested - cumulative[segment_indices]) / lengths[segment_indices]
    sampled = (starts[segment_indices] + fractions[:, None] * vectors[segment_indices])
    tangents = (vectors[segment_indices] / lengths[segment_indices, None])
    return np.ascontiguousarray(sampled, dtype=np.float64), np.ascontiguousarray(tangents, dtype=np.float64)


def prepare_polyline_segments(polyline: Any) -> PreparedPolyline | None:
    points = as_xy_array(polyline)
    if points is None:
        return None

    starts = points[:-1]
    vectors = np.diff(points, axis=0)
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared_lengths > _SEGMENT_SQUARED_TOLERANCE
    if not np.any(valid):
        return None

    return _contiguous_float64(starts[valid]), _contiguous_float64(vectors[valid]), _contiguous_float64(squared_lengths[valid])


def point_to_prepared_polyline_distance(point: Any, prepared_polyline: PreparedPolyline | None) -> float:
    coordinates = as_xy_array(point)
    if coordinates is None or prepared_polyline is None:
        return float('inf')
    starts, vectors, squared_lengths = prepared_polyline
    relative = _float64_array(coordinates[None, :] - starts)
    projection = _float64_array(np.einsum("ij,ij->i", relative, vectors) / squared_lengths)
    projection = _float64_array(np.clip(projection, 0, 1))
    closest_points = _float64_array(starts + projection[:, None] * vectors)
    offsets = _float64_array(closest_points - coordinates[None, :])
    squared_distances = _float64_array(np.einsum("ij,ij->i", offsets, offsets))

    return math.sqrt(float(np.min(squared_distances)))


def points_to_prepared_polyline_distances(points: Any, prepared_polyline: PreparedPolyline | None, *, chunk_size: int = 256) -> FloatArray | None:
    if prepared_polyline is None:
        return None

    try:
        query_points = _float64_array(points)
    except (ValueError, TypeError):
        return None

    if (query_points.ndim != 2) or (query_points.shape[1] < 2) or len(query_points) == 0:
        return None

    query_points = _contiguous_float64(query_points[:, :2])
    if not np.all(np.isfinite(query_points)):
        return None

    starts, vectors, squared_lengths = prepared_polyline
    distances = np.empty(len(query_points), dtype=np.float64)
    normalized_chunk_size = max(1, int(chunk_size))

    for start_index in range(0, len(query_points), normalized_chunk_size):
        chunk = query_points[start_index:start_index + normalized_chunk_size]
        point_offsets = _float64_array(np.expand_dims(chunk, axis=1) - np.expand_dims(starts, axis=0))

        projection_numerators = _float64_array(np.einsum("pse,se->ps", point_offsets, vectors))
        projection_denominators = _float64_array(np.expand_dims(squared_lengths, axis=0))
        projection = _float64_array(projection_numerators / projection_denominators)
        projection = _float64_array(np.clip(projection, 0.0, 1.0))

        expanded_starts = _float64_array(np.expand_dims(starts, axis=0))
        expanded_projection = _float64_array(np.expand_dims(projection, axis=2))
        expanded_vectors = _float64_array(np.expand_dims(vectors, axis=0))
        expanded_chunk = _float64_array(np.expand_dims(chunk, axis=1))

        closest_offsets = _float64_array(expanded_starts + expanded_projection * expanded_vectors - expanded_chunk)
        squared_distances = np.einsum("psi, psi->ps", closest_offsets, closest_offsets)
        distances[start_index:start_index + len(chunk)] = np.sqrt(np.min(squared_distances, axis=1))
    return distances


def point_to_polyline_distance(point: Any, polyline: Any) -> float:
    prepared = prepare_polyline_segments(polyline)
    return point_to_prepared_polyline_distance(point, prepared)


def closest_segment_distance_and_tangent(point: Any, polyline: Any) -> tuple[float, XYArray | None]:
    """
    Return the distance and unit tangent of the closest polyline segment.
    When several segments have equal distance, the first stored segment is selected, matching NumPy's ``argmin`` tie behavior.
    """
    coordinates = _as_xy_point(point)
    prepared = prepare_polyline_segments(polyline)
    if prepared is None or coordinates is None:
        return float('inf'), None

    starts, vectors, squared_lengths = prepared
    relative = _float64_array(coordinates[None, :] - starts)
    projection_numerators = _float64_array(np.einsum("ij,ij->i", relative, vectors))
    projection = _float64_array(projection_numerators / squared_lengths)
    projection = _float64_array(np.clip(projection, 0.0, 1.0))
    expanded_projection = _float64_array(np.expand_dims(projection, axis=1))
    closest = _float64_array(starts + expanded_projection * vectors)
    distances = np.linalg.norm(closest - coordinates, axis=1)
    best = int(np.argmin(distances))
    segment_length = math.sqrt(float(squared_lengths[best]))
    if segment_length < 1e-12:
        return float(distances[best]), None

    tangent = _contiguous_float64(vectors[best] / segment_length)
    return float(distances[best]), tangent
