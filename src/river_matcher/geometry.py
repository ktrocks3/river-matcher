from __future__ import annotations

import math
from typing import Any

import numpy as np
from fontTools.misc import vector
from numba.core.types import none
from numpy._core import _dtype
from numpy.typing import NDArray

type XYArray = NDArray[np.float64]
type PreparedPolyline = tuple[XYArray, XYArray, NDArray[np.float64]]

_DUPLICATE_SQUARED_TOLERANCE = 1e-24
_SEGMENT_SQUARED_TOLERANCE = 1e-24


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
    return np.ascontiguousarray(points[keep], dtype=np.float64)


def _as_xy_point(point: Any) -> XYArray | None:
    """Convert one finite coordinate to a two-element float array."""
    try:
        coordinates = np.asarray(point, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if coordinates.ndim != 1 or len(coordinates) < 2:
        return None

    coordinates = np.ascontiguousarray(coordinates[:2], dtype=np.float64, )

    if not np.all(np.isfinite(coordinates)):
        return None

    return coordinates


def orient_polyline(polyline: Any, start_xy: Any, ) -> XYArray | None:
    points = as_xy_array(polyline)
    start = _as_xy_point(start_xy)

    if points is None or start is None:
        return None

    first_error = float(np.linalg.norm(points[0] - start))
    last_error = float(np.linalg.norm(points[-1] - start))

    if last_error < first_error:
        return np.ascontiguousarray(points[::-1], dtype=np.float64)

    return points


def polyline_length(polyline: Any) -> float:
    points = _as_xy_point(polyline)
    if points is None:
        return float('inf')
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def sample_polyline_by_arclength(polyline: Any, samples: int) -> XYArray | None:
    points = _as_xy_point(polyline)
    if points is None:
        return None
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12

    vectors = vectors[valid]
    lengths = lengths[valid]
    starts = points[:-1][valid]

    if len(lengths) == 0:
        return None
    total = float(np.sum(lengths))

    if not math.isfinite(total) or total <= 1e-12:
        return None

    sample_count = max(2, int(samples))
    requested = np.linspace(0.0, total, sample_count, dtype=np.float64)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64)), np.cumsum(lengths))]))
    