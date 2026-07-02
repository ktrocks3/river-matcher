from __future__ import annotations

import math
from typing import Any

import numpy as np
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
        delta = points[index] - points[index - 1]