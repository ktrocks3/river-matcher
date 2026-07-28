from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]


def display_coordinates(value: ArrayLike) -> FloatArray:
    """Return a vertically flipped display copy without changing model geometry."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"Display coordinates must have shape (..., 2), got {points.shape}.")

    displayed = np.array(points, dtype=np.float64, copy=True, order="C")
    displayed[..., 1] *= -1.0
    return displayed


def display_bounds(bounds: Sequence[float]) -> tuple[float, float, float, float]:
    """Flip an ``(x_min, x_max, y_min, y_max)`` display window."""
    if len(bounds) != 4:
        raise ValueError(f"Display bounds must contain four values, got {len(bounds)}.")
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    return x_min, x_max, -y_max, -y_min
