from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

type VertexId = int
type EdgeId = int
type XYArray = NDArray[np.float64]
type Coordinates = dict[VertexId, tuple[float, float]]
type CandidatePair = tuple[VertexId, VertexId]
type CandidateSets = dict[VertexId, tuple[VertexId, ...]]
type CostTable = dict[EdgeId, dict[CandidatePair, float]]
type VertexMapping = dict[VertexId, VertexId]
type EdgeCosts = dict[EdgeId, float]


class Objective(StrEnum):
    ADDITIVE = "additive"
    BOTTLENECK = "bottleneck"


def _clean_polyline(polyline: XYArray) -> tuple[XYArray, float]:
    """Validate one coordinate polyline and calculate it's geometric length"""
    try:
        points = np.asarray(polyline, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'A polyline must contain numeric coordinates') from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"A polyline must have shape (n, 2) with n >= 2, got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"A polyline must contain finite values")

    # Remove consecutive duplicates because they create zero-length segments.
    keep = [0]
    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]
        if float
