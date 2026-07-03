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