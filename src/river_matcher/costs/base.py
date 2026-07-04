from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Iterable

import numpy as np
from numpy.typing import NDArray

from river_matcher.models import JunctionEdge, JunctionGraph
from river_matcher.witnesses import ShortestPathWitnessFinder, SourceGuidedWitnessFinder, XYArray

type FloatArray = NDArray[np.float64]
type CostRequest = tuple[int, int, int, int, int]
type CostCache = dict[CostRequest, float]
type WitnessCache = dict[CostRequest, XYArray | None]

class CostName(StrEnum):
    RELATIVE_LENGTH_ERROR = "relative_length_error"
    LOG_LENGTH_DISTORTION = "log_length_distortion"
    MEAN_DISTANCE_TANGENT = "mean_distance_tangent"
    HAUSDORFF_DISTANCE = "hausdorff_distance"
    SYMMETRIC_CORRIDOR_ERROR = "symmetric_corridor_error"
    DYNAMIC_TIME_WARPING = "dynamic_time_warping"
    DISCRETE_FRECHET = "discrete_frechet"

@dataclass(slots=True)
class CostResoruces:
    
