from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from river_matcher.geometry import PreparedPolyline, XYArray, orient_polyline, points_to_prepared_polyline_distances, polyline_length, prepare_polyline_segments, \
    sample_polyline_by_arclength
from river_matcher.models import JunctionEdge, JunctionGraph

type FloatArray = NDArray[np.float64]
type WitnessRequest = tuple[int, int, int, int, int]
type WitnessPaths = dict[WitnessRequest, XYArray | None]
type TargetPair = tuple[int,int]
type TargetPairPaths = dict[TargetPair, XYArray | None]
type Arc = tuple[int, float, XYArray]
type Adjacency = dict[int, list[Arc]]
type DistanceMap = dict[int, float]
type ParentMap = dict[int, int|None]
type ParentSegmentMap = dict[int, XYArray]
type ShortestPathTree = tuple[DistanceMap, ParentMap, ParentSegmentMap]

