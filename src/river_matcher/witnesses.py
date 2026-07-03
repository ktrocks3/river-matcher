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
type TargetPair = tuple[int, int]
type TargetPairPaths = dict[TargetPair, XYArray | None]
type Arc = tuple[int, float, XYArray]
type Adjacency = dict[int, list[Arc]]
type DistanceMap = dict[int, float]
type ParentMap = dict[int, int | None]
type ParentSegmentMap = dict[int, XYArray]
type ShortestPathTree = tuple[DistanceMap, ParentMap, ParentSegmentMap]

_TIE_TOLERANCE = 1e-15
_LENGTH_TIE_BREAK = 1e-6


@dataclass(frozen=True, slots=True)
class _TargetEdgeRecord:
    id: int
    u: int
    v: int
    length: float
    forward: XYArray
    reverse: XYArray
    samples: XYArray | None = None


@dataclass(slots=True)
class WitnessTiming:
    adjacency_seconds: float = 0.0
    adjacency_builds: int = 0
    dijkstra_seconds: float = 0.0
    dijkstra_runs: int = 0


def _float64_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _readonly_xy(value: Any) -> XYArray:
    points = cast(XYArray, np.ascontiguousarray(value, dtype=np.float64))
    points.setflags(write=False)
    return points


def _prepare_target_edges(target: JunctionGraph, *, samples: int | None = None) -> tuple[_TargetEdgeRecord, ...]:
    records: list[_TargetEdgeRecord] = []
    for edge in target.edges:
        forward = orient_polyline(edge.polyline, target.coordinates[edge.u], target.coordinates[edge.v])
        if forward is None:
            raise RuntimeError(f"Could not orient target edge e{edge.id} from vertex {edge.u} to vertex {edge.v}.")
        length = polyline_length(forward)
        if not math.isfinite(length) or length <= 1e-12:
            f"Could not orient target edge e{edge.id} from vertex {edge.u} to vertex {edge.v}."

        stored_forward = _readonly_xy(forward)
        stored_reverse = _readonly_xy(forward[::-1])
        sampled: XYArray | None = None

        if samples is not None:
            sampled = sample_polyline_by_arclength(stored_forward, samples)
            if sampled is None:
                raise RuntimeError(f"Could not sample target edge e{edge.id}.")
            sampled = _readonly_xy(sampled)
        records.append(_TargetEdgeRecord(id=edge.id, u=edge.u, v=edge.v, length=length, forward=stored_forward, reverse=stored_reverse, samples=sampled))
    return tuple(records)


def _dijkstra(adjacency: Mapping[int, list[Arc]], start: int) -> ShortestPathTree:
    queue: list[tuple[float, int]] = [(0.0, start)]
    distances: DistanceMap = {start: 0.0}
    parent: ParentMap = {start: None}
    parent_segment: ParentSegmentMap = {}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current] + _TIE_TOLERANCE:
            continue

        for neighbour, weight, segment in adjacency.get(current, []):
            new_distance = distance + weight
            if 
