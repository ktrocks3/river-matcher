from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

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
    SYMMETRIC_CORRIDOR_EXCEEDANCE = "symmetric_corridor_exceedance"
    DYNAMIC_TIME_WARPING = "dynamic_time_warping"
    DISCRETE_FRECHET = "discrete_frechet"


@dataclass(slots=True)
class CostResources:
    """ Shared graph-dependent resources used by multiple cost functions.
        One factory owns one resource container, allowing costs with compatible witness settings to reuse path trees, corridor graphs and path geometry."""
    source: JunctionGraph
    target: JunctionGraph
    shortest_path: ShortestPathWitnessFinder = field(init=False)
    _guided_paths: dict[tuple[float, int], SourceGuidedWitnessFinder] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.shortest_path = ShortestPathWitnessFinder(self.target)

    def guided_paths(self, *, rho: float, edge_samples: int) -> SourceGuidedWitnessFinder:
        key = (rho, edge_samples)
        if key not in self._guided_paths:
            self._guided_paths[key] = SourceGuidedWitnessFinder(self.source, self.target, rho=key[0], edge_samples=key[1])
        return self._guided_paths[key]

    def clear_caches(self) -> None:
        self.shortest_path.clear_cache()
        for finder in self._guided_paths.values():
            finder.clear_cache()


class BaseEdgeCost(ABC):
    """Common cached interface implemented by every local edge cost."""
    name: ClassVar[CostName]
    label: ClassVar[str]

    def __init__(self, resources: CostResources) -> None:
        self.resources = resources
        self._source_edges = {edge.id: edge for edge in resources.source.edges}
        self._target_vertices = set(resources.target.vertices)
        self._cost_cache: CostCache = {}
        self._witness_cache: WitnessCache = {}

    @staticmethod
    def _request(edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> CostRequest:
        return int(edge_id), int(source_u), int(source_v), int(target_u), int(target_v)

    @abstractmethod
    def _compute(self, request: CostRequest) -> float:
        """Compute one uncached cost."""

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return None

    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float:
        key = self._request(edge_id, source_u, source_v, target_u, target_v)
        if key in self._cost_cache:
            return self._cost_cache[key]
        value = float(self._compute(key))
        if math.isnan(value):
            value = math.inf
        elif value < 0.0:
            raise RuntimeError(f"{self.name} produced negative cost {value!r} for request {key}.")
        self._cost_cache[key] = value
        return value

    def batch(self, requests: Iterable[CostRequest]) -> FloatArray:
        """ Evaluate several requests.
            Expensive subclasses should override this method with a vectorized, compiled or library-backed implementation."""
        normalized = [self._request(*req) for req in requests]
        output = np.empty(len(normalized), dtype=float)
        for i, req in enumerate(normalized):
            output[i] = self(*req)
        return output

    def edge_ok(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int, threshold: float) -> bool:
        return self(edge_id, source_u, source_v, target_u, target_v) <= float(threshold)

    def witness(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> XYArray | None:
        key = self._request(edge_id, source_u, source_v, target_u, target_v)
        value = self(*key)
        if not math.isfinite(value):
            return None
        if key not in self._witness_cache:
            self._witness_cache[key] = self._compute_witness(key)
        return self._witness_cache[key]

    def clear_cache(self) -> None:
        self._cost_cache.clear()
        self._witness_cache.clear()

    def _source_edge(self, request: CostRequest, ) -> JunctionEdge | None:
        edge_id, source_u, source_v, _, _ = request
        edge = self._source_edges.get(edge_id)
        if edge is None:
            return None
        if (source_u == edge.u and source_v == edge.v) or (source_u == edge.v and source_v == edge.u):
            return edge
        return None

    def _valid_target_pair(self, request: CostRequest, ) -> bool:
        _, _, _, target_u, target_v = request
        return target_u != target_v and target_u in self._target_vertices and target_v in self._target_vertices

    def _remember_witness(self, request: CostRequest, witness: XYArray | None, ) -> None:
        self._witness_cache[request] = witness
