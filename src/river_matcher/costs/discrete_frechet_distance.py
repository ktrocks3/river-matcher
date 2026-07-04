from __future__ import annotations

import math
from typing import cast

import numpy as np
from curvesimilarities import dfd

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.geometry import XYArray, sample_polyline_by_arclength
from river_matcher.witnesses import SourceGuidedWitnessFinder

type SourceSampleKey = tuple[int, int, int]


class DiscreteFrechetDistance(BaseEdgeCost):
    name = CostName.DISCRETE_FRECHET_DISTANCE
    label = "sampled discrete Fréchet distance"

    def __init__(self, resources: CostResources, *, rho: float = 10.0, edge_samples: int = 12, curve_samples: int = 64) -> None:
        super().__init__(resources)

        resolved_curve_samples = int(curve_samples)

        if resolved_curve_samples < 2:
            raise ValueError("Curve sample count must be at least 2.")

        self.rho = float(rho)
        self.edge_samples = int(edge_samples)
        self.curve_samples = resolved_curve_samples
        self._finder: SourceGuidedWitnessFinder = resources.guided_paths(rho=self.rho, edge_samples=self.edge_samples)
        self._source_sample_cache: dict[SourceSampleKey, XYArray | None] = {}

    def _source_samples(self, edge_id: int, source_u: int, source_v: int) -> XYArray | None:
        key = (int(edge_id), int(source_u), int(source_v))

        if key in self._source_sample_cache:
            return self._source_sample_cache[key]

        source_polyline = self._finder.source_polyline(*key)

        if source_polyline is None:
            self._source_sample_cache[key] = None
            return None

        samples = sample_polyline_by_arclength(source_polyline, self.curve_samples)
        self._source_sample_cache[key] = samples

        if samples is not None:
            reverse_key = (key[0], key[2], key[1])
            reverse_samples = cast(XYArray, np.ascontiguousarray(samples[::-1], dtype=np.float64))
            reverse_samples.setflags(write=False)
            self._source_sample_cache.setdefault(reverse_key, reverse_samples)
        return samples

    def _compute(self, request: CostRequest) -> float:
        edge = self._source_edge(request)

        if edge is None or not self._valid_target_pair(request):
            return math.inf

        edge_id, source_u, source_v, target_u, target_v = request
        source_samples = self._source_samples(edge_id, source_u, source_v)
        witness = self._finder.path(edge_id, source_u, source_v, target_u, target_v)
        self._remember_witness(request, witness)

        if source_samples is None or witness is None:
            return math.inf

        witness_samples = sample_polyline_by_arclength(witness, self.curve_samples)

        if witness_samples is None:
            return math.inf

        value = float(dfd(source_samples, witness_samples))

        return value if math.isfinite(value) else math.inf

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self._finder.path(*request)

    def clear_cache(self) -> None:
        super().clear_cache()
        self._source_sample_cache.clear()
