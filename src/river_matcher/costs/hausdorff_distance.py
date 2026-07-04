from __future__ import annotations

import math

from shapely import LineString, hausdorff_distance

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.witnesses import SourceGuidedWitnessFinder, XYArray


class HausdorffDistance(BaseEdgeCost):
    """Symmetric discrete Hausdorff distance between a source edge and its source-guided target witness.
    The result uses the same coordinate units as the input graphs."""

    name = CostName.HAUSDORFF_DISTANCE
    label = "source-guided Hausdorff distance"

    def __init__(self, resources: CostResources, *, rho: float = 10.0, edge_samples: int = 12, densify: float | None = None) -> None:
        super().__init__(resources)

        if densify is not None:
            resolved_densify = float(densify)

            if not math.isfinite(resolved_densify) or resolved_densify <= 0.0 or resolved_densify > 1.0:
                raise ValueError("Hausdorff densify must be greater than 0 and at most 1.")
        else:
            resolved_densify = None

        self.rho = float(rho)
        self.edge_samples = int(edge_samples)
        self.densify = resolved_densify
        self._finder: SourceGuidedWitnessFinder = resources.guided_paths(rho=self.rho, edge_samples=self.edge_samples)
        self._source_geometry_cache: dict[int, LineString] = {}

    def _source_geometry(self, edge_id: int) -> LineString:
        if edge_id not in self._source_geometry_cache:
            edge = self._source_edges[edge_id]
            self._source_geometry_cache[edge_id] = LineString(edge.polyline)

        return self._source_geometry_cache[edge_id]

    def _compute(self, request: CostRequest) -> float:
        edge = self._source_edge(request)

        if edge is None or not self._valid_target_pair(request):
            return math.inf

        edge_id, source_u, source_v, target_u, target_v = request
        witness = self._finder.path(edge_id, source_u, source_v, target_u, target_v)
        self._remember_witness(request, witness)

        if witness is None:
            return math.inf

        source_geometry = self._source_geometry(edge_id)
        witness_geometry = LineString(witness)

        if self.densify is None:
            value = hausdorff_distance(source_geometry, witness_geometry)
        else:
            value = hausdorff_distance(source_geometry, witness_geometry, densify=self.densify)

        return float(value)

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self._finder.path(*request)

    def clear_cache(self) -> None:
        super().clear_cache()
        self._source_geometry_cache.clear()
