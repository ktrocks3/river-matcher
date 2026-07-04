from __future__ import annotations

import math

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.witnesses import XYArray

_MINIMUM_LENGTH = 1e-12


class RelativeLengthError(BaseEdgeCost):
    """ Absolute relative difference between source and target-network length.
        The value is zero when both lengths agree and remains a conventional lower-is-better cost for both additive and bottleneck optimization."""
    name = CostName.RELATIVE_LENGTH_ERROR
    label = "relative shortest-path length error"

    def __init__(self, resources: CostResources):
        super().__init__(resources)

    def _compute(self, request: CostRequest) -> float:
        edge = self._source_edge(request)
        if edge is None or not self._valid_target_pair(request):
            return math.inf
        source_length = float(edge.length)
        if (not math.isfinite(source_length)) or (source_length <= _MINIMUM_LENGTH):
            return math.inf
        target_u = request[3]
        target_v = request[4]
        target_length = self.resources.shortest_path.distance(target_u, target_v)

        if (not math.isfinite(target_length)) or (target_length <= _MINIMUM_LENGTH):
            return math.inf
        return abs(target_length / source_length - 1.0)

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self.resources.shortest_path.path(request[3], request[4])
