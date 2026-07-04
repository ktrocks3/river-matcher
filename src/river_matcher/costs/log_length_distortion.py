from __future__ import annotations

import math

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.witnesses import XYArray

_MINIMUM_LENGTH = 1e-12


class LogLengthDistortion(BaseEdgeCost):
    """Absolute logarithmic distortion of target-network length.
    Swapping the two lengths leaves the value unchanged because abs(log(a / b)) equals abs(log(b / a))."""

    name = CostName.LOG_LENGTH_DISTORTION
    label = "absolute log shortest-path length distortion"

    def __init__(self, resources: CostResources) -> None:
        super().__init__(resources)

    def _compute(self, request: CostRequest) -> float:
        edge = self._source_edge(request)

        if edge is None or not self._valid_target_pair(request):
            return math.inf

        source_length = float(edge.length)

        if not math.isfinite(source_length) or source_length <= _MINIMUM_LENGTH:
            return math.inf

        target_length = self.resources.shortest_path.distance(request[3], request[4])

        if not math.isfinite(target_length) or target_length <= _MINIMUM_LENGTH:
            return math.inf

        return abs(math.log(target_length / source_length))

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self.resources.shortest_path.path(request[3], request[4])
