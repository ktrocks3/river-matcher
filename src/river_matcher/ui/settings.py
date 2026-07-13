from __future__ import annotations

from enum import StrEnum

from river_matcher.costs.base import CostName


class DominanceMode(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"

    @property
    def display_name(self) -> str:
        return self.value.title()

    def enabled_for(self, cost_name: CostName | str) -> bool:
        if self is DominanceMode.ON:
            return True
        if self is DominanceMode.OFF:
            return False
        return CostName(cost_name) is CostName.DISCRETE_FRECHET_DISTANCE
