from __future__ import annotations

from collections.abc import Callable
from typing import Any

from river_matcher.costs.base import BaseEdgeCost, CostName, CostResources
from river_matcher.costs.hausdorff_distance import HausdorffDistance
from river_matcher.costs.log_length_distortion import LogLengthDistortion
from river_matcher.costs.mean_distance_tangent import MeanDistanceTangent
from river_matcher.costs.relative_length_error import RelativeLengthError
from river_matcher.costs.symmetric_corridor_exceedance import SymmetricCorridorExceedance
from river_matcher.models import JunctionGraph

type CostConstructor = Callable[..., BaseEdgeCost]

_COST_TYPES: dict[CostName, CostConstructor] = {CostName.RELATIVE_LENGTH_ERROR: RelativeLengthError, CostName.LOG_LENGTH_DISTORTION: LogLengthDistortion,
                                                CostName.MEAN_DISTANCE_TANGENT: MeanDistanceTangent, CostName.HAUSDORFF_DISTANCE: HausdorffDistance, CostName.SYMMETRIC_CORRIDOR_EXCEEDANCE: SymmetricCorridorExceedance, }


def available_costs() -> tuple[CostName, ...]:
    return tuple(_COST_TYPES)


def create_cost(name: CostName | str, source: JunctionGraph, target: JunctionGraph, **options: Any) -> BaseEdgeCost:
    """ Construct one standalone cost.
        Use CostFactory directly when several costs should share witness caches."""
    return CostFactory(source, target).create(name, **options)


class CostFactory:
    """ Construct edge costs while sharing graph-dependent path resources.
        Reusing one factory across experiments prevents each cost object from independently rebuilding identical target shortest-path structures."""

    def __init__(self, source: JunctionGraph, target: JunctionGraph) -> None:
        self.resources = CostResources(source, target)

    def create(self, name: CostName | str, **options: Any) -> BaseEdgeCost:
        try:
            resolved_name = CostName(name)
        except ValueError as exc:
            available = ", ".join(cost_name.value for cost_name in available_costs())
            raise ValueError(f"Unknown cost {name!r}. Available costs: {available}.") from exc
        constructor = _COST_TYPES.get(resolved_name)
        if constructor is None:
            raise NotImplementedError(f"Cost {resolved_name.value!r} is named but not implemented yet.")
        return constructor(self.resources, **options)
