from __future__ import annotations

import math

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.witnesses import XYArray

_MINIMUM_LENGTH = 1e-12
class RelativeLengthError(BaseEdgeCost):
    """ Absolute relative difference between source and target-network length.
        The value is zero when both lengths agree and remains a conventional lower-is-better cost for both additive and bottleneck optimization."""
    name = CostName.RELATIVE_LENGTH_ERROR
    label = "Relative Length Error"