from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from operator import index
from typing import Protocol

from river_matcher.decomposition import Bag, SourceDecomposition

type State = tuple[int, ...]
type SeparatorKey = tuple[int, ...]
type CostRequest = tuple[int, int, int, int, int]
type CandidateSets = Mapping[int, Iterable[int]]


class EdgeCost(Protocol):
    """Uniform local edge cost interface consumed by the DP"""

    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float: ...


class Objective(StrEnum):
    ADDITIVE = "additive"
    BOTTLENECK = "bottleneck"


@dataclass(frozen=True, slots=True)
class BagDPStatistics:
    bag: Bag
    enumerated_states: int
    feasible_states: int
    message_entries: int

