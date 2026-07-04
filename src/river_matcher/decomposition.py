from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping
import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_degree, treewidth_min_fill_in
from river_matcher.models import JunctionGraph

type Bag = frozenset[int]
type TreeEdge = tuple[Bag, Bag]
type OwnedEdge = tuple[int, int, int]
type OwnedEdgePosition = tuple[int, int, int]
type ChildPositions = tuple[Bag, tuple[int, ...]]

class TreewidthHeuristic(StrEnum):
    MINIMUM_FILL_IN = "minimum_fill_in"
    MINIMUM_DEGREE = "minimum_degree"

@dataclass(frozen=True, slots=True)
class BagPlan:
    """ Precomputed variable positions used by the dynamic program.
        Positions refer to the sorted ``variables`` tuple of this bag."""
    bag: Bag
    variables: tuple[int, ...]
    parent_positions: tuple[int, ...]
    child_positions: tuple[ChildPositions, ...]
    owned_edge_positions: tuple[OwnedEdgePosition, ...]

@dataclass(frozen=True, slots=True)
class SourceDecomposition:
    