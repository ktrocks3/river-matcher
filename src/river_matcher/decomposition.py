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
    """Rooted tree decomposition and source-edge ownership information."""
    width: int
    heuristic: TreewidthHeuristic
    minimum_fill_width: int
    minimum_degree_width: int
    root: Bag
    bags: tuple[Bag, ...]
    tree_edges: tuple[TreeEdge, ...]
    parent: Mapping[Bag, Bag | None]
    children: Mapping[Bag, tuple[Bag, ...]]
    depth: Mapping[Bag, int]
    postorder: tuple[Bag, ...]
    owned_edges: Mapping[Bag, tuple[OwnedEdge, ...]]
    bag_plans: Mapping[Bag, BagPlan]

    @property
    def bag_count(self) -> int:
        return len(self.bags)

    @property
    def maximum_bag_size(self) -> int:
        return self.width + 1

def _bag_key(bag: Bag) -> tuple[int, ...]:
    return tuple(sorted(bag))

def _canonical_tree_edge(first:Bag, second: Bag) -> TreeEdge: