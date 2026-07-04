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


def _canonical_tree_edge(first: Bag, second: Bag) -> TreeEdge:
    if _bag_key(first) <= _bag_key(second):
        return first, second
    return second, first


def build_simple_source_graph(source: JunctionGraph) -> nx.Graph:
    """ Return the simple graph used for tree decomposition.
        Parallel source edges collapse to one graph edge here, but remain distinct in ``assign_edge_owneds`` through their unique source-edge IDs
    """
    graph = nx.Graph()
    graph.add_nodes_from(source.vertices)
    graph.add_edges_from((edge.u, edge.v) for edge in source.edges)
    return graph


def _normalize_decomposition_tree(raw_tree: nx.Graph) -> nx.Graph:
    """Normalize NetworkX bag objects to ``frozenset[int]``"""
    normalized = nx.Graph()
    bag_lookup: dict[object, Bag] = {}

    for raw_bag in raw_tree.nodes:
        bag = frozenset(int(vertex) for vertex in raw_bag)
        if not bag:
            raise ValueError("Tree decomposition contains an empty bag")
        if bag in bag_lookup.values():
            raise ValueError(f"Tree decomposition contains duplicate normalized bag {_bag_key(bag)}.")
        bag_lookup[raw_bag] = bag
        normalized.add_node(bag)

    for raw_first, raw_second in raw_tree.edges:
        normalized.add_edge(bag_lookup[raw_first], bag_lookup[raw_second])
    return normalized


def choose_tree_decomposition(graph: nx.Graph) -> tuple[int, TreewidthHeuristic, int, int, nx.Graph]:
    """Run both NetworkX heuristics and retain the lower-width decomposition."""
    minimum_fill_width, minimum_fill_tree = treewidth_min_fill_in(graph)
    minimum_degree_width, minimum_degree_tree = treewidth_min_degree(graph)
    if minimum_fill_width <= minimum_degree_width:
        return minimum_fill_width, TreewidthHeuristic.MINIMUM_FILL_IN, minimum_fill_width, minimum_degree_width, _normalize_decomposition_tree(minimum_fill_tree)
    return minimum_degree_width, TreewidthHeuristic.MINIMUM_DEGREE, minimum_fill_width, minimum_degree_width, _normalize_decomposition_tree(minimum_degree_tree)


def root_tree_decomposition(tree: nx.Graph) -> tuple[Bag, dict[Bag, Bag | None], dict[Bag, tuple[Bag, ...]], dict[Bag, int], tuple[Bag, ...]]:
    """ Root a decomposition tree at a deterministic center bag.
        A center minimizes the maximum rooted depth. Lexicographic bag order breaks ties so repeated construction is deterministic for one decomposition.
    """
    if tree.number_of_nodes() == 0:
        raise ValueError("Cannot root an empty tree decomposition.")
    if not nx.is_tree(tree):
        raise ValueError("The decomposition graph must be a tree.")
    root = min((frozenset(int(vertex) for vertex in bag) for bag in nx.center(tree)), key=_bag_key)
    parent: dict[Bag, Bag | None] = {root: None}
    depth: dict[Bag, int] = {root: 0}
    child_lists: defaultdict[Bag, list[Bag]] = defaultdict(list)
    queue: deque[Bag] = deque([root])

    while queue:
        current = queue.popleft()

        for neighbor in sorted(tree.neighbors(current), key=_bag_key):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            child_lists[current].append(neighbor)
            queue.append(neighbor)
    bags = tuple(sorted(tree.nodes, key=_bag_key))
    children = {bag: tuple(child_lists.get(bag, ())) for bag in bags}
    postorder: list[Bag] = []
    stack: list[tuple[Bag, bool]] = [(root, False)]
    while stack:
        bag, expanded = stack.pop()
        if expanded:
            postorder.append(bag)
            continue
        stack.append((bag, True))
        for child in reversed(children[bag]):
            stack.append((child, False))
    return root, parent, children, depth, tuple(postorder)


def assign_edge_owners(source: JunctionGraph, bags: tuple[Bag, ...], parent: Mapping[Bag, Bag | None], depth: Mapping[Bag, int]) -> dict[Bag, tuple[OwnedEdge, ...]]:
    """ Assign every source multi-edge to the highest bag containing its endpoints.
        Parallel source edges receive separate ownership records even though they share the same endpoint pair."""
    bags_by_vertex: defaultdict[int, set[Bag]] = defaultdict(set)
    for bag in bags:
        for vertex in bag:
            bags_by_vertex[vertex].add(bag)

    owned_lists: dict[Bag, list[OwnedEdge]] = {bag: [] for bag in bags}
    for edge in sorted(source.edges, key=lambda x: x.id):
        containing_bags = bags_by_vertex[edge.u] & bags_by_vertex[edge.v]
        if not containing_bags:
            raise ValueError(f"No decomposition bag contains both endpoints of source edge e{edge.id}: ({edge.u}, {edge.v}).")
        owner = min(containing_bags, key=lambda x: (depth[x], _bag_key(x)))
        parent_bag = parent[owner]
        if parent_bag is not None and edge.u in parent_bag and edge.v in parent_bag:
            raise RuntimeError(f"Selected owner is not the highest bag for source edge e{edge.id}: ({edge.u}, {edge.v}).")
        owned_lists[owner].append((edge.id, edge.u, edge.v))
    return {bag: tuple(owned_lists[bag]) for bag in bags}


def build_bag_plans(postorder: tuple[Bag, ...], parent: Mapping[Bag, Bag | None], children: Mapping[Bag, tuple[Bag, ...]], owned_edges: Mapping[Bag, tuple[OwnedEdge, ...]]) -> \
dict[Bag, BagPlan]:
    """Precompute separator and owned-edge positions for every bag."""
    plans: dict[Bag, BagPlan] = {}
    for bag in postorder:
        variables = tuple(sorted(bag))
        positions = {vertex: index for index, vertex in enumerate(variables)}
        parent_bag = parent[bag]
        parent_positions = tuple(positions[vertex] for vertex in variables if parent_bag is not None and vertex in parent_bag)
        child_positions = tuple((child, tuple(positions[vertex] for vertex in variables if vertex in child),) for child in children[bag])
        owned_edge_positions = tuple((edge_id, positions[u], positions[v],) for edge_id, u, v in owned_edges[bag])
        plans[bag] = BagPlan(bag=bag, variables=variables, parent_positions=parent_positions, child_positions=child_positions, owned_edge_positions=owned_edge_positions)
    return plans


def validate_decomposition_tree(source: JunctionGraph, width: int, tree: nx.Graph) -> None:
    """ Validate the three defining tree-decomposition properties.
        Every source vertex must occur, every source edge must be covered, and the bags containing one source vertex must form a connected subtree."""
    if tree.number_of_nodes() == 0:
        raise ValueError("Tree decomposition has no bags.")
    if not nx.is_tree(tree):
        raise ValueError("The decomposition graph is not a tree.")

    bags = tuple(tree.nodes)
    source_vertices = set(source.vertices)
    covered_vertices: set[int] = set()
    bags_by_vertex: defaultdict[int, set[Bag]] = defaultdict(set)

    for bag in bags:
        if not bag:
            raise ValueError("Tree decomposition contains an empty bag.")
        unknown = set(bag) - source_vertices
        if unknown:
            raise ValueError(f"Tree-decomposition bag contains unknown source vertices: {sorted(unknown)}.")
        covered_vertices.update(bag)
        for vertex in bag:
            bags_by_vertex[vertex].add(bag)

    if covered_vertices != source_vertices:
        missing = sorted(source_vertices - covered_vertices)
        raise ValueError(f"The decomposition graph has {len(missing)}-bags.")
