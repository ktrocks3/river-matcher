from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_degree, treewidth_min_fill_in

from river_matcher.models import JunctionGraph

type Bag = frozenset[int]
type TreeEdge = tuple[Bag, Bag]
type OwnedEdge = tuple[int, int, int]
type OwnedEdgePosition = tuple[int, int, int]
type ChildPositions = tuple[Bag, tuple[int, ...]]
type SourceGraph = nx.Graph[int]
type DecompositionTree = nx.Graph[Bag]


class TreewidthHeuristic(StrEnum):
    MINIMUM_FILL_IN = "minimum_fill_in"
    MINIMUM_DEGREE = "minimum_degree"


def build_simple_source_graph(source: JunctionGraph) -> SourceGraph:
    graph = cast(SourceGraph, nx.Graph())
    graph.add_nodes_from(source.vertices)
    graph.add_edges_from((edge.u, edge.v) for edge in source.edges)
    return graph


def _normalize_decomposition_tree(raw_tree: DecompositionTree) -> DecompositionTree:
    normalized = cast(DecompositionTree, nx.Graph())
    bag_lookup: dict[object, Bag] = {}
    for raw_bag in raw_tree.nodes:
        bag = frozenset(int(vertex) for vertex in raw_bag)
        if not bag:
            raise ValueError("Tree decomposition contains an empty bag.")
        if bag in bag_lookup.values():
            raise ValueError("Tree decomposition contains duplicate normalized bag {_bag_key(bag)}.")
        bag_lookup[raw_bag] = bag
        normalized.add_node(bag)
    for raw_first, raw_second in raw_tree.edges:
        normalized.add_edge(bag_lookup[raw_first], bag_lookup[raw_second])
    return normalized


@dataclass(frozen=True, slots=True)
class BagPlan:
    """Precomputed variable positions used by the dynamic program.
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


def choose_tree_decomposition(graph: SourceGraph) -> tuple[int, TreewidthHeuristic, int, int, nx.Graph]:
    """Run both NetworkX heuristics and retain the lower-width decomposition."""
    minimum_fill_width, minimum_fill_tree = treewidth_min_fill_in(graph)
    minimum_degree_width, minimum_degree_tree = treewidth_min_degree(graph)
    if minimum_fill_width <= minimum_degree_width:
        return minimum_fill_width, TreewidthHeuristic.MINIMUM_FILL_IN, minimum_fill_width, minimum_degree_width, _normalize_decomposition_tree(minimum_fill_tree)
    return minimum_degree_width, TreewidthHeuristic.MINIMUM_DEGREE, minimum_fill_width, minimum_degree_width, _normalize_decomposition_tree(minimum_degree_tree)


def root_tree_decomposition(tree: DecompositionTree) -> tuple[Bag, dict[Bag, Bag | None], dict[Bag, tuple[Bag, ...]], dict[Bag, int], tuple[Bag, ...]]:
    """Root a decomposition tree at a deterministic center bag.
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
    """Assign every source multi-edge to the highest bag containing its endpoints.
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


def build_bag_plans(
    postorder: tuple[Bag, ...], parent: Mapping[Bag, Bag | None], children: Mapping[Bag, tuple[Bag, ...]], owned_edges: Mapping[Bag, tuple[OwnedEdge, ...]],
) -> dict[Bag, BagPlan]:
    """Precompute separator and owned-edge positions for every bag."""
    plans: dict[Bag, BagPlan] = {}
    for bag in postorder:
        variables = tuple(sorted(bag))
        positions = {vertex: index for index, vertex in enumerate(variables)}
        parent_bag = parent[bag]
        parent_positions = tuple(positions[vertex] for vertex in variables if parent_bag is not None and vertex in parent_bag)
        child_positions = tuple((child, tuple(positions[vertex] for vertex in variables if vertex in child)) for child in children[bag])
        owned_edge_positions = tuple((edge_id, positions[u], positions[v]) for edge_id, u, v in owned_edges[bag])
        plans[bag] = BagPlan(bag=bag, variables=variables, parent_positions=parent_positions, child_positions=child_positions, owned_edge_positions=owned_edge_positions)
    return plans


def validate_decomposition_tree(source: JunctionGraph, width: int, tree: DecompositionTree) -> None:
    """Validate the three defining tree-decomposition properties.
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
        raise ValueError(f"Tree decomposition does not cover source vertices {missing}.")
    for edge in source.edges:
        if not bags_by_vertex[edge.u] & bags_by_vertex[edge.v]:
            raise ValueError(f"No tree-decomposition bag covers source edge e{edge.id}: ({edge.u}, {edge.v}).")
    for vertex in sorted(source_vertices):
        containing_bags = bags_by_vertex[vertex]
        if not nx.is_connected(tree.subgraph(containing_bags)):
            raise ValueError(f"Bags containing source vertex {vertex} do not form a connected subtree.")
    actual_width = max(len(bag) for bag in bags) - 1
    if actual_width != width:
        raise ValueError(f"Reported treewidth is {width}, but the largest bag implies width {actual_width}.")


def _validate_rooted_structure(
    bags: tuple[Bag, ...],
    tree: DecompositionTree,
    root: Bag,
    parent: Mapping[Bag, Bag | None],
    children: Mapping[Bag, tuple[Bag, ...]],
    depth: Mapping[Bag, int],
    postorder: tuple[Bag, ...],
) -> None:
    bag_set = set(bags)
    if root not in bag_set:
        raise ValueError("Root bag is not part of the decomposition.")
    if set(parent) != bag_set:
        raise ValueError("Parent mapping does not contain every bag exactly once.")
    if set(children) != bag_set:
        raise ValueError("Children mapping does not contain every bag exactly once.")
    if set(depth) != bag_set:
        raise ValueError("Depth mapping does not contain every bag exactly once.")
    if parent[root] is not None or depth[root] != 0:
        raise ValueError("Root bag must have no parent and depth zero.")

    expected_tree_edges: set[TreeEdge] = set()
    for bag in bags:
        if bag == root:
            continue
        parent_bag = parent[bag]
        if parent_bag is None:
            raise ValueError(f"Non-root bag {_bag_key(bag)} has no parent.")

        if not tree.has_edge(parent_bag, bag):
            raise ValueError(f"Parent relation is not a decomposition edge: {_bag_key(parent_bag)} -> {_bag_key(bag)}.")
        if bag not in children[parent_bag]:
            raise ValueError(f"Parent and children mappings disagree for bag {_bag_key(bag)}.")
        if depth[bag] != depth[parent_bag] + 1:
            raise ValueError(f"Depth is inconsistent for bag {_bag_key(bag)}.")

        expected_tree_edges.add(_canonical_tree_edge(parent_bag, bag))
    actual_tree_edges = {_canonical_tree_edge(first, second) for first, second in tree.edges}
    if expected_tree_edges != actual_tree_edges:
        raise ValueError("Rooted parent relations do not reproduce the decomposition tree.")
    if len(postorder) != len(bags) or set(postorder) != bag_set:
        raise ValueError("Postorder must contain every bag exactly once.")
    postorder_index = {bag: index for index, bag in enumerate(postorder)}
    if postorder[-1] != root:
        raise ValueError("The root bag must be last in postorder.")
    for parent_bag, child_bags in children.items():
        for child in child_bags:
            if parent.get(child) != parent_bag:
                raise ValueError(f"Children mapping contains an invalid child relation for bag {_bag_key(child)}.")
            if postorder_index[child] >= postorder_index[parent_bag]:
                raise ValueError("A child bag appears after its parent in postorder.")


def _validate_edge_ownership(source: JunctionGraph, bags: tuple[Bag, ...], parent: Mapping[Bag, Bag | None], owned_edges: Mapping[Bag, tuple[OwnedEdge, ...]]) -> None:
    if set(owned_edges) != set(bags):
        raise ValueError("Owned-edge mapping does not contain every bag exactly once.")
    source_by_id = {edge.id: edge for edge in source.edges}
    seen: list[int] = []
    for bag in bags:
        for edge_id, u, v in owned_edges[bag]:
            edge = source_by_id.get(edge_id)
            if edge is None:
                raise ValueError(f"Unknown owned source edge ID {edge_id}.")
            if (u, v) != (edge.u, edge.v):
                raise ValueError(f"Owned record for source edge e{edge_id} has endpoints ({u}, {v}), expected ({edge.u}, {edge.v}).")
            if u not in bag or v not in bag:
                raise ValueError(f"Owner bag {_bag_key(bag)} does not contain both endpoints of source edge e{edge_id}.")
            parent_bag = parent[bag]
            if parent_bag is not None and u in parent_bag and v in parent_bag:
                raise ValueError(f"Source edge e{edge_id} is not owned by its highest containing bag.")
            seen.append(edge_id)
    expected = sorted(source_by_id)
    actual = sorted(seen)
    if actual != expected:
        raise ValueError(f"Source-edge ownership is incomplete or duplicated: expected IDs {expected}, found {actual}.")


def validate_source_decomposition(source: JunctionGraph, decomposition: SourceDecomposition) -> None:
    """Validate the complete rooted source decomposition and its bag plans."""
    tree = cast(DecompositionTree, nx.Graph())
    tree.add_nodes_from(decomposition.bags)
    tree.add_edges_from(decomposition.tree_edges)

    validate_decomposition_tree(source, decomposition.width, tree)
    _validate_rooted_structure(decomposition.bags, tree, decomposition.root, decomposition.parent, decomposition.children, decomposition.depth, decomposition.postorder)
    _validate_edge_ownership(source, decomposition.bags, decomposition.parent, decomposition.owned_edges)
    expected_plans = build_bag_plans(decomposition.postorder, decomposition.parent, decomposition.children, decomposition.owned_edges)
    if set(decomposition.bag_plans) != set(decomposition.bags):
        raise ValueError("Bag-plan mapping does not contain every bag exactly once.")

    for bag in decomposition.bags:
        if decomposition.bag_plans[bag] != expected_plans[bag]:
            raise ValueError(f"Bag plan is inconsistent for bag {_bag_key(bag)}.")


def build_source_decomposition(source: JunctionGraph, *, validate: bool = True) -> SourceDecomposition:
    """Build the rooted source tree decomposition used by the dynamic program."""
    if not source.vertices:
        raise ValueError("Cannot decompose a source graph without vertices.")

    simple_graph = build_simple_source_graph(source)
    width, heuristic, minimum_fill_width, minimum_degree_width, tree = choose_tree_decomposition(simple_graph)
    validate_decomposition_tree(source, width, tree)
    root, parent, children, depth, postorder = root_tree_decomposition(tree)

    bags = tuple(sorted(tree.nodes, key=_bag_key))
    tree_edges = tuple(sorted((_canonical_tree_edge(first, second) for first, second in tree.edges), key=lambda edge: (_bag_key(edge[0]), _bag_key(edge[1]))))
    owned_edges = assign_edge_owners(source, bags, parent, depth)
    bag_plans = build_bag_plans(postorder, parent, children, owned_edges)

    decomposition = SourceDecomposition(
        width=width,
        heuristic=heuristic,
        minimum_fill_width=minimum_fill_width,
        minimum_degree_width=minimum_degree_width,
        root=root,
        bags=bags,
        tree_edges=tree_edges,
        parent=parent,
        children=children,
        depth=depth,
        postorder=postorder,
        owned_edges=owned_edges,
        bag_plans=bag_plans,
    )

    if validate:
        validate_source_decomposition(source, decomposition)

    return decomposition
