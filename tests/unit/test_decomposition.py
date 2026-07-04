from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from river_matcher.decomposition import (BagPlan, TreewidthHeuristic, assign_edge_owners, build_bag_plans, build_simple_source_graph, build_source_decomposition,
                                         root_tree_decomposition, validate_decomposition_tree, validate_source_decomposition, )
from river_matcher.models import JunctionEdge, JunctionGraph

type EdgeSpec = tuple[int, int, int]
type Bag = frozenset[int]


def make_source(vertices: tuple[int, ...], edges: tuple[EdgeSpec, ...], ) -> JunctionGraph:
    coordinates = {vertex: (float(vertex), 0.0) for vertex in vertices}

    return JunctionGraph(name="source", coordinates=coordinates,
        edges=tuple(JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray([coordinates[u], coordinates[v], ], dtype=np.float64, ), ) for edge_id, u, v in edges), )


def make_tree(bags: tuple[Bag, ...], edges: tuple[tuple[Bag, Bag], ...], ) -> nx.Graph[Bag]:
    tree: nx.Graph[Bag] = nx.Graph()
    tree.add_nodes_from(bags)
    tree.add_edges_from(edges)
    return tree


def flatten_owned_edge_ids(decomposition: object, ) -> list[int]:
    assert hasattr(decomposition, "bags")
    assert hasattr(decomposition, "owned_edges")

    return sorted(edge_id for bag in decomposition.bags for edge_id, _, _ in decomposition.owned_edges[bag])


def test_simple_source_graph_collapses_parallel_edges_and_keeps_isolates() -> None:
    source = make_source(vertices=(0, 1, 2, 3), edges=((10, 0, 1), (11, 0, 1), (12, 1, 2),), )

    graph = build_simple_source_graph(source)

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert {frozenset(edge) for edge in graph.edges} == {frozenset((0, 1)), frozenset((1, 2)), }
    assert graph.number_of_edges() == 2


def test_path_graph_has_width_one_and_valid_decomposition() -> None:
    source = make_source(vertices=(0, 1, 2, 3), edges=((10, 0, 1), (11, 1, 2), (12, 2, 3),), )

    decomposition = build_source_decomposition(source)

    assert decomposition.width == 1
    assert decomposition.maximum_bag_size == 2
    assert decomposition.bag_count == 3
    assert flatten_owned_edge_ids(decomposition) == [10, 11, 12]

    validate_source_decomposition(source, decomposition)


def test_four_cycle_has_width_two() -> None:
    source = make_source(vertices=(0, 1, 2, 3), edges=((10, 0, 1), (11, 1, 2), (12, 2, 3), (13, 3, 0),), )

    decomposition = build_source_decomposition(source)

    assert decomposition.width == 2
    assert decomposition.maximum_bag_size == 3
    assert flatten_owned_edge_ids(decomposition) == [10, 11, 12, 13]

    validate_source_decomposition(source, decomposition)


def test_single_isolated_vertex_has_width_zero() -> None:
    source = make_source(vertices=(7,), edges=(), )

    decomposition = build_source_decomposition(source)

    assert decomposition.width == 0
    assert decomposition.bags == (frozenset((7,)),)
    assert decomposition.root == frozenset((7,))
    assert decomposition.tree_edges == ()
    assert decomposition.owned_edges[decomposition.root] == ()


def test_minimum_fill_is_selected_when_heuristic_widths_tie() -> None:
    source = make_source(vertices=(0, 1, 2, 3), edges=((10, 0, 1), (11, 1, 2), (12, 2, 3),), )

    decomposition = build_source_decomposition(source)

    assert decomposition.minimum_fill_width == 1
    assert decomposition.minimum_degree_width == 1
    assert decomposition.heuristic is TreewidthHeuristic.MINIMUM_FILL_IN


def test_repeated_construction_is_deterministic() -> None:
    source = make_source(vertices=(0, 1, 2, 3, 4), edges=((10, 0, 1), (11, 1, 2), (12, 2, 3), (13, 3, 4), (14, 1, 3),), )

    first = build_source_decomposition(source)
    second = build_source_decomposition(source)

    assert first == second


def test_parallel_source_edges_remain_separate_owned_records() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 0, 1), (12, 1, 2),), )

    decomposition = build_source_decomposition(source)

    records = {edge_id: (bag, u, v) for bag in decomposition.bags for edge_id, u, v in decomposition.owned_edges[bag]}

    assert set(records) == {10, 11, 12}
    assert records[10][0] == records[11][0]
    assert records[10][1:] == (0, 1)
    assert records[11][1:] == (0, 1)


def test_edge_is_owned_by_highest_bag_containing_both_endpoints() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 1, 2), (11, 1, 2),), )
    root = frozenset((0, 1, 2))
    child = frozenset((1, 2))
    bags = (root, child)
    parent = {root: None, child: root, }
    depth = {root: 0, child: 1, }

    owned = assign_edge_owners(source, bags, parent, depth, )

    assert owned[root] == ((10, 1, 2), (11, 1, 2),)
    assert owned[child] == ()


def test_root_is_deterministic_center_and_postorder_is_child_first() -> None:
    first = frozenset((0, 1))
    second = frozenset((1, 2))
    third = frozenset((2, 3))
    fourth = frozenset((3, 4))
    tree = make_tree(bags=(first, second, third, fourth,), edges=((first, second), (second, third), (third, fourth),), )

    root, parent, children, depth, postorder = root_tree_decomposition(tree)

    assert root == second
    assert parent[root] is None
    assert depth[root] == 0
    assert postorder[-1] == root
    assert set(postorder) == {first, second, third, fourth, }

    positions = {bag: index for index, bag in enumerate(postorder)}

    for parent_bag, child_bags in children.items():
        for child in child_bags:
            assert parent[child] == parent_bag
            assert depth[child] == depth[parent_bag] + 1
            assert positions[child] < positions[parent_bag]


def test_bag_plans_store_separator_and_edge_positions() -> None:
    root = frozenset((0, 1, 2))
    child = frozenset((1, 2, 3))
    parent = {root: None, child: root, }
    children = {root: (child,), child: (), }
    owned_edges = {root: ((10, 0, 2),), child: ((11, 1, 3),), }

    plans = build_bag_plans(postorder=(child, root), parent=parent, children=children, owned_edges=owned_edges, )

    assert plans[root] == BagPlan(bag=root, variables=(0, 1, 2), parent_positions=(), child_positions=((child, (1, 2),),), owned_edge_positions=((10, 0, 2),), )
    assert plans[child] == BagPlan(bag=child, variables=(1, 2, 3), parent_positions=(0, 1), child_positions=(), owned_edge_positions=((11, 0, 2),), )


def test_validation_rejects_missing_source_vertex() -> None:
    source = make_source(vertices=(0, 1), edges=(), )
    bag = frozenset((0,))
    tree = make_tree(bags=(bag,), edges=(), )

    with pytest.raises(ValueError, match="does not cover source vertices", ):
        validate_decomposition_tree(source, width=0, tree=tree, )


def test_validation_rejects_uncovered_source_edge() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),), )
    first = frozenset((0,))
    second = frozenset((1,))
    tree = make_tree(bags=(first, second), edges=((first, second),), )

    with pytest.raises(ValueError, match="No tree-decomposition bag covers source edge", ):
        validate_decomposition_tree(source, width=0, tree=tree, )


def test_validation_rejects_disconnected_vertex_bags() -> None:
    source = make_source(vertices=(0, 1, 2), edges=(), )
    first = frozenset((0,))
    middle = frozenset((1,))
    last = frozenset((0, 2))
    tree = make_tree(bags=(first, middle, last,), edges=((first, middle), (middle, last),), )

    with pytest.raises(ValueError, match="do not form a connected subtree", ):
        validate_decomposition_tree(source, width=1, tree=tree, )


def test_validation_rejects_incorrect_reported_width() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),), )
    bag = frozenset((0, 1))
    tree = make_tree(bags=(bag,), edges=(), )

    with pytest.raises(ValueError, match="largest bag implies width 1", ):
        validate_decomposition_tree(source, width=0, tree=tree, )


def test_rooting_rejects_empty_graph() -> None:
    tree: nx.Graph[Bag] = nx.Graph()

    with pytest.raises(ValueError, match="Cannot root an empty tree decomposition", ):
        root_tree_decomposition(tree)


def test_rooting_rejects_non_tree_graph() -> None:
    first = frozenset((0,))
    second = frozenset((1,))
    third = frozenset((2,))
    tree = make_tree(bags=(first, second, third,), edges=((first, second), (second, third), (third, first),), )

    with pytest.raises(ValueError, match="must be a tree", ):
        root_tree_decomposition(tree)


def test_complete_validation_rejects_duplicate_edge_ownership() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 1, 2),), )
    decomposition = build_source_decomposition(source)
    owned_edges = dict(decomposition.owned_edges)
    owner = next(bag for bag in decomposition.bags if owned_edges[bag])
    owned_edges[owner] = (*owned_edges[owner], owned_edges[owner][0],)
    invalid = replace(decomposition, owned_edges=owned_edges, )

    with pytest.raises(ValueError, match="incomplete or duplicated", ):
        validate_source_decomposition(source, invalid, )


def test_complete_validation_rejects_inconsistent_bag_plan() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 1, 2),), )
    decomposition = build_source_decomposition(source)
    bag_plans = dict(decomposition.bag_plans)
    bag_plans[decomposition.root] = replace(bag_plans[decomposition.root], parent_positions=(999,), )
    invalid = replace(decomposition, bag_plans=bag_plans, )

    with pytest.raises(ValueError, match="Bag plan is inconsistent", ):
        validate_source_decomposition(source, invalid, )
