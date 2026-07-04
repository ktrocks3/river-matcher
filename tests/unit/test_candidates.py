from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.candidates import compute_candidate_sets, compute_candidate_sets_reference
from river_matcher.models import JunctionEdge, JunctionGraph


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64))


def make_graph(name: str, coordinates: dict[int, tuple[float, float]], edges: tuple[JunctionEdge, ...] = ()) -> JunctionGraph:
    return JunctionGraph(name=name, coordinates=coordinates, edges=edges)


def test_target_edge_contributes_both_endpoints() -> None:
    source = make_graph("source", {1: (5.0, 1.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (10.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (10.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=1.0, top_k=10)

    assert candidates == {1: [10, 20]}


def test_candidate_radius_is_inclusive() -> None:
    source = make_graph("source", {1: (2.0, 1.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (4.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (4.0, 0.0)]),))

    included = compute_candidate_sets(source, target, rho=1.0, top_k=10)
    excluded = compute_candidate_sets(source, target, rho=0.999, top_k=10)

    assert included == {1: [10, 20]}
    assert excluded == {1: []}


def test_zero_radius_accepts_point_on_edge() -> None:
    source = make_graph("source", {1: (2.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (4.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (4.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=0.0, top_k=10)

    assert candidates == {1: [10, 20]}


def test_candidate_distance_uses_polyline_segments() -> None:
    source = make_graph("source", {1: (1.0, 1.1)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (2.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=0.11, top_k=10)

    assert candidates == {1: [10, 20]}


def test_candidates_are_ordered_by_edge_distance() -> None:
    source = make_graph("source", {1: (5.0, 0.5)})
    target = make_graph(
        "target", {10: (0.0, 0.0), 11: (10.0, 0.0), 12: (10.0, 2.0)}, (make_edge(0, 10, 11, [(0.0, 0.0), (10.0, 0.0)]), make_edge(1, 11, 12, [(0.0, 2.0), (10.0, 2.0)])),
    )

    candidates = compute_candidate_sets(source, target, rho=2.0, top_k=10)

    assert candidates == {1: [10, 11, 12]}


def test_duplicate_vertices_keep_their_first_distance_order() -> None:
    source = make_graph("source", {1: (5.0, 0.0)})
    target = make_graph(
        "target", {10: (0.0, 2.0), 20: (0.0, 1.0), 30: (10.0, 1.0)}, (make_edge(0, 10, 20, [(0.0, 2.0), (10.0, 2.0)]), make_edge(1, 20, 30, [(0.0, 1.0), (10.0, 1.0)])),
    )

    candidates = compute_candidate_sets(source, target, rho=3.0, top_k=10)

    assert candidates == {1: [20, 30, 10]}


def test_equal_distance_preserves_target_edge_and_endpoint_order() -> None:
    source = make_graph("source", {1: (5.0, 0.0)})
    target = make_graph(
        "target",
        {10: (10.0, -1.0), 20: (0.0, -1.0), 30: (0.0, 1.0), 40: (10.0, 1.0)},
        (make_edge(99, 20, 10, [(0.0, -1.0), (10.0, -1.0)]), make_edge(3, 30, 40, [(0.0, 1.0), (10.0, 1.0)])),
    )

    candidates = compute_candidate_sets(source, target, rho=1.0, top_k=10)

    assert candidates == {1: [20, 10, 30, 40]}


def test_candidate_order_does_not_sort_vertex_ids() -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {90: (-1.0, 0.0), 5: (1.0, 0.0)}, (make_edge(0, 90, 5, [(-1.0, 0.0), (1.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=0.0, top_k=10)

    assert candidates == {1: [90, 5]}


def test_top_k_truncates_after_deduplication() -> None:
    source = make_graph("source", {1: (5.0, 0.0)})
    target = make_graph(
        "target",
        {10: (0.0, 0.0), 20: (10.0, 0.0), 30: (0.0, 1.0), 40: (10.0, 1.0)},
        (make_edge(0, 10, 20, [(0.0, 0.0), (10.0, 0.0)]), make_edge(1, 30, 40, [(0.0, 1.0), (10.0, 1.0)])),
    )

    candidates = compute_candidate_sets(source, target, rho=2.0, top_k=3)

    assert candidates == {1: [10, 20, 30]}


def test_parallel_edges_do_not_duplicate_candidate_vertices() -> None:
    source = make_graph("source", {1: (1.0, 0.2)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (2.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (2.0, 0.0)]), make_edge(1, 10, 20, [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)])))

    candidates = compute_candidate_sets(source, target, rho=1.0, top_k=10)

    assert candidates == {1: [10, 20]}


def test_no_edge_within_radius_produces_empty_candidate_set() -> None:
    source = make_graph("source", {1: (100.0, 100.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (1.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=1.0, top_k=10)

    assert candidates == {1: []}


def test_there_is_no_nearest_vertex_fallback() -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (100.0, 100.0), 30: (101.0, 100.0)}, (make_edge(0, 20, 30, [(100.0, 100.0), (101.0, 100.0)]),))

    candidates = compute_candidate_sets(source, target, rho=1.0, top_k=10)

    assert candidates == {1: []}


def test_isolated_target_vertex_never_becomes_candidate() -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (10.0, 0.0), 30: (11.0, 0.0)}, (make_edge(0, 20, 30, [(10.0, 0.0), (11.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=20.0, top_k=10)

    assert 10 not in candidates[1]
    assert candidates == {1: [20, 30]}


def test_target_without_edges_produces_empty_sets() -> None:
    source = make_graph("source", {8: (0.0, 0.0), 2: (1.0, 1.0)})
    target = make_graph("target", {10: (0.0, 0.0)})

    candidates = compute_candidate_sets(source, target, rho=100.0, top_k=10)

    assert candidates == {2: [], 8: []}


def test_multiple_source_vertices_are_processed_independently() -> None:
    source = make_graph("source", {1: (1.0, 0.0), 2: (9.0, 5.0), 3: (100.0, 100.0)})
    target = make_graph(
        "target",
        {10: (0.0, 0.0), 20: (10.0, 0.0), 30: (0.0, 5.0), 40: (10.0, 5.0)},
        (make_edge(0, 10, 20, [(0.0, 0.0), (10.0, 0.0)]), make_edge(1, 30, 40, [(0.0, 5.0), (10.0, 5.0)])),
    )

    candidates = compute_candidate_sets(source, target, rho=0.5, top_k=10)

    assert candidates == {1: [10, 20], 2: [30, 40], 3: []}


def test_output_source_keys_are_deterministically_sorted() -> None:
    source = make_graph("source", {9: (0.0, 0.0), 1: (0.0, 0.0), 5: (0.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0), 20: (1.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 0.0)]),))

    candidates = compute_candidate_sets(source, target, rho=0.0, top_k=10)

    assert list(candidates) == [1, 5, 9]


@pytest.mark.parametrize("rho", [-1.0, float("-inf"), float("inf"), float("nan")])
def test_invalid_radius_is_rejected(rho: float) -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0)})

    with pytest.raises(ValueError, match="Candidate radius must be finite and nonnegative"):
        compute_candidate_sets(source, target, rho=rho, top_k=10)


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {10: (0.0, 0.0)})

    with pytest.raises(ValueError, match="Candidate limit must be at least 1"):
        compute_candidate_sets(source, target, rho=1.0, top_k=top_k)


def test_numba_and_reference_implementations_match_exactly() -> None:
    source = make_graph("source", {1: (0.0, 0.0), 2: (2.0, 0.8), 3: (4.0, -0.4), 4: (8.0, 2.0), 5: (20.0, 20.0)})
    target = make_graph(
        "target",
        {10: (0.0, 0.0), 20: (4.0, 0.0), 30: (8.0, 0.0), 40: (0.0, 2.0), 50: (8.0, 2.0)},
        (
            make_edge(0, 10, 20, [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0)]),
            make_edge(1, 20, 30, [(4.0, 0.0), (8.0, 0.0)]),
            make_edge(2, 40, 50, [(0.0, 2.0), (8.0, 2.0)]),
            make_edge(3, 10, 50, [(0.0, 0.0), (8.0, 2.0)]),
        ),
    )

    accelerated = compute_candidate_sets(source, target, rho=1.25, top_k=4)
    reference = compute_candidate_sets_reference(source, target, rho=1.25, top_k=4)

    assert accelerated == reference


def test_numba_and_reference_match_for_zero_radius() -> None:
    source = make_graph("source", {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 1.0)})
    target = make_graph(
        "target", {10: (0.0, 0.0), 20: (2.0, 0.0), 30: (2.0, 2.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (2.0, 0.0)]), make_edge(1, 20, 30, [(2.0, 0.0), (2.0, 2.0)])),
    )

    accelerated = compute_candidate_sets(source, target, rho=0.0, top_k=10)
    reference = compute_candidate_sets_reference(source, target, rho=0.0, top_k=10)

    assert accelerated == reference


def test_reference_and_accelerated_results_contain_finite_vertex_ids() -> None:
    source = make_graph("source", {1: (0.0, 0.0)})
    target = make_graph("target", {10: (-1.0, 0.0), 20: (1.0, 0.0)}, (make_edge(0, 10, 20, [(-1.0, 0.0), (1.0, 0.0)]),))

    for implementation in (compute_candidate_sets, compute_candidate_sets_reference):
        candidates = implementation(source, target, rho=0.0, top_k=10)
        assert all(isinstance(vertex, int) and math.isfinite(float(vertex)) for vertex in candidates[1])
