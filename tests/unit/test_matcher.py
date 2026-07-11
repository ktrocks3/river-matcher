from __future__ import annotations

import numpy as np
import pytest

from river_matcher.costs import CostName
from river_matcher.decomposition import build_source_decomposition
from river_matcher.dynamic_programming import Objective
from river_matcher.matcher import RiverGraphMatcher, match_graphs, match_graphs_both
from river_matcher.models import JunctionEdge, JunctionGraph

type EdgeSpec = tuple[int, int, int]
type CandidateSets = dict[int, tuple[int, ...]]


def make_graph(name: str, coordinates: dict[int, tuple[float, float]], edges: tuple[EdgeSpec, ...]) -> JunctionGraph:
    return JunctionGraph(name=name, coordinates=coordinates,
        edges=tuple(JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray([coordinates[u], coordinates[v]], dtype=np.float64)) for edge_id, u, v in edges))


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = make_graph("source", {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}, ((10, 0, 1), (11, 1, 2),))
    target = make_graph("target", {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (2.0, 0.0), 40: (10.0, 0.0)}, ((100, 10, 20), (101, 20, 30),))
    return source, target


def perfect_candidates() -> CandidateSets:
    return {0: (10,), 1: (20,), 2: (30,)}


def test_match_materializes_mapping_costs_and_witnesses() -> None:
    source, target = make_graphs()
    decomposition = build_source_decomposition(source)
    matcher = RiverGraphMatcher(source, target, candidate_sets=perfect_candidates(), decomposition=decomposition)

    result = matcher.match(CostName.RELATIVE_LENGTH_ERROR, Objective.ADDITIVE)

    assert result.feasible
    assert result.solution is not None
    assert result.cost_name is CostName.RELATIVE_LENGTH_ERROR
    assert result.solution.objective is Objective.ADDITIVE
    assert result.solution.value == pytest.approx(0.0)
    assert result.solution.mapping == {0: 10, 1: 20, 2: 30}

    assert result.decomposition is decomposition
    assert result.candidate_sets is matcher.candidate_sets
    assert [edge.edge_id for edge in result.solution.edges] == [10, 11]
    assert [edge.cost for edge in result.solution.edges] == pytest.approx([0.0, 0.0])

    np.testing.assert_allclose(result.solution.edges[0].witness, [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(result.solution.edges[1].witness, [[1.0, 0.0], [2.0, 0.0]])


def test_match_both_materializes_both_objectives() -> None:
    source, target = make_graphs()
    matcher = RiverGraphMatcher(source, target, candidate_sets=perfect_candidates())

    result = matcher.match_both("relative_length_error")

    assert result.additive_feasible
    assert result.bottleneck_feasible
    assert result.additive is not None
    assert result.bottleneck is not None

    assert result.additive.objective is Objective.ADDITIVE
    assert result.bottleneck.objective is Objective.BOTTLENECK
    assert result.additive.value == pytest.approx(0.0)
    assert result.bottleneck.value == pytest.approx(0.0)
    assert result.additive.mapping == result.bottleneck.mapping
    assert result.dp_statistics.unique_cost_requests == 2


def test_match_all_materializes_three_aggregations() -> None:
    source, target = make_graphs()
    matcher = RiverGraphMatcher(source, target, candidate_sets=perfect_candidates())

    result = matcher.match_all("relative_length_error")

    assert result.additive is not None
    assert result.bottleneck is not None
    assert result.length_weighted_additive is not None
    assert result.length_weighted_additive.objective is Objective.LENGTH_WEIGHTED_ADDITIVE
    assert result.length_weighted_additive.value == pytest.approx(0.0)


def test_mapping_evaluation_includes_length_weighted_additive_value() -> None:
    source = make_graph("source", {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (11.0, 0.0)}, ((10, 0, 1), (11, 1, 2),))
    target = make_graph("target", {10: (0.0, 0.0), 20: (2.0, 0.0), 30: (7.0, 0.0)}, ((100, 10, 20), (101, 20, 30),))
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,), 2: (30,)})

    evaluation = matcher.evaluate_mapping({0: 10, 1: 20, 2: 30}, CostName.RELATIVE_LENGTH_ERROR)

    assert evaluation.additive_value == pytest.approx(1.5)
    assert evaluation.bottleneck_value == pytest.approx(1.0)
    assert evaluation.length_weighted_additive_value == pytest.approx(6.0)


def test_length_weighted_match_keeps_unweighted_edge_costs() -> None:
    source = make_graph("source", {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (11.0, 0.0)}, ((10, 0, 1), (11, 1, 2),))
    target = make_graph("target", {10: (0.0, 0.0), 20: (2.0, 0.0), 30: (7.0, 0.0)}, ((100, 10, 20), (101, 20, 30),))
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,), 2: (30,)})

    result = matcher.match(CostName.RELATIVE_LENGTH_ERROR, Objective.LENGTH_WEIGHTED_ADDITIVE)

    assert result.solution is not None
    assert result.solution.objective is Objective.LENGTH_WEIGHTED_ADDITIVE
    assert [edge.cost for edge in result.solution.edges] == pytest.approx([1.0, 0.5])
    assert result.solution.value == pytest.approx(6.0)


def test_parallel_source_edges_remain_distinct_in_match_result() -> None:
    source = make_graph("source", {0: (0.0, 0.0), 1: (1.0, 0.0)}, ((11, 0, 1), (10, 0, 1),))
    target = make_graph("target", {10: (0.0, 0.0), 20: (1.0, 0.0)}, ((100, 10, 20),))
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,)})

    result = matcher.match("relative_length_error", "additive")

    assert result.solution is not None
    assert result.solution.value == pytest.approx(0.0)
    assert [edge.edge_id for edge in result.solution.edges] == [10, 11]
    assert result.dp_statistics.unique_cost_requests == 2

    for edge in result.solution.edges:
        assert edge.source_u == 0
        assert edge.source_v == 1
        assert edge.target_u == 10
        assert edge.target_v == 20
        assert edge.cost == pytest.approx(0.0)


def test_source_edge_orientation_is_preserved() -> None:
    source = make_graph("source", {0: (0.0, 0.0), 1: (1.0, 0.0)}, ((10, 1, 0),))
    target = make_graph("target", {10: (0.0, 0.0), 20: (1.0, 0.0)}, ((100, 10, 20),))
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,)})

    result = matcher.match("relative_length_error", "additive")

    assert result.solution is not None
    matched_edge = result.solution.edges[0]

    assert matched_edge.source_u == 1
    assert matched_edge.source_v == 0
    assert matched_edge.target_u == 20
    assert matched_edge.target_v == 10

    np.testing.assert_allclose(matched_edge.witness, [[1.0, 0.0], [0.0, 0.0]])


def test_unreachable_candidate_assignment_is_infeasible() -> None:
    source, target = make_graphs()
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,), 2: (40,)})

    result = matcher.match("relative_length_error", "additive")

    assert not result.feasible
    assert result.solution is None
    assert result.candidate_statistics.empty_domains == 0


def test_missing_candidate_domain_is_normalized_to_empty() -> None:
    source, target = make_graphs()
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,)})

    result = matcher.match("relative_length_error", "additive")

    assert matcher.candidate_sets == {0: (10,), 1: (20,), 2: ()}
    assert result.candidate_statistics.empty_domains == 1
    assert result.solution is None


def test_candidates_are_sorted_deduplicated_and_summarized() -> None:
    source, target = make_graphs()
    matcher = RiverGraphMatcher(source, target, candidate_sets={0: (20, 10, 20, 10), 1: (20,), 2: (30,)})

    statistics = matcher.candidate_statistics

    assert matcher.candidate_sets == {0: (10, 20), 1: (20,), 2: (30,)}
    assert statistics.source_vertices == 3
    assert statistics.empty_domains == 0
    assert statistics.total_candidates == 4
    assert statistics.minimum_candidates == 1
    assert statistics.maximum_candidates == 2


def test_unknown_source_vertex_in_candidates_is_rejected() -> None:
    source, target = make_graphs()

    with pytest.raises(ValueError, match="unknown source vertices"):
        RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,), 2: (30,), 999: (10,)})


def test_unknown_target_vertex_in_candidates_is_rejected() -> None:
    source, target = make_graphs()

    with pytest.raises(ValueError, match="unknown target vertices"):
        RiverGraphMatcher(source, target, candidate_sets={0: (10,), 1: (20,), 2: (999,)})


def test_candidate_generation_receives_requested_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = make_graphs()
    calls: list[tuple[JunctionGraph, JunctionGraph, float, int]] = []

    def fake_compute_candidate_sets(received_source: JunctionGraph, received_target: JunctionGraph, *, rho: float, top_k: int) -> CandidateSets:
        calls.append((received_source, received_target, rho, top_k,))
        return perfect_candidates()

    monkeypatch.setattr("river_matcher.matcher.compute_candidate_sets", fake_compute_candidate_sets)

    matcher = RiverGraphMatcher(source, target, candidate_rho=3.5, top_k=7)

    assert calls == [(source, target, 3.5, 7,)]
    assert matcher.candidate_sets == perfect_candidates()


def test_convenience_functions_prepare_and_solve_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = make_graphs()
    calls: list[tuple[float, int]] = []

    def fake_compute_candidate_sets(received_source: JunctionGraph, received_target: JunctionGraph, *, rho: float, top_k: int) -> CandidateSets:
        assert received_source is source
        assert received_target is target
        calls.append((rho, top_k))
        return perfect_candidates()

    monkeypatch.setattr("river_matcher.matcher.compute_candidate_sets", fake_compute_candidate_sets)

    single = match_graphs(source, target, "relative_length_error", "additive", candidate_rho=4.0, top_k=8)
    both = match_graphs_both(source, target, "relative_length_error", candidate_rho=5.0, top_k=9)

    assert single.solution is not None
    assert single.solution.value == pytest.approx(0.0)

    assert both.additive is not None
    assert both.bottleneck is not None
    assert both.additive.value == pytest.approx(0.0)
    assert both.bottleneck.value == pytest.approx(0.0)

    assert calls == [(4.0, 8), (5.0, 9)]


def test_supplied_decomposition_is_validated_against_source() -> None:
    source, target = make_graphs()
    decomposition = build_source_decomposition(source)
    incompatible_source = make_graph("incompatible", {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}, ((99, 0, 2),))

    with pytest.raises(ValueError, match="No tree-decomposition bag covers source edge"):
        RiverGraphMatcher(incompatible_source, target, candidate_sets=perfect_candidates(), decomposition=decomposition)
