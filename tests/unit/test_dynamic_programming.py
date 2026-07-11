from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest

from river_matcher.decomposition import build_source_decomposition
from river_matcher.dynamic_programming import DPSolution, Objective, solve_tree_dp, solve_tree_dp_both
from river_matcher.models import JunctionEdge, JunctionGraph

type EdgeSpec = tuple[int, int, int]
type CostRequest = tuple[int, int, int, int, int]
type CandidateSets = Mapping[int, tuple[int, ...]]
type CostTable = Mapping[CostRequest, float]


class TableCost:
    def __init__(self, values: CostTable, *, default: float = math.inf) -> None:
        self.values = dict(values)
        self.default = default
        self.calls: list[CostRequest] = []

    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float:
        request = edge_id, source_u, source_v, target_u, target_v
        self.calls.append(request)
        return self.values.get(request, self.default)


def objective_value(source: JunctionGraph, mapping: Mapping[int, int], values: CostTable, objective: Objective, edge_weights: Mapping[int, float] | None = None) -> float:
    weights = {} if edge_weights is None else edge_weights
    edge_values: list[float] = []
    for edge in source.edges:
        request = edge.id, edge.u, edge.v, mapping[edge.u], mapping[edge.v]
        value = float(values.get(request, math.inf))
        if not math.isfinite(value):
            return math.inf
        edge_values.append(weights.get(edge.id, 1.0) * value if objective is Objective.LENGTH_WEIGHTED_ADDITIVE else value)
    if objective is Objective.ADDITIVE:
        return sum(edge_values)
    if objective is Objective.LENGTH_WEIGHTED_ADDITIVE:
        return sum(edge_values)
    return max(edge_values, default=0.0)


def make_source(vertices: tuple[int, ...], edges: tuple[EdgeSpec, ...]) -> JunctionGraph:
    coordinates = {vertex: (float(vertex), 0.0) for vertex in vertices}

    return JunctionGraph(
        name="source",
        coordinates=coordinates,
        edges=tuple(JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray([coordinates[u], coordinates[v]], dtype=np.float64)) for edge_id, u, v in edges),
    )


def brute_force_optimum(source: JunctionGraph, candidate_sets: CandidateSets, values: CostTable, objective: Objective,
                        edge_weights: Mapping[int, float] | None = None) -> float | None:
    vertices = tuple(sorted(source.vertices))
    domains = tuple(tuple(sorted(set(candidate_sets.get(vertex, ())))) for vertex in vertices)

    if any(not domain for domain in domains):
        return None

    best = math.inf

    for state in product(*domains):
        mapping = dict(zip(vertices, state, strict=True))
        best = min(best, objective_value(source, mapping, values, objective, edge_weights))

    return None if not math.isfinite(best) else best


def assert_matches_brute_force(source: JunctionGraph, candidate_sets: CandidateSets, values: CostTable, objective: Objective, solution: DPSolution | None,
                               edge_weights: Mapping[int, float] | None = None) -> None:
    expected = brute_force_optimum(source, candidate_sets, values, objective, edge_weights)

    if expected is None:
        assert solution is None
        return

    assert solution is not None
    assert solution.objective is objective
    assert solution.value == pytest.approx(expected)
    assert set(solution.mapping) == set(source.vertices)

    for vertex, target in solution.mapping.items():
        assert target in candidate_sets[vertex]

    actual = objective_value(source, solution.mapping, values, objective, edge_weights)
    assert actual == pytest.approx(expected)


def test_additive_and_bottleneck_can_select_different_mappings() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1), (11, 0, 1)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 11), 1: (20, 21)}
    values = {
        (10, 0, 1, 10, 20): 0.0,
        (11, 0, 1, 10, 20): 9.0,
        (10, 0, 1, 11, 21): 5.0,
        (11, 0, 1, 11, 21): 5.0,
        (10, 0, 1, 10, 21): 100.0,
        (11, 0, 1, 10, 21): 100.0,
        (10, 0, 1, 11, 20): 100.0,
        (11, 0, 1, 11, 20): 100.0,
    }
    cost = TableCost(values)

    result = solve_tree_dp_both(decomposition, candidate_sets, cost)

    assert result.additive is not None
    assert result.bottleneck is not None

    assert result.additive.value == pytest.approx(9.0)
    assert result.additive.mapping == {0: 10, 1: 20}

    assert result.bottleneck.value == pytest.approx(5.0)
    assert result.bottleneck.mapping == {0: 11, 1: 21}

    assert result.statistics.enumerated_states == 4
    assert result.statistics.feasible_states == 4
    assert result.statistics.message_entries == 1
    assert result.statistics.unique_cost_requests == 8

    assert len(cost.calls) == 8
    assert len(set(cost.calls)) == 8
    assert {request[0] for request in cost.calls} == {10, 11}


def test_single_objective_solvers_match_joint_solver() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1), (11, 0, 1)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 11), 1: (20, 21)}
    values = {
        (10, 0, 1, 10, 20): 0.0,
        (11, 0, 1, 10, 20): 9.0,
        (10, 0, 1, 11, 21): 5.0,
        (11, 0, 1, 11, 21): 5.0,
        (10, 0, 1, 10, 21): 100.0,
        (11, 0, 1, 10, 21): 100.0,
        (10, 0, 1, 11, 20): 100.0,
        (11, 0, 1, 11, 20): 100.0,
    }

    both = solve_tree_dp_both(decomposition, candidate_sets, TableCost(values))
    additive = solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.ADDITIVE)
    bottleneck = solve_tree_dp(decomposition, candidate_sets, TableCost(values), "bottleneck")

    assert additive.solution == both.additive
    assert bottleneck.solution == both.bottleneck


def test_length_weighted_additive_uses_edge_weights_without_changing_local_costs() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1), (11, 0, 1)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 11), 1: (20, 21)}
    values = {
        (10, 0, 1, 10, 20): 0.0,
        (11, 0, 1, 10, 20): 9.0,
        (10, 0, 1, 11, 21): 5.0,
        (11, 0, 1, 11, 21): 5.0,
        (10, 0, 1, 10, 21): 100.0,
        (11, 0, 1, 10, 21): 100.0,
        (10, 0, 1, 11, 20): 100.0,
        (11, 0, 1, 11, 20): 100.0,
    }
    edge_weights = {10: 1.0, 11: 10.0}

    result = solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.LENGTH_WEIGHTED_ADDITIVE, edge_weights=edge_weights)

    assert_matches_brute_force(source, candidate_sets, values, Objective.LENGTH_WEIGHTED_ADDITIVE, result.solution, edge_weights)
    assert result.solution is not None
    assert result.solution.value == pytest.approx(55.0)
    assert result.solution.mapping == {0: 11, 1: 21}


def test_length_weighted_additive_defaults_missing_weights_to_one() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1), (11, 0, 1)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 11), 1: (20, 21)}
    values = {
        (10, 0, 1, 10, 20): 0.0,
        (11, 0, 1, 10, 20): 9.0,
        (10, 0, 1, 11, 21): 5.0,
        (11, 0, 1, 11, 21): 5.0,
    }

    result = solve_tree_dp(decomposition, candidate_sets, TableCost(values), "length_weighted_additive")

    assert result.solution is not None
    assert result.solution.value == pytest.approx(9.0)
    assert result.solution.mapping == {0: 10, 1: 20}


def test_length_weighted_additive_combines_child_messages() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 1, 2)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 20), 1: (10, 20), 2: (10, 20)}
    values = {
        (edge.id, edge.u, edge.v, target_u, target_v): float((edge.id - 9) * abs(target_u - target_v) // 10 + target_v // 10)
        for edge in source.edges
        for target_u in candidate_sets[edge.u]
        for target_v in candidate_sets[edge.v]
    }
    edge_weights = {10: 7.0, 11: 2.0}

    assert decomposition.bag_count > 1

    result = solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.LENGTH_WEIGHTED_ADDITIVE, edge_weights=edge_weights)

    assert_matches_brute_force(source, candidate_sets, values, Objective.LENGTH_WEIGHTED_ADDITIVE, result.solution, edge_weights)


def test_source_edge_orientation_is_preserved_in_cost_request() -> None:
    source = make_source(vertices=(1, 2), edges=((10, 2, 1),))
    decomposition = build_source_decomposition(source)
    candidate_sets = {1: (100,), 2: (200,)}
    request = 10, 2, 1, 200, 100
    cost = TableCost({request: 3.5})

    result = solve_tree_dp(decomposition, candidate_sets, cost, Objective.ADDITIVE)

    assert result.solution is not None
    assert result.solution.value == pytest.approx(3.5)
    assert result.solution.mapping == {1: 100, 2: 200}
    assert cost.calls == [request]


def test_noninjective_mapping_is_allowed() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (7, 8), 1: (7, 8)}
    values = {(10, 0, 1, 7, 7): 0.0, (10, 0, 1, 7, 8): 5.0, (10, 0, 1, 8, 7): 5.0, (10, 0, 1, 8, 8): 0.0}

    result = solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.ADDITIVE)

    assert result.solution is not None
    assert result.solution.value == pytest.approx(0.0)
    assert result.solution.mapping[0] == result.solution.mapping[1]


def test_duplicate_candidates_are_removed_and_ties_are_deterministic() -> None:
    source = make_source(vertices=(0,), edges=())
    decomposition = build_source_decomposition(source)

    result = solve_tree_dp(decomposition, {0: (9, 3, 9, 5)}, TableCost({}), Objective.ADDITIVE)

    assert result.solution is not None
    assert result.solution.value == pytest.approx(0.0)
    assert result.solution.mapping == {0: 3}
    assert result.statistics.enumerated_states == 3
    assert result.statistics.unique_cost_requests == 0


@pytest.mark.parametrize("candidate_sets", [{0: (), 1: (10,)}, {0: (10,)}])
def test_empty_or_missing_candidate_domain_is_infeasible(candidate_sets: CandidateSets) -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),))
    decomposition = build_source_decomposition(source)

    result = solve_tree_dp(decomposition, candidate_sets, TableCost({}), Objective.ADDITIVE)

    assert not result.feasible
    assert result.solution is None
    assert result.statistics.enumerated_states == 0
    assert result.statistics.unique_cost_requests == 0


@pytest.mark.parametrize("invalid_value", [math.inf, math.nan])
def test_nonfinite_local_cost_makes_state_infeasible(invalid_value: float) -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10,), 1: (20,)}
    values = {(10, 0, 1, 10, 20): invalid_value}

    result = solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.ADDITIVE)

    assert result.solution is None
    assert result.statistics.enumerated_states == 1
    assert result.statistics.feasible_states == 0


def test_negative_local_cost_is_rejected() -> None:
    source = make_source(vertices=(0, 1), edges=((10, 0, 1),))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10,), 1: (20,)}
    values = {(10, 0, 1, 10, 20): -1.0}

    with pytest.raises(ValueError, match="Local edge costs must be nonnegative"):
        solve_tree_dp(decomposition, candidate_sets, TableCost(values), Objective.ADDITIVE)


def test_separator_recurrence_matches_brute_force_on_fixed_instance() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 0, 1), (12, 1, 2)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 20), 1: (10, 20), 2: (10, 20)}
    values = {
        (edge.id, edge.u, edge.v, target_u, target_v): float(abs(target_u - target_v) // 10 + edge.id - 10)
        for edge in source.edges
        for target_u in candidate_sets[edge.u]
        for target_v in candidate_sets[edge.v]
    }

    result = solve_tree_dp_both(decomposition, candidate_sets, TableCost(values))

    assert_matches_brute_force(source, candidate_sets, values, Objective.ADDITIVE, result.additive)
    assert_matches_brute_force(source, candidate_sets, values, Objective.BOTTLENECK, result.bottleneck)


def test_random_instances_match_exhaustive_enumeration() -> None:
    source = make_source(vertices=(0, 1, 2), edges=((10, 0, 1), (11, 0, 1), (12, 1, 2)))
    decomposition = build_source_decomposition(source)
    candidate_sets = {0: (10, 20, 30), 1: (10, 20, 30), 2: (10, 20, 30)}
    rng = np.random.default_rng(20260704)

    for _ in range(50):
        values: dict[CostRequest, float] = {}

        for edge in source.edges:
            for target_u in candidate_sets[edge.u]:
                for target_v in candidate_sets[edge.v]:
                    request = edge.id, edge.u, edge.v, target_u, target_v

                    if rng.random() < 0.12:
                        values[request] = math.inf
                    else:
                        values[request] = float(rng.integers(0, 25))

        result = solve_tree_dp_both(decomposition, candidate_sets, TableCost(values))
        assert_matches_brute_force(source, candidate_sets, values, Objective.ADDITIVE, result.additive)
        assert_matches_brute_force(source, candidate_sets, values, Objective.BOTTLENECK, result.bottleneck)
