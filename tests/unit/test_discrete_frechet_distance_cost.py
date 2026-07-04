from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from river_matcher.costs import (
    CostFactory,
    CostName,
    DiscreteFrechetDistance,
)
from river_matcher.models import JunctionEdge, JunctionGraph

type FloatArray = NDArray[np.float64]
type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]], ) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64), )


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0), },
        edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0), ], ), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), ], ),), )
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (1.0, 1.0), 30: (2.0, 0.0), 40: (0.0, 1.0), 50: (2.0, 1.0), 60: (10.0, 0.0), 70: (11.0, 0.0), },
        edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.0, 1.0), ], ), make_edge(11, 20, 30, [(1.0, 1.0), (2.0, 0.0), ], ), make_edge(12, 40, 50, [(0.0, 1.0), (2.0, 1.0), ], ),
               make_edge(13, 60, 70, [(10.0, 0.0), (11.0, 0.0), ], ),), )

    return source, target


def make_cost(*, rho: float = 2.0, edge_samples: int = 12, curve_samples: int = 33, ) -> DiscreteFrechetDistance:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(CostName.DISCRETE_FRECHET_DISTANCE, rho=rho, edge_samples=edge_samples, curve_samples=curve_samples, )

    assert isinstance(cost, DiscreteFrechetDistance)
    return cost


def discrete_frechet_reference(first: FloatArray, second: FloatArray, ) -> float:
    """
    Independent Eiter-Mannila dynamic program using Euclidean point distance.
    """
    rows = len(first)
    columns = len(second)
    table = np.full((rows, columns), np.inf, dtype=np.float64, )

    for row in range(rows):
        for column in range(columns):
            distance = float(np.linalg.norm(first[row] - second[column]))

            if row == 0 and column == 0:
                table[row, column] = distance
            elif row == 0:
                table[row, column] = max(table[row, column - 1], distance, )
            elif column == 0:
                table[row, column] = max(table[row - 1, column], distance, )
            else:
                previous = min(table[row - 1, column], table[row - 1, column - 1], table[row, column - 1], )
                table[row, column] = max(previous, distance, )

    return float(table[-1, -1])


def test_factory_creates_discrete_frechet_cost() -> None:
    cost = make_cost()

    assert (cost.name is CostName.DISCRETE_FRECHET_DISTANCE)
    assert cost.label == "sampled discrete Fréchet distance"


def test_identical_source_and_witness_have_zero_cost() -> None:
    cost = make_cost()

    value = cost(1, 1, 2, 10, 30, )

    assert value == pytest.approx(0.0)


def test_parallel_lines_have_unit_discrete_frechet_distance() -> None:
    cost = make_cost()

    value = cost(0, 1, 2, 40, 50, )

    assert value == pytest.approx(1.0)


def test_cost_is_specific_to_parallel_source_edge() -> None:
    cost = make_cost()

    straight_source = cost(0, 1, 2, 10, 30, )
    curved_source = cost(1, 1, 2, 10, 30, )

    assert straight_source > 0.0
    assert curved_source == pytest.approx(0.0)


def test_cost_is_invariant_under_reversing_both_mappings() -> None:
    cost = make_cost()

    forward = cost(1, 1, 2, 10, 30, )
    reverse = cost(1, 2, 1, 30, 10, )

    assert reverse == pytest.approx(forward)


def test_source_sample_cache_stores_both_orientations() -> None:
    cost = make_cost()

    forward = cost._source_samples(1, 1, 2, )
    reverse = cost._source_samples(1, 2, 1, )

    assert forward is not None
    assert reverse is not None
    assert not reverse.flags.writeable

    np.testing.assert_allclose(reverse, forward[::-1], )

    assert set(cost._source_sample_cache) == {(1, 1, 2), (1, 2, 1), }


def test_witness_returns_source_guided_target_path() -> None:
    cost = make_cost()

    witness = cost.witness(1, 1, 2, 10, 30, )

    assert witness is not None
    assert not witness.flags.writeable

    np.testing.assert_allclose(witness, [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], ], )


def test_reverse_witness_reverses_geometry() -> None:
    cost = make_cost()

    forward = cost.witness(1, 1, 2, 10, 30, )
    reverse = cost.witness(1, 2, 1, 30, 10, )

    assert forward is not None
    assert reverse is not None

    np.testing.assert_allclose(reverse, forward[::-1], )


def test_curve_sample_count_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="Curve sample count must be at least 2", ):
        make_cost(curve_samples=1)


@pytest.mark.parametrize("cost_request", [(999, 1, 2, 10, 30), (1, 1, 999, 10, 30), (1, 1, 2, 10, 10), (1, 1, 2, 10, 999), (1, 1, 2, 10, 60), ], )
def test_invalid_or_unreachable_requests_return_infinity(cost_request: CostRequest, ) -> None:
    cost = make_cost()

    assert math.isinf(cost(*cost_request))
    assert cost.witness(*cost_request) is None


def test_batch_preserves_request_order() -> None:
    cost = make_cost()
    requests: list[CostRequest] = [(1, 1, 2, 10, 30), (0, 1, 2, 40, 50), (0, 1, 2, 10, 60), ]

    values = cost.batch(requests)

    assert values.dtype == np.float64
    assert values.shape == (3,)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)
    assert math.isinf(values[2])


def test_repeated_requests_use_local_caches() -> None:
    cost = make_cost()
    cost_request: CostRequest = (1, 1, 2, 10, 30,)

    first = cost(*cost_request)
    first_witness = cost.witness(*cost_request)
    second = cost(*cost_request)
    second_witness = cost.witness(*cost_request)

    assert first == second == pytest.approx(0.0)
    assert first_witness is second_witness
    assert len(cost._cost_cache) == 1
    assert len(cost._witness_cache) == 1
    assert len(cost._source_sample_cache) == 2


def test_factory_reuses_guided_finder_for_matching_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("discrete_frechet_distance", rho=2.0, edge_samples=8, curve_samples=16, )
    second = factory.create("discrete_frechet_distance", rho=2.0, edge_samples=8, curve_samples=64, )

    assert isinstance(first, DiscreteFrechetDistance)
    assert isinstance(second, DiscreteFrechetDistance)
    assert first._finder is second._finder


def test_factory_separates_guided_finders_with_different_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("discrete_frechet_distance", rho=1.0, edge_samples=8, )
    second = factory.create("discrete_frechet_distance", rho=2.0, edge_samples=8, )
    third = factory.create("discrete_frechet_distance", rho=1.0, edge_samples=16, )

    assert isinstance(first, DiscreteFrechetDistance)
    assert isinstance(second, DiscreteFrechetDistance)
    assert isinstance(third, DiscreteFrechetDistance)

    assert first._finder is not second._finder
    assert first._finder is not third._finder
    assert second._finder is not third._finder


def test_clear_cache_removes_local_cost_state() -> None:
    cost = make_cost()

    cost(1, 1, 2, 10, 30, )

    assert cost._cost_cache
    assert cost._witness_cache
    assert cost._source_sample_cache

    cost.clear_cache()

    assert cost._cost_cache == {}
    assert cost._witness_cache == {}
    assert cost._source_sample_cache == {}


def test_library_matches_independent_reference() -> None:
    rng = np.random.default_rng(20260704)

    from curvesimilarities import dfd

    for _ in range(40):
        first = np.ascontiguousarray(rng.normal(size=(11, 2)), dtype=np.float64, )
        second = np.ascontiguousarray(rng.normal(size=(13, 2)), dtype=np.float64, )

        expected = discrete_frechet_reference(first, second, )
        actual = float(dfd(first, second, ))

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, )
