from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.costs import CostFactory, CostName, HausdorffDistance
from river_matcher.models import JunctionEdge, JunctionGraph

type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64))


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0), },
                           edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0)]), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),))
    target = JunctionGraph(name="target", coordinates={10: (0.0, 1.0), 20: (2.0, 1.0), 30: (0.0, 0.0), 40: (1.0, 1.0), 50: (2.0, 0.0), },
                           edges=(make_edge(10, 10, 20, [(0.0, 1.0), (2.0, 1.0)]), make_edge(11, 30, 40, [(0.0, 0.0), (1.0, 1.0)]),
                                  make_edge(12, 40, 50, [(1.0, 1.0), (2.0, 0.0)]),))

    return source, target


def make_cost(*, rho: float = 2.0, edge_samples: int = 12, densify: float | None = None) -> HausdorffDistance:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    cost = factory.create(CostName.HAUSDORFF_DISTANCE, rho=rho, edge_samples=edge_samples, densify=densify)

    assert isinstance(cost, HausdorffDistance)
    return cost


def test_factory_creates_hausdorff_cost() -> None:
    cost = make_cost()

    assert cost.name is CostName.HAUSDORFF_DISTANCE
    assert cost.label == "source-guided Hausdorff distance"


def test_parallel_lines_have_unit_hausdorff_distance() -> None:
    cost = make_cost()

    value = cost(0, 1, 2, 10, 20)

    assert value == pytest.approx(1.0)


def test_identical_source_and_witness_geometry_have_zero_cost() -> None:
    cost = make_cost()

    value = cost(1, 1, 2, 30, 50)

    assert value == pytest.approx(0.0)


def test_cost_is_specific_to_parallel_source_edge() -> None:
    cost = make_cost()

    straight_source = cost(0, 1, 2, 30, 50)
    curved_source = cost(1, 1, 2, 30, 50)

    assert straight_source == pytest.approx(1.0)
    assert curved_source == pytest.approx(0.0)


def test_cost_is_invariant_under_reversing_both_mappings() -> None:
    cost = make_cost()

    forward = cost(1, 1, 2, 30, 50)
    reverse = cost(1, 2, 1, 50, 30)

    assert reverse == pytest.approx(forward)


def test_witness_returns_source_guided_target_path() -> None:
    cost = make_cost()

    witness = cost.witness(1, 1, 2, 30, 50)

    assert witness is not None
    assert not witness.flags.writeable

    np.testing.assert_allclose(witness, [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])


def test_reverse_witness_reverses_geometry() -> None:
    cost = make_cost()

    forward = cost.witness(1, 1, 2, 30, 50)
    reverse = cost.witness(1, 2, 1, 50, 30)

    assert forward is not None
    assert reverse is not None

    np.testing.assert_allclose(reverse, forward[::-1])


def test_densified_hausdorff_distance_is_supported() -> None:
    cost = make_cost(densify=0.5)

    value = cost(0, 1, 2, 30, 50)

    assert value == pytest.approx(1.0)


@pytest.mark.parametrize("densify", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_invalid_densify_is_rejected(densify: float) -> None:
    with pytest.raises(ValueError, match="Hausdorff densify must be greater than 0 and at most 1"):
        make_cost(densify=densify)


@pytest.mark.parametrize("cost_request", [(999, 1, 2, 30, 50), (1, 1, 999, 30, 50), (1, 1, 2, 30, 30), (1, 1, 2, 30, 999), (1, 1, 2, 10, 30)])
def test_invalid_or_unreachable_requests_return_infinity(cost_request: CostRequest) -> None:
    cost = make_cost()

    assert math.isinf(cost(*cost_request))
    assert cost.witness(*cost_request) is None


def test_batch_preserves_request_order() -> None:
    cost = make_cost()
    requests: list[CostRequest] = [(0, 1, 2, 10, 20), (1, 1, 2, 30, 50), (0, 1, 2, 10, 30)]

    values = cost.batch(requests)

    assert values.dtype == np.float64
    assert values.shape == (3,)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.0)
    assert math.isinf(values[2])


def test_repeated_requests_use_cost_and_witness_caches() -> None:
    cost = make_cost()
    cost_request: CostRequest = (1, 1, 2, 30, 50,)

    first = cost(*cost_request)
    first_witness = cost.witness(*cost_request)
    second = cost(*cost_request)
    second_witness = cost.witness(*cost_request)

    assert first == second == pytest.approx(0.0)
    assert first_witness is second_witness
    assert len(cost._cost_cache) == 1
    assert len(cost._witness_cache) == 1
    assert len(cost._source_geometry_cache) == 1


def test_factory_reuses_guided_finder_for_matching_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("hausdorff_distance", rho=2.0, edge_samples=8)
    second = factory.create("hausdorff_distance", rho=2.0, edge_samples=8, densify=0.5)

    assert isinstance(first, HausdorffDistance)
    assert isinstance(second, HausdorffDistance)
    assert first._finder is second._finder


def test_factory_separates_guided_finders_with_different_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("hausdorff_distance", rho=1.0, edge_samples=8)
    second = factory.create("hausdorff_distance", rho=2.0, edge_samples=8)
    third = factory.create("hausdorff_distance", rho=1.0, edge_samples=16)

    assert isinstance(first, HausdorffDistance)
    assert isinstance(second, HausdorffDistance)
    assert isinstance(third, HausdorffDistance)

    assert first._finder is not second._finder
    assert first._finder is not third._finder
    assert second._finder is not third._finder


def test_clear_cache_removes_local_cost_state() -> None:
    cost = make_cost()

    cost(1, 1, 2, 30, 50)

    assert cost._cost_cache
    assert cost._witness_cache
    assert cost._source_geometry_cache

    cost.clear_cache()

    assert cost._cost_cache == {}
    assert cost._witness_cache == {}
    assert cost._source_geometry_cache == {}
