from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.costs import CostFactory, CostName, LogLengthDistortion, RelativeLengthError, available_costs, create_cost
from river_matcher.models import JunctionEdge, JunctionGraph

type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64))


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0)},
                           edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0)]), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),))
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (3.0, 0.0), 30: (10.0, 0.0), 40: (11.0, 0.0)},
                           edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]), make_edge(11, 30, 40, [(10.0, 0.0), (11.0, 0.0)]),))

    return source, target


def test_factory_creates_both_implemented_costs() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    relative = factory.create(CostName.RELATIVE_LENGTH_ERROR)
    logarithmic = factory.create("log_length_distortion")

    assert isinstance(relative, RelativeLengthError)
    assert isinstance(logarithmic, LogLengthDistortion)


def test_factory_costs_share_graph_resources() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    relative = factory.create("relative_length_error")
    logarithmic = factory.create("log_length_distortion")

    assert relative.resources is logarithmic.resources
    assert (relative.resources.shortest_path is logarithmic.resources.shortest_path)


def test_available_costs_only_lists_implemented_costs() -> None:
    assert available_costs() == (CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION, CostName.MEAN_DISTANCE_TANGENT, CostName.HAUSDORFF_DISTANCE,
                                 CostName.SYMMETRIC_CORRIDOR_EXCEEDANCE, CostName.DISCRETE_FRECHET_DISTANCE,)


def test_factory_rejects_unknown_cost_name() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    with pytest.raises(ValueError, match="Unknown cost"):
        factory.create("not-a-cost")


def test_factory_reports_named_but_unimplemented_cost() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        factory.create(CostName.DYNAMIC_TIME_WARPING_DISTANCE)


def test_standalone_factory_function_constructs_cost() -> None:
    source, target = make_graphs()

    cost = create_cost("relative_length_error", source, target)

    assert isinstance(cost, RelativeLengthError)


def test_relative_length_error_uses_shortest_path_length() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("relative_length_error")

    value = cost(0, 1, 2, 10, 20)

    assert value == pytest.approx(0.5)


def test_relative_length_error_is_source_edge_specific() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("relative_length_error")

    straight = cost(0, 1, 2, 10, 20)
    curved = cost(1, 1, 2, 10, 20)

    curved_length = 2.0 * math.sqrt(2.0)

    assert straight == pytest.approx(0.5)
    assert curved == pytest.approx(abs(3.0 / curved_length - 1.0))
    assert curved != pytest.approx(straight)


def test_log_length_distortion_uses_absolute_log_ratio() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("log_length_distortion")

    value = cost(0, 1, 2, 10, 20)

    assert value == pytest.approx(abs(math.log(3.0 / 2.0)))


def test_log_length_distortion_is_source_edge_specific() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("log_length_distortion")

    curved_length = 2.0 * math.sqrt(2.0)
    value = cost(1, 1, 2, 10, 20)

    assert value == pytest.approx(abs(math.log(3.0 / curved_length)))


@pytest.mark.parametrize("name", [CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION])
def test_length_costs_accept_reversed_orientations(name: CostName) -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(name)

    forward = cost(0, 1, 2, 10, 20)
    reverse = cost(0, 2, 1, 20, 10)

    assert reverse == pytest.approx(forward)


@pytest.mark.parametrize("name", [CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION])
def test_length_costs_return_shortest_path_witness(name: CostName) -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(name)

    path = cost.witness(0, 1, 2, 10, 20)

    assert path is not None
    assert not path.flags.writeable

    np.testing.assert_allclose(path, [[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]])


@pytest.mark.parametrize("name", [CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION])
def test_reversed_witness_has_reversed_geometry(name: CostName) -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(name)

    forward = cost.witness(0, 1, 2, 10, 20)
    reverse = cost.witness(0, 2, 1, 20, 10)

    assert forward is not None
    assert reverse is not None

    np.testing.assert_allclose(reverse, forward[::-1])


@pytest.mark.parametrize("cost_request", [(999, 1, 2, 10, 20), (0, 1, 999, 10, 20), (0, 1, 2, 10, 10), (0, 1, 2, 10, 999), (0, 1, 2, 10, 30)])
@pytest.mark.parametrize("name", [CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION])
def test_invalid_or_unreachable_requests_return_infinity(name: CostName, cost_request: CostRequest) -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(name)

    assert math.isinf(cost(*cost_request))
    assert cost.witness(*cost_request) is None


@pytest.mark.parametrize("name", [CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION])
def test_batch_preserves_request_order(name: CostName) -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(name)
    requests: list[CostRequest] = [(0, 1, 2, 10, 20), (1, 1, 2, 10, 20), (0, 1, 2, 10, 30)]

    values = cost.batch(requests)

    assert values.dtype == np.float64
    assert values.shape == (3,)
    assert values[0] == pytest.approx(cost(*requests[0]))
    assert values[1] == pytest.approx(cost(*requests[1]))
    assert math.isinf(values[2])


def test_edge_ok_uses_inclusive_threshold() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("relative_length_error")

    assert cost.edge_ok(0, 1, 2, 10, 20, threshold=0.5)
    assert not cost.edge_ok(0, 1, 2, 10, 20, threshold=0.499)


def test_repeated_requests_use_cost_cache() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("relative_length_error")
    request: CostRequest = (0, 1, 2, 10, 20,)

    first = cost(*request)
    second = cost(*request)

    assert first == second
    assert len(cost._cost_cache) == 1


def test_two_costs_reuse_same_shortest_path_tree() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)
    relative = factory.create("relative_length_error")
    logarithmic = factory.create("log_length_distortion")

    relative(0, 1, 2, 10, 20)
    logarithmic(0, 1, 2, 10, 20)

    assert (len(factory.resources.shortest_path._tree_cache) == 1)


def test_clear_cache_removes_cost_and_witness_entries() -> None:
    source, target = make_graphs()
    cost = CostFactory(source, target).create("relative_length_error")

    cost(0, 1, 2, 10, 20)
    cost.witness(0, 1, 2, 10, 20)

    assert cost._cost_cache
    assert cost._witness_cache

    cost.clear_cache()

    assert cost._cost_cache == {}
    assert cost._witness_cache == {}
