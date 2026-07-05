from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.costs import CostFactory, CostName, MeanDistanceTangent
from river_matcher.costs.mean_distance_tangent import _directed_mean_distance_tangent, _sample_prepared_curve
from river_matcher.geometry import prepare_polyline_segments
from river_matcher.models import JunctionEdge, JunctionGraph

type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64))


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0)},
        edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0)]), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])))
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (1.0, 1.0), 30: (2.0, 0.0), 40: (0.0, 1.0), 50: (2.0, 1.0), 60: (10.0, 0.0), 70: (11.0, 0.0)},
        edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.0, 1.0)]), make_edge(11, 20, 30, [(1.0, 1.0), (2.0, 0.0)]), make_edge(12, 40, 50, [(0.0, 1.0), (2.0, 1.0)]),
               make_edge(13, 60, 70, [(10.0, 0.0), (11.0, 0.0)]),))

    return source, target


def make_cost(*, rho: float = 2.0, edge_samples: int = 12, curve_samples: int = 33, tangent_weight: float = 1.0) -> MeanDistanceTangent:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(CostName.MEAN_DISTANCE_TANGENT, rho=rho, edge_samples=edge_samples, curve_samples=curve_samples, tangent_weight=tangent_weight)

    assert isinstance(cost, MeanDistanceTangent)
    return cost


def test_numba_kernel_returns_zero_for_identical_line() -> None:
    sample_points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    sample_tangents = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    segment_starts = np.asarray([[0.0, 0.0]], dtype=np.float64)
    segment_vectors = np.asarray([[2.0, 0.0]], dtype=np.float64)
    segment_squared_lengths = np.asarray([4.0], dtype=np.float64)

    value = _directed_mean_distance_tangent(sample_points, sample_tangents, segment_starts, segment_vectors, segment_squared_lengths, 3.0)

    assert value == pytest.approx(0.0)


def test_numba_kernel_applies_full_penalty_to_perpendicular_tangent() -> None:
    sample_points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    sample_tangents = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    segment_starts = np.asarray([[0.0, 0.0]], dtype=np.float64)
    segment_vectors = np.asarray([[2.0, 0.0]], dtype=np.float64)
    segment_squared_lengths = np.asarray([4.0], dtype=np.float64)

    value = _directed_mean_distance_tangent(sample_points, sample_tangents, segment_starts, segment_vectors, segment_squared_lengths, 3.0)

    assert value == pytest.approx(3.0)


def test_numba_kernel_treats_antiparallel_tangents_as_aligned() -> None:
    sample_points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    sample_tangents = np.asarray([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    segment_starts = np.asarray([[0.0, 0.0]], dtype=np.float64)
    segment_vectors = np.asarray([[2.0, 0.0]], dtype=np.float64)
    segment_squared_lengths = np.asarray([4.0], dtype=np.float64)

    value = _directed_mean_distance_tangent(sample_points, sample_tangents, segment_starts, segment_vectors, segment_squared_lengths, 10.0)

    assert value == pytest.approx(0.0)


def test_numba_kernel_returns_mean_point_to_segment_distance() -> None:
    sample_points = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    sample_tangents = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    segment_starts = np.asarray([[0.0, 0.0]], dtype=np.float64)
    segment_vectors = np.asarray([[2.0, 0.0]], dtype=np.float64)
    segment_squared_lengths = np.asarray([4.0], dtype=np.float64)

    value = _directed_mean_distance_tangent(sample_points, sample_tangents, segment_starts, segment_vectors, segment_squared_lengths, 5.0)

    assert value == pytest.approx(2.0)


def test_factory_creates_mean_distance_tangent_cost() -> None:
    cost = make_cost()

    assert cost.name is CostName.MEAN_DISTANCE_TANGENT
    assert cost.label == "symmetric mean distance and tangent error"


def test_identical_source_and_witness_geometry_have_zero_cost() -> None:
    cost = make_cost()

    value = cost(1, 1, 2, 10, 30)

    assert value == pytest.approx(0.0)


def test_parallel_lines_have_unit_mean_distance() -> None:
    cost = make_cost(tangent_weight=10.0)

    value = cost(0, 1, 2, 40, 50)

    assert value == pytest.approx(1.0)


def test_cost_is_specific_to_parallel_source_edge() -> None:
    cost = make_cost()

    straight_source = cost(0, 1, 2, 10, 30)
    curved_source = cost(1, 1, 2, 10, 30)

    assert straight_source > 0.0
    assert curved_source == pytest.approx(0.0)


def test_tangent_weight_increases_cost_for_directional_mismatch() -> None:
    no_tangent_penalty = make_cost(tangent_weight=0.0)
    weighted = make_cost(tangent_weight=5.0)

    unweighted_value = no_tangent_penalty(0, 1, 2, 10, 30)
    weighted_value = weighted(0, 1, 2, 10, 30)

    assert weighted_value > unweighted_value


def test_cost_is_invariant_under_reversing_both_mappings() -> None:
    cost = make_cost()

    forward = cost(1, 1, 2, 10, 30)
    reverse = cost(1, 2, 1, 30, 10)

    assert reverse == pytest.approx(forward)


def test_witness_returns_source_guided_target_path() -> None:
    cost = make_cost()

    witness = cost.witness(1, 1, 2, 10, 30)

    assert witness is not None
    assert not witness.flags.writeable

    np.testing.assert_allclose(witness, [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])


def test_reverse_witness_reverses_geometry() -> None:
    cost = make_cost()

    forward = cost.witness(1, 1, 2, 10, 30)
    reverse = cost.witness(1, 2, 1, 30, 10)

    assert forward is not None
    assert reverse is not None

    np.testing.assert_allclose(reverse, forward[::-1])


def test_curve_sample_count_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="Curve sample count must be at least 2"):
        make_cost(curve_samples=1)


@pytest.mark.parametrize("tangent_weight", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_tangent_weight_is_rejected(tangent_weight: float) -> None:
    with pytest.raises(ValueError, match="Tangent weight must be finite and nonnegative"):
        make_cost(tangent_weight=tangent_weight)


@pytest.mark.parametrize("cost_request", [(999, 1, 2, 10, 30), (1, 1, 999, 10, 30), (1, 1, 2, 10, 10), (1, 1, 2, 10, 999), (1, 1, 2, 10, 60)])
def test_invalid_or_unreachable_requests_return_infinity(cost_request: CostRequest) -> None:
    cost = make_cost()

    assert math.isinf(cost(*cost_request))
    assert cost.witness(*cost_request) is None


def test_batch_preserves_request_order() -> None:
    cost = make_cost()
    requests: list[CostRequest] = [(1, 1, 2, 10, 30), (0, 1, 2, 40, 50), (0, 1, 2, 10, 60)]

    values = cost.batch(requests)

    assert values.dtype == np.float64
    assert values.shape == (3,)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)
    assert math.isinf(values[2])


def test_repeated_requests_use_local_caches() -> None:
    cost = make_cost()
    cost_request: CostRequest = (1, 1, 2, 10, 30)

    first = cost(*cost_request)
    first_witness = cost.witness(*cost_request)
    second = cost(*cost_request)
    second_witness = cost.witness(*cost_request)

    assert first == second == pytest.approx(0.0)
    assert first_witness is second_witness
    assert len(cost._cost_cache) == 1
    assert len(cost._witness_cache) == 1
    assert len(cost._source_sample_cache) == 1
    assert len(cost._source_prepared_cache) == 1


def test_factory_reuses_guided_finder_for_matching_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("mean_distance_tangent", rho=2.0, edge_samples=8, curve_samples=16, tangent_weight=1.0)
    second = factory.create("mean_distance_tangent", rho=2.0, edge_samples=8, curve_samples=64, tangent_weight=5.0)

    assert isinstance(first, MeanDistanceTangent)
    assert isinstance(second, MeanDistanceTangent)
    assert first._finder is second._finder


def test_factory_separates_guided_finders_with_different_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("mean_distance_tangent", rho=1.0, edge_samples=8)
    second = factory.create("mean_distance_tangent", rho=2.0, edge_samples=8)
    third = factory.create("mean_distance_tangent", rho=1.0, edge_samples=16)

    assert isinstance(first, MeanDistanceTangent)
    assert isinstance(second, MeanDistanceTangent)
    assert isinstance(third, MeanDistanceTangent)

    assert first._finder is not second._finder
    assert first._finder is not third._finder
    assert second._finder is not third._finder


def test_clear_cache_removes_local_cost_state() -> None:
    cost = make_cost()

    cost(1, 1, 2, 10, 30)

    assert cost._cost_cache
    assert cost._witness_cache
    assert cost._source_sample_cache
    assert cost._source_prepared_cache

    cost.clear_cache()

    assert cost._cost_cache == {}
    assert cost._witness_cache == {}
    assert cost._source_sample_cache == {}
    assert cost._source_prepared_cache == {}


def test_sampling_accepts_distinct_arclength_positions_with_equal_coordinates() -> None:
    polyline = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    prepared = prepare_polyline_segments(polyline)

    assert prepared is not None

    sampled = _sample_prepared_curve(prepared, samples=3)

    assert sampled is not None
    points, tangents = sampled

    np.testing.assert_allclose(points, [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), np.ones(3))
    assert np.all(np.isfinite(tangents))
