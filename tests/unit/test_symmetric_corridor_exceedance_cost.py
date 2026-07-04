from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.costs import (
    CostFactory,
    CostName,
    SymmetricCorridorExceedance,
)
from river_matcher.costs.symmetric_corridor_exceedance import (
    _directed_mean_corridor_exceedance,
)
from river_matcher.models import JunctionEdge, JunctionGraph

type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]], ) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64), )


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0), },
                           edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0), ], ), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), ], ),), )
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (1.0, 1.0), 30: (2.0, 0.0), 40: (0.0, 1.0), 50: (2.0, 1.0), 60: (10.0, 0.0), 70: (11.0, 0.0), },
                           edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.0, 1.0), ], ), make_edge(11, 20, 30, [(1.0, 1.0), (2.0, 0.0), ], ),
                                  make_edge(12, 40, 50, [(0.0, 1.0), (2.0, 1.0), ], ), make_edge(13, 60, 70, [(10.0, 0.0), (11.0, 0.0), ], ),), )

    return source, target


def make_cost(*, rho: float = 2.0, edge_samples: int = 12, curve_samples: int = 33, corridor_radius: float | None = 0.5, ) -> SymmetricCorridorExceedance:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(CostName.SYMMETRIC_CORRIDOR_EXCEEDANCE, rho=rho, edge_samples=edge_samples, curve_samples=curve_samples,
                                              corridor_radius=corridor_radius, )

    assert isinstance(cost, SymmetricCorridorExceedance)
    return cost


def line_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray,]:
    starts = np.asarray([[0.0, 0.0]], dtype=np.float64, )
    vectors = np.asarray([[2.0, 0.0]], dtype=np.float64, )
    squared_lengths = np.asarray([4.0], dtype=np.float64, )

    return starts, vectors, squared_lengths


def test_numba_kernel_returns_zero_inside_corridor() -> None:
    points = np.asarray([[0.0, 0.25], [1.0, 0.25], [2.0, 0.25], ], dtype=np.float64, )
    starts, vectors, squared_lengths = line_segments()

    value = _directed_mean_corridor_exceedance(points, starts, vectors, squared_lengths, 0.5, )

    assert value == pytest.approx(0.0)


def test_numba_kernel_returns_zero_on_corridor_boundary() -> None:
    points = np.asarray([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5], ], dtype=np.float64, )
    starts, vectors, squared_lengths = line_segments()

    value = _directed_mean_corridor_exceedance(points, starts, vectors, squared_lengths, 0.5, )

    assert value == pytest.approx(0.0)


def test_numba_kernel_returns_one_at_twice_corridor_radius() -> None:
    points = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], ], dtype=np.float64, )
    starts, vectors, squared_lengths = line_segments()

    value = _directed_mean_corridor_exceedance(points, starts, vectors, squared_lengths, 0.5, )

    assert value == pytest.approx(1.0)


def test_numba_kernel_averages_point_contributions() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 1.0], ], dtype=np.float64, )
    starts, vectors, squared_lengths = line_segments()

    value = _directed_mean_corridor_exceedance(points, starts, vectors, squared_lengths, 0.5, )

    assert value == pytest.approx(0.5)


def test_factory_creates_symmetric_corridor_cost() -> None:
    cost = make_cost()

    assert (cost.name is CostName.SYMMETRIC_CORRIDOR_EXCEEDANCE)
    assert cost.label == "symmetric mean corridor exceedance"


def test_identical_source_and_witness_have_zero_cost() -> None:
    cost = make_cost()

    value = cost(1, 1, 2, 10, 30, )

    assert value == pytest.approx(0.0)


def test_parallel_lines_inside_corridor_have_zero_cost() -> None:
    cost = make_cost(corridor_radius=1.0)

    value = cost(0, 1, 2, 40, 50, )

    assert value == pytest.approx(0.0)


def test_parallel_lines_at_twice_radius_have_unit_cost() -> None:
    cost = make_cost(corridor_radius=0.5)

    value = cost(0, 1, 2, 40, 50, )

    assert value == pytest.approx(1.0)


def test_corridor_radius_defaults_to_witness_radius() -> None:
    cost = make_cost(rho=1.0, corridor_radius=None, )

    assert cost.corridor_radius == pytest.approx(1.0)
    assert cost(0, 1, 2, 40, 50, ) == pytest.approx(0.0)


def test_cost_is_specific_to_parallel_source_edge() -> None:
    cost = make_cost(corridor_radius=0.25)

    straight_source = cost(0, 1, 2, 10, 30, )
    curved_source = cost(1, 1, 2, 10, 30, )

    assert straight_source > 0.0
    assert curved_source == pytest.approx(0.0)


def test_cost_is_invariant_under_reversing_both_mappings() -> None:
    cost = make_cost(corridor_radius=0.25)

    forward = cost(0, 1, 2, 10, 30, )
    reverse = cost(0, 2, 1, 30, 10, )

    assert reverse == pytest.approx(forward)


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


@pytest.mark.parametrize("corridor_radius", [0.0, -1.0, float("inf"), float("-inf"), float("nan"), ], )
def test_invalid_corridor_radius_is_rejected(corridor_radius: float, ) -> None:
    with pytest.raises(ValueError, match="Corridor radius must be positive and finite", ):
        make_cost(corridor_radius=corridor_radius)


@pytest.mark.parametrize("cost_request", [(999, 1, 2, 10, 30), (1, 1, 999, 10, 30), (1, 1, 2, 10, 10), (1, 1, 2, 10, 999), (1, 1, 2, 10, 60), ], )
def test_invalid_or_unreachable_requests_return_infinity(cost_request: CostRequest, ) -> None:
    cost = make_cost()

    assert math.isinf(cost(*cost_request))
    assert cost.witness(*cost_request) is None


def test_batch_preserves_request_order() -> None:
    cost = make_cost(corridor_radius=0.5)
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
    assert len(cost._source_sample_cache) == 1
    assert len(cost._source_prepared_cache) == 1


def test_factory_reuses_guided_finder_for_matching_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("symmetric_corridor_exceedance", rho=2.0, edge_samples=8, curve_samples=16, corridor_radius=0.5, )
    second = factory.create("symmetric_corridor_exceedance", rho=2.0, edge_samples=8, curve_samples=64, corridor_radius=2.0, )

    assert isinstance(first, SymmetricCorridorExceedance)
    assert isinstance(second, SymmetricCorridorExceedance)
    assert first._finder is second._finder


def test_factory_separates_guided_finders_with_different_options() -> None:
    source, target = make_graphs()
    factory = CostFactory(source, target)

    first = factory.create("symmetric_corridor_exceedance", rho=1.0, edge_samples=8, )
    second = factory.create("symmetric_corridor_exceedance", rho=2.0, edge_samples=8, )
    third = factory.create("symmetric_corridor_exceedance", rho=1.0, edge_samples=16, )

    assert isinstance(first, SymmetricCorridorExceedance)
    assert isinstance(second, SymmetricCorridorExceedance)
    assert isinstance(third, SymmetricCorridorExceedance)

    assert first._finder is not second._finder
    assert first._finder is not third._finder
    assert second._finder is not third._finder


def test_clear_cache_removes_local_cost_state() -> None:
    cost = make_cost()

    cost(1, 1, 2, 10, 30, )

    assert cost._cost_cache
    assert cost._witness_cache
    assert cost._source_sample_cache
    assert cost._source_prepared_cache

    cost.clear_cache()

    assert cost._cost_cache == {}
    assert cost._witness_cache == {}
    assert cost._source_sample_cache == {}
    assert cost._source_prepared_cache == {}


def directed_mean_corridor_exceedance_reference(sample_points: np.ndarray, segment_starts: np.ndarray, segment_vectors: np.ndarray, segment_squared_lengths: np.ndarray,
                                                corridor_radius: float, ) -> float:
    total = 0.0

    for point in sample_points:
        best_distance = math.inf

        for start, vector, squared_length in zip(segment_starts, segment_vectors, segment_squared_lengths, strict=True, ):
            projection = float(np.dot(point - start, vector) / squared_length)
            projection = min(max(projection, 0.0), 1.0)

            closest = start + projection * vector
            distance = float(np.linalg.norm(closest - point))
            best_distance = min(best_distance, distance)

        if best_distance > corridor_radius:
            total += (best_distance / corridor_radius - 1.0)

    return total / len(sample_points)


def test_numba_kernel_matches_python_reference() -> None:
    rng = np.random.default_rng(20260704)

    for _ in range(50):
        sample_points = rng.normal(size=(17, 2))
        segment_starts = rng.normal(size=(9, 2))
        segment_vectors = rng.normal(size=(9, 2))

        # Prevent invalid zero-length segments.
        short = (np.linalg.norm(segment_vectors, axis=1) < 0.1)
        segment_vectors[short, 0] += 1.0

        segment_squared_lengths = np.sum(segment_vectors * segment_vectors, axis=1, )
        corridor_radius = float(rng.uniform(0.1, 3.0))

        expected = (directed_mean_corridor_exceedance_reference(sample_points, segment_starts, segment_vectors, segment_squared_lengths, corridor_radius, ))
        actual = _directed_mean_corridor_exceedance(sample_points, segment_starts, segment_vectors, segment_squared_lengths, corridor_radius, )

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, )
