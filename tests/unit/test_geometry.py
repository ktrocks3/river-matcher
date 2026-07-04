from __future__ import annotations

import math

import numpy as np
import pytest

from river_matcher.geometry import (
    PreparedPolyline,
    as_xy_array,
    closest_segment_distance_and_tangent,
    orient_polyline,
    point_to_polyline_distance,
    point_to_prepared_polyline_distance,
    points_to_prepared_polyline_distances,
    polyline_length,
    prepare_polyline_segments,
    sample_polyline_by_arclength,
    sample_polyline_with_tangents,
)


def test_as_xy_array_normalizes_dtype_layout_and_extra_columns() -> None:
    points = as_xy_array([[0, 1, 100], [2, 3, 200]])

    assert points is not None
    assert points.shape == (2, 2)
    assert points.dtype == np.float64
    assert points.flags.c_contiguous

    np.testing.assert_array_equal(points, np.asarray([[0.0, 1.0], [2.0, 3.0]]))


def test_as_xy_array_removes_consecutive_duplicate_points() -> None:
    points = as_xy_array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert points is not None

    np.testing.assert_array_equal(points, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_as_xy_array_removes_near_duplicate_points() -> None:
    points = as_xy_array([[0.0, 0.0], [5e-13, 0.0], [1.0, 0.0]])

    assert points is not None

    np.testing.assert_array_equal(points, np.asarray([[0.0, 0.0], [1.0, 0.0]]))


@pytest.mark.parametrize(
    "polyline",
    [
        None,
        [],
        [0.0, 1.0],
        [[0.0, 0.0]],
        [[[0.0, 0.0], [1.0, 1.0]]],
        [[0.0], [1.0]],
        [[0.0, 0.0], [float("nan"), 1.0]],
        [[0.0, 0.0], [float("inf"), 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        "not-a-polyline",
    ],
)
def test_as_xy_array_rejects_invalid_polylines(polyline: object) -> None:
    assert as_xy_array(polyline) is None


def test_orient_polyline_preserves_forward_orientation() -> None:
    points = orient_polyline([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], start_xy=(0.0, 0.0), end_xy=(2.0, 0.0))

    assert points is not None

    np.testing.assert_array_equal(points, np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))


def test_orient_polyline_reverses_when_endpoint_error_is_lower() -> None:
    points = orient_polyline([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]], start_xy=(0.0, 0.0), end_xy=(2.0, 0.0))

    assert points is not None

    np.testing.assert_array_equal(points, np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))


def test_orient_polyline_preserves_input_orientation_on_tie() -> None:
    original = np.asarray([[0.0, 0.0], [1.0, 0.0]])

    points = orient_polyline(original, start_xy=(0.5, 0.0), end_xy=(0.5, 0.0))

    assert points is not None
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize(
    ("polyline", "start", "end"),
    [
        (None, (0.0, 0.0), (1.0, 0.0)),
        ([[0.0, 0.0], [1.0, 0.0]], None, (1.0, 0.0)),
        ([[0.0, 0.0], [1.0, 0.0]], (0.0, 0.0), None),
        ([[0.0, 0.0], [1.0, 0.0]], (float("nan"), 0.0), (1.0, 0.0)),
    ],
)
def test_orient_polyline_rejects_invalid_inputs(polyline: object, start: object, end: object) -> None:
    assert orient_polyline(polyline, start, end) is None


def test_polyline_length_sums_all_segment_lengths() -> None:
    length = polyline_length([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

    assert length == pytest.approx(10.0)


def test_polyline_length_removes_duplicate_points() -> None:
    length = polyline_length([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])

    assert length == pytest.approx(5.0)


def test_polyline_length_returns_infinity_for_invalid_input() -> None:
    assert math.isinf(polyline_length([[1.0, 1.0], [1.0, 1.0]]))


def test_sample_polyline_by_arclength_on_straight_segment() -> None:
    sampled = sample_polyline_by_arclength([[0.0, 0.0], [4.0, 0.0]], samples=5)

    assert sampled is not None

    np.testing.assert_allclose(sampled, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]))


def test_sample_polyline_by_arclength_across_bend() -> None:
    sampled = sample_polyline_by_arclength([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], samples=5)

    assert sampled is not None

    np.testing.assert_allclose(sampled, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]))


def test_sample_polyline_by_arclength_returns_at_least_two_samples() -> None:
    sampled = sample_polyline_by_arclength([[0.0, 0.0], [2.0, 0.0]], samples=1)

    assert sampled is not None
    assert sampled.shape == (2, 2)

    np.testing.assert_allclose(sampled, np.asarray([[0.0, 0.0], [2.0, 0.0]]))


def test_sample_polyline_by_arclength_rejects_invalid_polyline() -> None:
    assert sample_polyline_by_arclength([[1.0, 1.0]], 10) is None


def test_sample_polyline_with_tangents_across_bend() -> None:
    sampled, tangents = sample_polyline_with_tangents([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], samples=5)

    assert sampled is not None
    assert tangents is not None

    np.testing.assert_allclose(sampled, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]))

    np.testing.assert_allclose(tangents, np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))


def test_sample_polyline_with_tangents_rejects_invalid_polyline() -> None:
    sampled, tangents = sample_polyline_with_tangents([[1.0, 1.0]], samples=10)

    assert sampled is None
    assert tangents is None


def test_prepare_polyline_segments_returns_expected_arrays() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])

    assert prepared is not None
    starts, vectors, squared_lengths = prepared

    np.testing.assert_array_equal(starts, np.asarray([[0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(vectors, np.asarray([[2.0, 0.0], [0.0, 3.0]]))
    np.testing.assert_array_equal(squared_lengths, np.asarray([4.0, 9.0]))

    assert starts.dtype == np.float64
    assert vectors.dtype == np.float64
    assert squared_lengths.dtype == np.float64
    assert starts.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert squared_lengths.flags.c_contiguous


def test_prepare_polyline_segments_discards_very_short_segments() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [1e-7, 0.0], [1.0, 0.0]])

    assert prepared is not None
    starts, vectors, squared_lengths = prepared

    assert starts.shape == (1, 2)
    assert vectors.shape == (1, 2)
    assert squared_lengths.shape == (1,)

    np.testing.assert_allclose(starts[0], [1e-7, 0.0])
    np.testing.assert_allclose(vectors[0], [1.0 - 1e-7, 0.0])


def test_prepare_polyline_segments_rejects_only_short_segments() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [1e-7, 0.0]])

    assert prepared is None


def test_point_to_prepared_polyline_distance_uses_segment_projection() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [4.0, 0.0]])

    distance = point_to_prepared_polyline_distance((2.0, 3.0), prepared)

    assert distance == pytest.approx(3.0)


def test_point_to_prepared_polyline_distance_clamps_to_endpoint() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [4.0, 0.0]])

    distance = point_to_prepared_polyline_distance((7.0, 4.0), prepared)

    assert distance == pytest.approx(5.0)


def test_point_to_prepared_polyline_distance_selects_nearest_segment() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    distance = point_to_prepared_polyline_distance((3.0, 1.0), prepared)

    assert distance == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("point", "prepared"),
    [(None, None), ((0.0, 0.0), None), (None, prepare_polyline_segments([[0.0, 0.0], [1.0, 0.0]])), ((float("nan"), 0.0), prepare_polyline_segments([[0.0, 0.0], [1.0, 0.0]]))],
)
def test_point_to_prepared_polyline_distance_returns_infinity_for_invalid_input(point: object, prepared: PreparedPolyline | None) -> None:
    assert math.isinf(point_to_prepared_polyline_distance(point, prepared))


def test_points_to_prepared_polyline_distances_matches_scalar_queries() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    points = np.asarray([[1.0, 1.0], [3.0, 1.0], [-1.0, 0.0], [2.0, 2.0]])

    batched = points_to_prepared_polyline_distances(points, prepared, chunk_size=2)

    assert batched is not None

    scalar = np.asarray([point_to_prepared_polyline_distance(point, prepared) for point in points])

    np.testing.assert_allclose(batched, scalar)
    np.testing.assert_allclose(batched, [1.0, 1.0, 1.0, 0.0])


def test_points_to_prepared_polyline_distances_handles_one_point_chunks() -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [4.0, 0.0]])

    distances = points_to_prepared_polyline_distances([[0.0, 1.0], [2.0, 2.0], [4.0, 3.0]], prepared, chunk_size=1)

    assert distances is not None
    np.testing.assert_allclose(distances, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("points", [None, [], [0.0, 1.0], [[[0.0, 0.0]]], [[float("nan"), 0.0]]])
def test_points_to_prepared_polyline_distances_rejects_invalid_points(points: object) -> None:
    prepared = prepare_polyline_segments([[0.0, 0.0], [1.0, 0.0]])

    assert points_to_prepared_polyline_distances(points, prepared) is None


def test_points_to_prepared_polyline_distances_rejects_missing_preparation() -> None:
    assert points_to_prepared_polyline_distances([[0.0, 0.0]], None) is None


def test_point_to_polyline_distance_prepares_and_queries_polyline() -> None:
    distance = point_to_polyline_distance(point=(2.0, 3.0), polyline=[[0.0, 0.0], [4.0, 0.0]])

    assert distance == pytest.approx(3.0)


def test_point_to_polyline_distance_returns_infinity_for_invalid_polyline() -> None:
    distance = point_to_polyline_distance(point=(0.0, 0.0), polyline=[[1.0, 1.0]])

    assert math.isinf(distance)


def test_closest_segment_distance_and_tangent() -> None:
    distance, tangent = closest_segment_distance_and_tangent(point=(3.0, 1.0), polyline=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    assert distance == pytest.approx(1.0)
    assert tangent is not None
    np.testing.assert_allclose(tangent, [0.0, 1.0])


def test_closest_segment_tie_uses_first_stored_segment() -> None:
    distance, tangent = closest_segment_distance_and_tangent(point=(1.0, 0.0), polyline=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    assert distance == pytest.approx(0.0)
    assert tangent is not None
    np.testing.assert_allclose(tangent, [1.0, 0.0])


def test_closest_segment_distance_and_tangent_rejects_invalid_input() -> None:
    distance, tangent = closest_segment_distance_and_tangent(point=None, polyline=[[0.0, 0.0], [1.0, 0.0]])

    assert math.isinf(distance)
    assert tangent is None
