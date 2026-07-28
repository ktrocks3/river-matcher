from __future__ import annotations

import numpy as np
import pytest

from river_matcher.visualization import display_bounds, display_coordinates


def test_display_coordinates_flips_y_without_mutating_model_geometry() -> None:
    model_coordinates = np.asarray(((2.0, 3.0), (5.0, -7.0)), dtype=np.float64)

    displayed = display_coordinates(model_coordinates)

    np.testing.assert_array_equal(displayed, ((2.0, -3.0), (5.0, 7.0)))
    np.testing.assert_array_equal(model_coordinates, ((2.0, 3.0), (5.0, -7.0)))
    assert not np.shares_memory(displayed, model_coordinates)


def test_display_coordinates_accepts_a_single_point() -> None:
    displayed = display_coordinates((12.5, 8.0))

    np.testing.assert_array_equal(displayed, (12.5, -8.0))


def test_display_coordinates_rejects_non_xy_data() -> None:
    with pytest.raises(ValueError, match="shape"):
        display_coordinates((1.0, 2.0, 3.0))


def test_display_bounds_preserves_increasing_axis_limits() -> None:
    assert display_bounds((200.0, 500.0, 100.0, 320.0)) == (200.0, 500.0, -320.0, -100.0)
