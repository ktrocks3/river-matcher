from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from river_matcher.models import JunctionEdge, JunctionGraph
from river_matcher.witnesses import ShortestPathWitnessFinder, SourceGuidedWitnessFinder


def make_edge(edge_id: int, u: int, v: int, polyline: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(polyline, dtype=np.float64))


def make_graph(name: str, coordinates: dict[int, tuple[float, float]], edges: tuple[JunctionEdge, ...] = ()) -> JunctionGraph:
    return JunctionGraph(name=name, coordinates=coordinates, edges=edges)


def shortest_path_target() -> JunctionGraph:
    return make_graph(
        "target",
        {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (0.0, 2.0), 40: (2.0, 0.0)},
        (
            make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 0.0)]),
            make_edge(1, 20, 40, [(1.0, 0.0), (2.0, 0.0)]),
            make_edge(2, 10, 30, [(0.0, 0.0), (0.0, 2.0)]),
            make_edge(3, 30, 40, [(0.0, 2.0), (2.0, 0.0)]),
        ),
    )


def arch_source() -> JunctionGraph:
    return make_graph("source", {1: (0.0, 0.0), 2: (4.0, 0.0)}, (make_edge(7, 1, 2, [(0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)]),))


def arch_target() -> JunctionGraph:
    return make_graph(
        "target",
        {10: (0.0, 0.0), 20: (1.0, 2.0), 30: (3.0, 2.0), 40: (4.0, 0.0)},
        (
            make_edge(0, 10, 40, [(0.0, 0.0), (4.0, 0.0)]),
            make_edge(1, 10, 20, [(0.0, 0.0), (1.0, 2.0)]),
            make_edge(2, 20, 30, [(1.0, 2.0), (3.0, 2.0)]),
            make_edge(3, 30, 40, [(3.0, 2.0), (4.0, 0.0)]),
        ),
    )


def parallel_source_graph() -> JunctionGraph:
    return make_graph(
        "source", {1: (0.0, 0.0), 2: (4.0, 0.0)}, (make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 2.0), (4.0, 0.0)]), make_edge(1, 1, 2, [(0.0, 0.0), (2.0, -2.0), (4.0, 0.0)]))
    )


def parallel_route_target() -> JunctionGraph:
    return make_graph(
        "target",
        {10: (0.0, 0.0), 20: (2.0, 2.0), 30: (2.0, -2.0), 40: (4.0, 0.0)},
        (
            make_edge(0, 10, 20, [(0.0, 0.0), (2.0, 2.0)]),
            make_edge(1, 20, 40, [(2.0, 2.0), (4.0, 0.0)]),
            make_edge(2, 10, 30, [(0.0, 0.0), (2.0, -2.0)]),
            make_edge(3, 30, 40, [(2.0, -2.0), (4.0, 0.0)]),
        ),
    )


def test_shortest_path_returns_geometric_distance_and_polyline() -> None:
    finder = ShortestPathWitnessFinder(shortest_path_target())

    distance = finder.distance(10, 40)
    path = finder.path(10, 40)

    assert distance == pytest.approx(2.0)
    assert path is not None

    np.testing.assert_allclose(path, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_shortest_path_reverse_request_reuses_cached_geometry() -> None:
    finder = ShortestPathWitnessFinder(shortest_path_target())

    forward = finder.path(10, 40)
    tree_count = len(finder._tree_cache)
    reverse = finder.path(40, 10)

    assert forward is not None
    assert reverse is not None
    assert len(finder._tree_cache) == tree_count == 1

    np.testing.assert_allclose(reverse, forward[::-1])


def test_shortest_path_parallel_edges_keep_selected_geometry() -> None:
    target = make_graph("parallel", {10: (0.0, 0.0), 20: (2.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]), make_edge(1, 10, 20, [(0.0, 0.0), (2.0, 0.0)])))
    finder = ShortestPathWitnessFinder(target)

    path = finder.path(10, 20)

    assert path is not None
    assert finder.distance(10, 20) == pytest.approx(2.0)

    np.testing.assert_allclose(path, [[0.0, 0.0], [2.0, 0.0]])


def test_equal_length_parallel_edges_use_first_stored_edge() -> None:
    target = make_graph(
        "parallel-tie", {10: (0.0, 0.0), 20: (2.0, 0.0)}, (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]), make_edge(1, 10, 20, [(0.0, 0.0), (1.0, -1.0), (2.0, 0.0)]))
    )
    finder = ShortestPathWitnessFinder(target)

    path = finder.path(10, 20)

    assert path is not None

    np.testing.assert_allclose(path, [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])


def test_path_concatenation_removes_repeated_junction_point() -> None:
    target = make_graph(
        "detailed",
        {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (2.0, 0.0)},
        (make_edge(0, 10, 20, [(0.0, 0.0), (0.5, 0.2), (1.0, 0.0)]), make_edge(1, 20, 30, [(1.0, 0.0), (1.5, -0.2), (2.0, 0.0)])),
    )
    finder = ShortestPathWitnessFinder(target)

    path = finder.path(10, 30)

    assert path is not None
    assert path.shape == (5, 2)

    np.testing.assert_allclose(path, [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0], [1.5, -0.2], [2.0, 0.0]])


def test_shortest_path_handles_same_invalid_and_disconnected_vertices() -> None:
    target = make_graph(
        "disconnected",
        {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (10.0, 0.0), 40: (11.0, 0.0)},
        (make_edge(0, 10, 20, [(0.0, 0.0), (1.0, 0.0)]), make_edge(1, 30, 40, [(10.0, 0.0), (11.0, 0.0)])),
    )
    finder = ShortestPathWitnessFinder(target)

    assert finder.distance(10, 10) == pytest.approx(0.0)
    assert finder.path(10, 10) is None

    assert math.isinf(finder.distance(10, 40))
    assert finder.path(10, 40) is None

    assert math.isinf(finder.distance(10, 999))
    assert finder.path(10, 999) is None


def test_shortest_path_batch_reuses_start_tree() -> None:
    finder = ShortestPathWitnessFinder(shortest_path_target())

    paths = finder.paths([(10, 40), (40, 10), (10, 30)])

    assert set(paths) == {(10, 40), (40, 10), (10, 30)}
    assert all(path is not None for path in paths.values())
    assert len(finder._tree_cache) == 1


def test_returned_shortest_path_is_read_only() -> None:
    finder = ShortestPathWitnessFinder(shortest_path_target())

    path = finder.path(10, 40)

    assert path is not None
    assert not path.flags.writeable

    with pytest.raises(ValueError):
        path[0, 0] = 100.0


def test_source_guided_path_prefers_corridor_over_shorter_route() -> None:
    source = arch_source()
    target = arch_target()

    ordinary = ShortestPathWitnessFinder(target)
    guided = SourceGuidedWitnessFinder(source, target, rho=0.1, edge_samples=12)

    ordinary_path = ordinary.path(10, 40)
    guided_path = guided.path(7, 1, 2, 10, 40)

    assert ordinary_path is not None
    assert guided_path is not None

    np.testing.assert_allclose(ordinary_path, [[0.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(guided_path, [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])


def test_source_polyline_is_oriented_by_requested_source_endpoints() -> None:
    finder = SourceGuidedWitnessFinder(arch_source(), arch_target(), rho=0.1)

    forward = finder.source_polyline(7, 1, 2)
    reverse = finder.source_polyline(7, 2, 1)

    assert forward is not None
    assert reverse is not None
    assert not forward.flags.writeable
    assert not reverse.flags.writeable

    np.testing.assert_allclose(reverse, forward[::-1])

    assert finder.source_polyline(7, 1, 999) is None
    assert finder.source_polyline(999, 1, 2) is None


def test_parallel_source_edges_select_different_target_routes() -> None:
    finder = SourceGuidedWitnessFinder(parallel_source_graph(), parallel_route_target(), rho=0.1, edge_samples=12)

    upper = finder.path(0, 1, 2, 10, 40)
    lower = finder.path(1, 1, 2, 10, 40)

    assert upper is not None
    assert lower is not None

    np.testing.assert_allclose(upper, [[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]])
    np.testing.assert_allclose(lower, [[0.0, 0.0], [2.0, -2.0], [4.0, 0.0]])


def test_source_guided_reverse_request_uses_reverse_cache_entry() -> None:
    finder = SourceGuidedWitnessFinder(parallel_source_graph(), parallel_route_target(), rho=0.1)

    forward = finder.path(0, 1, 2, 10, 40)
    runs_after_forward = finder.timing.dijkstra_runs
    reverse = finder.path(0, 2, 1, 40, 10)

    assert forward is not None
    assert reverse is not None
    assert finder.timing.dijkstra_runs == runs_after_forward == 1

    np.testing.assert_allclose(reverse, forward[::-1])


def test_source_guided_cache_reuses_adjacency_and_start_tree() -> None:
    finder = SourceGuidedWitnessFinder(parallel_source_graph(), parallel_route_target(), rho=0.1)

    assert finder.path(0, 1, 2, 10, 20) is not None
    assert finder.path(0, 1, 2, 10, 40) is not None

    assert finder.timing.adjacency_builds == 1
    assert finder.timing.dijkstra_runs == 1

    assert finder.path(0, 1, 2, 20, 40) is not None

    assert finder.timing.adjacency_builds == 1
    assert finder.timing.dijkstra_runs == 2


def test_source_guided_clear_cache_resets_caches_and_timing() -> None:
    finder = SourceGuidedWitnessFinder(parallel_source_graph(), parallel_route_target(), rho=0.1)

    assert finder.path(0, 1, 2, 10, 40) is not None
    assert finder.timing.adjacency_builds == 1
    assert finder.timing.dijkstra_runs == 1

    finder.clear_cache()

    assert finder.timing.adjacency_builds == 0
    assert finder.timing.dijkstra_runs == 0
    assert finder._adjacency_cache == {}
    assert finder._tree_cache == {}
    assert finder._path_cache == {}

    assert finder.path(0, 1, 2, 10, 40) is not None
    assert finder.timing.adjacency_builds == 1
    assert finder.timing.dijkstra_runs == 1


@pytest.mark.parametrize("rho", [0.0, -1.0, float("inf"), float("nan")])
def test_source_guided_rejects_invalid_radius(rho: float) -> None:
    with pytest.raises(ValueError, match="Witness radius must be positive and finite"):
        SourceGuidedWitnessFinder(arch_source(), arch_target(), rho=rho)


def test_source_guided_rejects_too_few_edge_samples() -> None:
    with pytest.raises(ValueError, match="Witness edge sample count must be at least 2"):
        SourceGuidedWitnessFinder(arch_source(), arch_target(), rho=1.0, edge_samples=1)


def test_source_guided_batch_rejects_wrong_request_length() -> None:
    finder = SourceGuidedWitnessFinder(arch_source(), arch_target(), rho=0.1)
    invalid_requests: Any = [(7, 1, 2, 10)]

    with pytest.raises(ValueError, match="Each witness request must contain"):
        finder.paths(invalid_requests)


def test_source_guided_invalid_requests_return_none() -> None:
    finder = SourceGuidedWitnessFinder(arch_source(), arch_target(), rho=0.1)

    assert finder.path(999, 1, 2, 10, 40) is None
    assert finder.path(7, 1, 999, 10, 40) is None
    assert finder.path(7, 1, 2, 10, 10) is None
    assert finder.path(7, 1, 2, 10, 999) is None


def test_witness_finders_handle_target_without_edges() -> None:
    source = arch_source()
    target = make_graph("empty-target", {10: (0.0, 0.0), 20: (1.0, 0.0)})

    ordinary = ShortestPathWitnessFinder(target)
    guided = SourceGuidedWitnessFinder(source, target, rho=1.0)

    assert math.isinf(ordinary.distance(10, 20))
    assert ordinary.path(10, 20) is None
    assert guided.path(7, 1, 2, 10, 20) is None
