import math
from typing import Any

import numpy as np
import pytest

from river_matcher.models import JunctionEdge, JunctionGraph, MatchResult, Objective


def make_edge(edge_id=0, u=1, v=2, polyline=None):
    if polyline is None:
        polyline = [[0.0, 0.0], [3.0, 4.0]]

    return JunctionEdge(id=edge_id, u=u, v=v, polyline=polyline)


def make_graph(edges=None):
    if edges is None:
        edges = (make_edge(),)

    return JunctionGraph(name="test_graph", coordinates={1: (0.0, 0.0), 2: (3.0, 4.0)}, edges=tuple(edges))


def test_edge_converts_identifiers_to_integers():
    edge_data: dict[str, Any] = {"id": "7", "u": "1", "v": "2", "polyline": [[0.0, 0.0], [1.0, 0.0]], }

    edge = JunctionEdge(**edge_data)

    assert edge.id == 7
    assert edge.u == 1
    assert edge.v == 2


def test_edge_calculates_polyline_length():
    edge = make_edge(polyline=[[0.0, 0.0], [3.0, 4.0]])

    assert edge.length == pytest.approx(5.0)


def test_edge_removes_consecutive_duplicate_points():
    edge = make_edge(polyline=[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], ])

    expected = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], ], dtype=np.float64)

    np.testing.assert_array_equal(edge.polyline, expected)
    assert edge.length == pytest.approx(2.0)


def test_edge_stores_contiguous_float64_polyline():
    input_points = np.asarray([[0, 0], [1, 1], [2, 2], ], dtype=np.int32)[::2]

    edge = make_edge(polyline=input_points)

    assert edge.polyline.dtype == np.float64
    assert edge.polyline.flags.c_contiguous


def test_edge_polyline_is_read_only():
    edge = make_edge()

    assert not edge.polyline.flags.writeable

    with pytest.raises(ValueError):
        edge.polyline[0, 0] = 10.0


def test_negative_edge_id_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        make_edge(edge_id=-1)


def test_self_loop_is_rejected():
    with pytest.raises(ValueError, match="Self-loop"):
        make_edge(u=1, v=1)


@pytest.mark.parametrize("polyline", [[], [[0.0, 0.0]], [0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], ])
def test_invalid_polyline_shape_is_rejected(polyline):
    with pytest.raises(ValueError, match="shape"):
        make_edge(polyline=polyline)


@pytest.mark.parametrize("polyline", [[[0.0, 0.0], [math.nan, 1.0]], [[0.0, 0.0], [math.inf, 1.0]], [[0.0, 0.0], [-math.inf, 1.0]], ])
def test_nonfinite_polyline_coordinates_are_rejected(polyline):
    with pytest.raises(ValueError, match="NaN or infinite"):
        make_edge(polyline=polyline)


def test_zero_length_polyline_is_rejected():
    with pytest.raises(ValueError, match="distinct points"):
        make_edge(polyline=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], ])


def test_parallel_edges_with_different_ids_are_accepted():
    edge_0 = make_edge(edge_id=0, polyline=[[0.0, 0.0], [3.0, 4.0]])
    edge_1 = make_edge(edge_id=1, polyline=[[0.0, 0.0], [1.0, 3.0], [3.0, 4.0]])

    graph = make_graph(edges=(edge_0, edge_1))

    assert len(graph.edges) == 2
    assert graph.edge_by_id[0] is edge_0
    assert graph.edge_by_id[1] is edge_1


def test_duplicate_edge_ids_are_rejected():
    edge_0 = make_edge(edge_id=4, polyline=[[0.0, 0.0], [3.0, 4.0]])
    edge_1 = make_edge(edge_id=4, polyline=[[0.0, 0.0], [2.0, 4.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="already in graph"):
        make_graph(edges=(edge_0, edge_1))


def test_missing_edge_endpoint_is_rejected():
    edge = make_edge(edge_id=0, u=1, v=99)

    with pytest.raises(ValueError, match="missing endpoint 99"):
        JunctionGraph(name="missing_endpoint", coordinates={1: (0.0, 0.0)}, edges=(edge,))


def test_graph_normalizes_name_coordinates_and_vertex_order():
    graph = JunctionGraph(name="  example  ", coordinates={"5": [5, 6], "2": np.asarray([2.0, 3.0])}, edges=())

    assert graph.name == "example"
    assert graph.coordinates == {5: (5.0, 6.0), 2: (2.0, 3.0)}
    assert graph.vertices == (2, 5)


def test_empty_graph_name_is_rejected():
    with pytest.raises(ValueError, match="Junction graph must have a name"):
        JunctionGraph(name="   ", coordinates={1: (0.0, 0.0)}, edges=())


def test_graph_without_vertices_is_rejected():
    with pytest.raises(ValueError, match="at least one vertex"):
        JunctionGraph(name="empty", coordinates={}, edges=())


def test_duplicate_vertex_ids_after_integer_conversion_are_rejected():
    with pytest.raises(ValueError, match="Duplicate vertex ID"):
        JunctionGraph(name="duplicate_vertices", coordinates={1: (0.0, 0.0), "1": (1.0, 1.0)}, edges=())


def test_feasible_match_result_normalizes_values():
    result = MatchResult(objective="additive", feasible=True, value=7, phi={"1": "10", "2": "20"}, edge_costs={"3": 7})

    assert result.objective is Objective.ADDITIVE
    assert result.feasible
    assert result.value == 7.0
    assert result.phi == {1: 10, 2: 20}
    assert result.edge_costs == {3: 7.0}


def test_feasible_match_requires_finite_value():
    with pytest.raises(ValueError, match="finite objective value"):
        MatchResult(objective=Objective.ADDITIVE, feasible=True, value=float("inf"))


def test_feasible_match_rejects_nonfinite_edge_cost():
    with pytest.raises(ValueError, match="non-finite edge costs"):
        MatchResult(objective=Objective.BOTTLENECK, feasible=True, value=2.0, phi={1: 10}, edge_costs={0: float("inf")})


def test_infeasible_factory_creates_empty_result():
    result = MatchResult.infeasible(Objective.BOTTLENECK)

    assert result.objective is Objective.BOTTLENECK
    assert not result.feasible
    assert result.value == float("inf")
    assert result.phi == {}
    assert result.edge_costs == {}


def test_infeasible_match_rejects_finite_value():
    with pytest.raises(ValueError, match="positive infinity"):
        MatchResult(objective=Objective.ADDITIVE, feasible=False, value=0.0)


def test_infeasible_match_rejects_partial_mapping():
    with pytest.raises(ValueError, match="cannot contain a vertex mapping"):
        MatchResult(objective=Objective.ADDITIVE, feasible=False, value=float("inf"), phi={1: 10})
