from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from river_matcher.preprocessing import compress_degree_two_chains, filter_raw_graph, load_embedded_graph, load_junction_graph, preprocess_raw_graph


def raw_edge(edge_id: int, u: int, v: int, path: object, delta: float = 1.0) -> dict:
    return {"id": edge_id, "u": u, "v": v, "delta": delta, "path": path}


def graph_signature(graph) -> tuple:
    """Return a comparable representation that includes edge geometry."""
    return graph.name, graph.coordinates, tuple((edge.id, edge.u, edge.v, edge.length, edge.polyline.tolist()) for edge in graph.edges)


def test_filter_removes_invalid_vertices_and_unusable_edges() -> None:
    vertices = {1: (0.0, 0.0), 2: (-1.0, -1.0), 3: (float("nan"), 0.0), 4: (float("inf"), 0.0), 5: (2.0, 0.0)}
    edges = [
        raw_edge(10, 1, 5, [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
        raw_edge(11, 1, 2, [(0.0, 0.0), (1.0, 0.0)]),
        raw_edge(12, 1, 1, [(0.0, 0.0), (1.0, 0.0)]),
        raw_edge(13, 1, 5, [(0.0, 0.0)]),
        raw_edge(14, 1, 5, [(0.0, 0.0), (float("nan"), 0.0)]),
        raw_edge(15, 99, 5, [(0.0, 0.0), (2.0, 0.0)]),
    ]
    filtered_vertices, filtered_edges = filter_raw_graph(vertices, edges)

    assert filtered_vertices == {1: (0.0, 0.0), 5: (2.0, 0.0)}
    assert filtered_edges == [{"id": 10, "u": 1, "v": 5, "delta": 1.0, "path": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]}]


def test_filter_accepts_custom_invalid_vertex_marker() -> None:
    vertices = {1: (999.0, 999.0), 2: (-1.0, -1.0), 3: (0.0, 0.0)}

    filtered_vertices, filtered_edges = filter_raw_graph(vertices, [], invalid_vertex_marker=(999.0, 999.0))

    assert filtered_vertices == {2: (-1.0, -1.0), 3: (0.0, 0.0)}
    assert filtered_edges == []


@pytest.mark.parametrize("path", [[], [(0.0, 0.0)], [(1.0, 1.0), (1.0, 1.0)], [(0.0, 0.0), (float("nan"), 1.0)], [(0.0, 0.0), (float("inf"), 1.0)], [0.0, 1.0], "not-a-polyline"])
def test_filter_rejects_unusable_polylines(path: object) -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0)}

    _, filtered_edges = filter_raw_graph(vertices, [raw_edge(1, 1, 2, path)])

    assert filtered_edges == []


def test_filter_skips_malformed_edge_records() -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0)}

    edges = [{}, {"id": 1}, {"id": "bad", "u": 1, "v": 2, "delta": 1.0, "path": []}, {"id": 2, "u": 1, "v": 2, "delta": "bad", "path": []}]

    _, filtered_edges = filter_raw_graph(vertices, edges)

    assert filtered_edges == []


def test_compresses_one_degree_two_chain() -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}

    edges = [raw_edge(10, 1, 2, [(0.0, 0.0), (1.0, 0.0)]), raw_edge(11, 2, 3, [(2.0, 0.0), (1.0, 0.0)])]  # Stored backwards to verify that compression orients each segment.

    junctions, compressed = compress_degree_two_chains(vertices, edges)

    assert junctions == {1, 3}
    assert len(compressed) == 1

    edge = compressed[0]

    assert edge["u"] == 1
    assert edge["v"] == 3
    assert edge["raw_edge_ids"] == (10, 11)
    assert edge["chain_vertices"] == (1, 2, 3)

    np.testing.assert_allclose(edge["polyline"], np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_branch_vertices_are_not_suppressed() -> None:
    vertices = {1: (0.0, 0.0), 2: (-1.0, 0.0), 3: (1.0, 0.0), 4: (0.0, 1.0)}

    edges = [raw_edge(1, 1, 2, [(0.0, 0.0), (-1.0, 0.0)]), raw_edge(2, 1, 3, [(0.0, 0.0), (1.0, 0.0)]), raw_edge(3, 1, 4, [(0.0, 0.0), (0.0, 1.0)])]

    junctions, compressed = compress_degree_two_chains(vertices, edges)

    assert junctions == {1, 2, 3, 4}
    assert len(compressed) == 3


def test_parallel_raw_edges_remain_parallel() -> None:
    vertices = {1: (0.0, 0.0), 2: (2.0, 0.0)}

    edges = [raw_edge(20, 1, 2, [(0.0, 0.0), (2.0, 0.0)]), raw_edge(10, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])]

    junctions, compressed = compress_degree_two_chains(vertices, edges)

    assert junctions == {1, 2}
    assert len(compressed) == 2
    assert all(edge["u"] == 1 and edge["v"] == 2 for edge in compressed)
    assert {edge["raw_edge_ids"] for edge in compressed} == {(10,), (20,)}


def test_pure_degree_two_cycle_becomes_two_parallel_edges() -> None:
    vertices = {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, 1.0)}

    edges = [raw_edge(10, 1, 2, [(0.0, 0.0), (2.0, 0.0)]), raw_edge(11, 2, 3, [(2.0, 0.0), (1.0, 1.0)]), raw_edge(12, 3, 1, [(1.0, 1.0), (0.0, 0.0)])]

    junctions, compressed = compress_degree_two_chains(vertices, edges)

    assert junctions == {1, 2}
    assert len(compressed) == 2
    assert all(edge["u"] == 1 and edge["v"] == 2 for edge in compressed)

    raw_chains = {edge["raw_edge_ids"] for edge in compressed}

    assert (10,) in raw_chains
    assert (12, 11) in raw_chains


def test_compressed_edge_orientation_is_canonical() -> None:
    vertices = {2: (0.0, 0.0), 5: (2.0, 0.0)}

    edges = [raw_edge(10, 5, 2, [(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])]

    _, compressed = compress_degree_two_chains(vertices, edges)

    assert len(compressed) == 1
    assert compressed[0]["u"] == 2
    assert compressed[0]["v"] == 5

    np.testing.assert_allclose(compressed[0]["polyline"], np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_preprocessing_assigns_fresh_edge_ids() -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0)}

    graph = preprocess_raw_graph("example", vertices, [raw_edge(99, 1, 2, [(0.0, 0.0), (1.0, 0.0)])])

    assert len(graph.edges) == 1
    assert graph.edges[0].id == 0
    assert graph.edges[0].id != 99


def test_preprocessing_assigns_deterministic_ids_to_parallel_edges() -> None:
    vertices = {1: (0.0, 0.0), 2: (2.0, 0.0)}

    straight = raw_edge(20, 1, 2, [(0.0, 0.0), (2.0, 0.0)])
    curved = raw_edge(10, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])

    first = preprocess_raw_graph("parallel", vertices, [straight, curved])
    second = preprocess_raw_graph("parallel", vertices, [curved, straight])

    assert graph_signature(first) == graph_signature(second)
    assert [edge.id for edge in first.edges] == [0, 1]

    # Raw edge 10 sorts before raw edge 20, so the curved path receives ID 0.
    assert len(first.edges[0].polyline) == 3
    assert len(first.edges[1].polyline) == 2


def test_preprocessing_retains_largest_component() -> None:
    vertices = {1: (0.0, 0.0), 2: (-1.0, 0.0), 3: (1.0, 0.0), 4: (0.0, 1.0), 10: (10.0, 0.0), 11: (11.0, 0.0)}

    edges = [
        raw_edge(1, 1, 2, [(0.0, 0.0), (-1.0, 0.0)]),
        raw_edge(2, 1, 3, [(0.0, 0.0), (1.0, 0.0)]),
        raw_edge(3, 1, 4, [(0.0, 0.0), (0.0, 1.0)]),
        raw_edge(4, 10, 11, [(10.0, 0.0), (11.0, 0.0)]),
    ]

    graph = preprocess_raw_graph("components", vertices, edges)

    assert set(graph.vertices) == {1, 2, 3, 4}
    assert len(graph.edges) == 3


def test_preprocessing_can_retain_all_components() -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0), 10: (10.0, 0.0), 11: (11.0, 0.0)}

    edges = [raw_edge(1, 1, 2, [(0.0, 0.0), (1.0, 0.0)]), raw_edge(2, 10, 11, [(10.0, 0.0), (11.0, 0.0)])]

    graph = preprocess_raw_graph("components", vertices, edges, keep_largest_component=False)

    assert set(graph.vertices) == {1, 2, 10, 11}
    assert len(graph.edges) == 2


def test_largest_component_tie_uses_lowest_vertex_id() -> None:
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0), 10: (10.0, 0.0), 11: (11.0, 0.0)}

    edges = [raw_edge(1, 1, 2, [(0.0, 0.0), (1.0, 0.0)]), raw_edge(2, 10, 11, [(10.0, 0.0), (11.0, 0.0)])]

    graph = preprocess_raw_graph("tie", vertices, edges)

    assert set(graph.vertices) == {1, 2}
    assert len(graph.edges) == 1


def test_compressed_self_loop_is_removed() -> None:
    """
    The cycle 1-2-3-1 compresses to a self-loop at junction 1.

    The separate tail 1-4 remains a valid junction edge.
    """
    vertices = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0), 4: (-1.0, 0.0)}

    edges = [
        raw_edge(1, 1, 2, [(0.0, 0.0), (1.0, 0.0)]),
        raw_edge(2, 2, 3, [(1.0, 0.0), (0.5, 1.0)]),
        raw_edge(3, 3, 1, [(0.5, 1.0), (0.0, 0.0)]),
        raw_edge(4, 1, 4, [(0.0, 0.0), (-1.0, 0.0)]),
    ]

    graph = preprocess_raw_graph("lollipop", vertices, edges)

    assert set(graph.vertices) == {1, 4}
    assert len(graph.edges) == 1
    assert graph.edges[0].u == 1
    assert graph.edges[0].v == 4


def test_isolated_vertex_is_a_valid_junction_graph() -> None:
    graph = preprocess_raw_graph("isolated", {7: (3.0, 4.0)}, [])

    assert graph.vertices == (7,)
    assert graph.coordinates == {7: (3.0, 4.0)}
    assert graph.edges == ()


def test_no_valid_vertices_is_rejected() -> None:
    vertices = {1: (-1.0, -1.0), 2: (float("nan"), 0.0)}

    with pytest.raises(ValueError, match="contains no valid vertices"):
        preprocess_raw_graph("invalid", vertices, [])


def test_load_junction_graph_reads_and_preprocesses_file(tmp_path: Path) -> None:
    path = tmp_path / "river_1955.txt"
    path.write_text("\n".join(["3", "1 0.0 0.0", "2 1.0 0.0", "3 2.0 0.0", "2", "10 1 2 1.0 0.0 0.0 1.0 0.0", "11 2 3 1.0 1.0 0.0 2.0 0.0"]), encoding="utf-8")

    graph = load_junction_graph(path)

    assert graph.name == "river_1955"
    assert graph.vertices == (1, 3)
    assert len(graph.edges) == 1
    assert graph.edges[0].id == 0
    assert graph.edges[0].u == 1
    assert graph.edges[0].v == 3
    assert graph.edges[0].length == pytest.approx(2.0)


def test_load_junction_graph_accepts_name_override(tmp_path: Path) -> None:
    path = tmp_path / "source_file.txt"
    path.write_text("\n".join(["1", "5 2.0 3.0", "0"]), encoding="utf-8")

    graph = load_junction_graph(path, name="custom-name")

    assert graph.name == "custom-name"


def test_load_embedded_graph_preserves_degree_two_vertices(tmp_path: Path) -> None:
    path = tmp_path / "river.txt"
    path.write_text(
        "\n".join(
            [
                "5",
                "1 0.0 0.0",
                "2 1.0 0.0",
                "3 2.0 0.0",
                "10 10.0 0.0",
                "11 11.0 0.0",
                "3",
                "10 1 2 1.0 0.0 0.0 1.0 0.0",
                "11 2 3 1.0 1.0 0.0 2.0 0.0",
                "12 10 11 1.0 10.0 0.0 11.0 0.0",
            ],
        ),
        encoding="utf-8",
    )

    graph = load_embedded_graph(path)

    assert graph.name == "river_original"
    assert graph.vertices == (1, 2, 3)
    assert [(edge.id, edge.u, edge.v) for edge in graph.edges] == [(0, 1, 2), (1, 2, 3)]
