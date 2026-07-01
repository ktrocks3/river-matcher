from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

from river_matcher.graph_io import read_topotide_graph


def write_graph(tmp_path: Path, content: str, name: str = "graph.txt") -> Path:
    """Write one temporary TopoTide graph fixture."""
    normalized = textwrap.dedent(content).strip()
    path = tmp_path / name
    path.write_text(f"{normalized}\n" if normalized else "", encoding="utf-8")
    return path


def test_reads_valid_topotide_graph(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        2

        1 0.0 0.0
        2 3.0 4.0

        2
        10 1 2 0.5 0.0 0.0 1.0 1.0 3.0 4.0
        11 1 2 1.25 0.0 0.0 3.0 4.0
        """, )

    vertices, edges = read_topotide_graph(path)

    assert vertices == {1: (0.0, 0.0), 2: (3.0, 4.0), }

    assert edges == [{"id": 10, "u": 1, "v": 2, "delta": 0.5, "path": [(0.0, 0.0), (1.0, 1.0), (3.0, 4.0), ], },
        {"id": 11, "u": 1, "v": 2, "delta": 1.25, "path": [(0.0, 0.0), (3.0, 4.0), ], }, ]


def test_accepts_string_path(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        1
        8 2.5 7.5
        0
        """, )

    vertices, edges = read_topotide_graph(str(path))

    assert vertices == {8: (2.5, 7.5)}
    assert edges == []


def test_accepts_zero_declared_vertices_and_edges(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        0
        0
        """, )

    vertices, edges = read_topotide_graph(path)

    assert vertices == {}
    assert edges == []


def test_reader_preserves_data_that_preprocessing_must_validate(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        1
        1 -1.0 -1.0
        3
        5 1 999 nan
        6 1 1 inf 0.0 0.0
        7 1 1 -inf 0.0 0.0 1.0 1.0
        """, )

    vertices, edges = read_topotide_graph(path)

    assert vertices == {1: (-1.0, -1.0)}

    assert edges[0]["u"] == 1
    assert edges[0]["v"] == 999
    assert math.isnan(edges[0]["delta"])
    assert edges[0]["path"] == []

    assert math.isinf(edges[1]["delta"])
    assert edges[1]["delta"] > 0
    assert edges[1]["path"] == [(0.0, 0.0)]

    assert math.isinf(edges[2]["delta"])
    assert edges[2]["delta"] < 0
    assert edges[2]["path"] == [(0.0, 0.0), (1.0, 1.0)]


def test_blank_lines_do_not_change_reported_source_line_numbers(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        1


        one 0.0 0.0
        0
        """, )

    with pytest.raises(ValueError, match=r"Line 4: invalid vertex record"):
        read_topotide_graph(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = write_graph(tmp_path, "")

    with pytest.raises(ValueError, match="TopoTide graph file is empty"):
        read_topotide_graph(path)


@pytest.mark.parametrize(("content", "message"), [("""
            not-an-integer
            """, "vertex count must be an integer",), ("""
            -1
            """, "vertex count must be nonnegative",), ("""
            0
            not-an-integer
            """, "edge count must be an integer",), ("""
            0
            -1
            """, "edge count must be nonnegative",), ], )
def test_invalid_counts_are_rejected(tmp_path: Path, content: str, message: str, ) -> None:
    path = write_graph(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        read_topotide_graph(path)


@pytest.mark.parametrize(("content", "message"), [("""
            1
            """, "vertex 1 of 1",), ("""
            1
            1 0.0 0.0
            """, "the edge count",), ("""
            0
            1
            """, "edge 1 of 1",), ], )
def test_unexpected_end_of_file_is_rejected(tmp_path: Path, content: str, message: str, ) -> None:
    path = write_graph(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        read_topotide_graph(path)


@pytest.mark.parametrize("vertex_line", ["1", "1 0.0", "1 0.0 1.0 2.0", ], )
def test_vertex_with_wrong_field_count_is_rejected(tmp_path: Path, vertex_line: str, ) -> None:
    path = write_graph(tmp_path, f"""
        1
        {vertex_line}
        0
        """, )

    with pytest.raises(ValueError, match="a vertex must contain exactly"):
        read_topotide_graph(path)


@pytest.mark.parametrize("vertex_line", ["vertex 0.0 1.0", "1 x 1.0", "1 0.0 y", ], )
def test_vertex_with_invalid_numeric_values_is_rejected(tmp_path: Path, vertex_line: str, ) -> None:
    path = write_graph(tmp_path, f"""
        1
        {vertex_line}
        0
        """, )

    with pytest.raises(ValueError, match="invalid vertex record"):
        read_topotide_graph(path)


def test_duplicate_vertex_ids_after_integer_conversion_are_rejected(tmp_path: Path, ) -> None:
    path = write_graph(tmp_path, """
        2
        1 0.0 0.0
        01 1.0 1.0
        0
        """, )

    with pytest.raises(ValueError, match="duplicate raw vertex ID 1"):
        read_topotide_graph(path)


@pytest.mark.parametrize("edge_line", ["1", "1 2", "1 2 3", ], )
def test_edge_with_too_few_fields_is_rejected(tmp_path: Path, edge_line: str, ) -> None:
    path = write_graph(tmp_path, f"""
        0
        1
        {edge_line}
        """, )

    with pytest.raises(ValueError, match="an edge must contain at least"):
        read_topotide_graph(path)


@pytest.mark.parametrize("edge_line", ["1 2 3 0.5 10.0", "1 2 3 0.5 10.0 20.0 30.0", ], )
def test_edge_with_unpaired_coordinates_is_rejected(tmp_path: Path, edge_line: str, ) -> None:
    path = write_graph(tmp_path, f"""
        0
        1
        {edge_line}
        """, )

    with pytest.raises(ValueError, match="edge coordinates must occur in x/y pairs", ):
        read_topotide_graph(path)


@pytest.mark.parametrize("edge_line", ["edge 1 2 0.5", "1 source 2 0.5", "1 1 target 0.5", "1 1 2 delta", "1 1 2 0.5 x 0.0", ], )
def test_edge_with_invalid_numeric_values_is_rejected(tmp_path: Path, edge_line: str, ) -> None:
    path = write_graph(tmp_path, f"""
        0
        1
        {edge_line}
        """, )

    with pytest.raises(ValueError, match="invalid edge record"):
        read_topotide_graph(path)


def test_duplicate_edge_ids_after_integer_conversion_are_rejected(tmp_path: Path, ) -> None:
    path = write_graph(tmp_path, """
        0
        2
        7 1 2 0.5
        007 2 3 1.5
        """, )

    with pytest.raises(ValueError, match="duplicate raw edge ID 7"):
        read_topotide_graph(path)


def test_unexpected_trailing_content_is_rejected(tmp_path: Path) -> None:
    path = write_graph(tmp_path, """
        0
        0
        unexpected content
        """, )

    with pytest.raises(ValueError, match="unexpected trailing content"):
        read_topotide_graph(path)


def test_missing_file_reports_the_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(OSError, match="Could not read TopoTide graph file"):
        read_topotide_graph(path)
