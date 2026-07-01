from __future__ import annotations

from pathlib import Path
from typing import Any

type XY = tuple[float, float]
type RawVertices = dict[int, XY]
type RawEdge = dict[str, Any]
type RawEdges = list[RawEdge]
type RawGraph = tuple[RawVertices, RawEdges]


def _nonempty_lines(path: Path) -> list[tuple[int, str]]:
    """Read nonempty lines while retaining their original line numbers"""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [(line_number, stripped) for line_number, line in enumerate(handle, start=1) if (stripped := line.strip())]
    except OSError as exc:
        raise OSError(f"Could not read TopoTide graph file: {path}") from exc


def _parse_nonnegative_count(text: str, line_number: int, label: str) -> int:
    """Parse a nonnegative vertex or edge count."""
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: {label} count must be an integer, got {text!r}") from exc

    if count < 0:
        raise ValueError(f"Line {line_number}: {label} count must be nonnegative, got {text!r}")
    return count


def _parse_vertex(text: str, line_number: int) -> tuple[int, XY]:
    """Parse a vertex; vertex_id x y"""
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Line {line_number}: a vertex must contain exactly 'vertex_id x y', got {len(parts)} fields")
    try:
        vertex_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid vertex record {text!r}") from exc
    return vertex_id, (x, y)


def _parse_edge(text: str, line_number: int) -> RawEdge:
    """Parse one edge: edge_id u v delta x0 y0 x1 y1 ..."""
    parts = text.split()
    if len(parts) < 4:
        raise ValueError(f"Line {line_number}: an edge must contain at least 'edge_id u v delta'")
    coordinate_fields = parts[4:]
    if len(coordinate_fields) % 2 != 0:
        raise ValueError(f"Line {line_number}: edge coordinates must occur in x/y pairs, got {len(coordinate_fields)} coordinate values.")

    try:
        edge_id, u, v, delta, coordinates = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), [float(val) for val in coordinate_fields]
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid edge record {text!r}.") from exc

    path = [(coordinates[index], coordinates[index + 1]) for index in range(0, len(coordinates), 2)]
    return {"id": edge_id, "u": u, "v": v, "delta": delta, "path": path}


def read_topotide_graph(path: str | Path) -> RawGraph:
    """Read a raw TopoTide graph"""
    path = Path(path)
    lines = _nonempty_lines(path)

    if not lines:
        raise ValueError(f"TopoTide graph file is empty: {path}")

    cursor = 0

    def next_line(expected: str) -> tuple[int, str]:
        nonlocal cursor

        if cursor >= len(lines):
            raise ValueError(f"Unexpected end of file while reading {expected}: {path}")
        line = lines[cursor]
        cursor += 1
        return line

    line_number, text = next_line("the vertex count")
    vertex_count = _parse_nonnegative_count(text, line_number, "vertex")
    vertices: RawVertices = {}
    for vertex_index in range(vertex_count):
        line_number, text = next_line(f"vertex {vertex_index + 1} of {vertex_count}")
        vertex_id, coordinates = _parse_vertex(text, line_number)

        if vertex_id in vertices:
            raise ValueError(f"Line {line_number}: duplicate raw vertex ID {vertex_id}.")

        vertices[vertex_id] = coordinates

    line_number, text = next_line(f"the edge count")
    edge_count = _parse_nonnegative_count(text, line_number, "edge")
    edges: RawEdges = []
    seen_edges_id: set[int] = set()

    for edge_index in range(edge_count):
        line_number, text = next_line(f"edge {edge_index + 1} of {edge_count}")
        edge = _parse_edge(text, line_number)
        edge_id = int(edge["id"])

        if edge_id in seen_edges_id:
            raise ValueError(f"Line {line_number}: duplicate raw edge ID {edge_id}")

        seen_edges_id.add(edge_id)
        edges.append(edge)

    if cursor != len(lines):
        line_number, text = lines[cursor]
        raise ValueError(f"Line {line_number}: unexpected trailing content {text!r}.")

    return vertices, edges
