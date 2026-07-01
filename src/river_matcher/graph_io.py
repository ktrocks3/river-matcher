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
        raise OSError(f"Could not open TopoTide graph file: {path}") from exc


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
        raise ValueError(f"Line {line_number}: a edge must contain at least 'edge_id u v delta'")
    coordinate_fields = parts[4:]
    if len(coordinate_fields) % 2 != 0:
        raise ValueError(f"Line {line_number}: edge coordinates must occur in x/y pairs, got {len(coordinate_fields)} coordinate values.")

    try:
        edge_id, u, v, delta, coordinates = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), [float(val) for val in coordinate_fields]
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid edge record {text!r}.") from exc

    path = [(coordinates[index], coordinates[index + 1]) for index in range(0, len(coordinates), 2)]
    return {"id": edge_id, "u": u, "v": v, "delta": delta, "coordinates": path}
