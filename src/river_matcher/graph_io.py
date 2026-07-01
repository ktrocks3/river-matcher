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

def _parse_nonnegative_count()