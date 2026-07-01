from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

type XYArray = NDArray[np.float64]
type Adjacency = dict[int, list[tuple[int, int]]]
type CompressedEdge = dict[str, Any]


def _clean_raw_polyline(path: object) -> XYArray | None:
    """Return a finite positive-length coordinate polyline, or None when unusable."""
    try:
        points = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        return None

    keep = [0]
    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]
        if float(np.dot(delta, delta)) > 1e-24:
            keep.append(index)

    points = np.ascontiguousarray(points[keep], dtype=np.float64)
    if len(points) < 2:
        return None

    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if not np.isfinite(length) or length <= 1e-12:
        return None

    return points

def filter_raw_graph(vertices: RawVertices, edges: Iterable[RawEdge], *, invalid_vertex_marker: tuple[float, float] = (-1.0, -1.0)):
    