from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from river_matcher.graph_io import RawEdge, RawEdges, RawVertices

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
    """ Remove unusable raw vertices and edges. A retained edge must: 1. have two different endpoints; 2.  refer to retained vertices;
            3 have a finite positive-length coordinate polyline. """
    marker = (float(invalid_vertex_marker[0]), float(invalid_vertex_marker[1]))
    valid_vertices: RawVertices = {}

    for raw_vertex, raw_coordinates in vertices.items():
        vertex = int(raw_vertex)
        try:
            point = np.asarray(raw_coordinates, dtype=np.float64)
        except (TypeError, ValueError):
            continue

        if point.shape != (2,) or not np.all(np.isfinite(point)):
            continue

        coordinates = (float(point[0]), float(point[1]))
        if coordinates == marker:
            continue

        valid_vertices[vertex] = coordinates

    valid_edges: RawEdges = []
    for raw_edge in edges:
        try:
            edge_id, u, v, delta = int(raw_edge['id']), int(raw_edge['u']), int(raw_edge['v']), float(raw_edge['delta'])
        except (KeyError, ValueError, TypeError):
            continue

        polyline = _clean_raw_polyline(raw_edge.get("path"))
        if u == v or u not in valid_vertices or v not in valid_vertices or polyline is None:
            continue

        valid_edges.append({"id": edge_id, "u": u, "v": v, "delta": delta, "path": [(float(point[0]), float(point[1]) for point in polyline)]})

    return valid_vertices, valid_edges


def _build_adjacency(vertices: RawVertices, edges: RawEdges) -> Adjacency:
    """Build multigraph adjacency using raw-edge list positions as identities"""
    adjacency: Adjacency = {int(vertex): [] for vertex in vertices}

    for edge_index, edge in enumerate(edges):
        u, v = int(edge['u']), int(edge['v'])
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))

    for vertex in adjacency:
        adjacency[vertex].sort(key=lambda item: (int(edges[item[1]]["id"]), item[0], item[1]))

    return adjacency


def _connected_components(adjacency: Adjacency) -> list[set[int]]:
    """Return deterministic connected components, including isolated vertices."""
    components: list[set[int]] = []
    seen: set[int] = set()

    for start in sorted(adjacency):
        if start in seen:
            continue

        component = {start}
        seen.add(start)
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbour, _ in adjacency[current]:
                if neighbour in seen:
                    continue

                seen.add(neighbour)
                component.add(neighbour)
                queue.append(neighbour)
        components.append(component)
    return components


def _junction_vertices(adjacency: Adjacency) -> set[int]:
    """ Select vertices retained after degree-2 suppression. A component consisting entirely of degree-2 vertices is a cycle. Two deterministic anchor vertices are retained
    so the cycle becomes two parallel junction edges rather than disappearing. """
    junctions = {vertex for vertex, incident_edges in adjacency.items() if len(incident_edges) == 1}
    for component in _connected_components(adjacency):
        if len(component) < 2:
            continue

        if all(len(adjacency[vertex]) == 2 for vertex in component):
            first, second = sorted(component)[:2]
            junctions.add(first)
            junctions.add(second)

    return junctions

def _orient_polyline(edge: RawEdge, from_vertex: int, coordinates: RawVertices) -> XYArray:
    """Orient a raw edge polyline """
