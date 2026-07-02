from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from river_matcher.graph_io import RawEdge, RawEdges, RawVertices, read_topotide_graph
from river_matcher.models import JunctionGraph, JunctionEdge

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

        valid_edges.append({"id": edge_id, "u": u, "v": v, "delta": delta, "path": [(float(point[0]), float(point[1])) for point in polyline]})

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
    junctions = {vertex for vertex, incident_edges in adjacency.items() if len(incident_edges) != 2}
    for component in _connected_components(adjacency):
        if len(component) < 2:
            continue

        if all(len(adjacency[vertex]) == 2 for vertex in component):
            first, second = sorted(component)[:2]
            junctions.add(first)
            junctions.add(second)

    return junctions


def _orient_polyline(edge: RawEdge, from_vertex: int, coordinates: RawVertices) -> XYArray:
    """Orient a raw edge polyline away from the specified endpoint"""
    points = np.asarray(edge["path"], dtype=np.float64)
    endpoint = np.asarray(coordinates[from_vertex], dtype=np.float64)

    start_error = float(np.linalg.norm(points[0] - endpoint))
    end_error = float(np.linalg.norm(points[-1] - endpoint))

    if end_error < start_error:
        return np.ascontiguousarray(points[::-1], dtype=np.float64)
    return np.ascontiguousarray(points, dtype=np.float64)


def _concatenate_polyline(first: XYArray, second: XYArray) -> XYArray:
    """Join two oriented edge polylines without duplicating a shared endpoint"""
    seam = first[-1] - second[0]
    if float(np.dot(seam, seam)) <= 1e-24:
        return np.ascontiguousarray(np.vstack((first, second[1:])), dtype=np.float64)
    return np.ascontiguousarray(np.vstack((first, second)), dtype=np.float64)


def _canonicalize_compressed_edge(u: int, v: int, polyline: XYArray, raw_edge_ids: list[int], chain_vertices: list[int]) -> CompressedEdge:
    """Give an undirected compressed edge a deterministic orientation"""
    if u <= v:
        return {"u": u, "v": v, "polyline": polyline, "raw_edge_ids": tuple(raw_edge_ids), "chain_vertices": tuple(chain_vertices)}
    return {"u": v, "v": u, "polyline": np.ascontiguousarray(polyline[::-1], dtype=np.float64), "raw_edge_ids": tuple(reversed(raw_edge_ids)),
            "chain_vertices": tuple(reversed(chain_vertices))}


def compress_degree_two_chains(vertices: RawVertices, edges: RawEdges) -> tuple[set[int], list[CompressedEdge]]:
    """Suppress degree two vertices while preserving parallel compressed edges"""
    adjacency = _build_adjacency(vertices, edges)
    junctions = _junction_vertices(adjacency)
    visited_edges: set[int] = set()
    compressed_edges: list[CompressedEdge] = []

    for start in sorted(junctions):
        for neighbour, edge_index in adjacency[start]:
            if edge_index in visited_edges:
                continue

            edge = edges[edge_index]
            polyline = _orient_polyline(edge, start, vertices)
            raw_edge_ids = [int(edge["id"])]
            chain_vertices = [start, neighbour]

            visited_edges.add(edge_index)
            previous_edge = edge_index
            current = neighbour

            while current not in junctions:
                continuation = [(next_vertex, next_edge) for next_vertex, next_edge in adjacency[current] if next_edge != previous_edge]
                if len(continuation) != 1:
                    raise RuntimeError(f"Degree-2 vertex {current} has {len(continuation)} possible continuations")

                next_vertex, next_edge_index = continuation[0]
                if next_edge_index in visited_edges:
                    raise RuntimeError(f"Raw edge {edges[next_edge_index]['id']} was encountered twice during compression.")
                next_edge = edges[next_edge_index]
                next_polyline = _orient_polyline(next_edge, current, vertices)
                polyline = _concatenate_polyline(polyline, next_polyline)
                raw_edge_ids.append(int(next_edge['id']))
                chain_vertices.append(next_vertex)

                visited_edges.add(next_edge_index)
                previous_edge = next_edge_index
                current = next_vertex
            compressed_edges.append(_canonicalize_compressed_edge(start, current, polyline, raw_edge_ids, chain_vertices))

    if len(visited_edges) != len(edges):
        missing = sorted(int(edges[index]['id']) for index in range(len(edges)) if edges[index]['id'] not in visited_edges)
        raise RuntimeError(f'Compression did not visit raw edges {missing}')
    return junctions, compressed_edges


def _largest_component(nodes: set[int], edges: list[CompressedEdge]) -> tuple[set[int], list[CompressedEdge]]:
    adjacency: dict[int, set[int]] = {node: set() for node in nodes}
    for edge in edges:
        u, v = int(edge["u"]), int(edge["v"])
        adjacency[u].add(v)
        adjacency[v].add(u)
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
            for neighbour in sorted(adjacency[current]):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                component.add(neighbour)
                queue.append(neighbour)

            components.append(component)

    if not components:
        return set(), []

    def component_key(comp: set[int]) -> tuple[int, int, int]:
        edge_count = sum(int(e["u"]) in comp and int(e["v"]) in comp for e in edges)
        return len(comp), edge_count, -min(comp)

    largest = max(components, key=component_key)
    retained_edges = [edge for edge in edges if int(edge["u"]) in largest and int(edge["v"]) in largest]
    return set(largest), retained_edges


def _compressed_edge_sort_key(edge: CompressedEdge) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return int(edge["u"]), int(edge["v"]), tuple(int(value) for value in edge["raw_edge_ids"]), tuple(int(value) for value in edge["chain_vertices"])


def preprocess_raw_graph(name: str, vertices: RawVertices, edges: Iterable[RawEdge], *, keep_largest_component: bool = True) -> JunctionGraph:
    filtered_vertices, filtered_edges = filter_raw_graph(vertices, edges)
    if not filtered_vertices:
        raise ValueError(f"Graph {name!r} contains no valid vertices after filtering.")
    junction_nodes, compressed_edges = compress_degree_two_chains(filtered_vertices, filtered_edges)
    # Chains returning to their starting junctions become self loops, which we don't currently allow
    compressed_edges = [edge for edge in compressed_edges if int(edge["u"]) != int(edge["v"])]
    if keep_largest_component:
        junction_nodes, compressed_edges = _largest_component(junction_nodes, compressed_edges)
    if not junction_nodes:
        raise ValueError(f"Graph {name!r} contains no junction vertices after preprocessing.")
    retained_coordinates = {vertex: filtered_vertices[vertex] for vertex in sorted(junction_nodes)}
    ordered_edges = sorted(compressed_edges, key=_compressed_edge_sort_key)
    junction_edges = tuple(JunctionEdge(id=edge_id, u=int(edge["u"]), v=int(edge["v"]), polyline=edge["polyline"]) for edge_id, edge in enumerate(ordered_edges))
    return JunctionGraph(name=name, coordinates=retained_coordinates, edges=junction_edges)


def load_junction_graph(path: str | Path, *, name: str | None = None, keep_largest_component: bool = True) -> JunctionGraph:
    path = Path(path)
    vertices, edges = read_topotide_graph(path)
    return preprocess_raw_graph(name=name or path.stem, vertices=vertices, edges=edges, keep_largest_component=keep_largest_component)
