from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from river_matcher.geometry import (
    PreparedPolyline,
    XYArray,
    orient_polyline,
    points_to_prepared_polyline_distances,
    polyline_length,
    prepare_polyline_segments,
    sample_polyline_by_arclength,
)
from river_matcher.models import JunctionEdge, JunctionGraph

type FloatArray = NDArray[np.float64]
type WitnessRequest = tuple[int, int, int, int, int]
type WitnessPaths = dict[WitnessRequest, XYArray | None]
type TargetPair = tuple[int, int]
type TargetPairPaths = dict[TargetPair, XYArray | None]
type Arc = tuple[int, float, XYArray]
type Adjacency = dict[int, list[Arc]]
type DistanceMap = dict[int, float]
type ParentMap = dict[int, int | None]
type ParentSegmentMap = dict[int, XYArray]
type ShortestPathTree = tuple[DistanceMap, ParentMap, ParentSegmentMap]

_TIE_TOLERANCE = 1e-15
_LENGTH_TIE_BREAK = 1e-6


@dataclass(frozen=True, slots=True)
class _TargetEdgeRecord:
    id: int
    u: int
    v: int
    length: float
    forward: XYArray
    reverse: XYArray
    samples: XYArray | None = None


@dataclass(slots=True)
class WitnessTiming:
    adjacency_seconds: float = 0.0
    adjacency_builds: int = 0
    dijkstra_seconds: float = 0.0
    dijkstra_runs: int = 0


def _float64_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _readonly_xy(value: Any) -> XYArray:
    points = cast(XYArray, np.ascontiguousarray(value, dtype=np.float64))
    points.setflags(write=False)
    return points


def _prepare_target_edges(target: JunctionGraph, *, samples: int | None = None) -> tuple[_TargetEdgeRecord, ...]:
    records: list[_TargetEdgeRecord] = []
    for edge in target.edges:
        forward = orient_polyline(edge.polyline, target.coordinates[edge.u], target.coordinates[edge.v])
        if forward is None:
            raise RuntimeError(f"Could not orient target edge e{edge.id} from vertex {edge.u} to vertex {edge.v}.")
        length = polyline_length(forward)
        if not math.isfinite(length) or length <= 1e-12:
            f"Could not orient target edge e{edge.id} from vertex {edge.u} to vertex {edge.v}."

        stored_forward = _readonly_xy(forward)
        stored_reverse = _readonly_xy(forward[::-1])
        sampled: XYArray | None = None

        if samples is not None:
            sampled = sample_polyline_by_arclength(stored_forward, samples)
            if sampled is None:
                raise RuntimeError(f"Could not sample target edge e{edge.id}.")
            sampled = _readonly_xy(sampled)
        records.append(_TargetEdgeRecord(id=edge.id, u=edge.u, v=edge.v, length=length, forward=stored_forward, reverse=stored_reverse, samples=sampled))
    return tuple(records)


def _dijkstra(adjacency: Mapping[int, list[Arc]], start: int) -> ShortestPathTree:
    queue: list[tuple[float, int]] = [(0.0, start)]
    distances: DistanceMap = {start: 0.0}
    parent: ParentMap = {start: None}
    parent_segment: ParentSegmentMap = {}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current] + _TIE_TOLERANCE:
            continue

        for neighbour, weight, segment in adjacency.get(current, []):
            new_distance = distance + weight
            if new_distance < distances.get(neighbour, math.inf):
                distances[neighbour] = new_distance
                parent[neighbour] = current
                parent_segment[neighbour] = segment
                heapq.heappush(queue, (new_distance, neighbour))
    return distances, parent, parent_segment


def _reconstruct_path(end: int, parent: ParentMap, parent_segment: ParentSegmentMap) -> XYArray | None:
    """Concatenate the oriented edge polylines in one shortest-path tree."""
    if end not in parent:
        return None
    nodes: list[int] = []
    current: int | None = end
    while current is not None:
        nodes.append(current)
        current = parent[current]
    nodes.reverse()
    segments = [parent_segment[node] for node in nodes[1:]]
    if not segments:
        return None
    pieces: list[XYArray] = [segments[0]]
    pieces.extend(segment[1:] for segment in segments[1:])
    return _readonly_xy(np.vstack(pieces))


class ShortestPathWitnessFinder:
    """ Resolve ordinary geometric-length the shortest paths in a target multigraph.
        Parallel edges remain separate adjacency entries and retain their own geometry during path reconstruction."""

    def __init__(self, target: JunctionGraph) -> None:
        self.target = target
        self._target_vertices = set(target.vertices)
        self._records = _prepare_target_edges(target)
        self._adjacency = self._build_adjacency()
        self._tree_cache: dict[int, ShortestPathTree] = {}
        self._path_cache: TargetPairPaths = {}

    def _build_adjacency(self) -> Adjacency:
        adjacency: defaultdict[int, list[Arc]] = defaultdict(list)
        for record in self._records:
            adjacency[record.u].append((record.v, record.length, record.forward))
            adjacency[record.v].append((record.u, record.length, record.reverse))
        return dict(adjacency)

    def _tree(self, start: int) -> ShortestPathTree:
        if start in self._tree_cache:
            return self._tree_cache[start]
        if start not in self._target_vertices:
            tree: ShortestPathTree = ({}, {}, {})
        else:
            tree = _dijkstra(self._adjacency, start)
        self._tree_cache[start] = tree
        return tree

    def distance(self, start: int, end: int) -> float:
        """Return the geometric shortest-path length."""
        if (start not in self._target_vertices) or (end not in self._target_vertices):
            return math.inf
        if start == end:
            return 0.0
        distances, _, _ = self._tree(start)
        return distances.get(end, math.inf)

    def path(self, start: int, end: int) -> XYArray | None:
        """Return the geometric shortest-path polyline from start to end."""
        key = (start, end)
        if key in self._path_cache:
            return self._path_cache[key]
        if (start == end) or (start not in self._target_vertices) or (end not in self._target_vertices):
            self._path_cache[key] = None
            return None
        _, parent, parent_segment = self._tree(start)
        path = _reconstruct_path(end, parent, parent_segment)
        self._path_cache[key] = path

        if path is not None:
            self._path_cache.setdefault((end, start), _readonly_xy(path[::-1]))
        return path

    def paths(self, pairs: Iterable[TargetPair]) -> TargetPairPaths:
        output: TargetPairPaths = {}
        for raw_start, raw_end in pairs:
            key = (int(raw_start), int(raw_end))
            output[key] = self.path(*key)
        return output

    def clear_cache(self) -> None:
        self._tree_cache.clear()
        self._path_cache.clear()


class SourceGuidedWitnessFinder:
    """ Resolve target paths using one corridor-weighted graph per source edge.
        Target edge weights are estimated from equally spaced samples and remain distinct for parallel target edges."""

    def __init__(self, source: JunctionGraph, target: JunctionGraph, *, rho: float, edge_samples: int = 12) -> None:
        radius = float(rho)
        sample_count = int(edge_samples)

        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"Witness radius must be positive and finite, got {radius}")
        if sample_count < 2:
            raise ValueError(f"Witness edge sample count must be at least 2, got {sample_count}")

        self.source: JunctionGraph = source
        self.target: JunctionGraph = target
        self.rho: float = radius
        self.edge_samples: int = sample_count
        self.timing = WitnessTiming()
        self._source_edges: dict[int, JunctionEdge] = {edge.id: edge for edge in self.source.edges}
        self._target_vertices = set(target.vertices)
        self._target_records = _prepare_target_edges(target, samples=sample_count)
        target_samples = [record.samples for record in self._target_records if record.samples is not None]
        if len(target_samples) != len(self._target_records):
            raise RuntimeError("At least one prepared target edge has no samples.")
        if target_samples:
            self._target_samples = _readonly_xy(np.vstack(target_samples))
        else:
            self._target_samples = _readonly_xy(np.empty((0, 2), dtype=np.float64))
        self._target_lengths = _float64_array([record.length for record in self._target_records])
        self._canonical_source_cache: dict[int, XYArray | None] = {}
        self._source_cache: dict[tuple[int, int, int], XYArray | None] = {}
        self._adjacency_cache: dict[int, Adjacency] = {}
        self._tree_cache: dict[tuple[int, int], ShortestPathTree] = {}
        self._path_cache: WitnessPaths = {}

    def _canonical_source_polyline(self, edge_id: int) -> XYArray | None:
        if edge_id in self._canonical_source_cache:
            return self._canonical_source_cache[edge_id]
        edge = self._source_edges.get(edge_id, None)
        if edge is None:
            self._canonical_source_cache[edge_id] = None
            return None
        polyline = orient_polyline(edge.polyline, self.source.coordinates[edge.u], self.source.coordinates[edge.v])
        if polyline is None:
            self._canonical_source_cache[edge_id] = None
            return None
        stored = _readonly_xy(polyline)
        self._canonical_source_cache[edge_id] = stored
        return stored

    def source_polyline(self, edge_id: int, u: int, v: int) -> XYArray | None:
        """Return one source multi-edge oriented from u toward v."""
        key = (edge_id, u, v)
        if key in self._source_cache:
            return self._source_cache[key]
        edge = self._source_edges.get(edge_id, None)
        canonical = self._canonical_source_polyline(edge_id)

        if edge is None or canonical is None:
            self._source_cache[key] = None
            return None

        if u == edge.u and v == edge.v:
            polyline = canonical
        elif u == edge.v and v != edge.u:
            polyline = _readonly_xy(canonical[::-1])
        else:
            self._source_cache[key] = None
            return None
        self._source_cache[key] = polyline
        if polyline is not None:
            self._source_cache.setdefault((edge_id, v, u), _readonly_xy(polyline[::-1]))
        return polyline

    def _corridor_weights(self, prepared_source: PreparedPolyline) -> FloatArray:
        if not self._target_records:
            return _float64_array(np.empty(0, dtype=np.float64))
        distances = points_to_prepared_polyline_distances(self._target_samples, prepared_source)
        if distances is None:
            return _float64_array(self._target_lengths * (1 + _LENGTH_TIE_BREAK))
        distance_matrix = _float64_array(distances.reshape(len(self._target_records), self.edge_samples))
        finite_rows = np.all(np.isfinite(distance_matrix), axis=1)
        fractions = _float64_array(np.mean(distance_matrix > self.rho, axis=1))
        fractions = fractions.copy()
        fractions[~finite_rows] = 1.0
        return _float64_array(fractions * self._target_lengths + _LENGTH_TIE_BREAK * self._target_lengths)

    def _adjacency_for_source(self, edge_id: int) -> Adjacency:
        if edge_id in self._adjacency_cache:
            return self._adjacency_cache[edge_id]
        started = time.perf_counter()
        source = self._canonical_source_polyline(edge_id)
        prepared_source = prepare_polyline_segments(source)
        adjacency: defaultdict[int, list[Arc]] = defaultdict(list)
        if prepared_source is not None:
            weights = self._corridor_weights(prepared_source)
            for index, record in enumerate(self._target_records):
                weight = float(weights[index])
                adjacency[record.u].append((record.v, weight, record.forward))
                adjacency[record.v].append((record.u, weight, record.reverse))
        result = dict(adjacency)
        self._adjacency_cache[edge_id] = result
        self.timing.adjacency_seconds += time.perf_counter() - started
        self.timing.adjacency_seconds += 1
        return result

    def _tree(self, edge_id: int, start: int) -> ShortestPathTree:
        key = (edge_id, start)
        if key in self._tree_cache:
            return self._tree_cache[key]
        started = time.perf_counter()
        if (edge_id not in self._source_edges) or (start not in self._target_vertices):
            tree: ShortestPathTree = ({}, {}, {})
        else:
            tree = _dijkstra(self._adjacency_for_source(edge_id), start)
        self._tree_cache[key] = tree
        self.timing.dijkstra_seconds += time.perf_counter() - started
        self.timing.dijkstra_runs += 1
        return tree

    def path(self, edge_id: int, u: int, v: int, target_start: int, target_end: int) -> XYArray | None:
        key = (edge_id, u, v, target_start, target_end)
        if key in self._path_cache:
            return self._path_cache[key]
        if target_start == target_end or target_start not in self._target_vertices or target_end not in self._target_vertices or self.source_polyline(edge_id, u, v) is None:
            self._path_cache[key] = None
            return None
        _, parent, parent_segment = self._tree(edge_id, target_start)
        path = _reconstruct_path(target_end, parent, parent_segment)
        self._path_cache[key] = path

        if path is not None:
            self._path_cache.setdefault((edge_id, v, u, target_end, target_start), _readonly_xy(path[::-1]))
        return path

    def paths(self, requests: Iterable[WitnessRequest]) -> WitnessPaths:
        """ Resolve several witness requests.
            The tree cache ensures that requests sharing a source edge and mapped start vertex reuse one Dijkstra run."""
        output: WitnessPaths = {}
        for raw_request in requests:
            request = tuple(int(value) for value in raw_request)
            if len(request) != 5:
                raise ValueError("Each witness request must contain (edge_id, u, v, target_start, target_end).")
            key: WitnessRequest = request
            output[key] = self.path(*key)
        return output

    def clear_cache(self):
        self._canonical_source_cache.clear()
        self._source_cache.clear()
        self._adjacency_cache.clear()
        self._tree_cache.clear()
        self._path_cache.clear()
        self.timing = WitnessTiming()
