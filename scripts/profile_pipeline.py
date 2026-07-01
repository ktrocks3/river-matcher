from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

type VertexId = int
type EdgeId = int
type XYArray = NDArray[np.float64]
type Coordinates = dict[VertexId, tuple[float, float]]
type CandidatePair = tuple[VertexId, VertexId]
type CandidateSets = dict[VertexId, tuple[VertexId, ...]]
type CostTable = dict[EdgeId, dict[CandidatePair, float]]
type VertexMapping = dict[VertexId, VertexId]
type EdgeCosts = dict[EdgeId, float]


class Objective(StrEnum):
    ADDITIVE = "additive"
    BOTTLENECK = "bottleneck"


def _clean_polyline(polyline: XYArray) -> tuple[XYArray, float]:
    """Validate one coordinate polyline and calculate it's geometric length"""
    try:
        points = np.asarray(polyline, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'A polyline must contain numeric coordinates') from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"A polyline must have shape (n, 2) with n >= 2, got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"A polyline must contain finite values")

    # Remove consecutive duplicates because they create zero-length segments.
    keep = [0]
    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]
        if float(np.dot(delta, delta)) > 1e-24:
            keep.append(index)

    points = np.ascontiguousarray(points[keep], dtype=np.float64)
    if len(points) < 2:
        raise ValueError(f"A polyline must contain at least two points")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(np.max(segment_lengths))

    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"A polyline must contain finite values")

    # Geometry stored in the model should not be changed accidentally
    points.setflags(write=False)
    points = cast(XYArray, np.ascontiguousarray(points[keep], dtype=np.float64), )

    return points, length


@dataclass(frozen=True, slots=True, eq=False)
class JunctionEdge:
    """One embedded edge in a junction multigraph"""
    id: EdgeId
    u: VertexId
    v: VertexId
    polyline: XYArray = field(repr=False)
    length: float = field(repr=False)

    def __post_init__(self) -> None:
        edge_id, u, v = int(self.id), int(self.u), int(self.v)
        if edge_id < 0:
            raise ValueError(f"Edge id {edge_id} must be >= 0")
        polyline, length = _clean_polyline(self.polyline)

        object.__setattr__(self, "id", edge_id)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "polyline", polyline)
        object.__setattr__(self, "length", length)


@dataclass(frozen=True, slots=True, eq=False)
class JunctionGraph:
    """The whole embedded junction graph"""
    name: str
    coordinates: Coordinates = field(repr=False)
    edges: tuple[JunctionEdge, ...] = field(repr=False)
    vertices: tuple[VertexId, ...] = field(repr=False)
    edge_by_id: dict[EdgeId, JunctionEdge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError(f"Junction graph must have a name")

        coordinates: Coordinates = {}
        for raw_vertex, raw_point in self.coordinates.items():
            vertex = int(raw_vertex)
            if vertex in coordinates:
                raise ValueError(f"Vertex {vertex} already in graph")

            try:
                point = np.asarray(raw_point, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Vertex {vertex} must contain numeric values") from exc

            if point.shape != (2,):
                raise ValueError(f"Vertex {vertex} must have shape (2, ), got {point.shape}")
            if not np.all(np.isfinite(point)):
                raise ValueError(f"Vertex {vertex} must contain finite values")

            coordinates[vertex] = (float(point[0]), float(point[1]))

        if not coordinates:
            raise ValueError(f"Junction graph must contain at least one vertex")

        edges = tuple(self.edges)
        edge_by_id = {edge.id: edge for edge in edges}

        for edge in edges:
            if not isinstance(edge, JunctionEdge):
                raise ValueError(f"Edge {edge} must contain a JunctionEdge, got {type(edge).__name__}")
            if edge.id in edge_by_id:
                raise ValueError(f"Edge {edge.id} already in graph")
            if edge.u not in coordinates:
                raise ValueError(f"Edge {edge.u} not in graph")
            if edge.v not in coordinates:
                raise ValueError(f"Edge {edge.v} not in graph")
            edge_by_id[edge.id] = edge
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "edge_by_id", edge_by_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vertices", tuple(sorted(coordinates)))


@dataclass(frozen=True, slots=True, eq=False)
class MatchResult:
    """Phi: The result of one additive or bottleneck matching run"""
    objective: Objective
    feasible: bool
    value: float
    phi: VertexMapping = field(default_factory=dict, repr=False)
    edge_costs: EdgeCosts = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        objective = Objective(self.objective)
        feasible = bool(self.feasible)
        value = float(self.value)
        phi = {int(source): int(target) for source, target in self.phi.items()}
        edge_costs = {int(edge_id): }