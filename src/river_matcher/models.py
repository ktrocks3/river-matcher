from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

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
        raise ValueError("A polyline must contain numeric coordinates") from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"A polyline must have shape (n, 2) with n >= 2, got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("A polyline cannot contain NaN or infinite coordinates.")

    # Remove consecutive duplicates because they create zero-length segments.
    keep = [0]
    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]
        if float(np.dot(delta, delta)) > 1e-24:
            keep.append(index)

    points = np.ascontiguousarray(points[keep], dtype=np.float64)

    if len(points) < 2:
        raise ValueError("A polyline must contain at least two distinct points.")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(np.sum(segment_lengths))

    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("A polyline must have positive finite length.")

    points.setflags(write=False)

    return points, length


@dataclass(frozen=True, slots=True, eq=False)
class JunctionEdge:
    """One embedded edge in a junction multigraph."""

    id: EdgeId
    u: VertexId
    v: VertexId
    polyline: XYArray = field(repr=False)
    length: float = field(init=False)

    def __post_init__(self) -> None:
        edge_id, u, v = int(self.id), int(self.u), int(self.v)
        if edge_id < 0:
            raise ValueError(f"Edge ID must be nonnegative, got {edge_id}.")
        if u == v:
            raise ValueError(f"Self-loop e{edge_id}: ({u}, {v}) is not allowed.")
        polyline, length = _clean_polyline(self.polyline)

        object.__setattr__(self, "id", edge_id)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "polyline", polyline)
        object.__setattr__(self, "length", length)


@dataclass(frozen=True, slots=True, eq=False)
class JunctionGraph:
    """A clean embedded junction multigraph used by the matcher."""

    name: str
    coordinates: Coordinates = field(repr=False)
    edges: tuple[JunctionEdge, ...] = field(repr=False)
    vertices: tuple[VertexId, ...] = field(init=False)
    edge_by_id: dict[EdgeId, JunctionEdge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Junction graph must have a name")

        coordinates: Coordinates = {}
        for raw_vertex, raw_point in self.coordinates.items():
            vertex = int(raw_vertex)
            if vertex in coordinates:
                raise ValueError(f"Duplicate vertex ID: {vertex}")

            try:
                point = np.asarray(raw_point, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Vertex {vertex} must contain numeric values") from exc

            if point.shape != (2,):
                raise ValueError(f"Vertex {vertex} must have shape (2), got {point.shape}")
            if not np.all(np.isfinite(point)):
                raise ValueError(f"Vertex {vertex} contains NaN or infinite coordinates")

            coordinates[vertex] = (float(point[0]), float(point[1]))

        if not coordinates:
            raise ValueError("Junction graph must contain at least one vertex")

        edges = tuple(self.edges)
        edge_by_id: dict[EdgeId, JunctionEdge] = {}

        for edge in edges:
            if not isinstance(edge, JunctionEdge):
                raise TypeError("Every graph edge must be a JunctionEdge instance, got {type(edge).__name__}.")
            if edge.id in edge_by_id:
                raise ValueError(f"Edge ID {edge.id} is already in graph.")
            if edge.u not in coordinates:
                raise ValueError(f"Edge e{edge.id} refers to missing endpoint {edge.u}.")
            if edge.v not in coordinates:
                raise ValueError(f"Edge e{edge.id} refers to missing endpoint {edge.v}.")

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
        edge_costs = {int(edge_id): float(cost) for edge_id, cost in self.edge_costs.items()}

        if feasible:
            if not math.isfinite(value):
                raise ValueError("A feasible match must have a finite objective value.")
            invalid_edges = [edge_id for edge_id, cost in edge_costs.items() if not math.isfinite(cost)]
            if invalid_edges:
                raise ValueError("A feasible match cannot contain non-finite edge costs: {invalid_edges}.")
        else:
            if value != float("inf"):
                raise ValueError("An infeasible match must use positive infinity as its value.")
            if phi:
                raise ValueError("An infeasible match cannot contain a vertex mapping")
            if edge_costs:
                raise ValueError("An infeasible match cannot contain edge costs")

        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "edge_costs", edge_costs)

    @classmethod
    def infeasible(cls, objective: Objective):
        return cls(objective=objective, feasible=False, value=float("inf"))
