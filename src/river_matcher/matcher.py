from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from operator import index
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from river_matcher.candidates import compute_candidate_sets
from river_matcher.costs.base import CostName
from river_matcher.costs.factory import CostFactory
from river_matcher.decomposition import SourceDecomposition, build_source_decomposition, validate_source_decomposition
from river_matcher.dynamic_programming import DPStatistics, DPSolution, Objective, solve_tree_dp, solve_tree_dp_both
from river_matcher.models import JunctionGraph

type FloatArray = NDArray[np.float64]
type RawCandidateSets = Mapping[int, Iterable[int]]
type NormalizedCandidateSets = Mapping[int, tuple[int, ...]]


class MatchEdgeCost(Protocol):
    name: CostName

    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float: ...

    def witness(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> FloatArray | None: ...


@dataclass(frozen=True, slots=True)
class CandidateStatistics:
    source_vertices: int
    empty_domains: int
    total_candidates: int
    minimum_candidates: int
    maximum_candidates: int


@dataclass(frozen=True, slots=True)
class MatchedEdge:
    edge_id: int
    source_u: int
    source_v: int
    target_u: int
    target_v: int
    cost: float
    witness: FloatArray


@dataclass(frozen=True, slots=True)
class MatchSolution:
    objective: Objective
    value: float
    mapping: Mapping[int, int]
    edges: tuple[MatchedEdge, ...]


@dataclass(frozen=True, slots=True)
class MatchResult:
    cost_name: CostName
    candidate_sets: NormalizedCandidateSets
    candidate_statistics: CandidateStatistics
    decomposition: SourceDecomposition
    solution: MatchSolution | None
    dp_statistics: DPStatistics

    @property
    def feasible(self) -> bool:
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class BothMatchResult:
    cost_name: CostName
    candidate_sets: NormalizedCandidateSets
    candidate_statistics: CandidateStatistics
    decomposition: SourceDecomposition
    additive: MatchSolution | None
    bottleneck: MatchSolution | None
    dp_statistics: DPStatistics

    @property
    def additive_feasible(self) -> bool:
        return self.additive is not None

    @property
    def bottleneck_feasible(self) -> bool:
        return self.bottleneck is not None


def _normalize_candidate_sets(source: JunctionGraph, target: JunctionGraph, candidate_sets: RawCandidateSets) -> dict[int, tuple[int, ...]]:
    source_vertices = set(source.vertices)
    target_vertices = set(target.vertices)
    raw: dict[int, Iterable[int]] = {}

    for raw_source_vertex, candidates in candidate_sets.items():
        source_vertex = index(raw_source_vertex)

        if source_vertex in raw:
            raise ValueError(f"Candidate sets contain duplicate source vertex {source_vertex} after integer normalization.")

        raw[source_vertex] = candidates

    unknown_source_vertices = sorted(set(raw) - source_vertices)

    if unknown_source_vertices:
        raise ValueError(f"Candidate sets contain unknown source vertices {unknown_source_vertices}.")

    normalized: dict[int, tuple[int, ...]] = {}

    for source_vertex in sorted(source_vertices):
        candidates = tuple(sorted({index(candidate) for candidate in raw.get(source_vertex, ())}))
        unknown_target_vertices = sorted(set(candidates) - target_vertices)

        if unknown_target_vertices:
            raise ValueError(f"Candidate set for source vertex {source_vertex} contains unknown target vertices " f"{unknown_target_vertices}.")

        normalized[source_vertex] = candidates

    return normalized


def _candidate_statistics(candidate_sets: NormalizedCandidateSets) -> CandidateStatistics:
    sizes = tuple(len(candidates) for candidates in candidate_sets.values())

    return CandidateStatistics(source_vertices=len(sizes), empty_domains=sum(size == 0 for size in sizes), total_candidates=sum(sizes), minimum_candidates=min(sizes, default=0),
        maximum_candidates=max(sizes, default=0), )


def _realized_value(edges: tuple[MatchedEdge, ...], objective: Objective) -> float:
    values = (edge.cost for edge in edges)

    if objective is Objective.ADDITIVE:
        return sum(values)

    return max(values, default=0.0)


def _materialize_solution(source: JunctionGraph, edge_cost: MatchEdgeCost, solution: DPSolution | None) -> MatchSolution | None:
    if solution is None:
        return None

    mapping = {int(source_vertex): int(target_vertex) for source_vertex, target_vertex in solution.mapping.items()}
    expected_vertices = set(source.vertices)
    actual_vertices = set(mapping)

    if actual_vertices != expected_vertices:
        missing = sorted(expected_vertices - actual_vertices)
        extra = sorted(actual_vertices - expected_vertices)

        raise RuntimeError(f"Recovered mapping has incorrect source vertices; missing={missing}, extra={extra}.")

    matched_edges: list[MatchedEdge] = []

    for edge in sorted(source.edges, key=lambda item: item.id):
        target_u = mapping[edge.u]
        target_v = mapping[edge.v]
        value = float(edge_cost(edge.id, edge.u, edge.v, target_u, target_v))

        if not math.isfinite(value):
            raise RuntimeError(f"Recovered solution contains nonfinite cost for source edge e{edge.id}: " f"({edge.u}, {edge.v}) -> ({target_u}, {target_v}).")

        witness = edge_cost.witness(edge.id, edge.u, edge.v, target_u, target_v)

        if witness is None:
            raise RuntimeError(f"Recovered solution has no witness for source edge e{edge.id}: " f"({edge.u}, {edge.v}) -> ({target_u}, {target_v}).")

        matched_edges.append(MatchedEdge(edge_id=edge.id, source_u=edge.u, source_v=edge.v, target_u=target_u, target_v=target_v, cost=value, witness=witness))

    edges = tuple(matched_edges)
    realized_value = _realized_value(edges, solution.objective)

    if not math.isclose(realized_value, solution.value, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(f"Recovered {solution.objective.value} solution value is inconsistent: " f"DP={solution.value}, realized={realized_value}.")

    ordered_mapping = {source_vertex: mapping[source_vertex] for source_vertex in sorted(mapping)}

    return MatchSolution(objective=solution.objective, value=solution.value, mapping=ordered_mapping, edges=edges)


class RiverGraphMatcher:
    """
    Prepared exact matching problem for one source-target graph pair.

    Candidate sets, the source decomposition, and cost resources are reused
    across repeated cost methods and objectives.
    """

    __slots__ = ("candidate_sets", "cost_factory", "decomposition", "source", "target")

    def __init__(self, source: JunctionGraph, target: JunctionGraph, *, candidate_rho: float = 10.0, top_k: int = 25, candidate_sets: RawCandidateSets | None = None,
            decomposition: SourceDecomposition | None = None, validate_decomposition: bool = True, ) -> None:
        self.source = source
        self.target = target

        raw_candidate_sets = compute_candidate_sets(source, target, rho=candidate_rho, top_k=top_k) if candidate_sets is None else candidate_sets
        self.candidate_sets = _normalize_candidate_sets(source, target, raw_candidate_sets)

        if decomposition is None:
            self.decomposition = build_source_decomposition(source, validate=validate_decomposition)
        else:
            if validate_decomposition:
                validate_source_decomposition(source, decomposition)

            self.decomposition = decomposition

        self.cost_factory = CostFactory(source, target)

    @property
    def candidate_statistics(self) -> CandidateStatistics:
        return _candidate_statistics(self.candidate_sets)

    def match(self, cost_name: CostName | str, objective: Objective | str, **cost_options: object) -> MatchResult:
        resolved_objective = Objective(objective)
        edge_cost = self.cost_factory.create(cost_name, **cost_options)
        result = solve_tree_dp(self.decomposition, self.candidate_sets, edge_cost, resolved_objective)

        return MatchResult(cost_name=edge_cost.name, candidate_sets=self.candidate_sets, candidate_statistics=self.candidate_statistics, decomposition=self.decomposition,
            solution=_materialize_solution(self.source, edge_cost, result.solution), dp_statistics=result.statistics, )

    def match_both(self, cost_name: CostName | str, **cost_options: object) -> BothMatchResult:
        edge_cost = self.cost_factory.create(cost_name, **cost_options)
        result = solve_tree_dp_both(self.decomposition, self.candidate_sets, edge_cost)

        return BothMatchResult(cost_name=edge_cost.name, candidate_sets=self.candidate_sets, candidate_statistics=self.candidate_statistics, decomposition=self.decomposition,
            additive=_materialize_solution(self.source, edge_cost, result.additive), bottleneck=_materialize_solution(self.source, edge_cost, result.bottleneck),
            dp_statistics=result.statistics, )


def match_graphs(source: JunctionGraph, target: JunctionGraph, cost_name: CostName | str, objective: Objective | str, *, candidate_rho: float = 10.0, top_k: int = 25,
        cost_options: Mapping[str, object] | None = None, ) -> MatchResult:
    """Prepare and solve one exact graph-matching objective."""
    matcher = RiverGraphMatcher(source, target, candidate_rho=candidate_rho, top_k=top_k)

    return matcher.match(cost_name, objective, **dict(cost_options or {}))


def match_graphs_both(source: JunctionGraph, target: JunctionGraph, cost_name: CostName | str, *, candidate_rho: float = 10.0, top_k: int = 25,
        cost_options: Mapping[str, object] | None = None) -> BothMatchResult:
    """Prepare and solve both exact graph-matching objectives."""
    matcher = RiverGraphMatcher(source, target, candidate_rho=candidate_rho, top_k=top_k)

    return matcher.match_both(cost_name, **dict(cost_options or {}))
