from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray

from river_matcher.cancellation import CancellationToken
from river_matcher.candidates import compute_candidate_sets
from river_matcher.compatibility import CompatibilityStatistics, TargetConnectivityCompatibility, enforce_arc_consistency
from river_matcher.costs.base import BaseEdgeCost, CostName
from river_matcher.costs.factory import CostFactory
from river_matcher.decomposition import SourceDecomposition, build_source_decomposition, validate_source_decomposition
from river_matcher.dynamic_programming import DPStatistics, DPSolution, Objective, solve_tree_dp, solve_tree_dp_all, solve_tree_dp_both, solve_tree_feasibility
from river_matcher.models import JunctionGraph
from river_matcher.preflight import MatchingPreflight, estimate_matching

type FloatArray = NDArray[np.float64]
type RawCandidateSets = Mapping[int, Iterable[int]]
type NormalizedCandidateSets = Mapping[int, tuple[int, ...]]
type ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CandidateStatistics:
    source_vertices: int
    empty_domains: int
    total_candidates: int
    minimum_candidates: int
    maximum_candidates: int


@dataclass(frozen=True, slots=True)
class MatchTiming:
    arc_consistency_seconds: float = 0.0
    feasibility_dp_seconds: float = 0.0
    cost_setup_seconds: float = 0.0
    cost_dp_seconds: float = 0.0
    materialization_seconds: float = 0.0
    uncached_local_cost_seconds: float = 0.0
    uncached_local_cost_calls: int = 0
    local_cost_cache_hits: int = 0
    witness_adjacency_seconds: float = 0.0
    witness_adjacency_builds: int = 0
    witness_dijkstra_seconds: float = 0.0
    witness_dijkstra_runs: int = 0
    feasibility_reused: bool = False

    @property
    def measured_pipeline_seconds(self) -> float:
        return (self.arc_consistency_seconds + self.feasibility_dp_seconds + self.cost_setup_seconds + self.cost_dp_seconds + self.materialization_seconds)


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
class MappingEvaluation:
    cost_name: CostName
    mapping: Mapping[int, int]
    edges: tuple[MatchedEdge, ...]
    additive_value: float
    bottleneck_value: float
    length_weighted_additive_value: float
    invalid_edge_ids: tuple[int, ...]
    candidate_violations: tuple[int, ...]
    effective_candidate_violations: tuple[int, ...]
    timing: MatchTiming | None = None

    @property
    def feasible(self) -> bool:
        return not self.invalid_edge_ids


@dataclass(frozen=True, slots=True)
class MatchResult:
    cost_name: CostName
    candidate_sets: NormalizedCandidateSets
    candidate_statistics: CandidateStatistics
    decomposition: SourceDecomposition
    solution: MatchSolution | None
    dp_statistics: DPStatistics
    effective_candidate_sets: NormalizedCandidateSets | None = None
    effective_candidate_statistics: CandidateStatistics | None = None
    preflight: MatchingPreflight | None = None
    effective_preflight: MatchingPreflight | None = None
    compatibility_statistics: CompatibilityStatistics | None = None
    feasibility_statistics: DPStatistics | None = None
    timing: MatchTiming | None = None

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
    effective_candidate_sets: NormalizedCandidateSets | None = None
    effective_candidate_statistics: CandidateStatistics | None = None
    preflight: MatchingPreflight | None = None
    effective_preflight: MatchingPreflight | None = None
    compatibility_statistics: CompatibilityStatistics | None = None
    feasibility_statistics: DPStatistics | None = None
    timing: MatchTiming | None = None

    @property
    def additive_feasible(self) -> bool:
        return self.additive is not None

    @property
    def bottleneck_feasible(self) -> bool:
        return self.bottleneck is not None


@dataclass(frozen=True, slots=True)
class AggregationMatchResult:
    cost_name: CostName
    candidate_sets: NormalizedCandidateSets
    candidate_statistics: CandidateStatistics
    decomposition: SourceDecomposition
    additive: MatchSolution | None
    bottleneck: MatchSolution | None
    length_weighted_additive: MatchSolution | None
    dp_statistics: DPStatistics
    effective_candidate_sets: NormalizedCandidateSets | None = None
    effective_candidate_statistics: CandidateStatistics | None = None
    preflight: MatchingPreflight | None = None
    effective_preflight: MatchingPreflight | None = None
    compatibility_statistics: CompatibilityStatistics | None = None
    feasibility_statistics: DPStatistics | None = None
    timing: MatchTiming | None = None


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
            raise ValueError(f"Candidate set for source vertex {source_vertex} contains unknown target vertices {unknown_target_vertices}.")

        normalized[source_vertex] = candidates

    return normalized


def _normalize_mapping(source: JunctionGraph, target: JunctionGraph, mapping: Mapping[int, int]) -> dict[int, int]:
    source_vertices = set(source.vertices)
    target_vertices = set(target.vertices)
    normalized: dict[int, int] = {}

    for raw_source, raw_target in mapping.items():
        source_vertex = index(raw_source)
        target_vertex = index(raw_target)

        if source_vertex in normalized:
            raise ValueError(f"Mapping contains duplicate source vertex {source_vertex} after integer normalization.")

        normalized[source_vertex] = target_vertex

    actual_sources = set(normalized)
    missing = sorted(source_vertices - actual_sources)
    extra = sorted(actual_sources - source_vertices)

    if missing or extra:
        raise ValueError(f"Mapping must assign every source vertex exactly once; missing={missing}, extra={extra}.")

    unknown_targets = sorted(set(normalized.values()) - target_vertices)

    if unknown_targets:
        raise ValueError(f"Mapping contains unknown target vertices {unknown_targets}.")

    return {source_vertex: normalized[source_vertex] for source_vertex in sorted(normalized)}


def _candidate_statistics(candidate_sets: NormalizedCandidateSets) -> CandidateStatistics:
    sizes = tuple(len(candidates) for candidates in candidate_sets.values())
    return CandidateStatistics(source_vertices=len(sizes), empty_domains=sum(size == 0 for size in sizes), total_candidates=sum(sizes), minimum_candidates=min(sizes, default=0),
        maximum_candidates=max(sizes, default=0), )


def _edge_weights(source: JunctionGraph) -> dict[int, float]:
    return {edge.id: edge.length for edge in source.edges}


def _realized_value(source: JunctionGraph, edges: tuple[MatchedEdge, ...], objective: Objective) -> float:
    if objective is Objective.ADDITIVE:
        return sum(edge.cost for edge in edges)

    if objective is Objective.LENGTH_WEIGHTED_ADDITIVE:
        return sum(source.edge_by_id[edge.edge_id].length * edge.cost for edge in edges)

    return max((edge.cost for edge in edges), default=0.0)


def _materialize_solution(source: JunctionGraph, edge_cost: BaseEdgeCost, solution: DPSolution | None, *,
        cancellation_token: CancellationToken | None = None, ) -> MatchSolution | None:
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

    for edge_index, edge in enumerate(sorted(source.edges, key=lambda item: item.id)):
        if cancellation_token is not None and edge_index % 32 == 0:
            cancellation_token.check()

        target_u = mapping[edge.u]
        target_v = mapping[edge.v]
        value = float(edge_cost(edge.id, edge.u, edge.v, target_u, target_v))

        if not math.isfinite(value):
            raise RuntimeError(f"Recovered solution contains nonfinite cost for source edge e{edge.id}: "
                               f"({edge.u}, {edge.v}) -> ({target_u}, {target_v}).")

        witness = edge_cost.witness(edge.id, edge.u, edge.v, target_u, target_v)

        if witness is None:
            raise RuntimeError(f"Recovered solution has no witness for source edge e{edge.id}: "
                               f"({edge.u}, {edge.v}) -> ({target_u}, {target_v}).")

        matched_edges.append(MatchedEdge(edge_id=edge.id, source_u=edge.u, source_v=edge.v, target_u=target_u, target_v=target_v, cost=value, witness=witness, ))

    edges = tuple(matched_edges)
    realized_value = _realized_value(source, edges, solution.objective)

    if not math.isclose(realized_value, solution.value, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(f"Recovered {solution.objective.value} solution value is inconsistent: "
                           f"DP={solution.value}, realized={realized_value}.")

    ordered_mapping = {source_vertex: mapping[source_vertex] for source_vertex in sorted(mapping)}
    return MatchSolution(objective=solution.objective, value=solution.value, mapping=ordered_mapping, edges=edges)


def _empty_statistics() -> DPStatistics:
    return DPStatistics(bags=(), unique_cost_requests=0)


def _cost_key(name: CostName | str, options: Mapping[str, object]) -> tuple[CostName, str]:
    resolved = CostName(name)
    serialized = json.dumps(dict(options), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return resolved, serialized


class RiverGraphMatcher:
    """Prepared exact matching problem with reusable pruning, feasibility, cost and path caches."""

    __slots__ = ("candidate_sets", "cost_factory", "decomposition", "preflight", "source", "target", "_arc_consistency_seconds", "_compatibility", "_compatibility_statistics",
                 "_costs", "_effective_candidate_sets", "_effective_preflight", "_feasibility_dp_seconds", "_feasibility_statistics", "_feasible", "_lock",)

    def __init__(self, source: JunctionGraph, target: JunctionGraph, *, candidate_rho: float = 10.0, top_k: int = 25, candidate_sets: RawCandidateSets | None = None,
            decomposition: SourceDecomposition | None = None, validate_decomposition: bool = True, ) -> None:
        self.source = source
        self.target = target
        raw_candidate_sets = (compute_candidate_sets(source, target, rho=candidate_rho, top_k=top_k) if candidate_sets is None else candidate_sets)
        self.candidate_sets = _normalize_candidate_sets(source, target, raw_candidate_sets)

        if decomposition is None:
            self.decomposition = build_source_decomposition(source, validate=validate_decomposition)
        else:
            if validate_decomposition:
                validate_source_decomposition(source, decomposition)
            self.decomposition = decomposition

        self.preflight = estimate_matching(self.decomposition, self.candidate_sets)
        self.cost_factory = CostFactory(source, target)
        self._compatibility = TargetConnectivityCompatibility(source, target)
        self._arc_consistency_seconds = 0.0
        self._feasibility_dp_seconds = 0.0
        self._effective_candidate_sets: dict[int, tuple[int, ...]] | None = None
        self._compatibility_statistics: CompatibilityStatistics | None = None
        self._effective_preflight: MatchingPreflight | None = None
        self._feasible: bool | None = None
        self._feasibility_statistics: DPStatistics | None = None
        self._costs: dict[tuple[CostName, str], BaseEdgeCost] = {}
        self._lock = threading.RLock()

    @property
    def candidate_statistics(self) -> CandidateStatistics:
        return _candidate_statistics(self.candidate_sets)

    @property
    def effective_candidate_sets(self) -> NormalizedCandidateSets:
        return self.candidate_sets if self._effective_candidate_sets is None else self._effective_candidate_sets

    @property
    def effective_candidate_statistics(self) -> CandidateStatistics:
        return _candidate_statistics(self.effective_candidate_sets)

    @property
    def effective_preflight(self) -> MatchingPreflight:
        return self.preflight if self._effective_preflight is None else self._effective_preflight

    def prepare_feasibility(self, *, cancellation_token: CancellationToken | None = None, progress: ProgressCallback | None = None, ) -> bool:
        with self._lock:
            if self._feasible is not None:
                return self._feasible

            if cancellation_token is not None:
                cancellation_token.check()

            if progress is not None:
                progress("Pruning incompatible candidates")

            started = time.perf_counter()
            effective, compatibility_statistics = enforce_arc_consistency(self._compatibility, self.candidate_sets, cancellation_token=cancellation_token, )
            self._arc_consistency_seconds += time.perf_counter() - started
            effective_preflight = estimate_matching(self.decomposition, effective)

            if effective_preflight.empty_domains:
                feasible = False
                feasibility_statistics = _empty_statistics()
            else:
                if progress is not None:
                    progress("Checking global feasibility")

                started = time.perf_counter()
                feasibility = solve_tree_feasibility(self.decomposition, effective, self._compatibility, cancellation_token=cancellation_token, )
                self._feasibility_dp_seconds += time.perf_counter() - started
                feasible = feasibility.feasible
                feasibility_statistics = feasibility.statistics

            if cancellation_token is not None:
                cancellation_token.check()

            self._effective_candidate_sets = effective
            self._compatibility_statistics = compatibility_statistics
            self._effective_preflight = effective_preflight
            self._feasible = feasible
            self._feasibility_statistics = feasibility_statistics
            return feasible

    def _cost(self, cost_name: CostName | str, options: Mapping[str, object]) -> BaseEdgeCost:
        key = _cost_key(cost_name, options)
        cost = self._costs.get(key)

        if cost is None:
            cost = self.cost_factory.create(key[0], **dict(options))
            self._costs[key] = cost

        return cost

    def _common_metadata(self) -> dict[str, object]:
        return {"candidate_sets": self.candidate_sets, "candidate_statistics": self.candidate_statistics, "decomposition": self.decomposition,
            "effective_candidate_sets": self.effective_candidate_sets, "effective_candidate_statistics": self.effective_candidate_statistics, "preflight": self.preflight,
            "effective_preflight": self.effective_preflight, "compatibility_statistics": self._compatibility_statistics, "feasibility_statistics": self._feasibility_statistics, }

    def _feasibility_timing(self, arc_before: float, feasibility_before: float, reused: bool) -> MatchTiming:
        return MatchTiming(arc_consistency_seconds=self._arc_consistency_seconds - arc_before, feasibility_dp_seconds=self._feasibility_dp_seconds - feasibility_before,
            feasibility_reused=reused, )

    def evaluate_mapping(self, mapping: Mapping[int, int], cost_name: CostName | str, *, cancellation_token: CancellationToken | None = None,
            progress: ProgressCallback | None = None, **cost_options: object, ) -> MappingEvaluation:
        """Evaluate one fixed vertex mapping without running the optimization DP.

        The mapping score is the sum (additive) or maximum (bottleneck) of the
        selected local edge costs. Witnesses are recomputed using the selected
        cost's own witness construction and options.
        """
        with self._lock:
            normalized = _normalize_mapping(self.source, self.target, mapping)
            resolved_name = CostName(cost_name)

            if cancellation_token is not None:
                cancellation_token.check()

            if progress is not None:
                progress(f"Scoring imported mapping with {resolved_name.value.replace('_', ' ')}")

            started = time.perf_counter()
            edge_cost = self._cost(resolved_name, cost_options)
            cost_setup_seconds = time.perf_counter() - started
            cost_seconds_before = edge_cost.timing.uncached_seconds
            cost_calls_before = edge_cost.timing.uncached_calls
            cost_hits_before = edge_cost.timing.cache_hits
            witness_before = self.cost_factory.resources.guided_timing()
            started = time.perf_counter()
            matched_edges: list[MatchedEdge] = []
            invalid_edge_ids: list[int] = []

            for edge_index, edge in enumerate(sorted(self.source.edges, key=lambda item: item.id)):
                if cancellation_token is not None and edge_index % 32 == 0:
                    cancellation_token.check()

                target_u = normalized[edge.u]
                target_v = normalized[edge.v]
                value = float(edge_cost(edge.id, edge.u, edge.v, target_u, target_v))

                if not math.isfinite(value):
                    invalid_edge_ids.append(edge.id)
                    continue

                witness = edge_cost.witness(edge.id, edge.u, edge.v, target_u, target_v)

                if witness is None:
                    invalid_edge_ids.append(edge.id)
                    continue

                matched_edges.append(MatchedEdge(edge_id=edge.id, source_u=edge.u, source_v=edge.v, target_u=target_u, target_v=target_v, cost=value, witness=witness, ))

            evaluation_seconds = time.perf_counter() - started
            witness_after = self.cost_factory.resources.guided_timing()
            edges = tuple(matched_edges)
            feasible = not invalid_edge_ids
            additive_value = sum(edge.cost for edge in edges) if feasible else math.inf
            bottleneck_value = max((edge.cost for edge in edges), default=0.0) if feasible else math.inf
            length_weighted_additive_value = (sum(self.source.edge_by_id[edge.edge_id].length * edge.cost for edge in edges) if feasible else math.inf)
            candidate_violations = tuple(source_vertex for source_vertex, target_vertex in normalized.items() if target_vertex not in self.candidate_sets[source_vertex])
            effective_candidate_violations = tuple(
                source_vertex for source_vertex, target_vertex in normalized.items() if target_vertex not in self.effective_candidate_sets[source_vertex])
            timing = MatchTiming(cost_setup_seconds=cost_setup_seconds, materialization_seconds=evaluation_seconds,
                uncached_local_cost_seconds=edge_cost.timing.uncached_seconds - cost_seconds_before, uncached_local_cost_calls=edge_cost.timing.uncached_calls - cost_calls_before,
                local_cost_cache_hits=edge_cost.timing.cache_hits - cost_hits_before, witness_adjacency_seconds=witness_after.adjacency_seconds - witness_before.adjacency_seconds,
                witness_adjacency_builds=witness_after.adjacency_builds - witness_before.adjacency_builds,
                witness_dijkstra_seconds=witness_after.dijkstra_seconds - witness_before.dijkstra_seconds,
                witness_dijkstra_runs=witness_after.dijkstra_runs - witness_before.dijkstra_runs, )
            return MappingEvaluation(cost_name=resolved_name, mapping=normalized, edges=edges, additive_value=additive_value, bottleneck_value=bottleneck_value,
                length_weighted_additive_value=length_weighted_additive_value, invalid_edge_ids=tuple(invalid_edge_ids), candidate_violations=candidate_violations,
                effective_candidate_violations=effective_candidate_violations, timing=timing, )

    def match(self, cost_name: CostName | str, objective: Objective | str, *, cancellation_token: CancellationToken | None = None, progress: ProgressCallback | None = None,
            **cost_options: object, ) -> MatchResult:
        resolved_name = CostName(cost_name)
        resolved_objective = Objective(objective)
        feasibility_reused = self._feasible is not None
        arc_before = self._arc_consistency_seconds
        feasibility_before = self._feasibility_dp_seconds
        feasible = self.prepare_feasibility(cancellation_token=cancellation_token, progress=progress)
        timing = self._feasibility_timing(arc_before, feasibility_before, feasibility_reused)
        metadata = self._common_metadata()

        if not feasible:
            return MatchResult(cost_name=resolved_name, solution=None, dp_statistics=self._feasibility_statistics or _empty_statistics(), timing=timing, **metadata, )

        if progress is not None:
            progress(f"Evaluating {resolved_name.value.replace('_', ' ')}")

        started = time.perf_counter()
        edge_cost = self._cost(resolved_name, cost_options)
        cost_setup_seconds = time.perf_counter() - started
        cost_before = edge_cost.timing
        cost_seconds_before = cost_before.uncached_seconds
        cost_calls_before = cost_before.uncached_calls
        cost_hits_before = cost_before.cache_hits
        witness_before = self.cost_factory.resources.guided_timing()
        started = time.perf_counter()
        result = solve_tree_dp(self.decomposition, self.effective_candidate_sets, edge_cost, resolved_objective, compatibility=self._compatibility,
            cancellation_token=cancellation_token, edge_weights=_edge_weights(self.source), )
        cost_dp_seconds = time.perf_counter() - started
        started = time.perf_counter()
        solution = _materialize_solution(self.source, edge_cost, result.solution, cancellation_token=cancellation_token, )
        materialization_seconds = time.perf_counter() - started
        witness_after = self.cost_factory.resources.guided_timing()
        timing = MatchTiming(arc_consistency_seconds=timing.arc_consistency_seconds, feasibility_dp_seconds=timing.feasibility_dp_seconds, cost_setup_seconds=cost_setup_seconds,
            cost_dp_seconds=cost_dp_seconds, materialization_seconds=materialization_seconds, uncached_local_cost_seconds=edge_cost.timing.uncached_seconds - cost_seconds_before,
            uncached_local_cost_calls=edge_cost.timing.uncached_calls - cost_calls_before, local_cost_cache_hits=edge_cost.timing.cache_hits - cost_hits_before,
            witness_adjacency_seconds=witness_after.adjacency_seconds - witness_before.adjacency_seconds,
            witness_adjacency_builds=witness_after.adjacency_builds - witness_before.adjacency_builds,
            witness_dijkstra_seconds=witness_after.dijkstra_seconds - witness_before.dijkstra_seconds,
            witness_dijkstra_runs=witness_after.dijkstra_runs - witness_before.dijkstra_runs, feasibility_reused=timing.feasibility_reused, )
        return MatchResult(cost_name=edge_cost.name, solution=solution, dp_statistics=result.statistics, timing=timing, **metadata)

    def match_both(self, cost_name: CostName | str, *, cancellation_token: CancellationToken | None = None, progress: ProgressCallback | None = None,
            **cost_options: object, ) -> BothMatchResult:
        resolved_name = CostName(cost_name)
        feasibility_reused = self._feasible is not None
        arc_before = self._arc_consistency_seconds
        feasibility_before = self._feasibility_dp_seconds
        feasible = self.prepare_feasibility(cancellation_token=cancellation_token, progress=progress)
        timing = self._feasibility_timing(arc_before, feasibility_before, feasibility_reused)
        metadata = self._common_metadata()

        if not feasible:
            return BothMatchResult(cost_name=resolved_name, additive=None, bottleneck=None, dp_statistics=self._feasibility_statistics or _empty_statistics(), timing=timing,
                **metadata, )

        if progress is not None:
            progress(f"Evaluating {resolved_name.value.replace('_', ' ')}")

        started = time.perf_counter()
        edge_cost = self._cost(resolved_name, cost_options)
        cost_setup_seconds = time.perf_counter() - started
        cost_before = edge_cost.timing
        cost_seconds_before = cost_before.uncached_seconds
        cost_calls_before = cost_before.uncached_calls
        cost_hits_before = cost_before.cache_hits
        witness_before = self.cost_factory.resources.guided_timing()
        started = time.perf_counter()
        result = solve_tree_dp_both(self.decomposition, self.effective_candidate_sets, edge_cost, compatibility=self._compatibility, cancellation_token=cancellation_token, )
        cost_dp_seconds = time.perf_counter() - started
        started = time.perf_counter()
        additive = _materialize_solution(self.source, edge_cost, result.additive, cancellation_token=cancellation_token, )
        bottleneck = _materialize_solution(self.source, edge_cost, result.bottleneck, cancellation_token=cancellation_token, )
        materialization_seconds = time.perf_counter() - started
        witness_after = self.cost_factory.resources.guided_timing()
        timing = MatchTiming(arc_consistency_seconds=timing.arc_consistency_seconds, feasibility_dp_seconds=timing.feasibility_dp_seconds, cost_setup_seconds=cost_setup_seconds,
            cost_dp_seconds=cost_dp_seconds, materialization_seconds=materialization_seconds, uncached_local_cost_seconds=edge_cost.timing.uncached_seconds - cost_seconds_before,
            uncached_local_cost_calls=edge_cost.timing.uncached_calls - cost_calls_before, local_cost_cache_hits=edge_cost.timing.cache_hits - cost_hits_before,
            witness_adjacency_seconds=witness_after.adjacency_seconds - witness_before.adjacency_seconds,
            witness_adjacency_builds=witness_after.adjacency_builds - witness_before.adjacency_builds,
            witness_dijkstra_seconds=witness_after.dijkstra_seconds - witness_before.dijkstra_seconds,
            witness_dijkstra_runs=witness_after.dijkstra_runs - witness_before.dijkstra_runs, feasibility_reused=timing.feasibility_reused, )
        return BothMatchResult(cost_name=edge_cost.name, additive=additive, bottleneck=bottleneck, dp_statistics=result.statistics, timing=timing, **metadata, )

    def match_all(self, cost_name: CostName | str, *, cancellation_token: CancellationToken | None = None, progress: ProgressCallback | None = None,
            **cost_options: object, ) -> AggregationMatchResult:
        resolved_name = CostName(cost_name)
        feasibility_reused = self._feasible is not None
        arc_before = self._arc_consistency_seconds
        feasibility_before = self._feasibility_dp_seconds
        feasible = self.prepare_feasibility(cancellation_token=cancellation_token, progress=progress)
        timing = self._feasibility_timing(arc_before, feasibility_before, feasibility_reused)
        metadata = self._common_metadata()

        if not feasible:
            return AggregationMatchResult(cost_name=resolved_name, additive=None, bottleneck=None, length_weighted_additive=None,
                dp_statistics=self._feasibility_statistics or _empty_statistics(), timing=timing, **metadata, )

        if progress is not None:
            progress(f"Evaluating {resolved_name.value.replace('_', ' ')}")

        started = time.perf_counter()
        edge_cost = self._cost(resolved_name, cost_options)
        cost_setup_seconds = time.perf_counter() - started
        cost_before = edge_cost.timing
        cost_seconds_before = cost_before.uncached_seconds
        cost_calls_before = cost_before.uncached_calls
        cost_hits_before = cost_before.cache_hits
        witness_before = self.cost_factory.resources.guided_timing()
        started = time.perf_counter()
        result = solve_tree_dp_all(self.decomposition, self.effective_candidate_sets, edge_cost, edge_weights=_edge_weights(self.source), compatibility=self._compatibility,
            cancellation_token=cancellation_token, )
        cost_dp_seconds = time.perf_counter() - started
        started = time.perf_counter()
        additive = _materialize_solution(self.source, edge_cost, result.additive, cancellation_token=cancellation_token)
        bottleneck = _materialize_solution(self.source, edge_cost, result.bottleneck, cancellation_token=cancellation_token)
        length_weighted_additive = _materialize_solution(self.source, edge_cost, result.length_weighted_additive, cancellation_token=cancellation_token, )
        materialization_seconds = time.perf_counter() - started
        witness_after = self.cost_factory.resources.guided_timing()
        timing = MatchTiming(arc_consistency_seconds=timing.arc_consistency_seconds, feasibility_dp_seconds=timing.feasibility_dp_seconds, cost_setup_seconds=cost_setup_seconds,
            cost_dp_seconds=cost_dp_seconds, materialization_seconds=materialization_seconds, uncached_local_cost_seconds=edge_cost.timing.uncached_seconds - cost_seconds_before,
            uncached_local_cost_calls=edge_cost.timing.uncached_calls - cost_calls_before, local_cost_cache_hits=edge_cost.timing.cache_hits - cost_hits_before,
            witness_adjacency_seconds=witness_after.adjacency_seconds - witness_before.adjacency_seconds,
            witness_adjacency_builds=witness_after.adjacency_builds - witness_before.adjacency_builds,
            witness_dijkstra_seconds=witness_after.dijkstra_seconds - witness_before.dijkstra_seconds,
            witness_dijkstra_runs=witness_after.dijkstra_runs - witness_before.dijkstra_runs, feasibility_reused=timing.feasibility_reused, )
        return AggregationMatchResult(cost_name=edge_cost.name, additive=additive, bottleneck=bottleneck, length_weighted_additive=length_weighted_additive,
            dp_statistics=result.statistics, timing=timing, **metadata, )


def match_graphs(source: JunctionGraph, target: JunctionGraph, cost_name: CostName | str, objective: Objective | str, *, candidate_rho: float = 10.0, top_k: int = 25,
        cost_options: Mapping[str, object] | None = None, cancellation_token: CancellationToken | None = None, ) -> MatchResult:
    matcher = RiverGraphMatcher(source, target, candidate_rho=candidate_rho, top_k=top_k)
    return matcher.match(cost_name, objective, cancellation_token=cancellation_token, **dict(cost_options or {}), )


def match_graphs_both(source: JunctionGraph, target: JunctionGraph, cost_name: CostName | str, *, candidate_rho: float = 10.0, top_k: int = 25,
        cost_options: Mapping[str, object] | None = None, cancellation_token: CancellationToken | None = None, ) -> BothMatchResult:
    matcher = RiverGraphMatcher(source, target, candidate_rho=candidate_rho, top_k=top_k)
    return matcher.match_both(cost_name, cancellation_token=cancellation_token, **dict(cost_options or {}), )
