from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index

from river_matcher.cancellation import CancellationToken
from river_matcher.compatibility import TargetConnectivityCompatibility
from river_matcher.costs.base import BaseEdgeCost
from river_matcher.models import JunctionEdge, JunctionGraph

type CandidateDomains = Mapping[int, Sequence[int]]


@dataclass(frozen=True, slots=True)
class DominancePruningResult:
    candidate_sets: dict[int, list[int]]
    candidates_before: int
    candidates_after: int
    candidates_removed: int
    iterations: int
    dominance_comparisons: int
    local_cost_requests: int
    elapsed_seconds: float
    comparison_limit_reached: bool


def _oriented_targets(edge: JunctionEdge, source_vertex: int, candidate: int, opposite_candidate: int) -> tuple[int, int]:
    if source_vertex == edge.u:
        return candidate, opposite_candidate
    if source_vertex == edge.v:
        return opposite_candidate, candidate
    raise ValueError(f"Source vertex {source_vertex} is not incident to edge e{edge.id}.")


def _safe_deletions(candidates: tuple[int, ...], dominators: Mapping[int, set[int]]) -> set[int]:
    if len(candidates) <= 1:
        return set()

    retained_chain = {candidate for candidate in candidates if not dominators[candidate]}
    if not retained_chain:
        retained_chain.add(candidates[0])

    deletions: set[int] = set()
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            if candidate in retained_chain:
                continue
            if any(dominator in retained_chain for dominator in dominators[candidate]):
                retained_chain.add(candidate)
                deletions.add(candidate)
                changed = True

    return deletions


def prune_dominated_candidates(source: JunctionGraph, candidate_sets: CandidateDomains, edge_cost: BaseEdgeCost, compatibility: TargetConnectivityCompatibility, *,
                               comparison_limit: int | None = None, cancellation_token: CancellationToken | None = None) -> DominancePruningResult:
    """Remove candidates whose incident-edge costs are pointwise dominated.

    Every fixed-point pass reads one immutable snapshot. If the comparison limit
    interrupts a pass, that pass is discarded and domains from earlier complete
    passes are retained.
    """
    started = time.perf_counter()
    limit = None if comparison_limit is None else index(comparison_limit)
    if limit is not None and limit < 0:
        raise ValueError(f"Dominance comparison limit must be nonnegative, got {comparison_limit!r}.")

    source_vertices = set(source.vertices)
    unknown_sources = sorted(set(candidate_sets) - source_vertices)
    if unknown_sources:
        raise ValueError(f"Dominance candidate sets contain unknown source vertices {unknown_sources}.")

    domains = {source_vertex: sorted({index(candidate) for candidate in candidate_sets.get(source_vertex, ())}) for source_vertex in sorted(source.vertices)}
    candidates_before = sum(len(domain) for domain in domains.values())
    incident_edges: defaultdict[int, list[JunctionEdge]] = defaultdict(list)
    for edge in sorted(source.edges, key=lambda item: item.id):
        incident_edges[edge.u].append(edge)
        incident_edges[edge.v].append(edge)

    iterations = 0
    dominance_comparisons = 0
    local_cost_requests = 0
    work_items = 0
    comparison_limit_reached = False

    def checkpoint() -> None:
        nonlocal work_items
        work_items += 1
        if cancellation_token is not None and work_items % 256 == 0:
            cancellation_token.check()

    def dominates(snapshot: Mapping[int, tuple[int, ...]], source_vertex: int, candidate_a: int, candidate_b: int) -> bool:
        nonlocal local_cost_requests

        for edge in incident_edges.get(source_vertex, ()):
            opposite_source = edge.v if source_vertex == edge.u else edge.u
            for opposite_candidate in snapshot[opposite_source]:
                checkpoint()
                target_b_u, target_b_v = _oriented_targets(edge, source_vertex, candidate_b, opposite_candidate)
                if not compatibility.supports(edge.u, target_b_u, edge.v, target_b_v):
                    continue

                target_a_u, target_a_v = _oriented_targets(edge, source_vertex, candidate_a, opposite_candidate)
                if not compatibility.supports(edge.u, target_a_u, edge.v, target_a_v):
                    return False

                cost_b = float(edge_cost(edge.id, edge.u, edge.v, target_b_u, target_b_v))
                cost_a = float(edge_cost(edge.id, edge.u, edge.v, target_a_u, target_a_v))
                local_cost_requests += 2
                if cost_a > cost_b:
                    return False

        return True

    def compare(snapshot: Mapping[int, tuple[int, ...]], source_vertex: int, candidate_a: int, candidate_b: int) -> bool | None:
        nonlocal comparison_limit_reached, dominance_comparisons
        if limit is not None and dominance_comparisons >= limit:
            comparison_limit_reached = True
            return None
        dominance_comparisons += 1
        checkpoint()
        return dominates(snapshot, source_vertex, candidate_a, candidate_b)

    while True:
        iterations += 1
        snapshot = {source_vertex: tuple(domain) for source_vertex, domain in domains.items()}
        planned_deletions: dict[int, set[int]] = {}
        pass_complete = True

        for source_vertex in sorted(snapshot):
            if cancellation_token is not None:
                cancellation_token.check()

            candidates = snapshot[source_vertex]
            dominators = {candidate: set() for candidate in candidates}

            for first_index, smaller_candidate in enumerate(candidates):
                for larger_candidate in candidates[first_index + 1:]:
                    smaller_dominates = compare(snapshot, source_vertex, smaller_candidate, larger_candidate)
                    if smaller_dominates is None:
                        pass_complete = False
                        break
                    if smaller_dominates:
                        dominators[larger_candidate].add(smaller_candidate)
                        continue

                    larger_dominates = compare(snapshot, source_vertex, larger_candidate, smaller_candidate)
                    if larger_dominates is None:
                        pass_complete = False
                        break
                    if larger_dominates:
                        dominators[smaller_candidate].add(larger_candidate)

                if not pass_complete:
                    break

            if not pass_complete:
                break

            planned_deletions[source_vertex] = _safe_deletions(candidates, dominators)

        if not pass_complete:
            break

        removed_this_pass = sum(len(deletions) for deletions in planned_deletions.values())
        if removed_this_pass == 0:
            break

        domains = {source_vertex: [candidate for candidate in snapshot[source_vertex] if candidate not in planned_deletions[source_vertex]] for source_vertex in sorted(snapshot)}

    if cancellation_token is not None:
        cancellation_token.check()

    candidates_after = sum(len(domain) for domain in domains.values())
    return DominancePruningResult(candidate_sets={source_vertex: list(domains[source_vertex]) for source_vertex in sorted(domains)}, candidates_before=candidates_before,
                                  candidates_after=candidates_after, candidates_removed=candidates_before - candidates_after, iterations=iterations,
                                  dominance_comparisons=dominance_comparisons, local_cost_requests=local_cost_requests, elapsed_seconds=time.perf_counter() - started,
                                  comparison_limit_reached=comparison_limit_reached)
