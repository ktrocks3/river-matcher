from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from river_matcher.cancellation import CancellationToken
from river_matcher.models import JunctionGraph


type CandidateDomains = Mapping[int, Sequence[int]]


@dataclass(frozen=True, slots=True)
class CompatibilityStatistics:
    initial_candidates: int
    remaining_candidates: int
    removed_candidates: int
    revised_arcs: int
    empty_domains: int


class TargetConnectivityCompatibility:
    """
    Necessary and sufficient witness-existence relation for the current path finders.

    Both ordinary and source-guided witnesses retain every target edge with a finite
    nonnegative weight. A witness therefore exists exactly when the mapped endpoints
    are distinct and lie in the same target connected component.
    """

    __slots__ = ("component_by_target", "source_neighbors")

    def __init__(self, source: JunctionGraph, target: JunctionGraph) -> None:
        target_adjacency: defaultdict[int, list[int]] = defaultdict(list)

        for edge in target.edges:
            target_adjacency[edge.u].append(edge.v)
            target_adjacency[edge.v].append(edge.u)

        component_by_target: dict[int, int] = {}
        component = 0

        for start in target.vertices:
            if start in component_by_target:
                continue

            component_by_target[start] = component
            queue = deque([start])

            while queue:
                current = queue.popleft()

                for neighbor in target_adjacency.get(current, ()):
                    if neighbor in component_by_target:
                        continue
                    component_by_target[neighbor] = component
                    queue.append(neighbor)

            component += 1

        source_neighbors: defaultdict[int, set[int]] = defaultdict(set)

        for edge in source.edges:
            source_neighbors[edge.u].add(edge.v)
            source_neighbors[edge.v].add(edge.u)

        self.component_by_target = component_by_target
        self.source_neighbors = {vertex: frozenset(neighbors) for vertex, neighbors in source_neighbors.items()}

    def supports(self, source_u: int, target_u: int, source_v: int, target_v: int) -> bool:
        if source_v not in self.source_neighbors.get(source_u, ()):
            return True

        return (
            target_u != target_v
            and self.component_by_target.get(target_u) is not None
            and self.component_by_target.get(target_u) == self.component_by_target.get(target_v)
        )


def enforce_arc_consistency(
    compatibility: TargetConnectivityCompatibility,
    candidate_sets: CandidateDomains,
    *,
    cancellation_token: CancellationToken | None = None,
) -> tuple[dict[int, tuple[int, ...]], CompatibilityStatistics]:
    domains = {vertex: set(int(candidate) for candidate in candidates) for vertex, candidates in candidate_sets.items()}
    initial = sum(len(domain) for domain in domains.values())
    queue = deque(
        (source_u, source_v)
        for source_u, neighbors in compatibility.source_neighbors.items()
        for source_v in neighbors
    )
    revised_arcs = 0
    checks = 0

    while queue:
        source_u, source_v = queue.popleft()
        revised_arcs += 1
        domain_u = domains.get(source_u, set())
        domain_v = domains.get(source_v, set())
        removed: list[int] = []

        for target_u in tuple(domain_u):
            checks += 1

            if cancellation_token is not None and checks % 1024 == 0:
                cancellation_token.check()

            if not any(compatibility.supports(source_u, target_u, source_v, target_v) for target_v in domain_v):
                removed.append(target_u)

        if not removed:
            continue

        domain_u.difference_update(removed)

        for neighbor in compatibility.source_neighbors.get(source_u, ()):
            if neighbor != source_v:
                queue.append((neighbor, source_u))

    if cancellation_token is not None:
        cancellation_token.check()

    normalized = {vertex: tuple(sorted(domain)) for vertex, domain in domains.items()}
    remaining = sum(len(domain) for domain in normalized.values())
    statistics = CompatibilityStatistics(
        initial_candidates=initial,
        remaining_candidates=remaining,
        removed_candidates=initial - remaining,
        revised_arcs=revised_arcs,
        empty_domains=sum(len(domain) == 0 for domain in normalized.values()),
    )
    return normalized, statistics
