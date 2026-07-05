from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from river_matcher.decomposition import Bag, SourceDecomposition


@dataclass(frozen=True, slots=True)
class BagStateEstimate:
    bag: Bag
    candidate_product: int
    owned_edges: int
    children: int


@dataclass(frozen=True, slots=True)
class MatchingPreflight:
    empty_domains: int
    total_candidates: int
    minimum_candidates: int
    maximum_candidates: int
    estimated_state_upper_bound: int
    largest_candidate_product: int
    largest_bag: Bag | None
    largest_bags: tuple[BagStateEstimate, ...]

    @property
    def possible(self) -> bool:
        return self.empty_domains == 0


def estimate_matching(decomposition: SourceDecomposition, candidate_sets: Mapping[int, Sequence[int]], *, largest_count: int = 10) -> MatchingPreflight:
    sizes = tuple(len(candidate_sets.get(vertex, ())) for vertex in sorted(candidate_sets))
    rows: list[BagStateEstimate] = []
    total = 0

    for bag in decomposition.bags:
        plan = decomposition.bag_plans[bag]
        product = math.prod(len(candidate_sets.get(vertex, ())) for vertex in plan.variables)
        total += product
        rows.append(BagStateEstimate(bag=bag, candidate_product=product, owned_edges=len(plan.owned_edge_positions), children=len(plan.child_positions)))

    ordered = tuple(sorted(rows, key=lambda item: (-item.candidate_product, tuple(sorted(item.bag)))))
    largest = ordered[0] if ordered else None

    return MatchingPreflight(empty_domains=sum(size == 0 for size in sizes), total_candidates=sum(sizes), minimum_candidates=min(sizes, default=0),
        maximum_candidates=max(sizes, default=0), estimated_state_upper_bound=total, largest_candidate_product=0 if largest is None else largest.candidate_product,
        largest_bag=None if largest is None else largest.bag, largest_bags=ordered[:largest_count])
