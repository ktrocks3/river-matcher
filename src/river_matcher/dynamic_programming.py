from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from operator import index
from typing import Protocol

from river_matcher.decomposition import Bag, SourceDecomposition

type State = tuple[int, ...]
type SeparatorKey = tuple[int, ...]
type CostRequest = tuple[int, int, int, int, int]
type CandidateSets = Mapping[int, Iterable[int]]


class EdgeCost(Protocol):
    """Uniform local edge cost interface consumed by the DP"""

    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float: ...


class Objective(StrEnum):
    ADDITIVE = "additive"
    BOTTLENECK = "bottleneck"


@dataclass(frozen=True, slots=True)
class BagDPStatistics:
    bag: Bag
    enumerated_states: int
    feasible_states: int
    message_entries: int


@dataclass(frozen=True, slots=True)
class DPStatistics:
    bags: tuple[BagDPStatistics, ...]
    unique_cost_requests: int

    @property
    def enumerated_states(self) -> int:
        return sum(item.enumerated_states for item in self.bags)

    @property
    def feasible_states(self) -> int:
        return sum(item.feasible_states for item in self.bags)

    @property
    def message_entries(self) -> int:
        return sum(item.message_entries for item in self.bags)


@dataclass(frozen=True, slots=True)
class DPSolution:
    objective: Objective
    value: float
    mapping: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class DPSolveResult:
    objective: Objective
    solution: DPSolution | None
    statistics: DPStatistics

    @property
    def feasible(self) -> bool:
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class BothObjectiveResult:
    additive: DPSolution | None
    bottleneck: DPSolution | None
    statistics: DPStatistics


@dataclass(frozen=True, slots=True)
class _MessageEntry:
    value: float
    state: State
    child_keys: tuple[SeparatorKey, ...]


type MessageTable = dict[SeparatorKey, _MessageEntry]
type ObjectiveMessages = dict[Bag, MessageTable]
type AllMessages = dict[Objective, ObjectiveMessages]


class _CostEvaluator:
    """Memoize every edge-ID-aware local cost request once per DP run"""

    __slots__ = ("_cache", "_cost")

    def __init__(self, cost: EdgeCost) -> None:
        self._cache: dict[CostRequest, float] = {}
        self._cost = cost

    @property
    def unique_requests(self) -> int:
        return len(self._cache)

    def __call__(self, request: CostRequest) -> float:
        cached = self._cache.get(request)
        if cached is not None:
            return cached
        value = float(self._cost(*request))

        if value < 0.0:
            raise ValueError(f"Local edge costs must be nonnegative, but request {request} returned {value}.")

        if not math.isfinite(value):
            value = math.inf

        self._cache[request] = value
        return value


def _source_vertices(decomposition: SourceDecomposition) -> tuple[int, ...]:
    return tuple(sorted({vertex for bag in decomposition.bags for vertex in bag}))


def _normalize_candidate_sets(decomposition: SourceDecomposition, candidate_sets: CandidateSets) -> dict[int, tuple[int, ...]]:
    normalized: dict[int, tuple[int, ...]] = {}

    for vertex in _source_vertices(decomposition):
        normalized[vertex] = tuple(sorted({index(candidate) for candidate in candidate_sets.get(vertex, ())}))

    return normalized


def _is_better(value: float, state: State, previous: _MessageEntry | None) -> bool:
    return previous is None or value < previous.value or (value == previous.value and state < previous.state)


def _recover_solution(decomposition: SourceDecomposition, messages: AllMessages, objective: Objective) -> DPSolution | None:
    root_plan = decomposition.bag_plans[decomposition.root]
    if root_plan.parent_positions:
        raise RuntimeError("The root bag unexpectedly has a nonempty parent separator.")
    root_entry = messages[objective][decomposition.root].get(())
    if root_entry is None:
        return None
    mapping: dict[int, int] = {}
    stack: list[tuple[Bag, _MessageEntry]] = [(decomposition.root, root_entry)]
    while stack:
        bag, entry = stack.pop()
        plan = decomposition.bag_plans[bag]

        if len(entry.state) != len(plan.variables):
            raise RuntimeError(f"Stored state length is inconsistent for bag {tuple(sorted(bag))}.")
        for position, vertex in enumerate(plan.variables):
            target = entry.state[position]
            if vertex in mapping and mapping[vertex] != target:
                raise RuntimeError(f"Recovered assignments disagree for source vertex {vertex}: {mapping[vertex]} and {target}.")
            mapping[vertex] = target
        if len(entry.child_keys) != len(plan.child_positions):
            raise RuntimeError(f"Stored child-key count is inconsistent for bag {tuple(sorted(bag))}.")

        for child_index in range(len(plan.child_positions) - 1, -1, -1):
            child, _ = plan.child_positions[child_index]
            child_key = entry.child_keys[child_index]
            child_entry = messages[objective][child].get(child_key)

            if child_entry is None:
                raise RuntimeError(f"Missing child message for separator key {child_key} at bag {tuple(sorted(child))}.")
            stack.append((child, child_entry))
    ordered_mapping = {vertex: mapping[vertex] for vertex in sorted(mapping)}
    return DPSolution(objective=objective, value=root_entry.value, mapping=ordered_mapping)


def _solve(
    decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost, objectives: tuple[Objective, ...],
) -> tuple[dict[Objective, DPSolution | None], DPStatistics]:
    if not objectives:
        raise ValueError("At least one objective must be requested.")

    requested = tuple(dict.fromkeys(objectives))
    need_additive = Objective.ADDITIVE in requested
    need_bottleneck = Objective.BOTTLENECK in requested
    candidates = _normalize_candidate_sets(decomposition, candidate_sets)
    evaluator = _CostEvaluator(edge_cost)
    messages: AllMessages = {objective: {} for objective in objectives}
    bag_statistics: list[BagDPStatistics] = []
    for bag in decomposition.postorder:
        plan = decomposition.bag_plans[bag]
        tables: dict[Objective, MessageTable] = {objective: {} for objective in requested}
        candidate_lists = tuple(candidates[vertex] for vertex in plan.variables)
        child_infeasible = any(not messages[objective][child] for objective in requested for child, _ in plan.child_positions)

        if any(not domain for domain in candidate_lists) or child_infeasible:
            for objective in requested:
                messages[objective][bag] = tables[objective]
            bag_statistics.append(BagDPStatistics(bag=bag, enumerated_states=0, feasible_states=0, message_entries=0))
            continue
        enumerated_states = 0
        feasible_states = 0

        for state in product(*candidate_lists):
            enumerated_states += 1
            local_sum = 0.0
            local_max = 0.0
            valid = True

            for edge_id, u_position, v_position in plan.owned_edge_positions:
                source_u, source_v = plan.variables[u_position], plan.variables[v_position]
                target_u, target_v = state[u_position], state[v_position]
                value = evaluator((edge_id, source_u, source_v, target_u, target_v))

                if not math.isfinite(value):
                    valid = False
                    break

                local_sum += value
                local_max = max(local_max, value)

            if not valid:
                continue

            total_sum, total_max = local_sum, local_max
            child_keys: list[SeparatorKey] = []

            for child, positions in plan.child_positions:
                child_key = tuple(state[position] for position in positions)
                child_keys.append(child_key)

                if need_additive:
                    additive_entry = messages[Objective.ADDITIVE][child].get(child_key)
                    if additive_entry is None:
                        valid = False
                        break
                    total_sum += additive_entry.value

                if need_bottleneck:
                    bottleneck_entry = messages[Objective.BOTTLENECK][child].get(child_key)
                    if bottleneck_entry is None:
                        valid = False
                        break
                    total_max = max(total_max, bottleneck_entry.value)

            if not valid:
                continue

            feasible_states += 1
            parent_key = tuple(state[position] for position in plan.parent_positions)
            stored_child_keys = tuple(child_keys)

            if need_additive:
                previous = tables[Objective.ADDITIVE].get(parent_key)
                if _is_better(total_sum, state, previous):
                    tables[Objective.ADDITIVE][parent_key] = _MessageEntry(value=total_sum, state=state, child_keys=stored_child_keys)

            if need_bottleneck:
                previous = tables[Objective.BOTTLENECK].get(parent_key)
                if _is_better(total_max, state, previous):
                    tables[Objective.BOTTLENECK][parent_key] = _MessageEntry(value=total_max, state=state, child_keys=stored_child_keys)

        for objective in requested:
            messages[objective][bag] = tables[objective]
        entry_counts = {len(tables[objective]) for objective in requested}
        if len(entry_counts) != 1:
            raise RuntimeError(f"Objectives produced different feasible separator keys for bag {tuple(sorted(bag))}.")
        bag_statistics.append(BagDPStatistics(bag=bag, enumerated_states=enumerated_states, feasible_states=feasible_states, message_entries=entry_counts.pop()))

    statistics = DPStatistics(bags=tuple(bag_statistics), unique_cost_requests=evaluator.unique_requests)
    solutions = {objective: _recover_solution(decomposition, messages, objective) for objective in requested}

    return solutions, statistics


def solve_tree_dp(decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost, objective: Objective | str) -> DPSolveResult:
    resolved_objective = Objective(objective)
    solutions, statistics = _solve(decomposition, candidate_sets, edge_cost, (resolved_objective,))
    return DPSolveResult(objective=resolved_objective, solution=solutions[resolved_objective], statistics=statistics)


def solve_tree_dp_both(decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost) -> BothObjectiveResult:
    solutions, statistics = _solve(decomposition, candidate_sets, edge_cost, (Objective.ADDITIVE, Objective.BOTTLENECK))
    return BothObjectiveResult(additive=solutions[Objective.ADDITIVE], bottleneck=solutions[Objective.BOTTLENECK], statistics=statistics)
