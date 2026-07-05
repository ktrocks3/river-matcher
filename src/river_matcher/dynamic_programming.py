from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from operator import index
from typing import Protocol, cast

from river_matcher.cancellation import CancellationToken
from river_matcher.compatibility import TargetConnectivityCompatibility
from river_matcher.decomposition import Bag, BagPlan, SourceDecomposition

type State = tuple[int, ...]
type SeparatorKey = tuple[int, ...]
type CostRequest = tuple[int, int, int, int, int]
type CandidateSets = Mapping[int, Iterable[int]]


class EdgeCost(Protocol):
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
    partial_assignments: int = 0
    compatibility_prunes: int = 0
    cost_prunes: int = 0
    child_prunes: int = 0


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

    @property
    def partial_assignments(self) -> int:
        return sum(item.partial_assignments for item in self.bags)


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

_MISSING = object()


class _CostEvaluator:
    __slots__ = ("_cache", "_cost", "_token")

    def __init__(self, cost: EdgeCost, cancellation_token: CancellationToken | None) -> None:
        self._cache: dict[CostRequest, float] = {}
        self._cost = cost
        self._token = cancellation_token

    @property
    def unique_requests(self) -> int:
        return len(self._cache)

    def __call__(self, request: CostRequest) -> float:
        cached = self._cache.get(request, _MISSING)

        if cached is not _MISSING:
            return cast(float, cached)

        if self._token is not None:
            self._token.check()

        value = float(self._cost(*request))

        if value < 0.0:
            raise ValueError(f"Local edge costs must be nonnegative, but request {request} returned {value}.")

        if not math.isfinite(value):
            value = math.inf

        self._cache[request] = value
        return value


class _ZeroCost:
    def __call__(self, edge_id: int, source_u: int, source_v: int, target_u: int, target_v: int) -> float:
        return 0.0


def _source_vertices(decomposition: SourceDecomposition) -> tuple[int, ...]:
    return tuple(sorted({vertex for bag in decomposition.bags for vertex in bag}))


def _normalize_candidate_sets(decomposition: SourceDecomposition, candidate_sets: CandidateSets) -> dict[int, tuple[int, ...]]:
    return {vertex: tuple(sorted({index(candidate) for candidate in candidate_sets.get(vertex, ())})) for vertex in _source_vertices(decomposition)}


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


def _assignment_order(plan: BagPlan, candidates: Mapping[int, tuple[int, ...]], compatibility: TargetConnectivityCompatibility | None) -> tuple[int, ...]:
    def key(position: int) -> tuple[int, int, int]:
        vertex = plan.variables[position]
        degree = 0

        if compatibility is not None:
            degree = sum(neighbor in plan.bag for neighbor in compatibility.source_neighbors.get(vertex, ()))

        return len(candidates[vertex]), -degree, vertex

    return tuple(sorted(range(len(plan.variables)), key=key))


def _solve(decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost, objectives: tuple[Objective, ...], *,
           compatibility: TargetConnectivityCompatibility | None = None, cancellation_token: CancellationToken | None = None, evaluate_costs: bool = True) -> tuple[
    dict[Objective, DPSolution | None], DPStatistics]:
    if not objectives:
        raise ValueError("At least one objective must be requested.")

    requested = tuple(dict.fromkeys(objectives))
    need_additive = Objective.ADDITIVE in requested
    need_bottleneck = Objective.BOTTLENECK in requested
    candidates = _normalize_candidate_sets(decomposition, candidate_sets)
    evaluator = _CostEvaluator(edge_cost, cancellation_token)
    messages: AllMessages = {objective: {} for objective in requested}
    additive_messages = messages.get(Objective.ADDITIVE)
    bottleneck_messages = messages.get(Objective.BOTTLENECK)
    bag_statistics: list[BagDPStatistics] = []

    for bag in decomposition.postorder:
        if cancellation_token is not None:
            cancellation_token.check()

        plan = decomposition.bag_plans[bag]
        tables: dict[Objective, MessageTable] = {objective: {} for objective in requested}
        additive_table = tables.get(Objective.ADDITIVE)
        bottleneck_table = tables.get(Objective.BOTTLENECK)
        candidate_lists = tuple(candidates[vertex] for vertex in plan.variables)
        child_specs: list[tuple[tuple[int, ...], MessageTable | None, MessageTable | None]] = []
        child_infeasible = False

        for child, positions in plan.child_positions:
            additive_child = None if additive_messages is None else additive_messages[child]
            bottleneck_child = None if bottleneck_messages is None else bottleneck_messages[child]

            if (need_additive and not additive_child) or (need_bottleneck and not bottleneck_child):
                child_infeasible = True

            child_specs.append((positions, additive_child, bottleneck_child))

        if any(not domain for domain in candidate_lists) or child_infeasible:
            for objective in requested:
                messages[objective][bag] = tables[objective]

            bag_statistics.append(BagDPStatistics(bag=bag, enumerated_states=0, feasible_states=0, message_entries=0))
            continue

        order = _assignment_order(plan, candidates, compatibility)
        order_step = {position: step for step, position in enumerate(order)}
        assignments = [0] * len(plan.variables)
        edges_at_step: list[list[tuple[int, int, int, int, int]]] = [[] for _ in order]

        if evaluate_costs:
            for edge_id, u_position, v_position in plan.owned_edge_positions:
                step = max(order_step[u_position], order_step[v_position])
                edges_at_step[step].append((edge_id, plan.variables[u_position], plan.variables[v_position], u_position, v_position))

        constrained_positions_at_step: list[tuple[int, ...]] = [()] * len(order)
        component_by_target: Mapping[int, int] | None = None

        if compatibility is not None:
            component_by_target = compatibility.component_by_target

            for step, position in enumerate(order):
                neighbors = compatibility.source_neighbors.get(plan.variables[position], ())
                constrained_positions_at_step[step] = tuple(earlier_position for earlier_position in order[:step] if plan.variables[earlier_position] in neighbors)

        child_checks_at_step: list[list[int]] = [[] for _ in order]

        for child_index, (positions, _, _) in enumerate(child_specs):
            step = max((order_step[position] for position in positions), default=0)
            child_checks_at_step[step].append(child_index)

        resolved_child_keys: list[SeparatorKey | None] = [None] * len(child_specs)
        resolved_additive_entries: list[_MessageEntry | None] = [None] * len(child_specs)
        resolved_bottleneck_entries: list[_MessageEntry | None] = [None] * len(child_specs)
        enumerated_states = 0
        feasible_states = 0
        partial_assignments = 0
        compatibility_prunes = 0
        cost_prunes = 0
        child_prunes = 0

        def visit(step: int, local_sum: float, local_max: float) -> None:
            nonlocal enumerated_states, feasible_states, partial_assignments
            nonlocal compatibility_prunes, cost_prunes, child_prunes

            if cancellation_token is not None and partial_assignments % 1024 == 0:
                cancellation_token.check()

            if step == len(order):
                state = tuple(assignments)
                enumerated_states += 1
                total_sum = local_sum
                total_max = local_max

                for child_index in range(len(child_specs)):
                    if need_additive:
                        additive_entry = resolved_additive_entries[child_index]

                        if additive_entry is None:
                            raise RuntimeError("A feasible child entry was not retained until state completion.")

                        total_sum += additive_entry.value

                    if need_bottleneck:
                        bottleneck_entry = resolved_bottleneck_entries[child_index]

                        if bottleneck_entry is None:
                            raise RuntimeError("A feasible child entry was not retained until state completion.")

                        total_max = max(total_max, bottleneck_entry.value)

                feasible_states += 1
                parent_key = tuple(state[position] for position in plan.parent_positions)
                stored_child_keys = cast(tuple[SeparatorKey, ...], tuple(resolved_child_keys))

                if need_additive:
                    if additive_table is None:
                        raise RuntimeError("The additive message table is missing.")

                    previous = additive_table.get(parent_key)

                    if _is_better(total_sum, state, previous):
                        additive_table[parent_key] = _MessageEntry(total_sum, state, stored_child_keys)

                if need_bottleneck:
                    if bottleneck_table is None:
                        raise RuntimeError("The bottleneck message table is missing.")

                    previous = bottleneck_table.get(parent_key)

                    if _is_better(total_max, state, previous):
                        bottleneck_table[parent_key] = _MessageEntry(total_max, state, stored_child_keys)

                return

            position = order[step]

            for target_vertex in candidate_lists[position]:
                partial_assignments += 1

                if component_by_target is not None:
                    target_component = component_by_target.get(target_vertex)
                    compatible = True

                    for other_position in constrained_positions_at_step[step]:
                        other_target = assignments[other_position]

                        if target_vertex == other_target or target_component is None or target_component != component_by_target.get(other_target):
                            compatible = False
                            break

                    if not compatible:
                        compatibility_prunes += 1
                        continue

                assignments[position] = target_vertex
                next_sum = local_sum
                next_max = local_max
                valid = True

                for edge_id, source_u, source_v, u_position, v_position in edges_at_step[step]:
                    value = evaluator((edge_id, source_u, source_v, assignments[u_position], assignments[v_position]))

                    if not math.isfinite(value):
                        valid = False
                        cost_prunes += 1
                        break

                    next_sum += value
                    next_max = max(next_max, value)

                child_checks = child_checks_at_step[step]

                if valid:
                    for child_index in child_checks:
                        positions, additive_child, bottleneck_child = child_specs[child_index]
                        key = tuple(assignments[item] for item in positions)
                        resolved_child_keys[child_index] = key

                        if need_additive:
                            if additive_child is None:
                                raise RuntimeError("The additive child message table is missing.")

                            additive_entry = additive_child.get(key)

                            if additive_entry is None:
                                valid = False
                            else:
                                resolved_additive_entries[child_index] = additive_entry

                        if valid and need_bottleneck:
                            if bottleneck_child is None:
                                raise RuntimeError("The bottleneck child message table is missing.")

                            bottleneck_entry = bottleneck_child.get(key)

                            if bottleneck_entry is None:
                                valid = False
                            else:
                                resolved_bottleneck_entries[child_index] = bottleneck_entry

                        if not valid:
                            child_prunes += 1
                            break

                if valid:
                    visit(step + 1, next_sum, next_max)

                for child_index in child_checks:
                    resolved_child_keys[child_index] = None
                    resolved_additive_entries[child_index] = None
                    resolved_bottleneck_entries[child_index] = None

        visit(0, 0.0, 0.0)

        for objective in requested:
            messages[objective][bag] = tables[objective]

        entry_counts = {len(tables[objective]) for objective in requested}

        if len(entry_counts) != 1:
            raise RuntimeError(f"Objectives produced different feasible separator keys for bag {tuple(sorted(bag))}.")

        bag_statistics.append(BagDPStatistics(bag=bag, enumerated_states=enumerated_states, feasible_states=feasible_states, message_entries=entry_counts.pop(),
                                              partial_assignments=partial_assignments, compatibility_prunes=compatibility_prunes, cost_prunes=cost_prunes,
                                              child_prunes=child_prunes))

    statistics = DPStatistics(bags=tuple(bag_statistics), unique_cost_requests=evaluator.unique_requests if evaluate_costs else 0)
    solutions = {objective: _recover_solution(decomposition, messages, objective) for objective in requested}
    return solutions, statistics


def solve_tree_feasibility(decomposition: SourceDecomposition, candidate_sets: CandidateSets, compatibility: TargetConnectivityCompatibility, *,
                           cancellation_token: CancellationToken | None = None) -> DPSolveResult:
    solutions, statistics = _solve(decomposition, candidate_sets, _ZeroCost(), (Objective.ADDITIVE,), compatibility=compatibility, cancellation_token=cancellation_token,
                                   evaluate_costs=False)
    solution = solutions[Objective.ADDITIVE]
    return DPSolveResult(objective=Objective.ADDITIVE, solution=solution, statistics=statistics)


def solve_tree_dp(decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost, objective: Objective | str, *,
                  compatibility: TargetConnectivityCompatibility | None = None, cancellation_token: CancellationToken | None = None) -> DPSolveResult:
    resolved_objective = Objective(objective)
    solutions, statistics = _solve(decomposition, candidate_sets, edge_cost, (resolved_objective,), compatibility=compatibility, cancellation_token=cancellation_token)
    return DPSolveResult(objective=resolved_objective, solution=solutions[resolved_objective], statistics=statistics)


def solve_tree_dp_both(decomposition: SourceDecomposition, candidate_sets: CandidateSets, edge_cost: EdgeCost, *, compatibility: TargetConnectivityCompatibility | None = None,
                       cancellation_token: CancellationToken | None = None) -> BothObjectiveResult:
    solutions, statistics = _solve(decomposition, candidate_sets, edge_cost, (Objective.ADDITIVE, Objective.BOTTLENECK), compatibility=compatibility,
                                   cancellation_token=cancellation_token)
    return BothObjectiveResult(additive=solutions[Objective.ADDITIVE], bottleneck=solutions[Objective.BOTTLENECK], statistics=statistics)
