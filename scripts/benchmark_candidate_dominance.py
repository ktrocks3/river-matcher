#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SRC: Final[Path] = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from river_matcher.candidates import CandidateMode
from river_matcher.costs.base import CostName
from river_matcher.dynamic_programming import Objective
from river_matcher.ui.workers import GraphRepository, PairSession, load_matching_pair

BENCHMARK_PAIRS: Final[tuple[tuple[str, str, float], ...]] = (("1955e5", "1955e3", 10.0), ("1998e5", "1998e3", 10.0), ("2014e5", "2014e3", 10.0), ("2014e5", "1955e5", 12.5),
                                                              ("2014e5", "1955e3", 10.0), ("1998e3", "1955e3", 12.5), ("1998e5", "2014e3", 16.0))

BENCHMARK_COSTS: Final[tuple[CostName, ...]] = (CostName.DISCRETE_FRECHET_DISTANCE, CostName.RELATIVE_LENGTH_ERROR, CostName.LOG_LENGTH_DISTORTION)

COST_OPTIONS: Final[dict[CostName, dict[str, object]]] = {CostName.DISCRETE_FRECHET_DISTANCE: {"rho": 10.0, "edge_samples": 12, "curve_samples": 64},
                                                          CostName.RELATIVE_LENGTH_ERROR: {}, CostName.LOG_LENGTH_DISTORTION: {}}

TOP_K: Final[int] = 25
ADAPTIVE_MAX_POINTS_PER_SOURCE: Final[int] = 8
ADAPTIVE_MIN_SEPARATION: Final[float] = 1.0
OBJECTIVE_REL_TOL: Final[float] = 1e-10
OBJECTIVE_ABS_TOL: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    status: str
    objective_value: float | None
    total_seconds: float
    candidates_entering_dp: int | None
    estimated_states: int | None
    actual_states: int | None
    dominance_seconds: float | None = None
    candidates_before: int | None = None
    candidates_after: int | None = None
    candidates_removed: int | None = None
    dominance_iterations: int | None = None
    dominance_comparisons: int | None = None
    dominance_local_cost_requests: int | None = None
    comparison_limit_reached: bool = False
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare complete matches with exact candidate-dominance pruning disabled and enabled.")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "GraphExport")
    parser.add_argument("--cost", action="append", choices=[cost.value for cost in BENCHMARK_COSTS], help="Restrict to one registered benchmark cost.")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"), help="Restrict to one configured source/target pair.")
    parser.add_argument("--state-limit", type=int, default=10_000_000, help="Skip when the current raw preflight estimate exceeds N; use 0 to disable.")
    parser.add_argument("--dominance-comparison-limit", type=int, default=None, help="Stop dominance cleanly after N directed candidate comparisons.")
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.state_limit < 0:
        raise ValueError("--state-limit must be nonnegative")
    if args.dominance_comparison_limit is not None and args.dominance_comparison_limit < 0:
        raise ValueError("--dominance-comparison-limit must be nonnegative")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")


def discover_graphs(graph_dir: Path) -> dict[str, Path]:
    directory = graph_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Graph directory does not exist: {directory}")

    paths: dict[str, Path] = {}
    for path in sorted(directory.glob("*.txt")):
        if path.stem in paths:
            raise ValueError(f"Duplicate graph stem {path.stem!r} in {directory}")
        paths[path.stem] = path.resolve()
    return paths


def selected_pairs(args: argparse.Namespace) -> tuple[tuple[str, str, float], ...]:
    configured = {(source, target): (source, target, rho) for source, target, rho in BENCHMARK_PAIRS}
    if not args.pair:
        return BENCHMARK_PAIRS

    selected: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for raw_source, raw_target in args.pair:
        key = str(raw_source), str(raw_target)
        if key not in configured:
            available = ", ".join(f"{source}->{target}" for source, target, _ in BENCHMARK_PAIRS)
            raise ValueError(f"Unknown benchmark pair {key[0]}->{key[1]}. Available pairs: {available}")
        if key not in seen:
            selected.append(configured[key])
            seen.add(key)
    return tuple(selected)


def selected_costs(args: argparse.Namespace) -> tuple[CostName, ...]:
    if not args.cost:
        return BENCHMARK_COSTS
    return tuple(dict.fromkeys(CostName(cost) for cost in args.cost))


def run_complete_match(source_path: Path, target_path: Path, *, candidate_rho: float, cost_name: CostName, dominance_pruning: bool, state_limit: int,
                       dominance_comparison_limit: int | None) -> BenchmarkRun:
    started = time.perf_counter()

    try:
        repository = GraphRepository()
        source, target, swapped = load_matching_pair(repository, source_path, target_path, CandidateMode.ADAPTIVE_CLOSEST_POINTS)
        if swapped:
            raise ValueError(f"Configured direction {source_path.stem}->{target_path.stem} was reversed by sparse-to-dense normalization.")

        junction_target = repository.load(target.key.path)
        session = PairSession(source, target, candidate_rho=candidate_rho, top_k=TOP_K, candidate_mode=CandidateMode.ADAPTIVE_CLOSEST_POINTS, subdivision_points=0,
                              adaptive_max_points_per_source=ADAPTIVE_MAX_POINTS_PER_SOURCE, adaptive_min_separation=ADAPTIVE_MIN_SEPARATION, junction_target=junction_target)
        raw_preflight = session.preflight
        raw_candidates = session.matcher.candidate_statistics.total_candidates

        if raw_preflight.empty_domains:
            return BenchmarkRun("empty_domains", None, time.perf_counter() - started, raw_candidates, raw_preflight.estimated_state_upper_bound, None)
        if not dominance_pruning and 0 < state_limit < raw_preflight.estimated_state_upper_bound:
            return BenchmarkRun("state_limit", None, time.perf_counter() - started, raw_candidates, raw_preflight.estimated_state_upper_bound, None)

        result = session.matcher.match(cost_name, Objective.ADDITIVE, dominance_pruning=dominance_pruning, dominance_comparison_limit=dominance_comparison_limit,
                                       dominance_state_limit=(state_limit if dominance_pruning and state_limit > 0 else None), **COST_OPTIONS[cost_name])
        elapsed = time.perf_counter() - started
        if dominance_pruning and result.state_limit_reached:
            dominance = result.dominance_pruning

            if dominance is None or result.dominance_preflight is None:
                raise RuntimeError("Post-dominance state-limit result is missing statistics.")

            return BenchmarkRun(status="state_limit_after_dominance", objective_value=None, total_seconds=elapsed, candidates_entering_dp=dominance.candidates_after,
                                estimated_states=result.dominance_preflight.estimated_state_upper_bound, actual_states=None, dominance_seconds=dominance.elapsed_seconds,
                                candidates_before=dominance.candidates_before, candidates_after=dominance.candidates_after, candidates_removed=dominance.candidates_removed,
                                dominance_iterations=dominance.iterations, dominance_comparisons=dominance.dominance_comparisons,
                                dominance_local_cost_requests=dominance.local_cost_requests, comparison_limit_reached=dominance.comparison_limit_reached)
        solution = result.solution
        status = "ok" if solution is not None else "infeasible"
        value = None if solution is None else float(solution.value)

        if not dominance_pruning:
            statistics = result.effective_candidate_statistics or result.candidate_statistics
            preflight = result.effective_preflight or result.preflight
            return BenchmarkRun(status, value, elapsed, statistics.total_candidates, None if preflight is None else preflight.estimated_state_upper_bound,
                                result.dp_statistics.enumerated_states)

        dominance = result.dominance_pruning
        if dominance is None:
            raise RuntimeError("Dominance-enabled match returned no dominance statistics.")
        return BenchmarkRun(status, value, elapsed, dominance.candidates_after,
                            None if result.dominance_preflight is None else result.dominance_preflight.estimated_state_upper_bound, result.dp_statistics.enumerated_states,
                            dominance_seconds=dominance.elapsed_seconds, candidates_before=dominance.candidates_before, candidates_after=dominance.candidates_after,
                            candidates_removed=dominance.candidates_removed, dominance_iterations=dominance.iterations, dominance_comparisons=dominance.dominance_comparisons,
                            dominance_local_cost_requests=dominance.local_cost_requests, comparison_limit_reached=dominance.comparison_limit_reached)
    except Exception as exc:
        return BenchmarkRun("error", None, time.perf_counter() - started, None, None, None, error=f"{type(exc).__name__}: {exc}")


def _status_summary(runs: list[BenchmarkRun]) -> str:
    counts = Counter(run.status for run in runs)
    return ",".join(status if count == 1 else f"{status}x{count}" for status, count in sorted(counts.items()))


def _format_optional_ints(values: list[int | None]) -> str:
    available = sorted({value for value in values if value is not None})
    if not available:
        return "n/a"
    if len(available) == 1:
        return f"{available[0]:,}"
    return f"{available[0]:,}..{available[-1]:,}"


def _format_objectives(runs: list[BenchmarkRun]) -> str:
    values = [run.objective_value for run in runs if run.objective_value is not None]
    if not values:
        return "n/a"
    if all(math.isclose(values[0], value, rel_tol=OBJECTIVE_REL_TOL, abs_tol=OBJECTIVE_ABS_TOL) for value in values[1:]):
        return f"{values[0]:.12g}"
    return f"{min(values):.12g}..{max(values):.12g}"


def _mean_optional(values: list[float | None]) -> str:
    available = [value for value in values if value is not None]
    return "n/a" if not available else f"{fmean(available):.3f}s"


def print_comparison(source: str, target: str, rho: float, cost_name: CostName, unpruned: list[BenchmarkRun], pruned: list[BenchmarkRun]) -> None:
    completed_values = [(float(without.objective_value), float(with_pruning.objective_value)) for without, with_pruning in zip(unpruned, pruned, strict=True) if
                        without.objective_value is not None and with_pruning.objective_value is not None]
    equalities = [math.isclose(without_value, pruned_value, rel_tol=OBJECTIVE_REL_TOL, abs_tol=OBJECTIVE_ABS_TOL) for without_value, pruned_value in completed_values]
    differences = [abs(without_value - pruned_value) for without_value, pruned_value in completed_values]
    equal_text = "n/a" if not equalities else f"{str(all(equalities)).lower()} ({sum(equalities)}/{len(equalities)})"
    difference_text = "n/a" if not differences else f"{max(differences):.12g}"
    unpruned_total = sum(run.total_seconds for run in unpruned)
    pruned_total = sum(run.total_seconds for run in pruned)
    speedup = math.inf if pruned_total == 0.0 else unpruned_total / pruned_total

    print(f"\n{source}->{target} | rho={rho:g} | mode={CandidateMode.ADAPTIVE_CLOSEST_POINTS.value} | cost={cost_name.value} | aggregation={Objective.ADDITIVE.value}")
    print(f"  without: status={_status_summary(unpruned)}; value={_format_objectives(unpruned)}; mean_total={fmean(run.total_seconds for run in unpruned):.3f}s; "
          f"dp_candidates={_format_optional_ints([run.candidates_entering_dp for run in unpruned])}; "
          f"estimated_states={_format_optional_ints([run.estimated_states for run in unpruned])}; actual_states={_format_optional_ints([run.actual_states for run in unpruned])}")
    print(f"  with:    status={_status_summary(pruned)}; value={_format_objectives(pruned)}; mean_total={fmean(run.total_seconds for run in pruned):.3f}s; "
          f"dominance={_mean_optional([run.dominance_seconds for run in pruned])}; "
          f"candidates={_format_optional_ints([run.candidates_before for run in pruned])}->{_format_optional_ints([run.candidates_after for run in pruned])}; "
          f"dp_candidates={_format_optional_ints([run.candidates_entering_dp for run in pruned])}; "
          f"removed={_format_optional_ints([run.candidates_removed for run in pruned])}; iterations={_format_optional_ints([run.dominance_iterations for run in pruned])}; "
          f"comparisons={_format_optional_ints([run.dominance_comparisons for run in pruned])}; "
          f"local_cost_requests={_format_optional_ints([run.dominance_local_cost_requests for run in pruned])}; "
          f"limit_reached={any(run.comparison_limit_reached for run in pruned)}; "
          f"estimated_states={_format_optional_ints([run.estimated_states for run in pruned])}; actual_states={_format_optional_ints([run.actual_states for run in pruned])}")
    print(f"  compare: objectives_equal={equal_text}; max_abs_difference={difference_text}; total_speedup={speedup:.3f}x")

    errors = sorted({run.error for run in (*unpruned, *pruned) if run.error})
    for error in errors:
        print(f"  error: {error}")
    if equalities and not all(equalities):
        print("  *** WARNING: EXACT DOMINANCE CHANGED THE OPTIMAL OBJECTIVE VALUE ***")


def main() -> int:
    args = parse_args()
    validate_args(args)
    pairs = selected_pairs(args)
    costs = selected_costs(args)
    graph_paths = discover_graphs(args.graph_dir)

    required_stems = {stem for source, target, _ in pairs for stem in (source, target)}
    missing = sorted(required_stems - set(graph_paths))
    if missing:
        raise FileNotFoundError(f"Missing graph exports for: {', '.join(missing)}")

    total_unpruned_seconds = 0.0
    total_pruned_seconds = 0.0
    total_candidates_removed = 0
    completed_equal = 0
    skipped_or_failed = 0
    unequal = 0

    print(f"candidate mode: {CandidateMode.ADAPTIVE_CLOSEST_POINTS.value}; aggregation: {Objective.ADDITIVE.value}; repeats: {args.repeats}; "
          f"state limit: {args.state_limit}; dominance comparison limit: {args.dominance_comparison_limit}")
    for source, target, rho in pairs:
        for cost_name in costs:
            unpruned_runs: list[BenchmarkRun] = []
            pruned_runs: list[BenchmarkRun] = []

            for repeat_index in range(args.repeats):
                common_kwargs = {"source_path": graph_paths[source], "target_path": graph_paths[target], "candidate_rho": rho, "cost_name": cost_name,
                                 "state_limit": args.state_limit, "dominance_comparison_limit": args.dominance_comparison_limit, }

                # Alternate order to reduce systematic warm-cache bias.
                # Start with pruning so a potentially huge unpruned run does not block
                # the useful pruned result on the first repetition.
                if repeat_index % 2 == 0:
                    with_pruning = run_complete_match(dominance_pruning=True, **common_kwargs)
                    without = run_complete_match(dominance_pruning=False, **common_kwargs)
                else:
                    without = run_complete_match(dominance_pruning=False, **common_kwargs)
                    with_pruning = run_complete_match(dominance_pruning=True, **common_kwargs)
                unpruned_runs.append(without)
                pruned_runs.append(with_pruning)
                total_unpruned_seconds += without.total_seconds
                total_pruned_seconds += with_pruning.total_seconds
                total_candidates_removed += with_pruning.candidates_removed or 0

                if without.objective_value is None or with_pruning.objective_value is None:
                    skipped_or_failed += 1
                elif math.isclose(without.objective_value, with_pruning.objective_value, rel_tol=OBJECTIVE_REL_TOL, abs_tol=OBJECTIVE_ABS_TOL):
                    completed_equal += 1
                else:
                    unequal += 1

            print_comparison(source, target, rho, cost_name, unpruned_runs, pruned_runs)

    overall_speedup = math.inf if total_pruned_seconds == 0.0 else total_unpruned_seconds / total_pruned_seconds
    print("\nAggregate totals")
    print(f"  total unpruned wall time: {total_unpruned_seconds:.3f}s")
    print(f"  total pruned wall time, including dominance: {total_pruned_seconds:.3f}s")
    print(f"  overall speedup: {overall_speedup:.3f}x")
    print(f"  total candidates removed: {total_candidates_removed:,}")
    print(f"  completed equal-objective comparisons: {completed_equal}")
    print(f"  skipped or failed: {skipped_or_failed}")
    if unequal:
        print(f"  *** WARNING: {unequal} completed comparisons changed the objective ***")
    return 0 if unequal == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
