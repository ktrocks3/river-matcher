from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from river_matcher.candidates import compute_candidate_sets
from river_matcher.costs.base import BaseEdgeCost, CostName
from river_matcher.costs.factory import CostFactory
from river_matcher.decomposition import SourceDecomposition, build_source_decomposition
from river_matcher.dynamic_programming import BothObjectiveResult, DPSolution, Objective, solve_tree_dp_both
from river_matcher.models import JunctionGraph
from river_matcher.preprocessing import load_junction_graph

T = TypeVar("T")


def _timed(label: str, function: Callable[[], T], timings: dict[str, float]) -> T:
    started = time.perf_counter()
    try:
        return function()
    finally:
        timings[label] = time.perf_counter() - started


def _parse_cost_options(values: Sequence[str]) -> dict[str, object]:
    options: dict[str, object] = {}

    for value in values:
        key, separator, raw = value.partition("=")

        if not separator or not key:
            raise ValueError(f"Cost option must have the form key=value, received {value!r}.")

        if key in options:
            raise ValueError(f"Duplicate cost option {key!r}.")

        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw

    return options


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_report(candidate_sets: Mapping[int, Sequence[int]]) -> dict[str, object]:
    ordered = [(int(vertex), [int(candidate) for candidate in candidate_sets[vertex]]) for vertex in sorted(candidate_sets)]
    sizes = np.asarray([len(candidates) for _, candidates in ordered], dtype=np.int64)

    return {
        "vertices": int(len(ordered)),
        "empty_domains": int(np.count_nonzero(sizes == 0)),
        "total_candidates": int(sizes.sum()),
        "minimum_candidates": int(sizes.min()) if len(sizes) else 0,
        "median_candidates": float(np.median(sizes)) if len(sizes) else 0.0,
        "maximum_candidates": int(sizes.max()) if len(sizes) else 0,
        "digest": _json_digest(ordered),
    }


def _decomposition_report(decomposition: SourceDecomposition, candidate_sets: Mapping[int, Sequence[int]]) -> dict[str, object]:
    bag_rows: list[dict[str, object]] = []
    estimated_states = 0

    for bag in decomposition.bags:
        plan = decomposition.bag_plans[bag]
        states = math.prod(len(candidate_sets[vertex]) for vertex in plan.variables)
        estimated_states += states
        bag_rows.append(
            {
                "bag": list(plan.variables),
                "candidate_product": states,
                "owned_edges": len(plan.owned_edge_positions),
                "children": len(plan.child_positions),
            }
        )

    structure = {
        "root": sorted(decomposition.root),
        "bags": [sorted(bag) for bag in decomposition.bags],
        "tree_edges": [[sorted(first), sorted(second)] for first, second in decomposition.tree_edges],
        "owned_edges": {
            ",".join(map(str, sorted(bag))): [list(edge) for edge in decomposition.owned_edges[bag]]
            for bag in decomposition.bags
        },
    }

    return {
        "width": decomposition.width,
        "maximum_bag_size": decomposition.maximum_bag_size,
        "bag_count": decomposition.bag_count,
        "heuristic": decomposition.heuristic.value,
        "minimum_fill_width": decomposition.minimum_fill_width,
        "minimum_degree_width": decomposition.minimum_degree_width,
        "estimated_state_upper_bound": estimated_states,
        "largest_candidate_products": sorted(bag_rows, key=lambda row: int(row["candidate_product"]), reverse=True)[:10],
        "digest": _json_digest(structure),
    }


def _dp_report(result: BothObjectiveResult) -> dict[str, object]:
    return {
        "enumerated_states": result.statistics.enumerated_states,
        "feasible_states": result.statistics.feasible_states,
        "message_entries": result.statistics.message_entries,
        "unique_cost_requests": result.statistics.unique_cost_requests,
        "bags": [
            {
                "bag": sorted(statistics.bag),
                "enumerated_states": statistics.enumerated_states,
                "feasible_states": statistics.feasible_states,
                "message_entries": statistics.message_entries,
            }
            for statistics in result.statistics.bags
        ],
    }


def _materialize_solution(source: JunctionGraph, edge_cost: BaseEdgeCost, solution: DPSolution | None) -> dict[str, object]:
    if solution is None:
        return {
            "feasible": False,
            "value": None,
            "value_hex": None,
            "mapping_size": 0,
            "mapping_digest": None,
            "witness_count": 0,
            "witness_points": 0,
            "witness_digest": None,
        }

    mapping = {int(source_vertex): int(target_vertex) for source_vertex, target_vertex in solution.mapping.items()}
    mapping_rows = sorted(mapping.items())
    witness_hasher = hashlib.sha256()
    local_values: list[float] = []
    witness_count = 0
    witness_points = 0

    for edge in sorted(source.edges, key=lambda item: item.id):
        target_u = mapping[edge.u]
        target_v = mapping[edge.v]
        value = float(edge_cost(edge.id, edge.u, edge.v, target_u, target_v))

        if not math.isfinite(value):
            raise RuntimeError(f"Recovered solution has nonfinite cost for source edge e{edge.id}.")

        witness = edge_cost.witness(edge.id, edge.u, edge.v, target_u, target_v)

        if witness is None:
            raise RuntimeError(f"Recovered solution has no witness for source edge e{edge.id}.")

        points = np.asarray(witness, dtype=np.float64, order="C")

        if points.ndim != 2 or points.shape[1] != 2:
            raise RuntimeError(f"Witness for source edge e{edge.id} has invalid shape {points.shape}.")

        witness_hasher.update(int(edge.id).to_bytes(8, byteorder="little", signed=True))
        witness_hasher.update(np.asarray(points.shape, dtype="<i8").tobytes())
        witness_hasher.update(np.asarray(points, dtype="<f8", order="C").tobytes())
        local_values.append(value)
        witness_count += 1
        witness_points += len(points)

    realized_value = sum(local_values) if solution.objective is Objective.ADDITIVE else max(local_values, default=0.0)

    if not math.isclose(realized_value, solution.value, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(
            f"Recovered {solution.objective.value} value differs from the DP value: DP={solution.value}, realized={realized_value}."
        )

    return {
        "feasible": True,
        "value": solution.value,
        "value_hex": solution.value.hex(),
        "mapping_size": len(mapping),
        "mapping_digest": _json_digest(mapping_rows),
        "witness_count": witness_count,
        "witness_points": witness_points,
        "witness_digest": witness_hasher.hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_once(
    source_path: Path,
    target_path: Path,
    cost_name: CostName,
    candidate_rho: float,
    top_k: int,
    cost_options: Mapping[str, object],
    repeat: int,
) -> dict[str, object]:
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    source = _timed("load_source", lambda: load_junction_graph(source_path), timings)
    target = _timed("load_target", lambda: load_junction_graph(target_path), timings)
    candidate_sets = _timed(
        "candidate_generation",
        lambda: compute_candidate_sets(source, target, rho=candidate_rho, top_k=top_k),
        timings,
    )
    decomposition = _timed("source_decomposition", lambda: build_source_decomposition(source), timings)
    factory = _timed("cost_factory", lambda: CostFactory(source, target), timings)
    edge_cost = _timed("cost_creation", lambda: factory.create(cost_name, **dict(cost_options)), timings)
    dp_result = _timed(
        "dynamic_programming",
        lambda: solve_tree_dp_both(decomposition, candidate_sets, edge_cost),
        timings,
    )
    additive = _timed(
        "materialize_additive",
        lambda: _materialize_solution(source, edge_cost, dp_result.additive),
        timings,
    )
    bottleneck = _timed(
        "materialize_bottleneck",
        lambda: _materialize_solution(source, edge_cost, dp_result.bottleneck),
        timings,
    )
    timings["total"] = time.perf_counter() - total_started

    return {
        "repeat": repeat,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source_path.resolve()),
            "name": source.name,
            "vertices": len(source.vertices),
            "edges": len(source.edges),
        },
        "target": {
            "path": str(target_path.resolve()),
            "name": target.name,
            "vertices": len(target.vertices),
            "edges": len(target.edges),
        },
        "cost": {
            "name": edge_cost.name.value,
            "options": dict(cost_options),
        },
        "candidate_parameters": {
            "rho": candidate_rho,
            "top_k": top_k,
        },
        "candidates": _candidate_report(candidate_sets),
        "decomposition": _decomposition_report(decomposition, candidate_sets),
        "dynamic_programming": _dp_report(dp_result),
        "objectives": {
            "additive": additive,
            "bottleneck": bottleneck,
        },
        "timings_seconds": timings,
    }


def _stable_signature(run: Mapping[str, Any]) -> dict[str, object]:
    objectives = run["objectives"]
    assert isinstance(objectives, Mapping)

    return {
        "source": run["source"],
        "target": run["target"],
        "cost": run["cost"],
        "candidate_parameters": run["candidate_parameters"],
        "candidates": run["candidates"],
        "decomposition": run["decomposition"],
        "dynamic_programming": run["dynamic_programming"],
        "objectives": objectives,
    }


def _print_run(run: Mapping[str, Any]) -> None:
    source = run["source"]
    target = run["target"]
    candidates = run["candidates"]
    decomposition = run["decomposition"]
    dp = run["dynamic_programming"]
    objectives = run["objectives"]
    timings = run["timings_seconds"]

    print(f"\n=== repeat {run['repeat']} ===")
    print(
        f"{source['name']} ({source['vertices']} V, {source['edges']} E) -> "
        f"{target['name']} ({target['vertices']} V, {target['edges']} E)"
    )
    print(
        "candidates: "
        f"empty={candidates['empty_domains']}, total={candidates['total_candidates']}, "
        f"min={candidates['minimum_candidates']}, median={candidates['median_candidates']}, "
        f"max={candidates['maximum_candidates']}"
    )
    print(
        "decomposition: "
        f"width={decomposition['width']}, bags={decomposition['bag_count']}, "
        f"estimated states={decomposition['estimated_state_upper_bound']}"
    )
    print(
        "DP: "
        f"enumerated={dp['enumerated_states']}, feasible={dp['feasible_states']}, "
        f"messages={dp['message_entries']}, unique costs={dp['unique_cost_requests']}"
    )

    for objective in ("additive", "bottleneck"):
        result = objectives[objective]
        print(
            f"{objective}: feasible={result['feasible']}, value={result['value']}, "
            f"mapping={result['mapping_size']}, witnesses={result['witness_count']}"
        )

    print("timings:")

    for stage, seconds in timings.items():
        print(f"  {stage:24s} {seconds:10.3f} s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and verify the migrated matcher on one production graph pair.")
    parser.add_argument("source", type=Path, help="Source TopoTide .txt graph.")
    parser.add_argument("target", type=Path, help="Target TopoTide .txt graph.")
    parser.add_argument(
        "--cost",
        choices=tuple(cost.value for cost in CostName),
        default=CostName.RELATIVE_LENGTH_ERROR.value,
    )
    parser.add_argument("--candidate-rho", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--cost-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()

    if not arguments.source.is_file():
        raise FileNotFoundError(arguments.source)

    if not arguments.target.is_file():
        raise FileNotFoundError(arguments.target)

    if arguments.source.resolve() == arguments.target.resolve():
        raise ValueError("Source and target graph files must be different.")

    if not math.isfinite(arguments.candidate_rho) or arguments.candidate_rho < 0.0:
        raise ValueError("Candidate radius must be finite and nonnegative.")

    if arguments.top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if arguments.repeats < 1:
        raise ValueError("repeats must be at least 1.")

    cost_name = CostName(arguments.cost)
    cost_options = _parse_cost_options(arguments.cost_option)
    runs: list[dict[str, object]] = []

    for repeat in range(1, arguments.repeats + 1):
        run = _run_once(
            arguments.source,
            arguments.target,
            cost_name,
            arguments.candidate_rho,
            arguments.top_k,
            cost_options,
            repeat,
        )
        runs.append(run)
        _print_run(run)

    baseline = _stable_signature(runs[0])
    reproducible = all(_stable_signature(run) == baseline for run in runs[1:])
    output = arguments.output or Path("reports") / (
        f"end_to_end_{arguments.source.stem}_to_{arguments.target.stem}_{cost_name.value}.json"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "reproducible": reproducible,
        "runs": runs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print(f"\nreproducible across {len(runs)} run(s): {reproducible}")
    print(f"report: {output.resolve()}")
    return 0 if reproducible else 2


if __name__ == "__main__":
    raise SystemExit(main())
