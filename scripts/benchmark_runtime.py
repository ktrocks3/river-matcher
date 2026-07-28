#!/usr/bin/env python
"""
Controlled runtime benchmark for GeoMatcher.

Protocol
--------
* Graph files are loaded once per pair.
* Every warm-up and measured trial constructs a fresh RiverGraphMatcher.
* One or more warm-ups are discarded.
* Measured pair/cost trials are shuffled with a fixed seed.
* Figure generation is not performed.
* Preparation and solve time are reported separately.
* Result fingerprints are checked against the warm-up result so timing runs
  cannot silently change the computed mapping or objective values.

Run from the GeoMatcher repository root:

    uv run python scripts/benchmark_runtime.py experiments/runtime_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from river_matcher import RiverGraphMatcher, available_costs, load_junction_graph


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    output: list[str] = []
    previous_separator = False
    for character in value.strip().lower():
        if character.isalnum():
            output.append(character)
            previous_separator = False
        elif not previous_separator:
            output.append("_")
            previous_separator = True
    return "".join(output).strip("_") or "benchmark"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _mapping_hash(mapping: Mapping[Any, Any] | None) -> str | None:
    if mapping is None:
        return None
    payload = [
        [int(source), int(target)]
        for source, target in sorted(mapping.items())
    ]
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _result_signature(result: Any) -> dict[str, Any]:
    additive = result.additive
    bottleneck = result.bottleneck
    statistics = result.dp_statistics

    signature = {
        "additive_feasible": additive is not None,
        "bottleneck_feasible": bottleneck is not None,
        "additive_value": (
            float(additive.value) if additive is not None else None
        ),
        "bottleneck_value": (
            float(bottleneck.value) if bottleneck is not None else None
        ),
        "additive_mapping_hash": (
            _mapping_hash(additive.mapping) if additive is not None else None
        ),
        "bottleneck_mapping_hash": (
            _mapping_hash(bottleneck.mapping)
            if bottleneck is not None
            else None
        ),
        "enumerated_states": int(statistics.enumerated_states),
        "feasible_states": int(statistics.feasible_states),
        "message_entries": int(statistics.message_entries),
        "unique_cost_requests": int(statistics.unique_cost_requests),
    }
    signature["fingerprint"] = hashlib.sha256(
        _stable_json(signature).encode("utf-8")
    ).hexdigest()
    return signature


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialised = {
                key: (
                    _stable_json(value)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serialised)


def _summary(values: Sequence[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}

    return {
        f"{prefix}_minimum": float(np.min(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_maximum": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_standard_deviation": float(np.std(array)),
    }


def _normalise_costs(
    pair: Mapping[str, Any],
    global_costs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = pair.get("costs", global_costs)
    return [
        {
            "name": str(cost["name"]),
            "options": dict(cost.get("options", {})),
        }
        for cost in selected
    ]


def _validate_config(config: Mapping[str, Any]) -> None:
    pairs = config.get("pairs")
    if not pairs:
        raise ValueError("Benchmark configuration contains no pairs.")

    known_costs = {str(cost) for cost in available_costs()}
    global_costs = config.get("costs", [])

    for pair in pairs:
        for key in ("source", "target"):
            path = Path(pair[key]).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"{key} graph does not exist: {path}"
                )

        pair_costs = _normalise_costs(pair, global_costs)
        if not pair_costs:
            raise ValueError(
                f"Pair {pair.get('id', '<unnamed>')} contains no costs."
            )
        unknown = sorted(
            {
                cost["name"]
                for cost in pair_costs
                if cost["name"] not in known_costs
            }
        )
        if unknown:
            raise ValueError(
                "Unknown costs: "
                + ", ".join(unknown)
                + ". Available costs: "
                + ", ".join(sorted(known_costs))
            )


def _run_once(
    source: Any,
    target: Any,
    *,
    candidate_rho: float,
    top_k: int,
    cost_name: str,
    cost_options: Mapping[str, Any],
) -> tuple[dict[str, Any], float, float]:
    gc.collect()

    prepare_started = time.perf_counter()
    matcher = RiverGraphMatcher(
        source,
        target,
        candidate_rho=candidate_rho,
        top_k=top_k,
    )
    prepare_seconds = time.perf_counter() - prepare_started

    solve_started = time.perf_counter()
    result = matcher.match_both(cost_name, **dict(cost_options))
    solve_seconds = time.perf_counter() - solve_started

    return _result_signature(result), prepare_seconds, solve_seconds


def _pair_identifier(pair: Mapping[str, Any]) -> str:
    return _slug(
        str(
            pair.get(
                "id",
                f"{Path(pair['source']).stem}_to_{Path(pair['target']).stem}",
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GeoMatcher with fresh matcher instances, "
            "discarded warm-ups, and repeated measured trials."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)

    benchmark_name = _slug(
        str(config.get("benchmark_name", config_path.stem))
    )
    output_root = (
        arguments.output
        if arguments.output is not None
        else Path(config.get("output_dir", "benchmark_results"))
    ).expanduser().resolve() / benchmark_name
    output_root.mkdir(parents=True, exist_ok=True)

    warmups = int(config.get("warmups", 1))
    repetitions = int(config.get("repetitions", 5))
    seed = int(config.get("seed", 20260705))
    defaults = dict(config.get("defaults", {}))
    global_costs = list(config.get("costs", []))

    if warmups < 0:
        raise ValueError("warmups must be non-negative.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "benchmark_name": benchmark_name,
        "config_path": str(config_path),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "warmups": warmups,
        "repetitions": repetitions,
        "seed": seed,
        "protocol": (
            "Graphs loaded once per pair; fresh matcher per trial; "
            "warm-ups discarded; measured trials shuffled; no figures."
        ),
    }

    raw_trials: list[dict[str, Any]] = []
    expected_signatures: dict[tuple[str, str, str], dict[str, Any]] = {}
    pair_load_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    for pair in config["pairs"]:
        pair_id = _pair_identifier(pair)
        source_path = Path(pair["source"]).expanduser().resolve()
        target_path = Path(pair["target"]).expanduser().resolve()
        candidate_rho = float(
            pair.get(
                "candidate_rho",
                defaults.get("candidate_rho", 10.0),
            )
        )
        top_k = int(
            pair.get("top_k", defaults.get("top_k", 25))
        )
        costs = _normalise_costs(pair, global_costs)

        load_started = time.perf_counter()
        source = load_junction_graph(source_path)
        target = load_junction_graph(target_path)
        load_seconds = time.perf_counter() - load_started

        pair_load_rows.append(
            {
                "pair_id": pair_id,
                "source_name": source.name,
                "target_name": target.name,
                "source_vertices": len(source.vertices),
                "source_edges": len(source.edges),
                "target_vertices": len(target.vertices),
                "target_edges": len(target.edges),
                "candidate_rho": candidate_rho,
                "top_k": top_k,
                "load_graphs_seconds": load_seconds,
            }
        )

        for cost in costs:
            cost_name = cost["name"]
            cost_options = cost["options"]
            key = (
                pair_id,
                cost_name,
                _stable_json(cost_options),
            )

            for warmup_index in range(warmups):
                signature, _, _ = _run_once(
                    source,
                    target,
                    candidate_rho=candidate_rho,
                    top_k=top_k,
                    cost_name=cost_name,
                    cost_options=cost_options,
                )
                if key in expected_signatures:
                    if (
                        signature["fingerprint"]
                        != expected_signatures[key]["fingerprint"]
                    ):
                        raise RuntimeError(
                            f"Warm-up result changed for {pair_id} / "
                            f"{cost_name}."
                        )
                else:
                    expected_signatures[key] = signature

                print(
                    f"[warm-up {warmup_index + 1}/{warmups}] "
                    f"{pair_id} / {cost_name}"
                )

        jobs = [
            (repetition, cost)
            for repetition in range(1, repetitions + 1)
            for cost in costs
        ]
        random.Random(seed + sum(map(ord, pair_id))).shuffle(jobs)

        for order_index, (repetition, cost) in enumerate(jobs, start=1):
            cost_name = cost["name"]
            cost_options = cost["options"]
            key = (
                pair_id,
                cost_name,
                _stable_json(cost_options),
            )

            signature, prepare_seconds, solve_seconds = _run_once(
                source,
                target,
                candidate_rho=candidate_rho,
                top_k=top_k,
                cost_name=cost_name,
                cost_options=cost_options,
            )

            expected = expected_signatures.get(key)
            if expected is None:
                expected_signatures[key] = signature
            elif signature["fingerprint"] != expected["fingerprint"]:
                raise RuntimeError(
                    f"Measured result changed for {pair_id} / "
                    f"{cost_name}, repetition {repetition}."
                )

            row = {
                "pair_id": pair_id,
                "source_name": source.name,
                "target_name": target.name,
                "source_vertices": len(source.vertices),
                "source_edges": len(source.edges),
                "target_vertices": len(target.vertices),
                "target_edges": len(target.edges),
                "candidate_rho": candidate_rho,
                "top_k": top_k,
                "cost": cost_name,
                "cost_options": cost_options,
                "repetition": repetition,
                "execution_order_within_pair": order_index,
                "prepare_seconds": prepare_seconds,
                "solve_both_objectives_seconds": solve_seconds,
                "prepare_plus_solve_seconds": (
                    prepare_seconds + solve_seconds
                ),
                **signature,
            }
            raw_trials.append(row)

            print(
                f"[{pair_id}] {cost_name} "
                f"rep={repetition}/{repetitions}: "
                f"prepare={prepare_seconds:.3f}s, "
                f"solve={solve_seconds:.3f}s"
            )

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in raw_trials:
        grouped[
            (
                str(row["pair_id"]),
                str(row["cost"]),
                _stable_json(row["cost_options"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (_, _, _), rows in sorted(grouped.items()):
        first = rows[0]
        summary = {
            key: first[key]
            for key in (
                "pair_id",
                "source_name",
                "target_name",
                "source_vertices",
                "source_edges",
                "target_vertices",
                "target_edges",
                "candidate_rho",
                "top_k",
                "cost",
                "cost_options",
                "additive_feasible",
                "bottleneck_feasible",
                "additive_value",
                "bottleneck_value",
                "additive_mapping_hash",
                "bottleneck_mapping_hash",
                "enumerated_states",
                "feasible_states",
                "message_entries",
                "unique_cost_requests",
                "fingerprint",
            )
        }
        summary["repetitions"] = len(rows)
        summary.update(
            _summary(
                [float(row["prepare_seconds"]) for row in rows],
                "prepare_seconds",
            )
        )
        summary.update(
            _summary(
                [
                    float(row["solve_both_objectives_seconds"])
                    for row in rows
                ],
                "solve_both_objectives_seconds",
            )
        )
        summary.update(
            _summary(
                [
                    float(row["prepare_plus_solve_seconds"])
                    for row in rows
                ],
                "prepare_plus_solve_seconds",
            )
        )
        summary_rows.append(summary)

    total_seconds = time.perf_counter() - benchmark_started
    payload = {
        **metadata,
        "pair_loads": pair_load_rows,
        "trial_count": len(raw_trials),
        "summary_count": len(summary_rows),
        "benchmark_runtime_seconds": total_seconds,
        "trials": raw_trials,
        "summary": summary_rows,
    }

    _write_json(output_root / "benchmark.json", payload)
    _write_csv(output_root / "raw_trials.csv", raw_trials)
    _write_csv(output_root / "benchmark_summary.csv", summary_rows)
    _write_csv(output_root / "pair_loads.csv", pair_load_rows)

    print()
    print(f"Benchmark complete: {output_root}")
    print(f"Summary: {output_root / 'benchmark_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
