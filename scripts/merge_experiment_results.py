#!/usr/bin/env python
"""
Merge River-Matcher experiment suites into analysis-ready tables.

The script reads schema-v2 suite_summary.json files, follows each report path,
extracts complete mappings and edge-cost summaries, removes configured
screening duplicates, and writes compact CSV/JSON tables.

Run from the River-Matcher repository root:

    uv run python scripts/merge_experiment_results.py experiments/analysis_sources.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1


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
    return "".join(output).strip("_") or "analysis"


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalise_report_path(summary_path: Path, raw_path: str) -> Path:
    relative = PurePosixPath(raw_path.replace("\\", "/"))
    return summary_path.parent.joinpath(*relative.parts)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


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
            writer.writerow(
                {
                    key: (
                        _stable_json(value)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _mapping_hash(mapping: Mapping[str, Any] | None) -> str | None:
    if mapping is None:
        return None
    ordered = [
        [int(source), int(target)]
        for source, target in sorted(
            mapping.items(),
            key=lambda item: int(item[0]),
        )
    ]
    return hashlib.sha256(_stable_json(ordered).encode("utf-8")).hexdigest()


def _mapping_agreement(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[int, int, float | None]:
    common = sorted(set(first) & set(second), key=int)
    if not common:
        return 0, 0, None

    agreement = sum(
        int(first[source]) == int(second[source])
        for source in common
    )
    return agreement, len(common), agreement / len(common)


def _solution_extra(solution: Mapping[str, Any] | None) -> dict[str, Any]:
    if solution is None:
        return {
            "mapping_hash": None,
            "local_cost_sum": None,
            "mapping": None,
        }

    edges = solution.get("edges", [])
    return {
        "mapping_hash": _mapping_hash(solution.get("mapping", {})),
        "local_cost_sum": float(
            sum(float(edge["cost"]) for edge in edges)
        ),
        "mapping": solution.get("mapping", {}),
    }


def _load_suite(
    source_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_path = Path(source_config["path"]).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if int(summary.get("schema_version", 0)) != 2:
        raise ValueError(
            f"{summary_path} is not a schema-v2 suite summary."
        )

    include_pairs = set(source_config.get("include_pairs", []))
    exclude_pairs = set(source_config.get("exclude_pairs", []))
    rename_pairs = dict(source_config.get("rename_pairs", {}))

    selected_runs = []
    selected_solutions = []

    for run in summary["runs"]:
        original_pair = str(run["pair_id"])
        if include_pairs and original_pair not in include_pairs:
            continue
        if original_pair in exclude_pairs:
            continue

        pair_id = rename_pairs.get(original_pair, original_pair)
        report_path = _normalise_report_path(
            summary_path,
            str(run["report"]),
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        report_solutions = report.get("solutions", {})
        extras = {
            objective: _solution_extra(
                report_solutions.get(objective)
            )
            for objective in ("additive", "bottleneck")
        }

        canonical_run_id = (
            f"{pair_id}__{_slug(str(run['cost']))}"
            f"__rho_{str(run['candidate_rho']).replace('.', '_')}"
        )

        run_row = dict(run)
        run_row.update(
            {
                "run_id": canonical_run_id,
                "pair_id": pair_id,
                "suite_name": summary["suite_name"],
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "pair_category": report.get(
                    "metadata", {}
                ).get("pair_category"),
                "pair_notes": report.get(
                    "metadata", {}
                ).get("pair_notes"),
            }
        )
        selected_runs.append(run_row)

        source_solution_rows = {
            (str(row["pair_id"]), str(row["cost"]), str(row["objective"])): row
            for row in summary["solutions"]
        }

        for objective in ("additive", "bottleneck"):
            original_key = (
                original_pair,
                str(run["cost"]),
                objective,
            )
            source_row = source_solution_rows.get(original_key)
            if source_row is None:
                continue

            solution_row = dict(source_row)
            solution_row.update(
                {
                    "run_id": canonical_run_id,
                    "pair_id": pair_id,
                    "suite_name": summary["suite_name"],
                    "summary_path": str(summary_path),
                    "report_path": str(report_path),
                    "mapping_hash": extras[objective][
                        "mapping_hash"
                    ],
                    "local_cost_sum": extras[objective][
                        "local_cost_sum"
                    ],
                    "_mapping": extras[objective]["mapping"],
                }
            )
            selected_solutions.append(solution_row)

    metadata = {
        "suite_name": summary["suite_name"],
        "summary_path": str(summary_path),
        "selected_runs": len(selected_runs),
        "selected_solutions": len(selected_solutions),
    }
    return selected_runs, selected_solutions, metadata


def _deduplicate_runs(
    runs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in runs:
        key = (
            run["pair_id"],
            run["source_name"],
            run["target_name"],
            float(run["candidate_rho"]),
            int(run["top_k"]),
            run["cost"],
            _stable_json(run.get("cost_options", {})),
        )
        selected[key] = run
    return list(selected.values())


def _deduplicate_solutions(
    solutions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for solution in solutions:
        key = (
            solution["run_id"],
            solution["objective"],
        )
        selected[key] = solution
    return list(selected.values())


def _graph_pair_rows(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["pair_id"])].append(run)

    rows = []
    for pair_id, pair_runs in sorted(grouped.items()):
        first = pair_runs[0]
        rows.append(
            {
                "pair_id": pair_id,
                "category": first.get("pair_category"),
                "source_name": first["source_name"],
                "source_vertices": first["source_vertices"],
                "source_edges": first["source_edges"],
                "target_name": first["target_name"],
                "target_vertices": first["target_vertices"],
                "target_edges": first["target_edges"],
                "candidate_rho": first["candidate_rho"],
                "top_k": first["top_k"],
                "empty_candidate_domains": first[
                    "empty_candidate_domains"
                ],
                "total_candidates": first["total_candidates"],
                "minimum_candidates": first["minimum_candidates"],
                "median_candidates": first["median_candidates"],
                "maximum_candidates": first["maximum_candidates"],
                "treewidth": first["treewidth"],
                "bags": first["bags"],
                "maximum_bag_size": first["maximum_bag_size"],
                "enumerated_states": first["enumerated_states"],
                "feasible_states": first["feasible_states"],
                "message_entries": first["message_entries"],
                "notes": first.get("pair_notes"),
            }
        )
    return rows


def _objective_tradeoff_rows(
    runs: Sequence[Mapping[str, Any]],
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run_lookup = {str(run["run_id"]): run for run in runs}
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for solution in solutions:
        grouped[str(solution["run_id"])][
            str(solution["objective"])
        ] = solution

    rows = []
    for run_id, objective_rows in sorted(grouped.items()):
        additive = objective_rows.get("additive")
        bottleneck = objective_rows.get("bottleneck")
        if additive is None or bottleneck is None:
            continue

        additive_mapping = additive.get("_mapping") or {}
        bottleneck_mapping = bottleneck.get("_mapping") or {}
        agreement, compared, fraction = _mapping_agreement(
            additive_mapping,
            bottleneck_mapping,
        )

        additive_max = additive.get("local_cost_maximum")
        bottleneck_max = bottleneck.get("local_cost_maximum")
        reduction = None
        reduction_fraction = None
        if additive_max is not None and bottleneck_max is not None:
            reduction = float(additive_max) - float(bottleneck_max)
            if not math.isclose(float(additive_max), 0.0):
                reduction_fraction = reduction / float(additive_max)

        run = run_lookup[run_id]
        rows.append(
            {
                "run_id": run_id,
                "pair_id": run["pair_id"],
                "cost": run["cost"],
                "candidate_rho": run["candidate_rho"],
                "additive_local_cost_sum": additive[
                    "local_cost_sum"
                ],
                "bottleneck_local_cost_sum": bottleneck[
                    "local_cost_sum"
                ],
                "additive_worst_edge": additive_max,
                "bottleneck_worst_edge": bottleneck_max,
                "worst_edge_reduction": reduction,
                "worst_edge_reduction_fraction": reduction_fraction,
                "mapping_agreement_count": agreement,
                "mapping_vertices_compared": compared,
                "mapping_agreement_fraction": fraction,
            }
        )
    return rows


def _zero_cost_rows(
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": solution["pair_id"],
            "cost": solution["cost"],
            "objective": solution["objective"],
            "materialized_witnesses": solution[
                "materialized_witnesses"
            ],
            "zero_cost_edges": solution["zero_cost_edges"],
            "zero_cost_fraction": solution["zero_cost_fraction"],
            "positive_cost_edges": solution["positive_cost_edges"],
            "local_cost_positive_median": solution[
                "local_cost_positive_median"
            ],
            "local_cost_maximum": solution[
                "local_cost_maximum"
            ],
        }
        for solution in solutions
    ]


def _cost_agreement_rows(
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for solution in solutions:
        if solution.get("_mapping"):
            grouped[
                (
                    str(solution["pair_id"]),
                    str(solution["objective"]),
                )
            ].append(solution)

    rows = []
    for (pair_id, objective), entries in sorted(grouped.items()):
        entries = sorted(entries, key=lambda item: str(item["cost"]))
        for index, first in enumerate(entries):
            for second in entries[index + 1 :]:
                agreement, compared, fraction = _mapping_agreement(
                    first["_mapping"],
                    second["_mapping"],
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "objective": objective,
                        "cost_a": first["cost"],
                        "cost_b": second["cost"],
                        "agreement_count": agreement,
                        "vertices_compared": compared,
                        "agreement_fraction": fraction,
                    }
                )
    return rows


def _clean_solution_rows(
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in solution.items()
            if not key.startswith("_")
        }
        for solution in solutions
    ]


def _copy_benchmark(
    benchmark_path: str | None,
) -> list[dict[str, Any]]:
    if not benchmark_path:
        return []

    path = Path(benchmark_path).expanduser().resolve()
    if not path.exists():
        print(
            f"Benchmark summary not found yet: {path}. "
            "Continuing without runtime_benchmark.csv."
        )
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge River-Matcher experiment outputs."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the configured analysis output directory.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    analysis_name = _slug(
        str(config.get("analysis_name", config_path.stem))
    )
    output_root = (
        arguments.output
        if arguments.output is not None
        else Path(config.get("output_dir", "analysis_results"))
    ).expanduser().resolve() / analysis_name
    output_root.mkdir(parents=True, exist_ok=True)

    all_runs: list[dict[str, Any]] = []
    all_solutions: list[dict[str, Any]] = []
    source_metadata = []

    for source_config in config.get("suites", []):
        runs, solutions, metadata = _load_suite(source_config)
        all_runs.extend(runs)
        all_solutions.extend(solutions)
        source_metadata.append(metadata)

    runs = sorted(
        _deduplicate_runs(all_runs),
        key=lambda row: (
            str(row["pair_id"]),
            str(row["cost"]),
            float(row["candidate_rho"]),
        ),
    )
    solutions = sorted(
        _deduplicate_solutions(all_solutions),
        key=lambda row: (
            str(row["pair_id"]),
            str(row["cost"]),
            str(row["objective"]),
        ),
    )

    graph_pairs = _graph_pair_rows(runs)
    tradeoffs = _objective_tradeoff_rows(runs, solutions)
    zero_costs = _zero_cost_rows(solutions)
    cost_agreements = _cost_agreement_rows(solutions)
    clean_solutions = _clean_solution_rows(solutions)
    benchmark_rows = _copy_benchmark(
        config.get("benchmark_summary")
    )

    _write_csv(output_root / "runs.csv", runs)
    _write_csv(output_root / "solutions.csv", clean_solutions)
    _write_csv(output_root / "graph_pairs.csv", graph_pairs)
    _write_csv(
        output_root / "objective_tradeoffs.csv",
        tradeoffs,
    )
    _write_csv(
        output_root / "zero_cost_edges.csv",
        zero_costs,
    )
    _write_csv(
        output_root / "cost_mapping_agreement.csv",
        cost_agreements,
    )
    if benchmark_rows:
        _write_csv(
            output_root / "runtime_benchmark.csv",
            benchmark_rows,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_name": analysis_name,
        "config_path": str(config_path),
        "sources": source_metadata,
        "counts": {
            "runs": len(runs),
            "solutions": len(clean_solutions),
            "graph_pairs": len(graph_pairs),
            "objective_tradeoffs": len(tradeoffs),
            "zero_cost_rows": len(zero_costs),
            "cost_mapping_agreements": len(cost_agreements),
            "runtime_rows": len(benchmark_rows),
        },
        "files": {
            "runs": "runs.csv",
            "solutions": "solutions.csv",
            "graph_pairs": "graph_pairs.csv",
            "objective_tradeoffs": "objective_tradeoffs.csv",
            "zero_cost_edges": "zero_cost_edges.csv",
            "cost_mapping_agreement": "cost_mapping_agreement.csv",
            "runtime_benchmark": (
                "runtime_benchmark.csv" if benchmark_rows else None
            ),
        },
    }
    _write_json(output_root / "analysis_manifest.json", manifest)

    readme = """Generated analysis tables

runs.csv
    One row per graph pair and cost. Shared candidate, decomposition, DP,
    and one-off experiment timing fields occur only once.

solutions.csv
    One row per graph pair, cost, and objective.

graph_pairs.csv
    One compact row per selected graph pair.

objective_tradeoffs.csv
    Additive-versus-bottleneck comparison within each pair and cost,
    including local-cost sums, worst-edge reduction, and mapping agreement.

zero_cost_edges.csv
    Counts and fractions of zero-cost edges. A zero value is specific to
    the selected cost and must not automatically be interpreted as identical
    geometry.

cost_mapping_agreement.csv
    Pairwise mapping agreement between cost functions for each objective.

runtime_benchmark.csv
    Present only after benchmark_summary.csv is supplied in the analysis
    configuration. Use these repeated timings rather than one-off experiment
    timings in the thesis runtime table.
"""
    (output_root / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Analysis tables written to: {output_root}")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
