#!/usr/bin/env python
"""
Run a reproducible River-Matcher experiment suite without opening the UI.

For every graph pair and cost, this script:
  * loads and prepares the graph pair once;
  * solves additive and bottleneck objectives in one DP traversal;
  * records candidate, decomposition, DP, timing, mapping, edge-cost, and
    witness information in JSON;
  * writes an overlaid mapping PNG for each feasible objective;
  * writes a best/median/worst edge-detail PNG for each feasible objective;
  * writes one suite_summary.json containing compact rows for later analysis.

Run from the River-Matcher repository root:

    uv run python scripts/run_experiment_suite.py experiments.json

The configuration format is documented in experiments.example.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from river_matcher import (RiverGraphMatcher, available_costs, load_junction_graph)

SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    cleaned = []
    previous_underscore = False
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_underscore = False
        elif not previous_underscore:
            cleaned.append("_")
            previous_underscore = True
    return "".join(cleaned).strip("_") or "run"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float)):
        return _jsonable(enum_value)
    return repr(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_payload(matcher: RiverGraphMatcher) -> dict[str, Any]:
    sizes = [len(candidates) for candidates in matcher.candidate_sets.values()]
    stats = matcher.candidate_statistics

    return {"source_vertices": stats.source_vertices, "empty_domains": stats.empty_domains, "total_candidates": stats.total_candidates,
        "minimum_candidates": stats.minimum_candidates, "median_candidates": (float(statistics.median(sizes)) if sizes else 0.0),
        "mean_candidates": (float(statistics.fmean(sizes)) if sizes else 0.0), "maximum_candidates": stats.maximum_candidates,
        "domain_sizes": {str(vertex): len(candidates) for vertex, candidates in sorted(matcher.candidate_sets.items())}}


def _decomposition_payload(matcher: RiverGraphMatcher) -> dict[str, Any]:
    decomposition = matcher.decomposition
    raw_plans = getattr(decomposition, "bag_plans", ())
    if isinstance(raw_plans, Mapping):
        plans = tuple(raw_plans.values())
    else:
        plans = tuple(raw_plans)

    bags = []
    for plan in plans:
        bag = getattr(plan, "bag", ())
        bags.append(tuple(sorted(int(vertex) for vertex in bag)))

    width = getattr(decomposition, "width", None)
    if width is None and bags:
        width = max(len(bag) for bag in bags) - 1

    heuristic = getattr(decomposition, "heuristic", None)
    if heuristic is None:
        heuristic = getattr(decomposition, "method", None)

    return {"width": _jsonable(width), "heuristic": _jsonable(heuristic), "bags": len(bags), "maximum_bag_size": max(map(len, bags), default=0),
        "bag_sizes": [len(bag) for bag in bags]}


def _dp_payload(result: Any) -> dict[str, Any]:
    stats = result.dp_statistics
    bag_rows = []

    for item in stats.bags:
        bag_rows.append(
            {"bag": sorted(int(vertex) for vertex in item.bag), "bag_size": len(item.bag), "enumerated_states": item.enumerated_states, "feasible_states": item.feasible_states,
                "message_entries": item.message_entries})

    return {"enumerated_states": stats.enumerated_states, "feasible_states": stats.feasible_states, "message_entries": stats.message_entries,
        "unique_cost_requests": stats.unique_cost_requests, "bags": bag_rows}


ZERO_COST_TOLERANCE = 1e-12


def _edge_statistics(edges: Sequence[Any]) -> dict[str, Any]:
    values = np.asarray([float(edge.cost) for edge in edges], dtype=np.float64)

    if values.size == 0:
        return {"count": 0, "minimum": None, "q25": None, "median": None, "mean": None, "q75": None, "maximum": None, "standard_deviation": None, "zero_cost_count": 0,
            "zero_cost_fraction": None, "positive_cost_count": 0, "positive_cost_median": None}

    zero_mask = np.isclose(values, 0.0, rtol=0.0, atol=ZERO_COST_TOLERANCE)
    positive_values = values[~zero_mask]

    return {"count": int(values.size), "minimum": float(np.min(values)), "q25": float(np.quantile(values, 0.25)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "q75": float(np.quantile(values, 0.75)), "maximum": float(np.max(values)), "standard_deviation": float(np.std(values)),
        "zero_cost_count": int(np.count_nonzero(zero_mask)), "zero_cost_fraction": float(np.mean(zero_mask)), "positive_cost_count": int(positive_values.size),
        "positive_cost_median": (float(np.median(positive_values)) if positive_values.size else None)}


def _representative_edge_objects(edges: Sequence[Any]) -> list[tuple[str, Any]]:
    if not edges:
        return []

    ordered = sorted(edges, key=lambda edge: (float(edge.cost), int(edge.edge_id)))
    positive = [edge for edge in ordered if not math.isclose(float(edge.cost), 0.0, rel_tol=0.0, abs_tol=ZERO_COST_TOLERANCE)]
    median_positive = (positive[(len(positive) - 1) // 2] if positive else ordered[0])

    return [("Best", ordered[0]), ("Median positive", median_positive), ("Worst", ordered[-1]), ]


def _representative_edges(edges: Sequence[Any]) -> dict[str, int] | None:
    selected = _representative_edge_objects(edges)
    if not selected:
        return None

    return {"best": int(selected[0][1].edge_id), "median_positive": int(selected[1][1].edge_id), "worst": int(selected[2][1].edge_id)}


def _solution_payload(solution: Any | None) -> dict[str, Any] | None:
    if solution is None:
        return None

    return {"objective": solution.objective.value, "value": float(solution.value),
        "mapping": {str(source_vertex): int(target_vertex) for source_vertex, target_vertex in sorted(solution.mapping.items())},
        "edge_statistics": _edge_statistics(solution.edges), "representative_edge_ids": _representative_edges(solution.edges), "edges": [
            {"edge_id": int(edge.edge_id), "source_u": int(edge.source_u), "source_v": int(edge.source_v), "target_u": int(edge.target_u), "target_v": int(edge.target_v),
                "cost": float(edge.cost), "witness": np.asarray(edge.witness, dtype=np.float64).tolist()} for edge in
            sorted(solution.edges, key=lambda edge: int(edge.edge_id))]}


def _edge_lookup(graph: Any) -> dict[int, Any]:
    return {int(edge.id): edge for edge in graph.edges}


def _all_graph_lines(graph: Any) -> list[np.ndarray]:
    lines = []
    for edge in graph.edges:
        polyline = np.asarray(edge.polyline, dtype=np.float64)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            lines.append(polyline)
    return lines


def _finite_cost_range(edges: Sequence[Any], visual_range: Sequence[float] | None = None) -> tuple[float, float]:
    if visual_range is not None:
        if len(visual_range) != 2:
            raise ValueError("visual_range must contain exactly [minimum, maximum].")
        low = float(visual_range[0])
        high = float(visual_range[1])
        if (not math.isfinite(low) or not math.isfinite(high) or high <= low):
            raise ValueError(f"Invalid visual_range={visual_range!r}; "
                             "expected two finite numbers with maximum > minimum.")
        return low, high

    values = np.asarray([float(edge.cost) for edge in edges if math.isfinite(float(edge.cost))], dtype=np.float64)

    if values.size == 0:
        return 0.0, 1.0

    low = float(np.min(values))
    high = float(np.max(values))

    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        high = low + 1.0

    return low, high


def _source_edge_polyline(source_edge_by_id: Mapping[int, Any], edge_id: int) -> np.ndarray:
    source_edge = source_edge_by_id[edge_id]
    return np.asarray(source_edge.polyline, dtype=np.float64)


def _plot_overview(source: Any, target: Any, solution: Any, *, title: str, output_path: Path, dpi: int, visual_range: Sequence[float] | None, cost_label: str) -> None:
    source_edge_by_id = _edge_lookup(source)
    vmin, vmax = _finite_cost_range(solution.edges, visual_range)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(13, 9))

    target_lines = _all_graph_lines(target)
    if target_lines:
        ax.add_collection(LineCollection(target_lines, colors="0.78", linewidths=1.0, alpha=0.60, zorder=1))

    witness_lines = []
    witness_colors = []
    source_lines = []
    source_colors = []

    for matched_edge in solution.edges:
        cost = float(np.clip(matched_edge.cost, vmin, vmax))
        color = cmap(norm(cost))

        witness = np.asarray(matched_edge.witness, dtype=np.float64)
        if witness.ndim == 2 and witness.shape[0] >= 2:
            witness_lines.append(witness)
            witness_colors.append(color)

        source_polyline = _source_edge_polyline(source_edge_by_id, int(matched_edge.edge_id))
        source_lines.append(source_polyline)
        source_colors.append(color)

    if witness_lines:
        ax.add_collection(LineCollection(witness_lines, colors=witness_colors, linewidths=2.2, linestyles="dashed", alpha=0.75, zorder=2))

    if source_lines:
        ax.add_collection(LineCollection(source_lines, colors=source_colors, linewidths=5.0, alpha=0.34, zorder=3))
        ax.add_collection(LineCollection(source_lines, colors="black", linewidths=1.0, alpha=0.95, zorder=4))

    source_x = [float(source.coordinates[vertex][0]) for vertex in source.vertices]
    source_y = [float(source.coordinates[vertex][1]) for vertex in source.vertices]
    target_x = [float(target.coordinates[vertex][0]) for vertex in target.vertices]
    target_y = [float(target.coordinates[vertex][1]) for vertex in target.vertices]

    ax.scatter(target_x, target_y, s=8, c="0.60", alpha=0.45, linewidths=0.0, zorder=2)
    ax.scatter(source_x, source_y, s=18, facecolors="white", edgecolors="black", linewidths=0.45, zorder=5)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(cost_label)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.margins(0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_context(ax: Any, source: Any, target: Any) -> None:
    target_lines = _all_graph_lines(target)
    source_lines = _all_graph_lines(source)

    if target_lines:
        ax.add_collection(LineCollection(target_lines, colors="0.82", linewidths=0.8, alpha=0.55, zorder=1))
    if source_lines:
        ax.add_collection(LineCollection(source_lines, colors="0.55", linewidths=0.6, alpha=0.30, zorder=1))


def _plot_edge_details(source: Any, target: Any, solution: Any, *, title: str, output_path: Path, dpi: int) -> None:
    source_edge_by_id = _edge_lookup(source)
    selected = _representative_edge_objects(solution.edges)

    if not selected:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)

    for ax, (label, matched_edge) in zip(axes[0], selected):
        _draw_context(ax, source, target)

        source_polyline = _source_edge_polyline(source_edge_by_id, int(matched_edge.edge_id))
        witness = np.asarray(matched_edge.witness, dtype=np.float64)

        ax.plot(source_polyline[:, 0], source_polyline[:, 1], linewidth=3.0, label="Source edge", zorder=4)
        ax.plot(witness[:, 0], witness[:, 1], linewidth=3.0, linestyle="--", label="Target witness", zorder=4)

        source_u_xy = np.asarray(source.coordinates[int(matched_edge.source_u)], dtype=np.float64)
        source_v_xy = np.asarray(source.coordinates[int(matched_edge.source_v)], dtype=np.float64)
        target_u_xy = np.asarray(target.coordinates[int(matched_edge.target_u)], dtype=np.float64)
        target_v_xy = np.asarray(target.coordinates[int(matched_edge.target_v)], dtype=np.float64)

        ax.plot([source_u_xy[0], target_u_xy[0]], [source_u_xy[1], target_u_xy[1]], linewidth=0.9, linestyle=":", alpha=0.75, zorder=3)
        ax.plot([source_v_xy[0], target_v_xy[0]], [source_v_xy[1], target_v_xy[1]], linewidth=0.9, linestyle=":", alpha=0.75, zorder=3)

        points = np.vstack((source_polyline, witness))
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        padding = 0.18 * span

        ax.set_xlim(float(minimum[0] - padding[0]), float(maximum[0] + padding[0]))
        ax.set_ylim(float(minimum[1] - padding[1]), float(maximum[1] + padding[1]))
        ax.set_title(f"{label}: e{matched_edge.edge_id}\n"
                     f"cost={float(matched_edge.cost):.6g}, "
                     f"({matched_edge.source_u}, {matched_edge.source_v}) "
                     f"→ ({matched_edge.target_u}, {matched_edge.target_v})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[0][0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_distribution(solution: Any, *, title: str, output_path: Path, dpi: int) -> None:
    costs = np.asarray([float(edge.cost) for edge in solution.edges], dtype=np.float64)
    if costs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(30, max(5, int(math.sqrt(costs.size))))
    ax.hist(costs, bins=bins)
    ax.axvline(float(np.median(costs)), linestyle="--", linewidth=1.2, label=f"Median = {np.median(costs):.4g}")
    ax.set_title(title)
    ax.set_xlabel("Local edge cost")
    ax.set_ylabel("Source-edge count")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_pair(pair: Mapping[str, Any], *, costs: Sequence[Mapping[str, Any]], defaults: Mapping[str, Any], suite_root: Path, dpi: int, common_metadata: Mapping[str, Any]) -> \
tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_id = _slug(str(pair.get("id", f"{Path(pair['source']).stem}_to_{Path(pair['target']).stem}")))
    source_path = Path(pair["source"]).expanduser().resolve()
    target_path = Path(pair["target"]).expanduser().resolve()
    candidate_rho = float(pair.get("candidate_rho", defaults.get("candidate_rho", 10.0)))
    top_k = int(pair.get("top_k", defaults.get("top_k", 25)))

    pair_root = suite_root / pair_id
    pair_root.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    source = load_junction_graph(source_path)
    target = load_junction_graph(target_path)
    load_seconds = time.perf_counter() - load_started

    prepare_started = time.perf_counter()
    matcher = RiverGraphMatcher(source, target, candidate_rho=candidate_rho, top_k=top_k)
    prepare_seconds = time.perf_counter() - prepare_started

    pair_payload = {"schema_version": SCHEMA_VERSION, "created_at_utc": _utc_now(), "pair_id": pair_id,
        "source": {"name": source.name, "path": str(source_path), "vertices": len(source.vertices), "edges": len(source.edges)},
        "target": {"name": target.name, "path": str(target_path), "vertices": len(target.vertices), "edges": len(target.edges)},
        "parameters": {"candidate_rho": candidate_rho, "top_k": top_k}, "candidate_statistics": _candidate_payload(matcher), "decomposition": _decomposition_payload(matcher),
        "timings_seconds": {"load_graphs": load_seconds, "prepare_matcher": prepare_seconds},
        "metadata": {**common_metadata, "pair_notes": pair.get("notes"), "pair_category": pair.get("category")}}
    _write_json(pair_root / "pair.json", pair_payload)

    run_rows: list[dict[str, Any]] = []
    solution_rows: list[dict[str, Any]] = []

    decomposition_summary = _decomposition_payload(matcher)
    candidate_summary = _candidate_payload(matcher)

    for cost_config in costs:
        cost_name = str(cost_config["name"])
        cost_options = dict(cost_config.get("options", {}))
        visual_range = cost_config.get("visual_range")
        cost_label = str(cost_config.get("label", f"Local edge cost: {cost_name}"))
        cost_root = pair_root / _slug(cost_name)
        cost_root.mkdir(parents=True, exist_ok=True)

        run_started = time.perf_counter()
        result = matcher.match_both(cost_name, **cost_options)
        solve_seconds = time.perf_counter() - run_started

        figure_started = time.perf_counter()
        generated_figures: dict[str, dict[str, str]] = {}

        for objective_name, solution in (("additive", result.additive), ("bottleneck", result.bottleneck),):
            if solution is None:
                continue

            objective_figures = {"overview": str((cost_root / f"{objective_name}_overview.png").relative_to(suite_root)),
                "edge_details": str((cost_root / f"{objective_name}_best_median_positive_worst.png").relative_to(suite_root)),
                "cost_distribution": str((cost_root / f"{objective_name}_cost_distribution.png").relative_to(suite_root))}
            generated_figures[objective_name] = objective_figures

            _plot_overview(source, target, solution, title=(f"{source.name} → {target.name} | "
                                                            f"{cost_name} | {objective_name}"), output_path=cost_root / f"{objective_name}_overview.png", dpi=dpi,
                visual_range=visual_range, cost_label=cost_label)
            _plot_edge_details(source, target, solution, title=(f"{source.name} → {target.name} | "
                                                                f"{cost_name} | {objective_name}"), output_path=cost_root / f"{objective_name}_best_median_positive_worst.png",
                dpi=dpi)
            _plot_cost_distribution(solution, title=(f"Local costs | {source.name} → {target.name} | "
                                                     f"{cost_name} | {objective_name}"), output_path=cost_root / f"{objective_name}_cost_distribution.png", dpi=dpi)

        figure_seconds = time.perf_counter() - figure_started

        report = {**pair_payload,
            "cost": {"name": cost_name, "options": cost_options, "visual_range": visual_range, "label": cost_label, "visual_range_is_fixed": visual_range is not None},
            "feasible": {"additive": result.additive is not None, "bottleneck": result.bottleneck is not None}, "dp_statistics": _dp_payload(result),
            "solutions": {"additive": _solution_payload(result.additive), "bottleneck": _solution_payload(result.bottleneck)}, "figures": generated_figures,
            "timings_seconds": {**pair_payload["timings_seconds"], "solve_both_objectives": solve_seconds, "generate_figures": figure_seconds,
                "total_for_cost_after_preparation": (solve_seconds + figure_seconds)}}

        report_path = cost_root / "report.json"
        _write_json(report_path, report)

        run_id = f"{pair_id}__{_slug(cost_name)}"
        relative_report = str(report_path.relative_to(suite_root))

        run_rows.append({"run_id": run_id, "pair_id": pair_id, "source_name": source.name, "target_name": target.name, "source_vertices": len(source.vertices),
            "source_edges": len(source.edges), "target_vertices": len(target.vertices), "target_edges": len(target.edges), "candidate_rho": candidate_rho, "top_k": top_k,
            "cost": cost_name, "cost_options": cost_options, "visual_range": visual_range, "visual_range_is_fixed": visual_range is not None,
            "additive_feasible": result.additive is not None, "bottleneck_feasible": result.bottleneck is not None, "empty_candidate_domains": candidate_summary["empty_domains"],
            "total_candidates": candidate_summary["total_candidates"], "minimum_candidates": candidate_summary["minimum_candidates"],
            "median_candidates": candidate_summary["median_candidates"], "maximum_candidates": candidate_summary["maximum_candidates"], "treewidth": decomposition_summary["width"],
            "bags": decomposition_summary["bags"], "maximum_bag_size": decomposition_summary["maximum_bag_size"], "enumerated_states": (result.dp_statistics.enumerated_states),
            "feasible_states": (result.dp_statistics.feasible_states), "message_entries": (result.dp_statistics.message_entries),
            "unique_cost_requests": (result.dp_statistics.unique_cost_requests), "load_graphs_seconds": load_seconds, "prepare_matcher_seconds": prepare_seconds,
            "solve_both_objectives_seconds": solve_seconds, "generate_figures_seconds": figure_seconds, "report": relative_report})

        for objective_name, solution in (("additive", result.additive), ("bottleneck", result.bottleneck),):
            edge_stats = (_edge_statistics(solution.edges) if solution is not None else {})
            solution_rows.append({"run_id": run_id, "pair_id": pair_id, "cost": cost_name, "objective": objective_name, "feasible": solution is not None,
                "objective_value": (float(solution.value) if solution is not None else None), "mapped_vertices": (len(solution.mapping) if solution is not None else 0),
                "materialized_witnesses": (len(solution.edges) if solution is not None else 0), "local_cost_minimum": edge_stats.get("minimum"),
                "local_cost_q25": edge_stats.get("q25"), "local_cost_median": edge_stats.get("median"), "local_cost_positive_median": edge_stats.get("positive_cost_median"),
                "local_cost_mean": edge_stats.get("mean"), "local_cost_q75": edge_stats.get("q75"), "local_cost_maximum": edge_stats.get("maximum"),
                "local_cost_standard_deviation": edge_stats.get("standard_deviation"), "zero_cost_edges": edge_stats.get("zero_cost_count", 0),
                "zero_cost_fraction": edge_stats.get("zero_cost_fraction"), "positive_cost_edges": edge_stats.get("positive_cost_count", 0),
                "figures": generated_figures.get(objective_name, {}), "report": relative_report})

        print(f"[{pair_id}] {cost_name}: "
              f"additive={'yes' if result.additive else 'no'}, "
              f"bottleneck={'yes' if result.bottleneck else 'no'}, "
              f"states={result.dp_statistics.enumerated_states:,}, "
              f"solve={solve_seconds:.3f}s")

    return run_rows, solution_rows


def _validate_config(config: Mapping[str, Any]) -> None:
    if not config.get("pairs"):
        raise ValueError("Configuration contains no graph pairs.")
    if not config.get("costs"):
        raise ValueError("Configuration contains no costs.")

    known_costs = {str(cost) for cost in available_costs()}
    requested_costs = {str(item["name"]) for item in config["costs"]}
    unknown_costs = sorted(requested_costs - known_costs)

    if unknown_costs:
        raise ValueError("Unknown costs: " + ", ".join(unknown_costs) + ". Available costs: " + ", ".join(sorted(known_costs)))

    for pair in config["pairs"]:
        for key in ("source", "target"):
            path = Path(pair[key]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"{key} graph does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run a headless River-Matcher experiment suite and "
                                                  "export JSON reports and PNG figures."))
    parser.add_argument("config", type=Path, help="Path to the experiment-suite JSON file.")
    parser.add_argument("--output", type=Path, default=None, help=("Override the output directory from the configuration."))
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution. Default: 220.")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)

    suite_name = _slug(str(config.get("suite_name", config_path.stem)))
    configured_output = Path(config.get("output_dir", "experiment_results"))
    suite_root = (arguments.output if arguments.output is not None else configured_output).expanduser().resolve() / suite_name
    suite_root.mkdir(parents=True, exist_ok=True)

    common_metadata = {"suite_name": suite_name, "config_path": str(config_path), "git_commit": _git_commit(), "python": sys.version, "platform": platform.platform(),
        "argv": sys.argv}

    defaults = dict(config.get("defaults", {}))
    all_run_rows: list[dict[str, Any]] = []
    all_solution_rows: list[dict[str, Any]] = []
    suite_started = time.perf_counter()

    for pair in config["pairs"]:
        pair_run_rows, pair_solution_rows = _run_pair(pair, costs=config["costs"], defaults=defaults, suite_root=suite_root, dpi=arguments.dpi, common_metadata=common_metadata)
        all_run_rows.extend(pair_run_rows)
        all_solution_rows.extend(pair_solution_rows)

    suite_seconds = time.perf_counter() - suite_started
    suite_summary = {"schema_version": SCHEMA_VERSION, "created_at_utc": _utc_now(), "suite_name": suite_name, "metadata": common_metadata, "defaults": defaults,
        "pair_count": len(config["pairs"]), "cost_count": len(config["costs"]), "run_count": len(all_run_rows), "solution_count": len(all_solution_rows),
        "suite_runtime_seconds": suite_seconds, "runs": all_run_rows, "solutions": all_solution_rows}
    summary_path = suite_root / "suite_summary.json"
    _write_json(summary_path, suite_summary)

    print()
    print(f"Experiment suite complete: {suite_root}")
    print(f"Summary: {summary_path}")
    print(f"Runtime: {suite_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
