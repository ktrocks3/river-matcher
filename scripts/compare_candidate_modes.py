#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SRC: Final[Path] = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from river_matcher.candidates import (CandidateMode, compute_candidate_sets, compute_vertex_candidate_sets, prepare_candidate_target)
from river_matcher.decomposition import (SourceDecomposition, build_source_decomposition)
from river_matcher.dynamic_programming import Objective
from river_matcher.matcher import RiverGraphMatcher
from river_matcher.models import JunctionGraph
from river_matcher.preprocessing import (load_embedded_graph, load_junction_graph)

THESIS_PAIRS: Final[tuple[tuple[str, str, float], ...]] = (
    ("1998e5", "2014e3", 16.0),
    ("1998e3", "1955e3", 13.0),
    ("1998e5", "1955e3", 13.0),
    ("1998e5", "1955e5", 13.0),
    ("2014e3", "1955e3", 12.0),
    ("2014e5", "1955e5", 12.0),
    ("2014e5", "1998e5", 11.0),
    ("2014e3", "1998e3", 11.0),
    ("2014e5", "1998e3", 10.0),
)

COST_NAME: Final[str] = "discrete_frechet_distance"
COST_OPTIONS: Final[dict[str, object]] = {"rho": 10.0, "edge_samples": 12, "curve_samples": 64}


@dataclass(frozen=True, slots=True)
class GraphRecord:
    path: Path
    graph: JunctionGraph


@dataclass(frozen=True, slots=True)
class PreparedProblem:
    target: JunctionGraph
    matcher: RiverGraphMatcher


@dataclass(frozen=True, slots=True)
class MappingResult:
    mode: CandidateMode
    status: str
    target: JunctionGraph | None
    mapping: dict[int, int] | None
    value: float | None
    elapsed_seconds: float
    total_candidates: int | None
    minimum_candidates: int | None
    maximum_candidates: int | None
    estimated_states: int | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "GraphExport")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--fixed-radii", type=float, nargs="+", default=[10.0, 12.5])
    parser.add_argument("--state-limit", type=int, default=10_000_000, help=("Skip optimization when the raw preflight estimate exceeds this "
                                                                             "value. Use 0 to disable the limit."))
    parser.add_argument("--subdivision-points", type=int, default=2)
    parser.add_argument("--adaptive-max-points", type=int, default=8)
    parser.add_argument("--adaptive-min-separation", type=float, default=1.0)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument("--preflight-output", type=Path, default=ROOT / "candidate_mode_fixed_preflight.csv")
    parser.add_argument("--mapping-output", type=Path, default=ROOT / "discrete_frechet_mapping_comparison.csv")
    parser.add_argument("--skip-fixed-preflight", action="store_true", help="Skip the 504-job all-pairs fixed-radius preflight stage.")
    parser.add_argument("--skip-mapping-comparison", action="store_true", help="Skip the six thesis-pair discrete-Fréchet mapping stage.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.subdivision_points < 0:
        raise ValueError("--subdivision-points must be nonnegative")
    if args.adaptive_max_points < 1:
        raise ValueError("--adaptive-max-points must be at least 1")
    if args.adaptive_min_separation < 0:
        raise ValueError("--adaptive-min-separation must be nonnegative")
    if args.coordinate_tolerance < 0:
        raise ValueError("--coordinate-tolerance must be nonnegative")
    if any(not math.isfinite(rho) or rho < 0 for rho in args.fixed_radii):
        raise ValueError("Every fixed radius must be finite and nonnegative")


def graph_path(graph_dir: Path, stem: str) -> Path:
    path = graph_dir / f"{stem}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing graph export: {path}")
    return path.resolve()


def load_junction_records(graph_dir: Path) -> dict[str, GraphRecord]:
    directory = graph_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Graph directory does not exist: {directory}")

    records: dict[str, GraphRecord] = {}
    for path in sorted(directory.glob("*.txt")):
        graph = load_junction_graph(path)
        records[path.stem] = GraphRecord(path.resolve(), graph)
        print(f"loaded {path.name}: "
              f"{len(graph.vertices)} V / {len(graph.edges)} E")

    if len(records) < 2:
        raise RuntimeError(f"Need at least two graph exports in {directory}")
    return records


def sparse_to_dense_pairs(records: dict[str, GraphRecord]) -> list[tuple[GraphRecord, GraphRecord]]:
    values = list(records.values())
    pairs: list[tuple[GraphRecord, GraphRecord]] = []

    for i, left in enumerate(values):
        for right in values[i + 1:]:
            left_size = len(left.graph.vertices)
            right_size = len(right.graph.vertices)
            if left_size == right_size:
                continue
            source, target = ((left, right) if left_size < right_size else (right, left))
            pairs.append((source, target))

    pairs.sort(key=lambda pair: (pair[0].path.stem, pair[1].path.stem))
    return pairs


def build_target(source: JunctionGraph, target_path: Path, mode: CandidateMode, *, rho: float, subdivision_points: int, adaptive_max_points: int, adaptive_min_separation: float,
                 junction_cache: dict[Path, JunctionGraph], original_cache: dict[Path, JunctionGraph]) -> JunctionGraph:
    if mode is CandidateMode.ORIGINAL_TARGET_VERTICES:
        base = original_cache.get(target_path)
        if base is None:
            base = load_embedded_graph(target_path)
            original_cache[target_path] = base
    else:
        base = junction_cache[target_path]

    return prepare_candidate_target(source, base, candidate_mode=mode, rho=rho, subdivision_points=subdivision_points, adaptive_max_points_per_source=adaptive_max_points,
                                    adaptive_min_separation=adaptive_min_separation)


def candidate_sets_for_mode(source: JunctionGraph, target: JunctionGraph, mode: CandidateMode, *, rho: float, top_k: int) -> dict[int, list[int]]:
    if mode is CandidateMode.TARGET_JUNCTIONS:
        return compute_candidate_sets(source, target, rho=rho, top_k=top_k)
    return compute_vertex_candidate_sets(source, target, rho=rho, top_k=top_k)


def prepare_problem(source: JunctionGraph, target_path: Path, decomposition: SourceDecomposition, mode: CandidateMode, *, rho: float, top_k: int, subdivision_points: int,
                    adaptive_max_points: int, adaptive_min_separation: float, junction_cache: dict[Path, JunctionGraph],
                    original_cache: dict[Path, JunctionGraph]) -> PreparedProblem:
    target = build_target(source, target_path, mode, rho=rho, subdivision_points=subdivision_points, adaptive_max_points=adaptive_max_points,
                          adaptive_min_separation=adaptive_min_separation, junction_cache=junction_cache, original_cache=original_cache)
    candidate_sets = candidate_sets_for_mode(source, target, mode, rho=rho, top_k=top_k)
    matcher = RiverGraphMatcher(source, target, candidate_sets=candidate_sets, decomposition=decomposition)
    return PreparedProblem(target, matcher)


class CsvStreamWriter:
    """Write checkpoint rows without repeatedly reopening the CSV on Windows."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._handle = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "CsvStreamWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        return self

    def write(self, row: dict[str, object]) -> None:
        if self._handle is None:
            raise RuntimeError("CSV writer is not open")
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._handle.flush()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None


def run_fixed_preflight(records: dict[str, GraphRecord], args: argparse.Namespace, decomposition_cache: dict[Path, SourceDecomposition], junction_cache: dict[Path, JunctionGraph],
                        original_cache: dict[Path, JunctionGraph]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pairs = sparse_to_dense_pairs(records)
    total = len(pairs) * len(args.fixed_radii) * len(CandidateMode)
    job = 0

    print("\n=== Fixed-radius preflight comparison ===")
    with CsvStreamWriter(args.preflight_output) as csv_output:
        for source_record, target_record in pairs:
            decomposition = decomposition_cache.get(source_record.path)
            if decomposition is None:
                decomposition = build_source_decomposition(source_record.graph)
                decomposition_cache[source_record.path] = decomposition

            for rho in args.fixed_radii:
                for mode in CandidateMode:
                    job += 1
                    print(f"[preflight {job}/{total}] "
                          f"{source_record.path.stem} -> {target_record.path.stem}, "
                          f"rho={rho:g}, {mode.display_name}")
                    started = time.perf_counter()
                    status = "ok"
                    error = ""
                    try:
                        problem = prepare_problem(source_record.graph, target_record.path, decomposition, mode, rho=rho, top_k=args.top_k,
                                                  subdivision_points=args.subdivision_points, adaptive_max_points=args.adaptive_max_points,
                                                  adaptive_min_separation=args.adaptive_min_separation, junction_cache=junction_cache, original_cache=original_cache)
                        preflight = problem.matcher.preflight
                        statistics = problem.matcher.candidate_statistics
                    except Exception as exc:
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                        problem = None
                        preflight = None
                        statistics = None

                    rows.append({"source": source_record.path.stem, "target": target_record.path.stem, "rho": rho, "candidate_mode": mode.value, "top_k": args.top_k,
                                 "subdivision_points": args.subdivision_points, "adaptive_max_points_per_source": args.adaptive_max_points,
                                 "adaptive_min_separation": args.adaptive_min_separation, "matching_target_vertices": ("" if problem is None else len(problem.target.vertices)),
                                 "matching_target_edges": ("" if problem is None else len(problem.target.edges)),
                                 "empty_domains": ("" if preflight is None else preflight.empty_domains),
                                 "total_candidates": ("" if statistics is None else statistics.total_candidates),
                                 "minimum_candidates": ("" if statistics is None else statistics.minimum_candidates),
                                 "maximum_candidates": ("" if statistics is None else statistics.maximum_candidates),
                                 "estimated_state_upper_bound": ("" if preflight is None else preflight.estimated_state_upper_bound),
                                 "largest_candidate_product": ("" if preflight is None else preflight.largest_candidate_product),
                                 "elapsed_seconds": f"{time.perf_counter() - started:.6f}", "status": status, "error": error})
                    csv_output.write(rows[-1])

    return rows


def run_one_mapping(source: GraphRecord, target: GraphRecord, decomposition: SourceDecomposition, mode: CandidateMode, *, rho: float, args: argparse.Namespace,
                    junction_cache: dict[Path, JunctionGraph], original_cache: dict[Path, JunctionGraph]) -> MappingResult:
    started = time.perf_counter()

    try:
        problem = prepare_problem(source.graph, target.path, decomposition, mode, rho=rho, top_k=args.top_k, subdivision_points=args.subdivision_points,
                                  adaptive_max_points=args.adaptive_max_points, adaptive_min_separation=args.adaptive_min_separation, junction_cache=junction_cache,
                                  original_cache=original_cache)
        matcher = problem.matcher
        preflight = matcher.preflight
        stats = matcher.candidate_statistics

        if preflight.empty_domains:
            return MappingResult(mode, "empty_domains", problem.target, None, None, time.perf_counter() - started, stats.total_candidates, stats.minimum_candidates,
                                 stats.maximum_candidates, preflight.estimated_state_upper_bound)

        if (args.state_limit > 0 and preflight.estimated_state_upper_bound > args.state_limit):
            return MappingResult(mode, "state_limit", problem.target, None, None, time.perf_counter() - started, stats.total_candidates, stats.minimum_candidates,
                                 stats.maximum_candidates, preflight.estimated_state_upper_bound)

        result = matcher.match(COST_NAME, Objective.ADDITIVE, **COST_OPTIONS)
        solution = result.solution
        if solution is None:
            return MappingResult(mode, "globally_infeasible", problem.target, None, None, time.perf_counter() - started, stats.total_candidates, stats.minimum_candidates,
                                 stats.maximum_candidates, preflight.estimated_state_upper_bound)

        return MappingResult(mode, "ok", problem.target, dict(solution.mapping), float(solution.value), time.perf_counter() - started, stats.total_candidates,
                             stats.minimum_candidates, stats.maximum_candidates, preflight.estimated_state_upper_bound)

    except Exception as exc:
        return MappingResult(mode, "error", None, None, None, time.perf_counter() - started, None, None, None, None, error=f"{type(exc).__name__}: {exc}")


def compare_mapping_coordinates(source_vertices: set[int], baseline: MappingResult, other: MappingResult, *, tolerance: float) -> tuple[int, float, float, float] | None:
    if (baseline.mapping is None or baseline.target is None or other.mapping is None or other.target is None):
        return None

    distances: list[float] = []
    changed = 0

    for source_vertex in sorted(source_vertices):
        baseline_target = baseline.mapping[source_vertex]
        other_target = other.mapping[source_vertex]
        baseline_xy = np.asarray(baseline.target.coordinates[baseline_target], dtype=np.float64)
        other_xy = np.asarray(other.target.coordinates[other_target], dtype=np.float64)
        displacement = float(np.linalg.norm(other_xy - baseline_xy))
        distances.append(displacement)
        if displacement > tolerance:
            changed += 1

    count = len(distances)
    fraction = 0.0 if count == 0 else changed / count
    mean_distance = 0.0 if count == 0 else float(np.mean(distances))
    max_distance = 0.0 if count == 0 else max(distances)
    return changed, fraction, mean_distance, max_distance


def run_thesis_mapping_comparison(records: dict[str, GraphRecord], args: argparse.Namespace, decomposition_cache: dict[Path, SourceDecomposition],
                                  junction_cache: dict[Path, JunctionGraph], original_cache: dict[Path, JunctionGraph]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    print("\n=== Thesis-pair discrete Fréchet comparison ===")
    with CsvStreamWriter(args.mapping_output) as csv_output:
        for source_stem, target_stem, rho in THESIS_PAIRS:
            if source_stem not in records or target_stem not in records:
                raise FileNotFoundError(f"Missing thesis pair {source_stem} -> {target_stem}")

            source = records[source_stem]
            target = records[target_stem]
            decomposition = decomposition_cache.get(source.path)
            if decomposition is None:
                decomposition = build_source_decomposition(source.graph)
                decomposition_cache[source.path] = decomposition

            mode_results: dict[CandidateMode, MappingResult] = {}
            for mode in CandidateMode:
                print(f"{source_stem} -> {target_stem}, rho={rho:g}, "
                      f"{mode.display_name}")
                result = run_one_mapping(source, target, decomposition, mode, rho=rho, args=args, junction_cache=junction_cache, original_cache=original_cache)
                mode_results[mode] = result
                print(f"  status={result.status}, "
                      f"value={result.value}, "
                      f"states={result.estimated_states}, "
                      f"time={result.elapsed_seconds:.3f}s")

            baseline = mode_results[CandidateMode.TARGET_JUNCTIONS]
            source_vertices = set(source.graph.vertices)

            for mode in CandidateMode:
                result = mode_results[mode]
                comparison = compare_mapping_coordinates(source_vertices, baseline, result, tolerance=args.coordinate_tolerance)
                if comparison is None:
                    changed_vertices = ""
                    changed_fraction = ""
                    mean_displacement = ""
                    max_displacement = ""
                    mapping_changed = ""
                else:
                    (changed_vertices, changed_fraction, mean_displacement, max_displacement) = comparison
                    mapping_changed = changed_vertices > 0

                rows.append({"source": source_stem, "target": target_stem, "rho": rho, "cost": COST_NAME, "aggregation": Objective.ADDITIVE.value, "candidate_mode": mode.value,
                             "status": result.status, "objective_value": ("" if result.value is None else f"{result.value:.12g}"),
                             "total_candidates": ("" if result.total_candidates is None else result.total_candidates),
                             "minimum_candidates": ("" if result.minimum_candidates is None else result.minimum_candidates),
                             "maximum_candidates": ("" if result.maximum_candidates is None else result.maximum_candidates),
                             "estimated_state_upper_bound": ("" if result.estimated_states is None else result.estimated_states),
                             "mapping_changed_vs_target_junctions": mapping_changed, "changed_source_vertices": changed_vertices, "source_vertices": len(source_vertices),
                             "changed_source_fraction": ("" if changed_fraction == "" else f"{changed_fraction:.12g}"),
                             "mean_mapped_point_displacement": ("" if mean_displacement == "" else f"{mean_displacement:.12g}"),
                             "maximum_mapped_point_displacement": ("" if max_displacement == "" else f"{max_displacement:.12g}"), "coordinate_tolerance": args.coordinate_tolerance,
                             "elapsed_seconds": f"{result.elapsed_seconds:.6f}", "error": result.error or ""})
                csv_output.write(rows[-1])

    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\n=== Mapping-change summary ===")
    for mode in CandidateMode:
        if mode is CandidateMode.TARGET_JUNCTIONS:
            continue
        mode_rows = [row for row in rows if row["candidate_mode"] == mode.value and row["mapping_changed_vs_target_junctions"] != ""]
        changed = sum(bool(row["mapping_changed_vs_target_junctions"]) for row in mode_rows)
        comparable = len(mode_rows)
        changed_vertices = sum(int(row["changed_source_vertices"]) for row in mode_rows)
        total_vertices = sum(int(row["source_vertices"]) for row in mode_rows)
        vertex_fraction = (math.nan if total_vertices == 0 else changed_vertices / total_vertices)
        print(f"{mode.display_name}: mapping changed in "
              f"{changed}/{comparable} comparable pairs; "
              f"{changed_vertices}/{total_vertices} source assignments changed "
              f"({vertex_fraction:.1%})")


def main() -> int:
    args = parse_args()
    validate_args(args)

    records = load_junction_records(args.graph_dir)
    junction_cache = {record.path: record.graph for record in records.values()}
    original_cache: dict[Path, JunctionGraph] = {}
    decomposition_cache: dict[Path, SourceDecomposition] = {}

    if args.skip_fixed_preflight:
        preflight_rows: list[dict[str, object]] = []
        print("\nSkipping fixed-radius preflight comparison.")
    else:
        preflight_rows = run_fixed_preflight(records, args, decomposition_cache, junction_cache, original_cache)

    if args.skip_mapping_comparison:
        mapping_rows: list[dict[str, object]] = []
        print("\nSkipping thesis-pair mapping comparison.")
    else:
        mapping_rows = run_thesis_mapping_comparison(records, args, decomposition_cache, junction_cache, original_cache)
        print_summary(mapping_rows)

    if not args.skip_fixed_preflight:
        print(f"\nWrote {len(preflight_rows)} preflight rows to "
              f"{args.preflight_output.resolve()}")
    if not args.skip_mapping_comparison:
        print(f"Wrote {len(mapping_rows)} mapping rows to "
              f"{args.mapping_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
