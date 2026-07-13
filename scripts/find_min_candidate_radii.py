from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SRC: Final[Path] = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from river_matcher.candidates import (  # noqa: E402
    CandidateMode, compute_candidate_sets, compute_vertex_candidate_sets, merge_candidate_sets, prepare_candidate_target, )
from river_matcher.decomposition import (  # noqa: E402
    SourceDecomposition, build_source_decomposition, )
from river_matcher.models import JunctionGraph  # noqa: E402
from river_matcher.preflight import MatchingPreflight, estimate_matching  # noqa: E402
from river_matcher.preprocessing import (  # noqa: E402
    load_embedded_graph, load_junction_graph, )

CANDIDATE_SEMANTICS: Final[str] = "baseline_junctions_plus_mode_additions"

CSV_FIELDS: Final[tuple[str, ...]] = ("candidate_semantics", "source", "target", "source_vertices", "source_edges", "target_junction_vertices", "target_junction_edges",
                                      "candidate_mode", "minimum_rho", "top_k", "subdivision_points", "adaptive_max_points_per_source", "adaptive_min_separation",
                                      "matching_target_vertices", "matching_target_edges", "total_candidates", "minimum_candidates", "maximum_candidates",
                                      "estimated_state_upper_bound", "largest_candidate_product", "search_seconds", "status",)


@dataclass(frozen=True, slots=True)
class GraphRecord:
    path: Path
    junction: JunctionGraph

    @property
    def vertices(self) -> int:
        return len(self.junction.vertices)

    @property
    def edges(self) -> int:
        return len(self.junction.edges)


@dataclass(frozen=True, slots=True)
class RadiusEvaluation:
    rho: float
    target: JunctionGraph
    preflight: MatchingPreflight


@dataclass(frozen=True, slots=True)
class SearchResult:
    evaluation: RadiusEvaluation | None
    elapsed_seconds: float
    error: str | None = None


class CsvStreamWriter:
    """Write and flush one result row at a time."""

    def __init__(self, path: Path) -> None:
        self._handle: TextIO = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CsvStreamWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class PreflightEvaluator:
    """Evaluate one source-target-mode configuration at arbitrary radii."""

    def __init__(self, source: JunctionGraph, junction_target: JunctionGraph, original_target: JunctionGraph | None, decomposition: SourceDecomposition, *, mode: CandidateMode,
            top_k: int, subdivision_points: int, adaptive_max_points: int, adaptive_min_separation: float, ) -> None:
        self.source = source
        self.junction_target = junction_target
        self.original_target = original_target
        self.decomposition = decomposition
        self.mode = mode
        self.top_k = top_k
        self.subdivision_points = subdivision_points
        self.adaptive_max_points = adaptive_max_points
        self.adaptive_min_separation = adaptive_min_separation
        self._cache: dict[float, RadiusEvaluation] = {}

        self._fixed_target: JunctionGraph | None = None
        if mode is not CandidateMode.ADAPTIVE_CLOSEST_POINTS:
            self._fixed_target = self._build_target(0.0)

    def _base_target(self) -> JunctionGraph:
        if self.mode is CandidateMode.ORIGINAL_TARGET_VERTICES:
            if self.original_target is None:
                raise RuntimeError("Original target graph was not loaded for ORIGINAL_TARGET_VERTICES")
            return self.original_target
        return self.junction_target

    def _build_target(self, rho: float) -> JunctionGraph:
        if self.mode is CandidateMode.TARGET_JUNCTIONS:
            return self.junction_target

        return prepare_candidate_target(self.source, self._base_target(), candidate_mode=self.mode, rho=rho, subdivision_points=self.subdivision_points,
            adaptive_max_points_per_source=self.adaptive_max_points, adaptive_min_separation=self.adaptive_min_separation, )

    def _candidate_sets(self, matching_target: JunctionGraph, *, rho: float, ) -> dict[int, list[int]]:
        baseline = compute_candidate_sets(self.source, self.junction_target, rho=rho, top_k=self.top_k, )

        if self.mode is CandidateMode.TARGET_JUNCTIONS:
            return baseline

        matching_vertices = set(matching_target.vertices)
        junction_vertices = set(self.junction_target.vertices)

        missing = sorted(junction_vertices - matching_vertices)
        if missing:
            raise ValueError("Matching target is missing baseline junction vertices: "
                             f"{missing}")

        additional_vertices = matching_vertices - junction_vertices
        additional_limit = (self.adaptive_max_points if self.mode is CandidateMode.ADAPTIVE_CLOSEST_POINTS else max(1, len(additional_vertices)))

        additions = compute_vertex_candidate_sets(self.source, matching_target, rho=rho, top_k=additional_limit, eligible_vertices=additional_vertices, )
        candidate_sets = merge_candidate_sets(baseline, additions)

        unknown = sorted({candidate for domain in candidate_sets.values() for candidate in domain} - matching_vertices)
        if unknown:
            raise ValueError("Candidate IDs are not vertices of the matching target: "
                             f"{unknown}")

        return candidate_sets

    def evaluate(self, rho: float) -> RadiusEvaluation:
        radius = float(rho)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError(f"Radius must be finite and nonnegative, got {rho!r}")

        key = round(radius, 12)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        target = self._fixed_target
        if target is None:
            target = self._build_target(radius)

        candidate_sets = self._candidate_sets(target, rho=radius)
        preflight = estimate_matching(self.decomposition, candidate_sets)

        result = RadiusEvaluation(radius, target, preflight)
        self._cache[key] = result
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Find the minimum candidate radius for every sparse-to-dense pair "
                                                  "using baseline-plus-additions candidate semantics."))
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "GraphExport", help="Directory containing TopoTide .txt exports.", )
    parser.add_argument("--output", type=Path, default=ROOT / "candidate_radius_preflight.csv", help="CSV output path. Existing files are overwritten.", )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--precision", type=float, default=0.001, help="Reported minimum-radius precision.", )
    parser.add_argument("--initial-radius", type=float, default=1.0, help="Initial upper bound for exponential search.", )
    parser.add_argument("--max-radius", type=float, default=100.0, help="Stop if no feasible radius is found at or below this value.", )
    parser.add_argument("--subdivision-points", type=int, default=2)
    parser.add_argument("--adaptive-max-points", type=int, default=8)
    parser.add_argument("--adaptive-min-separation", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not math.isfinite(args.precision) or args.precision <= 0.0:
        raise ValueError("--precision must be finite and positive")
    if not math.isfinite(args.initial_radius) or args.initial_radius <= 0.0:
        raise ValueError("--initial-radius must be finite and positive")
    if not math.isfinite(args.max_radius) or args.max_radius <= 0.0:
        raise ValueError("--max-radius must be finite and positive")
    if args.initial_radius > args.max_radius:
        raise ValueError("--initial-radius cannot exceed --max-radius")
    if args.subdivision_points < 0:
        raise ValueError("--subdivision-points must be nonnegative")
    if args.adaptive_max_points < 1:
        raise ValueError("--adaptive-max-points must be at least 1")
    if (not math.isfinite(args.adaptive_min_separation) or args.adaptive_min_separation < 0.0):
        raise ValueError("--adaptive-min-separation must be finite and nonnegative")


def load_graphs(graph_dir: Path) -> list[GraphRecord]:
    directory = graph_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Graph directory does not exist: {directory}")

    paths = sorted(directory.glob("*.txt"))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two .txt graph exports in {directory}; found {len(paths)}")

    records: list[GraphRecord] = []
    print(f"Loading {len(paths)} graph exports from {directory}")
    for index, path in enumerate(paths, start=1):
        graph = load_junction_graph(path)
        records.append(GraphRecord(path.resolve(), graph))
        print(f"  [{index:>2}/{len(paths)}] {path.name}: "
              f"{len(graph.vertices)} V / {len(graph.edges)} E")
    return records


def directed_pairs(records: list[GraphRecord], ) -> list[tuple[GraphRecord, GraphRecord]]:
    pairs: list[tuple[GraphRecord, GraphRecord]] = []

    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            if left.vertices == right.vertices:
                print("Skipping equal-size pair "
                      f"{left.path.name} / {right.path.name} "
                      f"({left.vertices} junction vertices each)")
                continue

            source, target = ((left, right) if left.vertices < right.vertices else (right, left))
            pairs.append((source, target))

    pairs.sort(key=lambda pair: (pair[0].path.stem.lower(), pair[1].path.stem.lower(),))
    return pairs


def round_up(value: float, precision: float) -> float:
    units = math.ceil((value - 1e-12) / precision)
    return max(0.0, units * precision)


def find_minimum_radius(evaluator: PreflightEvaluator, *, precision: float, initial_radius: float, max_radius: float, ) -> SearchResult:
    started = time.perf_counter()

    try:
        zero = evaluator.evaluate(0.0)
        if zero.preflight.possible:
            return SearchResult(zero, time.perf_counter() - started)

        low = 0.0
        high = min(initial_radius, max_radius)
        high_result = evaluator.evaluate(high)

        while not high_result.preflight.possible and high < max_radius:
            low = high
            high = min(max_radius, 2.0 * high)
            high_result = evaluator.evaluate(high)

        if not high_result.preflight.possible:
            return SearchResult(None, time.perf_counter() - started, error=f"no radius <= {max_radius:g} removed all empty domains", )

        while high - low > precision / 4.0:
            middle = (low + high) / 2.0
            middle_result = evaluator.evaluate(middle)
            if middle_result.preflight.possible:
                high = middle
                high_result = middle_result
            else:
                low = middle

        reported = min(round_up(high, precision), max_radius)
        final = evaluator.evaluate(reported)

        while not final.preflight.possible and reported < max_radius:
            reported = min(max_radius, reported + precision)
            final = evaluator.evaluate(reported)

        if not final.preflight.possible:
            return SearchResult(None, time.perf_counter() - started, error=("binary search found a threshold, but the rounded radius "
                                                                            "was not preflight-feasible"), )

        return SearchResult(final, time.perf_counter() - started)

    except Exception as error:
        return SearchResult(None, time.perf_counter() - started, error=f"{type(error).__name__}: {error}", )


def csv_row(source: GraphRecord, target: GraphRecord, mode: CandidateMode, result: SearchResult, *, top_k: int, subdivision_points: int, adaptive_max_points: int,
        adaptive_min_separation: float, ) -> dict[str, object]:
    base: dict[str, object] = {"candidate_semantics": CANDIDATE_SEMANTICS, "source": source.path.stem, "target": target.path.stem, "source_vertices": source.vertices,
        "source_edges": source.edges, "target_junction_vertices": target.vertices, "target_junction_edges": target.edges, "candidate_mode": mode.value, "top_k": top_k,
        "subdivision_points": subdivision_points, "adaptive_max_points_per_source": adaptive_max_points, "adaptive_min_separation": adaptive_min_separation,
        "search_seconds": f"{result.elapsed_seconds:.6f}", }

    evaluation = result.evaluation
    if evaluation is None:
        return {**base, "minimum_rho": "", "matching_target_vertices": "", "matching_target_edges": "", "total_candidates": "", "minimum_candidates": "", "maximum_candidates": "",
            "estimated_state_upper_bound": "", "largest_candidate_product": "", "status": result.error or "failed", }

    preflight = evaluation.preflight
    return {**base, "minimum_rho": f"{evaluation.rho:.12g}", "matching_target_vertices": len(evaluation.target.vertices), "matching_target_edges": len(evaluation.target.edges),
        "total_candidates": preflight.total_candidates, "minimum_candidates": preflight.minimum_candidates, "maximum_candidates": preflight.maximum_candidates,
        "estimated_state_upper_bound": preflight.estimated_state_upper_bound, "largest_candidate_product": preflight.largest_candidate_product, "status": "ok", }


def main() -> int:
    args = parse_args()
    validate_args(args)

    records = load_graphs(args.graph_dir)
    pairs = directed_pairs(records)
    if not pairs:
        raise RuntimeError("No sparse-to-dense graph pairs were found.")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    decomposition_cache: dict[Path, SourceDecomposition] = {}
    original_target_cache: dict[Path, JunctionGraph] = {}
    total_jobs = len(pairs) * len(CandidateMode)
    job = 0

    print(f"\nCandidate semantics: {CANDIDATE_SEMANTICS}")
    print(f"Scanning {len(pairs)} directed pairs × {len(CandidateMode)} modes")
    print(f"Output: {output}\n")

    with CsvStreamWriter(output) as writer:
        for source_record, target_record in pairs:
            decomposition = decomposition_cache.get(source_record.path)
            if decomposition is None:
                decomposition = build_source_decomposition(source_record.junction)
                decomposition_cache[source_record.path] = decomposition

            for mode in CandidateMode:
                job += 1

                original_target: JunctionGraph | None = None
                if mode is CandidateMode.ORIGINAL_TARGET_VERTICES:
                    original_target = original_target_cache.get(target_record.path)
                    if original_target is None:
                        original_target = load_embedded_graph(target_record.path)
                        original_target_cache[target_record.path] = original_target

                print(f"[{job:>3}/{total_jobs}] "
                      f"{source_record.path.stem} -> {target_record.path.stem} | "
                      f"{mode.display_name}", flush=True, )

                evaluator = PreflightEvaluator(source_record.junction, target_record.junction, original_target, decomposition, mode=mode, top_k=args.top_k,
                    subdivision_points=args.subdivision_points, adaptive_max_points=args.adaptive_max_points, adaptive_min_separation=args.adaptive_min_separation, )
                result = find_minimum_radius(evaluator, precision=args.precision, initial_radius=args.initial_radius, max_radius=args.max_radius, )

                if result.evaluation is None:
                    print(f"    FAILED: {result.error}")
                else:
                    preflight = result.evaluation.preflight
                    print(f"    rho={result.evaluation.rho:g}; "
                          f"candidates={preflight.total_candidates}; "
                          f"range={preflight.minimum_candidates}-"
                          f"{preflight.maximum_candidates}; "
                          "estimated states="
                          f"{preflight.estimated_state_upper_bound:,}; "
                          f"{result.elapsed_seconds:.3f}s")

                writer.write(
                    csv_row(source_record, target_record, mode, result, top_k=args.top_k, subdivision_points=args.subdivision_points, adaptive_max_points=args.adaptive_max_points,
                        adaptive_min_separation=args.adaptive_min_separation, ))

    print(f"\nDone. Wrote {total_jobs} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
