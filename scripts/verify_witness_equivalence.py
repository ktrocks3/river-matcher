from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from river_matcher.candidates import compute_candidate_sets
from river_matcher.models import JunctionGraph
from river_matcher.preprocessing import load_junction_graph
from river_matcher.witnesses import ShortestPathWitnessFinder, SourceGuidedWitnessFinder

type XYArray = NDArray[np.float64]
type CandidateSets = dict[int, list[int]]
type WitnessRequest = tuple[int, int, int, int, int]
type TargetPair = tuple[int, int]
type WitnessPathMap = dict[WitnessRequest, XYArray | None]
type LegacyEdge = Mapping[str, Any]


class LegacyGraphData(Protocol):
    name: str
    vertices: Mapping[int, tuple[float, float]]
    nodes: set[int]
    edges: Sequence[LegacyEdge]
    edge_polylines: Mapping[int, object]


class LegacyGuidedFinder(Protocol):
    adjacency_cache: Mapping[object, object]
    shortest_path_tree_cache: Mapping[object, object]
    path_cache: Mapping[object, object]
    timing_stats: Mapping[str, float]

    def paths(self, requests: Sequence[WitnessRequest]) -> Mapping[WitnessRequest, object]: ...


@dataclass(frozen=True, slots=True)
class LegacyEdgeRecord:
    id: int
    u: int
    v: int
    polyline: XYArray


@dataclass(frozen=True, slots=True)
class EdgeMatch:
    legacy_id: int
    migrated_id: int
    u: int
    v: int


@dataclass(frozen=True, slots=True)
class RequestPair:
    legacy: WitnessRequest
    migrated: WitnessRequest


@dataclass(slots=True)
class ComparisonStats:
    total: int = 0
    mismatches: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, message: str, *, limit: int) -> None:
        self.mismatches += 1
        if len(self.examples) < limit:
            self.examples.append(message)


@dataclass(frozen=True, slots=True)
class DirectionSummary:
    source_edges: int
    requests: int
    skipped_equal_pairs: int
    ordinary_pairs: int
    ordinary_seconds: tuple[float, float]
    guided_seconds: tuple[float, float]
    legacy_adjacency_builds: int
    legacy_dijkstra_runs: int
    migrated_adjacency_builds: int
    migrated_dijkstra_runs: int


def _load_legacy_app(legacy_root: Path) -> ModuleType:
    root = legacy_root.resolve()
    app_path = root / "river_graph_matcher_app_v6.py"

    if not app_path.is_file():
        raise FileNotFoundError(f"Could not find legacy application at {app_path}.")

    module_name = "_legacy_river_graph_matcher_app_v6_witness_verifier"
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(root))

    try:
        spec = importlib.util.spec_from_file_location(module_name, app_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import specification for {app_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(root))

    required = (
        "read_topotide_graph",
        "filter_valid",
        "compress_to_junction_graph",
        "clean_junction_graph",
        "edge_polyline",
        "GraphData",
        "compute_candidate_sets_numba",
        "make_shortest_path_polyline_fn",
        "make_source_guided_witness_finder",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"Legacy application is missing callables: {', '.join(missing)}.")

    return module


def _load_legacy_graph(legacy_app: ModuleType, path: Path) -> LegacyGraphData:
    """Reproduce the lightweight graph preprocessing used by v6."""
    raw_vertices, raw_edges = legacy_app.read_topotide_graph(str(path))
    vertices, edges = legacy_app.filter_valid(raw_vertices, raw_edges)
    nodes, junction_edges = legacy_app.compress_to_junction_graph(vertices, edges)
    nodes, junction_edges = legacy_app.clean_junction_graph(nodes, junction_edges, keep_largest=True)

    normalized_edges = [dict(edge) for edge in junction_edges]

    for edge_id, edge in enumerate(normalized_edges):
        edge["_eid"] = edge_id

    edge_polylines: dict[int, XYArray] = {}

    for edge in normalized_edges:
        edge_id = int(edge["_eid"])
        polyline = legacy_app.edge_polyline(edge, vertices)

        if polyline is None:
            raise AssertionError(f"Legacy edge e{edge_id} has no valid polyline.")

        edge_polylines[edge_id] = _as_xy(polyline)

    graph = legacy_app.GraphData(path, path.stem, vertices, set(nodes), normalized_edges, edge_polylines)

    return cast(LegacyGraphData, graph)


def _as_xy(polyline: object) -> XYArray:
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        raise AssertionError(f"Invalid polyline shape {points.shape}.")
    points = np.ascontiguousarray(points[:, :2], dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise AssertionError("Polyline contains non-finite coordinates.")
    return points


def _canonical_polyline(polyline: object) -> XYArray:
    points = _as_xy(polyline)
    forward = tuple(float(value) for value in points.ravel())
    reverse = tuple(float(value) for value in points[::-1].ravel())
    return np.ascontiguousarray(points[::-1] if reverse < forward else points, dtype=np.float64)


def _polyline_length(polyline: XYArray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


def _geometry_sort_key(polyline: XYArray) -> tuple[float, int, tuple[float, ...]]:
    rounded = tuple(float(value) for value in np.round(polyline.ravel(), 12))
    return round(_polyline_length(polyline), 12), len(polyline), rounded


def _legacy_edges_by_endpoints(graph: LegacyGraphData) -> dict[tuple[int, int], list[LegacyEdgeRecord]]:
    groups: dict[tuple[int, int], list[LegacyEdgeRecord]] = {}
    for edge in graph.edges:
        edge_id = int(edge["_eid"])
        u = int(edge["a"])
        v = int(edge["b"])
        polyline = _canonical_polyline(graph.edge_polylines[edge_id])
        groups.setdefault((min(u, v), max(u, v)), []).append(LegacyEdgeRecord(edge_id, u, v, polyline))
    for records in groups.values():
        records.sort(key=lambda record: (_geometry_sort_key(record.polyline), record.id))
    return groups


def _match_source_edges(legacy: LegacyGraphData, migrated: JunctionGraph, *, atol: float) -> list[EdgeMatch]:
    legacy_groups = _legacy_edges_by_endpoints(legacy)
    migrated_groups: dict[tuple[int, int], list[Any]] = {}
    for edge in migrated.edges:
        migrated_groups.setdefault((min(edge.u, edge.v), max(edge.u, edge.v)), []).append(edge)
    for records in migrated_groups.values():
        records.sort(key=lambda edge: (_geometry_sort_key(_canonical_polyline(edge.polyline)), edge.id))

    if set(legacy_groups) != set(migrated_groups):
        raise AssertionError("Legacy and migrated source endpoint groups differ.")

    matches: list[EdgeMatch] = []
    for endpoints in sorted(legacy_groups):
        legacy_records = legacy_groups[endpoints]
        migrated_records = migrated_groups[endpoints]
        if len(legacy_records) != len(migrated_records):
            raise AssertionError(f"Parallel-edge multiplicity differs at {endpoints}.")

        unused = list(legacy_records)
        for migrated_edge in migrated_records:
            migrated_polyline = _canonical_polyline(migrated_edge.polyline)
            match_index = next(
                (
                    index
                    for index, record in enumerate(unused)
                    if record.polyline.shape == migrated_polyline.shape and np.allclose(record.polyline, migrated_polyline, rtol=0.0, atol=atol)
                ),
                None,
            )
            if match_index is None:
                raise AssertionError(f"No legacy geometry matches migrated edge e{migrated_edge.id} at {endpoints}.")
            legacy_record = unused.pop(match_index)
            matches.append(EdgeMatch(legacy_record.id, migrated_edge.id, endpoints[0], endpoints[1]))

        if unused:
            raise AssertionError(f"Unmatched legacy edges remain at {endpoints}.")

    matches.sort(key=lambda match: match.migrated_id)
    return matches


def _normalize_candidate_sets(candidate_sets: Mapping[Any, Any]) -> CandidateSets:
    return {int(source): [int(target) for target in targets] for source, targets in candidate_sets.items()}


def _verify_candidate_membership(legacy: CandidateSets, migrated: CandidateSets, source_vertices: Sequence[int]) -> CandidateSets:
    output: CandidateSets = {}
    if set(legacy) != set(source_vertices) or set(migrated) != set(source_vertices):
        raise AssertionError("Candidate source keys differ from the source graph vertices.")

    for vertex in source_vertices:
        legacy_values = legacy[vertex]
        migrated_values = migrated[vertex]
        if len(legacy_values) != len(set(legacy_values)) or len(migrated_values) != len(set(migrated_values)):
            raise AssertionError(f"Candidate list for source vertex {vertex} contains duplicates.")
        if sorted(legacy_values) != sorted(migrated_values):
            raise AssertionError(f"Candidate membership differs at source vertex {vertex}: legacy={legacy_values}, migrated={migrated_values}.")
        output[vertex] = sorted(migrated_values)

    return output


def _build_request_pairs(edge_matches: Sequence[EdgeMatch], candidates: CandidateSets) -> tuple[list[RequestPair], int]:
    requests: list[RequestPair] = []
    skipped_equal = 0

    for match in edge_matches:
        for target_start in candidates[match.u]:
            for target_end in candidates[match.v]:
                if target_start == target_end:
                    skipped_equal += 1
                    continue
                requests.append(
                    RequestPair(legacy=(match.legacy_id, match.u, match.v, target_start, target_end), migrated=(match.migrated_id, match.u, match.v, target_start, target_end))
                )

    return requests, skipped_equal


def _legacy_source_polylines(graph: LegacyGraphData) -> dict[object, object]:
    lookup: dict[object, object] = {}
    for edge in graph.edges:
        edge_id = int(edge["_eid"])
        u = int(edge["a"])
        v = int(edge["b"])
        polyline = _as_xy(graph.edge_polylines[edge_id])
        lookup[edge_id] = polyline
        lookup[(u, v)] = polyline
        lookup[(v, u)] = np.ascontiguousarray(polyline[::-1], dtype=np.float64)
    return lookup


def _normalize_path_map(raw: Mapping[WitnessRequest, object], keys: Sequence[WitnessRequest]) -> WitnessPathMap:
    output: WitnessPathMap = {}
    for key in keys:
        if key not in raw:
            raise AssertionError(f"Path result omitted request {key}.")
        value = raw[key]
        output[key] = None if value is None else _as_xy(value)

    return output


def _path_issue(expected: XYArray | None, actual: XYArray | None, *, atol: float) -> str | None:
    if expected is None or actual is None:
        return None if expected is actual else f"reachability differs: expected={'None' if expected is None else 'path'}, actual={'None' if actual is None else 'path'}"
    if expected.shape != actual.shape:
        return f"shape differs: expected={expected.shape}, actual={actual.shape}, lengths={_polyline_length(expected):.17g}/{_polyline_length(actual):.17g}"
    if np.allclose(expected, actual, rtol=0.0, atol=atol):
        return None
    maximum_error = float(np.max(np.abs(expected - actual)))
    return f"coordinates differ: max_error={maximum_error:.3e}, lengths={_polyline_length(expected):.17g}/{_polyline_length(actual):.17g}"


def _compare_paths(stats: ComparisonStats, label: str, key: object, expected: XYArray | None, actual: XYArray | None, *, atol: float, limit: int) -> None:
    stats.total += 1
    issue = _path_issue(expected, actual, atol=atol)
    if issue is not None:
        stats.add(f"{label} {key}: {issue}", limit=limit)


def _timing_value(stats: Mapping[str, float], key: str) -> int:
    return int(stats.get(key, 0.0))


def _verify_cache_count(errors: list[str], label: str, actual: int, expected: int) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected:,}, got {actual:,}.")


def _run_ordinary_verification(
    legacy_app: ModuleType,
    legacy_target: LegacyGraphData,
    migrated_target: JunctionGraph,
    requests: Sequence[RequestPair],
    stats: ComparisonStats,
    *,
    atol: float,
    example_limit: int,
) -> tuple[int, tuple[float, float]]:
    pairs = sorted({(request.migrated[3], request.migrated[4]) for request in requests})
    legacy_path = legacy_app.make_shortest_path_polyline_fn(legacy_target.edges, legacy_target.vertices)
    migrated_finder = ShortestPathWitnessFinder(migrated_target)

    started = time.perf_counter()
    legacy_first = {pair: None if (value := legacy_path(*pair)) is None else _as_xy(value) for pair in pairs}
    legacy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    migrated_first = {pair: path for pair, path in migrated_finder.paths(pairs).items()}
    migrated_seconds = time.perf_counter() - started

    legacy_repeat = {pair: None if (value := legacy_path(*pair)) is None else _as_xy(value) for pair in pairs}
    migrated_repeat = {pair: path for pair, path in migrated_finder.paths(pairs).items()}
    reverse_pairs = sorted({(target_end, target_start) for target_start, target_end in pairs})
    legacy_reverse = {pair: None if (value := legacy_path(*pair)) is None else _as_xy(value) for pair in reverse_pairs}
    migrated_reverse = {pair: path for pair, path in migrated_finder.paths(reverse_pairs).items()}

    for pair in pairs:
        reverse = (pair[1], pair[0])
        _compare_paths(stats, "ordinary legacy/migrated", pair, legacy_first[pair], migrated_first[pair], atol=atol, limit=example_limit)
        _compare_paths(stats, "ordinary legacy repeat", pair, legacy_first[pair], legacy_repeat[pair], atol=0.0, limit=example_limit)
        _compare_paths(stats, "ordinary migrated repeat", pair, migrated_first[pair], migrated_repeat[pair], atol=0.0, limit=example_limit)
        legacy_forward = legacy_first[pair]
        migrated_forward = migrated_first[pair]
        legacy_expected = None if legacy_forward is None else np.ascontiguousarray(legacy_forward[::-1], dtype=np.float64)
        migrated_expected = None if migrated_forward is None else np.ascontiguousarray(migrated_forward[::-1], dtype=np.float64)
        _compare_paths(stats, "ordinary legacy reverse", reverse, legacy_expected, legacy_reverse[reverse], atol=atol, limit=example_limit)
        _compare_paths(stats, "ordinary migrated reverse", reverse, migrated_expected, migrated_reverse[reverse], atol=atol, limit=example_limit)
        _compare_paths(stats, "ordinary reverse legacy/migrated", reverse, legacy_reverse[reverse], migrated_reverse[reverse], atol=atol, limit=example_limit)

    return len(pairs), (legacy_seconds, migrated_seconds)


def _run_guided_verification(
    legacy_app: ModuleType,
    legacy_source: LegacyGraphData,
    legacy_target: LegacyGraphData,
    migrated_source: JunctionGraph,
    migrated_target: JunctionGraph,
    requests: Sequence[RequestPair],
    stats: ComparisonStats,
    cache_errors: list[str],
    *,
    rho: float,
    edge_samples: int,
    atol: float,
    example_limit: int,
) -> tuple[tuple[float, float], tuple[int, int, int, int]]:
    legacy_finder = cast(
        LegacyGuidedFinder,
        legacy_app.make_source_guided_witness_finder(_legacy_source_polylines(legacy_source), legacy_target.edges, legacy_target.vertices, rho, edge_samples=edge_samples),
    )
    migrated_finder = SourceGuidedWitnessFinder(migrated_source, migrated_target, rho=rho, edge_samples=edge_samples)
    legacy_keys = [request.legacy for request in requests]
    migrated_keys = [request.migrated for request in requests]

    started = time.perf_counter()
    legacy_first = _normalize_path_map(legacy_finder.paths(legacy_keys), legacy_keys)
    legacy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    migrated_first = migrated_finder.paths(migrated_keys)
    migrated_seconds = time.perf_counter() - started

    expected_legacy_adjacencies = len({key[0] for key in legacy_keys})
    expected_migrated_adjacencies = len({key[0] for key in migrated_keys})
    expected_legacy_trees = len({(key[0], key[3]) for key in legacy_keys})
    expected_migrated_trees = len({(key[0], key[3]) for key in migrated_keys})
    legacy_after_first = (_timing_value(legacy_finder.timing_stats, "adjacency_builds"), _timing_value(legacy_finder.timing_stats, "dijkstra_runs"))
    migrated_after_first = (migrated_finder.timing.adjacency_builds, migrated_finder.timing.dijkstra_runs)
    _verify_cache_count(cache_errors, "legacy guided adjacency builds after first batch", legacy_after_first[0], expected_legacy_adjacencies)
    _verify_cache_count(cache_errors, "legacy guided Dijkstra runs after first batch", legacy_after_first[1], expected_legacy_trees)
    _verify_cache_count(cache_errors, "migrated guided adjacency builds after first batch", migrated_after_first[0], expected_migrated_adjacencies)
    _verify_cache_count(cache_errors, "migrated guided Dijkstra runs after first batch", migrated_after_first[1], expected_migrated_trees)

    legacy_repeat = _normalize_path_map(legacy_finder.paths(legacy_keys), legacy_keys)
    migrated_repeat = migrated_finder.paths(migrated_keys)
    _verify_cache_count(cache_errors, "legacy guided adjacency builds after repeat", _timing_value(legacy_finder.timing_stats, "adjacency_builds"), legacy_after_first[0])
    _verify_cache_count(cache_errors, "legacy guided Dijkstra runs after repeat", _timing_value(legacy_finder.timing_stats, "dijkstra_runs"), legacy_after_first[1])
    _verify_cache_count(cache_errors, "migrated guided adjacency builds after repeat", migrated_finder.timing.adjacency_builds, migrated_after_first[0])
    _verify_cache_count(cache_errors, "migrated guided Dijkstra runs after repeat", migrated_finder.timing.dijkstra_runs, migrated_after_first[1])

    legacy_reverse_keys = [(key[0], key[2], key[1], key[4], key[3]) for key in legacy_keys]
    migrated_reverse_keys = [(key[0], key[2], key[1], key[4], key[3]) for key in migrated_keys]
    legacy_reverse = _normalize_path_map(legacy_finder.paths(legacy_reverse_keys), legacy_reverse_keys)
    migrated_reverse = migrated_finder.paths(migrated_reverse_keys)

    all_reachable = True
    for request, legacy_reverse_key, migrated_reverse_key in zip(requests, legacy_reverse_keys, migrated_reverse_keys, strict=True):
        legacy_path = legacy_first[request.legacy]
        migrated_path = migrated_first[request.migrated]
        if legacy_path is None or migrated_path is None:
            all_reachable = False
        _compare_paths(stats, "guided legacy/migrated", request.migrated, legacy_path, migrated_path, atol=atol, limit=example_limit)
        _compare_paths(stats, "guided legacy repeat", request.legacy, legacy_path, legacy_repeat[request.legacy], atol=0.0, limit=example_limit)
        _compare_paths(stats, "guided migrated repeat", request.migrated, migrated_path, migrated_repeat[request.migrated], atol=0.0, limit=example_limit)
        legacy_expected = None if legacy_path is None else np.ascontiguousarray(legacy_path[::-1], dtype=np.float64)
        migrated_expected = None if migrated_path is None else np.ascontiguousarray(migrated_path[::-1], dtype=np.float64)
        _compare_paths(stats, "guided legacy reverse", legacy_reverse_key, legacy_expected, legacy_reverse[legacy_reverse_key], atol=atol, limit=example_limit)
        _compare_paths(stats, "guided migrated reverse", migrated_reverse_key, migrated_expected, migrated_reverse[migrated_reverse_key], atol=atol, limit=example_limit)
        _compare_paths(
            stats,
            "guided reverse legacy/migrated",
            migrated_reverse_key,
            legacy_reverse[legacy_reverse_key],
            migrated_reverse[migrated_reverse_key],
            atol=atol,
            limit=example_limit,
        )

    if all_reachable:
        _verify_cache_count(cache_errors, "legacy guided Dijkstra runs after reverse", _timing_value(legacy_finder.timing_stats, "dijkstra_runs"), legacy_after_first[1])
        _verify_cache_count(cache_errors, "migrated guided Dijkstra runs after reverse", migrated_finder.timing.dijkstra_runs, migrated_after_first[1])

    counts = (
        _timing_value(legacy_finder.timing_stats, "adjacency_builds"),
        _timing_value(legacy_finder.timing_stats, "dijkstra_runs"),
        migrated_finder.timing.adjacency_builds,
        migrated_finder.timing.dijkstra_runs,
    )
    return (legacy_seconds, migrated_seconds), counts


def _verify_direction(
    legacy_app: ModuleType,
    legacy_source: LegacyGraphData,
    legacy_target: LegacyGraphData,
    migrated_source: JunctionGraph,
    migrated_target: JunctionGraph,
    *,
    rho: float,
    top_k: int,
    edge_samples: int,
    atol: float,
    example_limit: int,
) -> tuple[DirectionSummary, ComparisonStats, ComparisonStats, list[str]]:
    edge_matches = _match_source_edges(legacy_source, migrated_source, atol=atol)
    source_vertices = sorted(migrated_source.vertices)
    legacy_candidates = _normalize_candidate_sets(
        legacy_app.compute_candidate_sets_numba(legacy_source.vertices, legacy_source.nodes, legacy_target.vertices, legacy_target.edges, rho=rho, top_k=top_k)
    )
    migrated_candidates = compute_candidate_sets(migrated_source, migrated_target, rho=rho, top_k=top_k)
    candidates = _verify_candidate_membership(legacy_candidates, migrated_candidates, source_vertices)
    requests, skipped_equal = _build_request_pairs(edge_matches, candidates)
    ordinary_stats = ComparisonStats()
    guided_stats = ComparisonStats()
    cache_errors: list[str] = []

    ordinary_pairs, ordinary_seconds = _run_ordinary_verification(legacy_app, legacy_target, migrated_target, requests, ordinary_stats, atol=atol, example_limit=example_limit)

    guided_seconds, cache_counts = _run_guided_verification(
        legacy_app,
        legacy_source,
        legacy_target,
        migrated_source,
        migrated_target,
        requests,
        guided_stats,
        cache_errors,
        rho=rho,
        edge_samples=edge_samples,
        atol=atol,
        example_limit=example_limit,
    )
    summary = DirectionSummary(
        source_edges=len(edge_matches),
        requests=len(requests),
        skipped_equal_pairs=skipped_equal,
        ordinary_pairs=ordinary_pairs,
        ordinary_seconds=ordinary_seconds,
        guided_seconds=guided_seconds,
        legacy_adjacency_builds=cache_counts[0],
        legacy_dijkstra_runs=cache_counts[1],
        migrated_adjacency_builds=cache_counts[2],
        migrated_dijkstra_runs=cache_counts[3],
    )
    return summary, ordinary_stats, guided_stats, cache_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare migrated ordinary and source-guided witnesses with the legacy v6 implementations.")
    parser.add_argument("--legacy-root", type=Path, required=True, help="Folder containing river_graph_matcher_app_v6.py and its sibling modules.")
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--edge-samples", type=int, default=12)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--max-differences", type=int, default=5)
    parser.add_argument("graph_a", type=Path)
    parser.add_argument("graph_b", type=Path)
    args = parser.parse_args()

    if not math.isfinite(args.rho) or args.rho <= 0.0:
        parser.error("--rho must be positive and finite.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.edge_samples < 2:
        parser.error("--edge-samples must be at least 2.")
    if not math.isfinite(args.atol) or args.atol < 0.0:
        parser.error("--atol must be finite and nonnegative.")
    if args.max_differences < 1:
        parser.error("--max-differences must be at least 1.")

    graph_paths = (args.graph_a.resolve(), args.graph_b.resolve())
    for path in graph_paths:
        if not path.is_file():
            parser.error(f"Graph file does not exist: {path}")

    legacy_app = _load_legacy_app(args.legacy_root)
    legacy_graphs = {path: _load_legacy_graph(legacy_app, path) for path in graph_paths}
    migrated_graphs = {path: load_junction_graph(path) for path in graph_paths}

    failures = 0
    directions = ((graph_paths[0], graph_paths[1]), (graph_paths[1], graph_paths[0]))
    for source_path, target_path in directions:
        label = f"{source_path.stem} -> {target_path.stem}"
        try:
            summary, ordinary_stats, guided_stats, cache_errors = _verify_direction(
                legacy_app,
                legacy_graphs[source_path],
                legacy_graphs[target_path],
                migrated_graphs[source_path],
                migrated_graphs[target_path],
                rho=args.rho,
                top_k=args.top_k,
                edge_samples=args.edge_samples,
                atol=args.atol,
                example_limit=args.max_differences,
            )
        except Exception as exc:
            failures += 1
            print(f"FAIL  {label}")
            print(f"      {type(exc).__name__}: {exc}")
            continue

        status = "PASS" if guided_stats.mismatches == 0 and not cache_errors else "FAIL"
        failures += status == "FAIL"
        print(f"{status}  {label}: rho={args.rho:g}, top_k={args.top_k}, edge_samples={args.edge_samples}")
        print(
            f"      source edges={summary.source_edges:,}, guided requests={summary.requests:,}, "
            f"skipped equal-endpoint pairs={summary.skipped_equal_pairs:,}, ordinary target pairs={summary.ordinary_pairs:,}"
        )
        print(
            f"      ordinary legacy={summary.ordinary_seconds[0]:.3f} s, migrated={summary.ordinary_seconds[1]:.3f} s; "
            f"guided legacy={summary.guided_seconds[0]:.3f} s, migrated={summary.guided_seconds[1]:.3f} s"
        )
        print(
            f"      guided cache counts: legacy adjacency={summary.legacy_adjacency_builds:,}, Dijkstra={summary.legacy_dijkstra_runs:,}; "
            f"migrated adjacency={summary.migrated_adjacency_builds:,}, Dijkstra={summary.migrated_dijkstra_runs:,}"
        )
        print(f"      ordinary geometry comparisons={ordinary_stats.total:,}, legacy/migrated differences={ordinary_stats.mismatches:,}")
        print(f"      guided path comparisons={guided_stats.total:,}, mismatches={guided_stats.mismatches:,}, cache errors={len(cache_errors):,}")

        for message in ordinary_stats.examples:
            print(f"      ORDINARY LEGACY DIFFERENCE: {message}")

        for message in guided_stats.examples:
            print(f"      GUIDED PATH: {message}")
        for message in cache_errors[: args.max_differences]:
            print(f"      CACHE: {message}")

    if failures:
        print(f"\n{failures} of {len(directions)} directions failed witness equivalence.")
        return 1

    print(
        "\nBoth graph directions matched legacy source-guided witness geometry, determinism, reverse behavior and cache counts. "
        "Ordinary legacy reconstruction differences were reported separately."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
