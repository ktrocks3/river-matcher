from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from river_matcher.candidates import compute_candidate_sets
from river_matcher.models import JunctionGraph
from river_matcher.preprocessing import load_junction_graph

type CandidateSets = dict[int, list[int]]
type CandidateDifference = tuple[int, tuple[int, ...], tuple[int, ...]]
_DEFAULT_CASES = ((10.0, 25), (5.0, 10), (20.0, 50))


@dataclass(frozen=True, slots=True)
class CrossImplementationComparison:
    ordered_mismatches: int
    membership_mismatches: int
    first_order_difference: CandidateDifference | None
    first_membership_difference: CandidateDifference | None


@dataclass(frozen=True, slots=True)
class LegacyGraph:
    name: str
    vertices: dict[int, tuple[float, float]]
    nodes: set[int]
    edges: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    vertices: int
    empty: int
    total: int
    minimum: int
    median: float
    maximum: int
    digest: str


def _compare_cross_implementation_candidates(
    expected_label: str, expected: CandidateSets, actual_label: str, actual: CandidateSets, source_ids: list[int]
) -> CrossImplementationComparison:
    _assert_complete_candidate_sets(expected_label, expected, source_ids)
    _assert_complete_candidate_sets(actual_label, actual, source_ids)

    ordered_mismatches = 0
    membership_mismatches = 0
    first_order_difference: CandidateDifference | None = None
    first_membership_difference: CandidateDifference | None = None

    for source_vertex in source_ids:
        expected_candidates = expected[source_vertex]
        actual_candidates = actual[source_vertex]

        if expected_candidates == actual_candidates:
            continue

        ordered_mismatches += 1
        difference = (source_vertex, tuple(expected_candidates), tuple(actual_candidates))

        if first_order_difference is None:
            first_order_difference = difference

        # Counter also detects unexpected duplicate candidates.
        if Counter(expected_candidates) != Counter(actual_candidates):
            membership_mismatches += 1

            if first_membership_difference is None:
                first_membership_difference = difference

    return CrossImplementationComparison(
        ordered_mismatches=ordered_mismatches,
        membership_mismatches=membership_mismatches,
        first_order_difference=first_order_difference,
        first_membership_difference=first_membership_difference,
    )


def _load_legacy_modules(legacy_root: Path) -> tuple[ModuleType, ModuleType]:
    """Import the exact legacy candidate implementations from one folder."""
    root = legacy_root.resolve()

    required = (root / "common_functions.py", root / "edge_factory_ui_numba_v4.py")
    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError("Missing legacy module files: " + ", ".join(missing))

    for module_name in ("edge_factory_ui_numba_v4", "common_functions", "edge_factory"):
        sys.modules.pop(module_name, None)

    sys.path.insert(0, str(root))
    importlib.invalidate_caches()

    common = importlib.import_module("common_functions")
    backend = importlib.import_module("edge_factory_ui_numba_v4")

    for name, module in (("common_functions", common), ("edge_factory_ui_numba_v4", backend)):
        module_file = getattr(module, "__file__", None)

        if module_file is None:
            raise ImportError(f"Imported legacy module {name!r} has no file path.")

        imported_path = Path(module_file).resolve()

        if imported_path.parent != root:
            raise ImportError(f"Imported {name!r} from {imported_path}, not from requested legacy root {root}.")

    required_functions = (
        (common, "read_topotide_graph"),
        (common, "filter_valid"),
        (common, "compress_to_junction_graph"),
        (common, "compute_S_vertices"),
        (backend, "compute_candidate_sets_numba"),
    )

    for module, function_name in required_functions:
        if not callable(getattr(module, function_name, None)):
            raise AttributeError(f"{module.__name__} does not provide callable {function_name!r}.")

    return common, backend


def _clean_legacy_junction_graph(nodes: set[int], edges: list[dict[str, Any]]) -> tuple[set[int], list[dict[str, Any]]]:
    """
    Apply the graph cleanup used by the v6 application.

    Self-loops are removed before retaining the largest connected component.
    """
    cleaned_edges = [edge for edge in edges if int(edge["a"]) != int(edge["b"])]

    adjacency: dict[int, set[int]] = defaultdict(set)

    for edge in cleaned_edges:
        u = int(edge["a"])
        v = int(edge["b"])
        adjacency[u].add(v)
        adjacency[v].add(u)

    component: dict[int, int] = {}
    component_id = 0

    for start in nodes:
        if start in component:
            continue

        component_id += 1
        queue = deque([start])
        component[start] = component_id

        while queue:
            current = queue.popleft()

            for neighbour in adjacency.get(current, ()):
                if neighbour in component:
                    continue

                component[neighbour] = component_id
                queue.append(neighbour)

    sizes = Counter(component.values())

    if not sizes:
        return set(), []

    largest_id = max(sizes.items(), key=lambda item: item[1])[0]
    retained_nodes = {vertex for vertex in nodes if component.get(vertex) == largest_id}
    retained_edges = [edge for edge in cleaned_edges if int(edge["a"]) in retained_nodes and int(edge["b"]) in retained_nodes]

    return retained_nodes, retained_edges


def _load_legacy_graph(common: ModuleType, path: Path) -> LegacyGraph:
    """Load one graph through the legacy v6 preprocessing path."""
    raw_vertices, raw_edges = common.read_topotide_graph(str(path))
    vertices, edges = common.filter_valid(raw_vertices, raw_edges)
    nodes, compressed_edges = common.compress_to_junction_graph(vertices, edges)

    normalized_nodes = {int(vertex) for vertex in nodes}
    normalized_edges = [dict(edge) for edge in compressed_edges]
    normalized_nodes, normalized_edges = _clean_legacy_junction_graph(normalized_nodes, normalized_edges)

    normalized_vertices = {int(vertex): (float(coordinates[0]), float(coordinates[1])) for vertex, coordinates in vertices.items()}

    return LegacyGraph(name=path.stem, vertices=normalized_vertices, nodes=normalized_nodes, edges=normalized_edges)


def _endpoint_multiplicities_new(graph: JunctionGraph) -> Counter[tuple[int, int]]:
    return Counter((min(edge.u, edge.v), max(edge.u, edge.v)) for edge in graph.edges)


def _endpoint_multiplicities_legacy(graph: LegacyGraph) -> Counter[tuple[int, int]]:
    return Counter((min(int(edge["a"]), int(edge["b"])), max(int(edge["a"]), int(edge["b"]))) for edge in graph.edges)


def _assert_graph_alignment(migrated: JunctionGraph, legacy: LegacyGraph) -> None:
    """
    Check the preprocessing assumptions required by candidate comparison.

    Full preprocessing geometry equivalence remains the responsibility of the
    dedicated preprocessing verification script.
    """
    migrated_nodes = {int(vertex) for vertex in migrated.vertices}

    if migrated_nodes != legacy.nodes:
        missing = sorted(legacy.nodes - migrated_nodes)
        extra = sorted(migrated_nodes - legacy.nodes)

        raise AssertionError(f"{migrated.name} junction-node sets differ; missing={missing[:10]}, extra={extra[:10]}.")

    migrated_multiplicity = _endpoint_multiplicities_new(migrated)
    legacy_multiplicity = _endpoint_multiplicities_legacy(legacy)

    if migrated_multiplicity != legacy_multiplicity:
        raise AssertionError(f"{migrated.name} endpoint multiplicities differ between migrated and legacy preprocessing.")


def _normalize_candidate_sets(candidate_sets: Mapping[Any, Any]) -> CandidateSets:
    return {int(source): [int(target) for target in targets] for source, targets in candidate_sets.items()}


def _assert_complete_candidate_sets(label: str, candidate_sets: CandidateSets, source_ids: list[int]) -> None:
    actual_order = list(candidate_sets)

    if actual_order != source_ids:
        missing = sorted(set(source_ids) - set(candidate_sets))
        extra = sorted(set(candidate_sets) - set(source_ids))

        raise AssertionError(f"{label} source keys or source-key order differ; missing={missing[:10]}, extra={extra[:10]}, first actual keys={actual_order[:10]}.")


def _first_list_difference(expected: list[int], actual: list[int]) -> int | None:
    common_length = min(len(expected), len(actual))

    for index in range(common_length):
        if expected[index] != actual[index]:
            return index

    if len(expected) != len(actual):
        return common_length

    return None


def _assert_candidate_sets_equal(expected_label: str, expected: CandidateSets, actual_label: str, actual: CandidateSets, source_ids: list[int]) -> None:
    _assert_complete_candidate_sets(expected_label, expected, source_ids)
    _assert_complete_candidate_sets(actual_label, actual, source_ids)

    for source_vertex in source_ids:
        expected_candidates = expected[source_vertex]
        actual_candidates = actual[source_vertex]

        if expected_candidates == actual_candidates:
            continue

        difference = _first_list_difference(expected_candidates, actual_candidates)

        raise AssertionError(
            f"Candidate sets differ at source vertex {source_vertex}, list index {difference}: {expected_label}={expected_candidates}, {actual_label}={actual_candidates}."
        )


def _summarize_candidate_sets(candidate_sets: CandidateSets, source_ids: list[int]) -> CandidateSummary:
    sizes = np.asarray([len(candidate_sets[vertex]) for vertex in source_ids], dtype=np.int64)
    serialized = json.dumps([[vertex, candidate_sets[vertex]] for vertex in source_ids], separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:16]

    return CandidateSummary(
        vertices=len(source_ids),
        empty=int(np.sum(sizes == 0)),
        total=int(np.sum(sizes)),
        minimum=int(np.min(sizes)),
        median=float(np.median(sizes)),
        maximum=int(np.max(sizes)),
        digest=digest,
    )


def _run_legacy_reference(common: ModuleType, source: LegacyGraph, target: LegacyGraph, *, rho: float, top_k: int) -> tuple[CandidateSets, float]:
    started = time.perf_counter()
    result = common.compute_S_vertices(source.vertices, source.nodes, target.vertices, target.edges, rho=rho, top_k=top_k)
    seconds = time.perf_counter() - started

    return _normalize_candidate_sets(result), seconds


def _run_legacy_numba(backend: ModuleType, source: LegacyGraph, target: LegacyGraph, *, rho: float, top_k: int) -> tuple[CandidateSets, float]:
    started = time.perf_counter()
    result = backend.compute_candidate_sets_numba(source.vertices, source.nodes, target.vertices, target.edges, rho=rho, top_k=top_k)
    seconds = time.perf_counter() - started

    return _normalize_candidate_sets(result), seconds


def _run_migrated(source: JunctionGraph, target: JunctionGraph, *, rho: float, top_k: int) -> tuple[CandidateSets, float]:
    started = time.perf_counter()
    result = compute_candidate_sets(source, target, rho=rho, top_k=top_k)
    seconds = time.perf_counter() - started

    return result, seconds


def _verify_case(
    common: ModuleType,
    backend: ModuleType,
    migrated_source: JunctionGraph,
    migrated_target: JunctionGraph,
    legacy_source: LegacyGraph,
    legacy_target: LegacyGraph,
    *,
    rho: float,
    top_k: int,
) -> CandidateSummary:
    source_ids = sorted(legacy_source.nodes)

    legacy_reference, reference_seconds = _run_legacy_reference(common, legacy_source, legacy_target, rho=rho, top_k=top_k)
    legacy_first, legacy_first_seconds = _run_legacy_numba(backend, legacy_source, legacy_target, rho=rho, top_k=top_k)
    legacy_warm, legacy_warm_seconds = _run_legacy_numba(backend, legacy_source, legacy_target, rho=rho, top_k=top_k)
    migrated_first, migrated_first_seconds = _run_migrated(migrated_source, migrated_target, rho=rho, top_k=top_k)
    migrated_warm, migrated_warm_seconds = _run_migrated(migrated_source, migrated_target, rho=rho, top_k=top_k)

    _assert_candidate_sets_equal("legacy reference", legacy_reference, "legacy Numba", legacy_first, source_ids)
    _assert_candidate_sets_equal("legacy Numba first run", legacy_first, "legacy Numba repeated run", legacy_warm, source_ids)
    cross_comparison = _compare_cross_implementation_candidates("legacy reference", legacy_reference, "migrated Numba", migrated_first, source_ids)

    if cross_comparison.membership_mismatches:
        difference = cross_comparison.first_membership_difference
        assert difference is not None
        source_vertex, legacy_candidates, migrated_candidates = difference
        raise AssertionError(
            f"Candidate membership differs after top-k truncation at source vertex {source_vertex}: legacy={list(legacy_candidates)}, "
            f"migrated={list(migrated_candidates)}. Total membership mismatches={cross_comparison.membership_mismatches:,}; "
            f"total ordered mismatches={cross_comparison.ordered_mismatches:,}."
        )
    _assert_candidate_sets_equal("migrated first run", migrated_first, "migrated repeated run", migrated_warm, source_ids)

    summary = _summarize_candidate_sets(migrated_first, source_ids)

    print(f"PASS  {legacy_source.name} -> {legacy_target.name}: rho={rho:g}, top_k={top_k}")
    print(
        f"      vertices={summary.vertices:,}, empty={summary.empty:,}, total={summary.total:,}, min={summary.minimum}, median={summary.median:.1f}, max={summary.maximum}, "
        f"sha256={summary.digest}"
    )
    print(f"      legacy reference={reference_seconds:.3f} s, legacy Numba first={legacy_first_seconds:.3f} s, warm={legacy_warm_seconds:.3f} s")
    print(f"      migrated Numba first= {migrated_first_seconds:.3f} s, warm={migrated_warm_seconds:.3f} s")
    print(f"      cross-implementation order differences={cross_comparison.ordered_mismatches:,}, membership differences={cross_comparison.membership_mismatches:,}")
    if cross_comparison.first_order_difference is not None:
        source_vertex, legacy_candidates, migrated_candidates = cross_comparison.first_order_difference
        print(f"      first order difference at source {source_vertex}: legacy={list(legacy_candidates)}, migrated={list(migrated_candidates)}")
    return summary


def _parse_cases(raw_cases: list[list[str]] | None, parser: argparse.ArgumentParser) -> tuple[tuple[float, int], ...]:
    if raw_cases is None:
        return _DEFAULT_CASES

    cases: list[tuple[float, int]] = []

    for raw_rho, raw_top_k in raw_cases:
        try:
            rho = float(raw_rho)
            top_k = int(raw_top_k)
        except ValueError:
            parser.error(f"Invalid --case values: {raw_rho!r} {raw_top_k!r}.")

        if not math.isfinite(rho) or rho < 0.0:
            parser.error(f"Case radius must be finite and nonnegative, got {raw_rho!r}.")

        if top_k < 1:
            parser.error(f"Case top_k must be at least 1, got {raw_top_k!r}.")

        cases.append((rho, top_k))

    return tuple(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare migrated candidate generation against both legacy candidate implementations in both graph directions.")
    parser.add_argument("--legacy-root", required=True, type=Path, help="Directory containing common_functions.py, edge_factory.py and edge_factory_ui_numba_v4.py.")
    parser.add_argument("--case", action="append", nargs=2, metavar=("RHO", "TOP_K"), help="Candidate parameters to verify. May be repeated. Defaults to 10/25, 5/10 and 20/50.")
    parser.add_argument("graph_a", type=Path, help="First TopoTide graph export.")
    parser.add_argument("graph_b", type=Path, help="Second TopoTide graph export.")
    args = parser.parse_args()

    cases = _parse_cases(args.case, parser)
    graph_paths = (args.graph_a.resolve(), args.graph_b.resolve())

    for path in graph_paths:
        if not path.is_file():
            parser.error(f"Graph file does not exist: {path}")

    common, backend = _load_legacy_modules(args.legacy_root)

    migrated_graphs = {path: load_junction_graph(path) for path in graph_paths}
    legacy_graphs = {path: _load_legacy_graph(common, path) for path in graph_paths}

    for path in graph_paths:
        _assert_graph_alignment(migrated_graphs[path], legacy_graphs[path])

        print(f"GRAPH {path.stem}: junction vertices={len(legacy_graphs[path].nodes):,}, junction edges= {len(legacy_graphs[path].edges):,}")

    failures: list[str] = []
    directions = ((graph_paths[0], graph_paths[1]), (graph_paths[1], graph_paths[0]))

    for source_path, target_path in directions:
        for rho, top_k in cases:
            try:
                _verify_case(
                    common, backend, migrated_graphs[source_path], migrated_graphs[target_path], legacy_graphs[source_path], legacy_graphs[target_path], rho=rho, top_k=top_k
                )
            except Exception as exc:
                label = f"{source_path.stem} -> {target_path.stem}, rho={rho:g}, top_k={top_k}"
                failures.append(label)

                print(f"FAIL  {label}")
                print(f"      {type(exc).__name__}: {exc}")

    total_cases = len(directions) * len(cases)

    if failures:
        print(f"\n{len(failures)} of {total_cases} candidate-equivalence cases failed.")
        return 1

    print(
        f"\nAll {total_cases} cases matched legacy candidate membership. Legacy-reference and legacy-Numba ordering also matched exactly; "
        f"migrated ordering differences were permitted and reported."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
