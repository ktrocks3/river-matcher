from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from river_matcher.models import JunctionGraph
from river_matcher.preprocessing import load_junction_graph

type XYArray = NDArray[np.float64]
type EdgeGeometry = tuple[float, XYArray]
type FloatArray = NDArray[np.float64]
type LegacyEdge = Mapping[str, Any]


class LegacyGraphData(Protocol):
    """Structural type for river_graph_matcher_app_v6.GraphData."""
    vertices: Mapping[int, tuple[float, float]]
    nodes: set[int]
    edges: Sequence[LegacyEdge]
    edge_polylines: Mapping[int, FloatArray]


def load_legacy_app(legacy_root: Path) -> ModuleType:
    """Import the legacy v6 application so its preprocessing code remains the reference."""
    legacy_root = legacy_root.resolve()
    app_path = legacy_root / "river_graph_matcher_app_v6.py"

    if not app_path.is_file():
        raise FileNotFoundError(f"Could not find legacy application at {app_path}.")

    module_name = "_legacy_river_graph_matcher_app_v6"
    spec = importlib.util.spec_from_file_location(module_name, app_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {app_path}.")

    sys.path.insert(0, str(legacy_root))

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(legacy_root))

    return module


def clean_polyline(polyline: object) -> XYArray:
    """Normalize geometry before comparing legacy and migrated edge records."""
    points = np.asarray(polyline, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        raise AssertionError(f"Invalid comparison polyline with shape {points.shape}.")

    points = points[:, :2]

    if not np.all(np.isfinite(points)):
        raise AssertionError("Comparison polyline contains non-finite coordinates.")

    keep = [0]

    for index in range(1, len(points)):
        delta = points[index] - points[keep[-1]]

        if float(np.dot(delta, delta)) > 1e-24:
            keep.append(index)

    points = np.ascontiguousarray(points[keep], dtype=np.float64)

    if len(points) < 2:
        raise AssertionError("Comparison polyline has fewer than two distinct points.")

    return points


def canonical_polyline(polyline: object) -> XYArray:
    """Choose one orientation-independent representation of an undirected edge."""
    points = clean_polyline(polyline)
    forward = tuple(float(value) for value in points.ravel())
    reverse = tuple(float(value) for value in points[::-1].ravel())

    if reverse < forward:
        return np.ascontiguousarray(points[::-1], dtype=np.float64)

    return points


def polyline_length(polyline: XYArray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


def geometry_sort_key(record: EdgeGeometry) -> tuple:
    """Sort parallel edges independently of their legacy or migrated IDs."""
    length, polyline = record
    rounded = tuple(float(value) for value in np.round(polyline.ravel(), 12))

    return round(length, 12), len(polyline), rounded,


def legacy_edge_groups(graph: LegacyGraphData) -> dict[tuple[int, int], list[EdgeGeometry]]:
    groups: dict[tuple[int, int], list[EdgeGeometry]] = defaultdict(list)

    for edge in graph.edges:
        edge_id = int(edge["_eid"])
        u = int(edge["a"])
        v = int(edge["b"])
        endpoints = (min(u, v), max(u, v))

        polyline = graph.edge_polylines.get(edge_id)

        if polyline is None:
            raise AssertionError(f"Legacy edge e{edge_id} has no polyline.")

        canonical = canonical_polyline(polyline)
        groups[endpoints].append((polyline_length(canonical), canonical))

    for records in groups.values():
        records.sort(key=geometry_sort_key)

    return dict(groups)


def migrated_edge_groups(graph: JunctionGraph) -> dict[tuple[int, int], list[EdgeGeometry]]:
    groups: dict[tuple[int, int], list[EdgeGeometry]] = defaultdict(list)

    for edge in graph.edges:
        endpoints = (min(edge.u, edge.v), max(edge.u, edge.v))
        canonical = canonical_polyline(edge.polyline)
        groups[endpoints].append((edge.length, canonical))

    for records in groups.values():
        records.sort(key=geometry_sort_key)

    return dict(groups)


def compare_coordinates(legacy_graph: LegacyGraphData, migrated_graph: JunctionGraph, *, atol: float, ) -> None:
    legacy_nodes = {int(vertex) for vertex in legacy_graph.nodes}
    migrated_nodes = set(migrated_graph.vertices)

    if legacy_nodes != migrated_nodes:
        missing = sorted(legacy_nodes - migrated_nodes)
        extra = sorted(migrated_nodes - legacy_nodes)

        raise AssertionError(f"Junction vertex sets differ: missing from migration={missing}, added by migration={extra}.")

    for vertex in sorted(legacy_nodes):
        legacy_xy = np.asarray(legacy_graph.vertices[vertex], dtype=np.float64)
        migrated_xy = np.asarray(migrated_graph.coordinates[vertex], dtype=np.float64)

        if not np.allclose(legacy_xy, migrated_xy, rtol=0.0, atol=atol):
            raise AssertionError(f"Coordinates differ at vertex {vertex}: legacy={legacy_xy.tolist()}, migrated={migrated_xy.tolist()}.")


def compare_endpoint_multiplicities(legacy_groups: dict[tuple[int, int], list[EdgeGeometry]], migrated_groups: dict[tuple[int, int], list[EdgeGeometry]], ) -> None:
    legacy_counts = Counter({endpoints: len(records) for endpoints, records in legacy_groups.items()})
    migrated_counts = Counter({endpoints: len(records) for endpoints, records in migrated_groups.items()})

    if legacy_counts != migrated_counts:
        all_endpoints = sorted(set(legacy_counts) | set(migrated_counts))
        differences = [(endpoints, legacy_counts.get(endpoints, 0), migrated_counts.get(endpoints, 0),) for endpoints in all_endpoints if
                       legacy_counts.get(endpoints, 0) != migrated_counts.get(endpoints, 0)]

        raise AssertionError(f"Endpoint multiplicities differ (endpoints, legacy, migrated): {differences}.")


def compare_edge_geometry(legacy_groups: dict[tuple[int, int], list[EdgeGeometry]], migrated_groups: dict[tuple[int, int], list[EdgeGeometry]], *, atol: float, ) -> None:
    for endpoints in sorted(legacy_groups):
        legacy_records = legacy_groups[endpoints]
        migrated_records = migrated_groups[endpoints]

        for parallel_index, (legacy_record, migrated_record) in enumerate(zip(legacy_records, migrated_records, strict=True)):
            legacy_length, legacy_polyline = legacy_record
            migrated_length, migrated_polyline = migrated_record

            if not np.isclose(legacy_length, migrated_length, rtol=0.0, atol=atol, ):
                raise AssertionError(f"Edge length differs for {endpoints}, parallel index {parallel_index}: legacy={legacy_length:.17g}, migrated={migrated_length:.17g}.")

            if legacy_polyline.shape != migrated_polyline.shape:
                raise AssertionError(
                    f"Polyline shape differs for {endpoints}, parallel index {parallel_index}: legacy={legacy_polyline.shape}, migrated={migrated_polyline.shape}.")

            if not np.allclose(legacy_polyline, migrated_polyline, rtol=0.0, atol=atol, ):
                maximum_error = float(np.max(np.abs(legacy_polyline - migrated_polyline)))

                raise AssertionError(f"Polyline geometry differs for {endpoints}, parallel index {parallel_index}; maximum absolute error {maximum_error:.3e}.")


def compare_graph(legacy_app: ModuleType, graph_path: Path, *, atol: float, ) -> tuple[int, int, int]:
    """Compare one production graph after complete legacy and migrated preprocessing."""
    graph_path = graph_path.resolve()

    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    legacy_graph = cast(LegacyGraphData, legacy_app.preprocess_graph_file(graph_path), )
    migrated_graph = load_junction_graph(graph_path)

    if migrated_graph.name != graph_path.stem:
        raise AssertionError(f"Migrated graph name is {migrated_graph.name!r}, expected {graph_path.stem!r}.")

    compare_coordinates(legacy_graph, migrated_graph, atol=atol)

    legacy_groups = legacy_edge_groups(legacy_graph)
    migrated_groups = migrated_edge_groups(migrated_graph)

    compare_endpoint_multiplicities(legacy_groups, migrated_groups)
    compare_edge_geometry(legacy_groups, migrated_groups, atol=atol, )

    parallel_groups = sum(len(records) > 1 for records in migrated_groups.values())

    return len(migrated_graph.vertices), len(migrated_graph.edges), parallel_groups,


def main() -> int:
    parser = argparse.ArgumentParser(description=("Compare the migrated preprocessing pipeline with the exact "
                                                  "legacy v6 preprocessing implementation."))
    parser.add_argument("--legacy-root", type=Path, required=True, help="Folder containing river_graph_matcher_app_v6.py and its sibling modules.", )
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for coordinates, lengths and polyline geometry.", )
    parser.add_argument("graphs", type=Path, nargs="+", help="TopoTide graph files to compare.", )
    args = parser.parse_args()

    if args.atol < 0.0 or not np.isfinite(args.atol):
        parser.error("--atol must be finite and nonnegative.")

    legacy_app = load_legacy_app(args.legacy_root)
    failures: list[tuple[Path, str]] = []

    for graph_path in args.graphs:
        try:
            vertex_count, edge_count, parallel_groups = compare_graph(legacy_app, graph_path, atol=args.atol, )
        except Exception as exc:
            failures.append((graph_path, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {graph_path}")
            print(f"      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {graph_path.stem}:  junction vertices={vertex_count:,}, junction edges={edge_count:,}, parallel endpoint groups={parallel_groups:,}")

    if failures:
        print(f"\n{len(failures)} of {len(args.graphs)} graphs differed.")
        return 1

    print(f"\nAll {len(args.graphs)} graphs matched the legacy v6 preprocessing result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
