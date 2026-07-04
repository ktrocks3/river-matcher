from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from river_matcher.geometry import (
    closest_segment_distance_and_tangent,
    point_to_prepared_polyline_distance,
    points_to_prepared_polyline_distances,
    prepare_polyline_segments,
    sample_polyline_by_arclength,
    sample_polyline_with_tangents,
)
from river_matcher.preprocessing import load_junction_graph

type FloatArray = NDArray[np.float64]
type LegacyFunction = Callable[..., Any]


@dataclass(slots=True)
class VerificationStats:
    edges: int = 0
    sampling_cases: int = 0
    query_points: int = 0
    max_sample_error: float = 0.0
    max_sample_tangent_error: float = 0.0
    max_scalar_distance_error: float = 0.0
    max_batch_distance_error: float = 0.0
    max_closest_distance_error: float = 0.0
    max_closest_tangent_error: float = 0.0


def _float_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def _load_legacy_backend(legacy_root: Path) -> ModuleType:
    """Import the exact legacy geometry backend used by the v6 application."""
    legacy_root = legacy_root.resolve()
    module_path = legacy_root / "edge_factory_ui_numba_v4.py"

    if not module_path.is_file():
        raise FileNotFoundError(f"Could not find legacy geometry backend at {module_path}.")

    module_name = "_legacy_edge_factory_ui_numba_v4"
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {module_path}.")

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

    required = (
        "_sample_polyline_points_by_arclength",
        "_sample_polyline_with_tangents",
        "_prepare_polyline_segments",
        "_point_to_prepared_polyline_distance",
        "_point_to_prepared_polyline_distances",
        "_closest_segment_distance_and_tangent",
    )

    missing = [name for name in required if not callable(getattr(module, name, None))]

    if missing:
        raise AttributeError(f"Legacy backend is missing required functions: {missing}.")

    return module


def _legacy_function(module: ModuleType, name: str) -> LegacyFunction:
    function = getattr(module, name, None)

    if not callable(function):
        raise AttributeError(f"Legacy backend does not provide callable {name!r}.")

    return function


def _require_array(value: Any, *, label: str) -> FloatArray:
    if value is None:
        raise AssertionError(f"{label} unexpectedly returned None.")

    array = _float_array(value)

    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{label} returned non-finite values.")

    return array


def _array_error(migrated: Any, legacy: Any, *, label: str, atol: float) -> float:
    migrated_array = _require_array(migrated, label=f"migrated {label}")
    legacy_array = _require_array(legacy, label=f"legacy {label}")

    if migrated_array.shape != legacy_array.shape:
        raise AssertionError(f"{label} shapes differ: migrated={migrated_array.shape}, legacy={legacy_array.shape}.")

    error = float(np.max(np.abs(migrated_array - legacy_array), initial=0.0))

    if error > atol:
        raise AssertionError(f"{label} differs; maximum absolute error {error:.17g} exceeds tolerance {atol:.17g}.")

    return error


def _scalar_error(migrated: float, legacy: float, *, label: str, atol: float) -> float:
    if not math.isfinite(migrated) or not math.isfinite(legacy):
        raise AssertionError(f"{label} returned non-finite values: migrated={migrated}, legacy={legacy}.")
    error = abs(migrated - legacy)
    if error > atol:
        raise AssertionError(f"{label} differs: migrated={migrated:.17g}, legacy={legacy:.17g}, error={error:.17g}, tolerance={atol:.17g}.")

    return error


def _query_points(polyline: FloatArray, *, samples: int, edge_length: float) -> FloatArray:
    points, tangents = sample_polyline_with_tangents(polyline, samples)
    if points is None or tangents is None:
        raise AssertionError("Could not generate deterministic production query points.")

    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offset = max(float(edge_length) * 1e-3, 1e-6)

    return np.ascontiguousarray(np.vstack((points, points + offset * normals, points - offset * normals)), dtype=np.float64)


def _compare_sampling(legacy: ModuleType, polyline: FloatArray, sample_counts: tuple[int, ...], *, graph_name: str, edge_id: int, atol: float, stats: VerificationStats) -> None:
    legacy_sample = _legacy_function(legacy, "_sample_polyline_points_by_arclength")
    legacy_sample_tangents = _legacy_function(legacy, "_sample_polyline_with_tangents")

    for samples in sample_counts:
        migrated_points = sample_polyline_by_arclength(polyline, samples)
        legacy_points = legacy_sample(polyline, samples)

        label = f"{graph_name} edge e{edge_id} arclength samples with n={samples}"
        error = _array_error(migrated_points, legacy_points, label=label, atol=atol)
        stats.max_sample_error = max(stats.max_sample_error, error)

        migrated_points, migrated_tangents = sample_polyline_with_tangents(polyline, samples)
        legacy_points, legacy_tangents = legacy_sample_tangents(polyline, samples)

        point_error = _array_error(migrated_points, legacy_points, label=f"{label} with tangents", atol=atol)
        tangent_error = _array_error(migrated_tangents, legacy_tangents, label=f"{label} tangents", atol=atol)

        stats.max_sample_error = max(stats.max_sample_error, point_error)
        stats.max_sample_tangent_error = max(stats.max_sample_tangent_error, tangent_error)
        stats.sampling_cases += 1


def _compare_distances(legacy: ModuleType, polyline: FloatArray, queries: FloatArray, *, graph_name: str, edge_id: int, atol: float, stats: VerificationStats) -> None:
    legacy_prepare = _legacy_function(legacy, "_prepare_polyline_segments")
    legacy_scalar = _legacy_function(legacy, "_point_to_prepared_polyline_distance")
    legacy_batch = _legacy_function(legacy, "_point_to_prepared_polyline_distances")
    legacy_closest = _legacy_function(legacy, "_closest_segment_distance_and_tangent")

    migrated_prepared = prepare_polyline_segments(polyline)
    legacy_prepared = legacy_prepare(polyline)

    if migrated_prepared is None or legacy_prepared is None:
        raise AssertionError(f"{graph_name} edge e{edge_id} could not be prepared.")

    for index, (migrated_part, legacy_part) in enumerate(zip(migrated_prepared, legacy_prepared, strict=True)):
        _array_error(migrated_part, legacy_part, label=f"{graph_name} edge e{edge_id} prepared segment array {index}", atol=atol)

    legacy_scalar_values = np.empty(len(queries), dtype=np.float64)

    for query_index, point in enumerate(queries):
        migrated_distance = point_to_prepared_polyline_distance(point, migrated_prepared)
        legacy_distance = float(legacy_scalar(point, legacy_prepared))
        legacy_scalar_values[query_index] = legacy_distance

        error = _scalar_error(migrated_distance, legacy_distance, label=f"{graph_name} edge e{edge_id} scalar distance query {query_index}", atol=atol)
        stats.max_scalar_distance_error = max(stats.max_scalar_distance_error, error)

        migrated_closest_distance, migrated_tangent = closest_segment_distance_and_tangent(point, polyline)
        legacy_closest_distance, legacy_tangent = legacy_closest(point, polyline)

        distance_error = _scalar_error(
            migrated_closest_distance, float(legacy_closest_distance), label=f"{graph_name} edge e{edge_id} closest distance query {query_index}", atol=atol
        )
        tangent_error = _array_error(migrated_tangent, legacy_tangent, label=f"{graph_name} edge e{edge_id} closest tangent query {query_index}", atol=atol)

        stats.max_closest_distance_error = max(stats.max_closest_distance_error, distance_error)
        stats.max_closest_tangent_error = max(stats.max_closest_tangent_error, tangent_error)

    migrated_batch = points_to_prepared_polyline_distances(queries, migrated_prepared, chunk_size=7)
    legacy_batch_values = legacy_batch(queries, legacy_prepared, chunk_size=7)

    legacy_batch_error = _array_error(migrated_batch, legacy_batch_values, label=f"{graph_name} edge e{edge_id} batched distances", atol=atol)
    scalar_batch_error = _array_error(migrated_batch, legacy_scalar_values, label=f"{graph_name} edge e{edge_id} batched versus legacy scalar distances", atol=atol)

    stats.max_batch_distance_error = max(stats.max_batch_distance_error, legacy_batch_error, scalar_batch_error)
    stats.query_points += len(queries)


def _verify_graph(legacy: ModuleType, graph_path: Path, *, sample_counts: tuple[int, ...], query_samples: int, atol: float) -> VerificationStats:
    graph = load_junction_graph(graph_path)
    stats = VerificationStats()

    for edge in graph.edges:
        polyline = np.ascontiguousarray(edge.polyline, dtype=np.float64)

        _compare_sampling(legacy, polyline, sample_counts, graph_name=graph.name, edge_id=edge.id, atol=atol, stats=stats)

        queries = _query_points(polyline, samples=query_samples, edge_length=edge.length)
        _compare_distances(legacy, polyline, queries, graph_name=graph.name, edge_id=edge.id, atol=atol, stats=stats)
        stats.edges += 1

    return stats


def _format_stats(graph_name: str, stats: VerificationStats) -> str:
    return (
        f"PASS  {graph_name}: edges={stats.edges:,}, sampling cases={stats.sampling_cases:,}, distance queries={stats.query_points:,}\n"
        f"      max sample error={stats.max_sample_error:.3e}, max sampled-tangent error={stats.max_sample_tangent_error:.3e}\n"
        f"      max scalar-distance error={stats.max_scalar_distance_error:.3e}, max batch-distance error={stats.max_batch_distance_error:.3e}\n"
        f"      max closest-distance error={stats.max_closest_distance_error:.3e}, max closest-tangent error={stats.max_closest_tangent_error:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare migrated geometry primitives against the legacy Numba-v4 backend on production junction-edge polylines.")
    parser.add_argument("--legacy-root", type=Path, required=True, help="Folder containing edge_factory_ui_numba_v4.py and its sibling legacy modules.")
    parser.add_argument("--samples", type=int, nargs="+", default=[2, 3, 5, 12, 32], help="Arclength sample counts checked for every production edge.")
    parser.add_argument("--query-samples", type=int, default=17, help="Base arclength points used to construct on-edge and normal-offset distance queries.")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute comparison tolerance; relative tolerance is zero.")
    parser.add_argument("graphs", type=Path, nargs="+", help="TopoTide graph exports to verify.")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0.0:
        parser.error("--atol must be finite and nonnegative.")

    if args.query_samples < 2:
        parser.error("--query-samples must be at least 2.")
    sample_counts = tuple(args.samples)

    if not sample_counts or any(samples < 2 for samples in sample_counts):
        parser.error("Every --samples value must be at least 2.")
    legacy = _load_legacy_backend(args.legacy_root)
    failures: list[tuple[Path, str]] = []

    for graph_path in args.graphs:
        try:
            stats = _verify_graph(legacy, graph_path.resolve(), sample_counts=sample_counts, query_samples=args.query_samples, atol=args.atol)
        except Exception as exc:
            failures.append((graph_path, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {graph_path}")
            print(f"      {type(exc).__name__}: {exc}")
        else:
            print(_format_stats(graph_path.stem, stats))
    if failures:
        print(f"\n{len(failures)} of {len(args.graphs)} production graphs differed.")
        return 1

    print(f"\nAll {len(args.graphs)} production graphs matched the legacy geometry backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
