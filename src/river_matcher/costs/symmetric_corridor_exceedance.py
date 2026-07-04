from __future__ import annotations

import math

from numba import njit

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources, FloatArray
from river_matcher.geometry import PreparedPolyline, XYArray, prepare_polyline_segments, sample_polyline_by_arclength
from river_matcher.witnesses import SourceGuidedWitnessFinder


@njit(cache=True, fastmath=False)
def _directed_mean_corridor_exceedance(sample_points: XYArray, segment_starts: XYArray, segment_vectors: XYArray, segment_squared_lengths: FloatArray,
                                       corridor_radius: float) -> float:
    sample_count = sample_points.shape[0]
    segment_count = segment_starts.shape[0]
    if sample_count == 0 or segment_count == 0:
        return math.inf

    total = 0.0
    for sample_index in range(sample_count):
        px, py = sample_points[sample_index, 0], sample_points[sample_index, 1]
        best_squared_distance = math.inf

        for segment_index in range(segment_count):
            sx, sy = segment_starts[segment_index, 0], segment_starts[segment_index, 1]
            vx, vy = segment_vectors[segment_index, 0], segment_vectors[segment_index, 1]
            squared_length = segment_squared_lengths[segment_index]
            projection = max(0, min(((px - sx) * vx + (py - sy) * vy) / squared_length, 1))

            dx = sx + projection * vx - px
            dy = sy + projection * vy - py
            squared_distance = dx * dx + dy * dy

            if squared_distance < best_squared_distance:
                best_squared_distance = squared_distance

        distance = math.sqrt(best_squared_distance)

        if distance > corridor_radius:
            total += distance / corridor_radius - 1.0

    return total / sample_count


class SymmetricCorridorExceedance(BaseEdgeCost):
    name = CostName.SYMMETRIC_CORRIDOR_EXCEEDANCE
    label = "symmetric mean corridor exceedance"

    def __init__(self, resources: CostResources, *, rho: float = 10.0, edge_samples: int = 12, curve_samples: int = 64, corridor_radius: float | None = None) -> None:
        super().__init__(resources)

        resolved_curve_samples = int(curve_samples)
        resolved_corridor_radius = float(rho if corridor_radius is None else corridor_radius)

        if resolved_curve_samples < 2:
            raise ValueError("Curve sample count must be at least 2.")

        if not math.isfinite(resolved_corridor_radius) or resolved_corridor_radius <= 0.0:
            raise ValueError("Corridor radius must be positive and finite.")

        self.rho = float(rho)
        self.edge_samples = int(edge_samples)
        self.curve_samples = resolved_curve_samples
        self.corridor_radius = resolved_corridor_radius
        self._finder: SourceGuidedWitnessFinder = (resources.guided_paths(rho=self.rho, edge_samples=self.edge_samples))
        self._source_sample_cache: dict[int, XYArray | None] = {}
        self._source_prepared_cache: dict[int, PreparedPolyline | None] = {}

    def _sample_curve(self, polyline: XYArray) -> XYArray | None:
        return sample_polyline_by_arclength(polyline, self.curve_samples)

    def _source_samples(self, edge_id: int) -> XYArray | None:
        if edge_id not in self._source_sample_cache:
            edge = self._source_edges[edge_id]
            self._source_sample_cache[edge_id] = (self._sample_curve(edge.polyline))

        return self._source_sample_cache[edge_id]

    def _source_prepared(self, edge_id: int) -> PreparedPolyline | None:
        if edge_id not in self._source_prepared_cache:
            edge = self._source_edges[edge_id]
            self._source_prepared_cache[edge_id] = (prepare_polyline_segments(edge.polyline))
        return self._source_prepared_cache[edge_id]

    def _compute(self, request: CostRequest) -> float:
        edge = self._source_edge(request)
        if edge is None or not self._valid_target_pair(request):
            return math.inf
        edge_id, source_u, source_v, target_u, target_v = request
        witness = self._finder.path(edge_id, source_u, source_v, target_u, target_v)
        self._remember_witness(request, witness)
        if witness is None:
            return math.inf
        source_samples = self._source_samples(edge_id)
        source_prepared = self._source_prepared(edge_id)
        witness_samples = self._sample_curve(witness)
        witness_prepared = prepare_polyline_segments(witness)
        if source_samples is None or source_prepared is None or witness_samples is None or witness_prepared is None:
            return math.inf
        source_starts, source_vectors, source_squared = source_prepared
        witness_starts, witness_vectors, witness_squared = witness_prepared
        source_to_witness = _directed_mean_corridor_exceedance(source_samples, witness_starts, witness_vectors, witness_squared, self.corridor_radius)
        witness_to_source = _directed_mean_corridor_exceedance(witness_samples, source_starts, source_vectors, source_squared, self.corridor_radius)
        if not math.isfinite(source_to_witness) or not math.isfinite(witness_to_source):
            return math.inf
        return 0.5 * (source_to_witness + witness_to_source)

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self._finder.path(*request)

    def clear_cache(self) -> None:
        super().clear_cache()
        self._source_sample_cache.clear()
        self._source_prepared_cache.clear()
