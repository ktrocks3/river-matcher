from __future__ import annotations

import math

import numpy as np
from numba import njit

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.geometry import PreparedPolyline, XYArray, prepare_polyline_segments
from river_matcher.witnesses import SourceGuidedWitnessFinder

type SampledCurve = tuple[XYArray, XYArray]

_VERTEX_PROJECTION_TOLERANCE = 1e-12
_TANGENT_NORM_TOLERANCE = 1e-12
_DOT_ALIGNMENT_TOLERANCE = 1e-12


def _sample_prepared_curve(prepared: PreparedPolyline, samples: int) -> SampledCurve | None:
    starts, vectors, squared_lengths = prepared
    lengths = np.sqrt(squared_lengths)

    if len(lengths) == 0 or not np.all(np.isfinite(lengths)) or np.any(lengths <= _TANGENT_NORM_TOLERANCE):
        return None

    total_length = float(np.sum(lengths))

    if not math.isfinite(total_length) or total_length <= _TANGENT_NORM_TOLERANCE:
        return None

    sample_count = max(2, int(samples))
    positions = np.linspace(0.0, total_length, sample_count, dtype=np.float64)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(lengths),))
    segment_indices = np.searchsorted(cumulative, positions, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)

    fractions = (positions - cumulative[segment_indices]) / lengths[segment_indices]
    fractions = np.clip(fractions, 0.0, 1.0)

    points = (starts[segment_indices] + fractions[:, None] * vectors[segment_indices])
    unit_vectors = vectors / lengths[:, None]
    tangents = unit_vectors[segment_indices].copy()

    for sample_index in range(sample_count):
        segment_index = int(segment_indices[sample_index])
        fraction = float(fractions[sample_index])
        adjacent_index = -1

        if fraction <= _VERTEX_PROJECTION_TOLERANCE and segment_index > 0:
            adjacent_index = segment_index - 1
        elif fraction >= 1.0 - _VERTEX_PROJECTION_TOLERANCE and segment_index + 1 < len(unit_vectors):
            adjacent_index = segment_index + 1

        if adjacent_index < 0:
            continue

        bisector = (unit_vectors[segment_index] + unit_vectors[adjacent_index])
        bisector_length = float(np.linalg.norm(bisector))

        # At a complete reversal either direction is equivalent because the
        # angular comparison uses the absolute tangent dot product.
        if bisector_length <= _TANGENT_NORM_TOLERANCE:
            tangents[sample_index] = unit_vectors[segment_index]
        else:
            tangents[sample_index] = (bisector / bisector_length)

    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(tangents, dtype=np.float64)


@njit(cache=True, fastmath=False)
def _directed_mean_distance_tangent(sample_points: XYArray, sample_tangents: XYArray, segment_starts: XYArray, segment_vectors: XYArray, segment_squared_lengths: np.ndarray,
                                    tangent_weight: float) -> float:
    sample_count, segment_count = sample_points.shape[0], segment_starts.shape[0]
    if sample_count == 0 or segment_count == 0:
        return math.inf
    total = 0.0

    for sample_index in range(sample_count):
        px, py = sample_points[sample_index, 0], sample_points[sample_index, 1]
        sample_tx, sample_ty = sample_tangents[sample_index, 0], sample_tangents[sample_index, 1]

        best_squared_distance, best_segment, best_projection = math.inf, -1, 0.0
        for segment_index in range(segment_count):
            sx, sy = segment_starts[segment_index, 0], segment_starts[segment_index, 1]
            vx, vy = segment_vectors[segment_index, 0], segment_vectors[segment_index, 1]
            squared_length = segment_squared_lengths[segment_index]
            projection = ((px - sx) * vx + (py - sy) * vy) / squared_length

            if projection < 0.0:
                projection = 0.0
            elif projection > 1.0:
                projection = 1.0
            dx = sx + projection * vx - px
            dy = sy + projection * vy - py
            squared_distance = dx * dx + dy * dy
            if squared_distance < best_squared_distance:
                best_squared_distance, best_segment, best_projection = squared_distance, segment_index, projection
        if best_segment < 0:
            return math.inf

        target_tx, target_ty = _target_tangent_at_projection(segment_vectors, segment_squared_lengths, best_segment, best_projection)
        tangent_dot = abs(sample_tx * target_tx + sample_ty * target_ty)
        if tangent_dot >= 1.0 - _DOT_ALIGNMENT_TOLERANCE:
            angular_error = 0.0
        else:
            if tangent_dot > 1.0:
                tangent_dot = 1.0
            angular_error = math.acos(tangent_dot) / (0.5 * math.pi)
        total += math.sqrt(best_squared_distance) + tangent_weight * angular_error

    return total / sample_count


@njit(cache=True, fastmath=False)
def _target_tangent_at_projection(segment_vectors: XYArray, segment_squared_lengths: np.ndarray, segment_index: int, projection: float) -> tuple[float, float]:
    """
    Return the local target tangent at a projected point.

    At an interior polyline vertex, the tangent is the normalized bisector
    of the incoming and outgoing segment directions.
    """
    current_length = math.sqrt(segment_squared_lengths[segment_index])
    current_x, current_y = segment_vectors[segment_index, 0] / current_length, segment_vectors[segment_index, 1] / current_length
    tangent_x, tangent_y = current_x, current_y

    if projection <= _VERTEX_PROJECTION_TOLERANCE and segment_index > 0:
        adjacent_index = segment_index - 1
        adjacent_length = math.sqrt(segment_squared_lengths[adjacent_index])
        tangent_x += segment_vectors[adjacent_index, 0] / adjacent_length
        tangent_y += segment_vectors[adjacent_index, 1] / adjacent_length
    elif projection >= 1.0 - _VERTEX_PROJECTION_TOLERANCE and segment_index + 1 < segment_vectors.shape[0]:
        adjacent_index = segment_index + 1
        adjacent_length = math.sqrt(segment_squared_lengths[adjacent_index])
        tangent_x += segment_vectors[adjacent_index, 0] / adjacent_length
        tangent_y += segment_vectors[adjacent_index, 1] / adjacent_length

    tangent_length = math.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)

    # A complete reversal has no unique bisector. Either adjacent direction is equivalent here because the metric uses an absolute tangent dot.
    if tangent_length <= _TANGENT_NORM_TOLERANCE:
        return current_x, current_y

    return tangent_x / tangent_length, tangent_y / tangent_length


class MeanDistanceTangent(BaseEdgeCost):
    name = CostName.MEAN_DISTANCE_TANGENT
    label = "symmetric mean distance and tangent error"

    def __init__(self, resources: CostResources, *, rho: float = 10.0, edge_samples: int = 12, curve_samples: int = 64, tangent_weight: float = 1.0) -> None:
        super().__init__(resources)

        resolved_curve_samples = int(curve_samples)
        resolved_tangent_weight = float(tangent_weight)

        if resolved_curve_samples < 2:
            raise ValueError("Curve sample count must be at least 2.")

        if not math.isfinite(resolved_tangent_weight) or resolved_tangent_weight < 0.0:
            raise ValueError("Tangent weight must be finite and nonnegative.")

        self.rho = float(rho)
        self.edge_samples = int(edge_samples)
        self.curve_samples = resolved_curve_samples
        self.tangent_weight = resolved_tangent_weight

        self._finder: SourceGuidedWitnessFinder = resources.guided_paths(rho=self.rho, edge_samples=self.edge_samples)

        self._source_sample_cache: dict[int, SampledCurve | None] = {}
        self._source_prepared_cache: dict[int, PreparedPolyline | None] = {}

    def _sample_curve(self, prepared: PreparedPolyline) -> SampledCurve | None:
        return _sample_prepared_curve(prepared, self.curve_samples)

    def _source_samples(self, edge_id: int) -> SampledCurve | None:
        if edge_id not in self._source_sample_cache:
            prepared = self._source_prepared(edge_id)
            self._source_sample_cache[edge_id] = (None if prepared is None else self._sample_curve(prepared))

        return self._source_sample_cache[edge_id]

    def _source_prepared(self, edge_id: int) -> PreparedPolyline | None:
        if edge_id not in self._source_prepared_cache:
            edge = self._source_edges[edge_id]
            self._source_prepared_cache[edge_id] = prepare_polyline_segments(edge.polyline)

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

        source_prepared = self._source_prepared(edge_id)
        witness_prepared = prepare_polyline_segments(witness)

        if source_prepared is None or witness_prepared is None:
            return math.inf

        source_samples = self._source_samples(edge_id)
        witness_samples = self._sample_curve(witness_prepared)

        if source_samples is None or witness_samples is None:
            return math.inf

        source_points, source_tangents = source_samples
        witness_points, witness_tangents = witness_samples

        witness_starts, witness_vectors, witness_squared = witness_prepared
        source_starts, source_vectors, source_squared = source_prepared

        source_to_witness = _directed_mean_distance_tangent(source_points, source_tangents, witness_starts, witness_vectors, witness_squared, self.tangent_weight)
        witness_to_source = _directed_mean_distance_tangent(witness_points, witness_tangents, source_starts, source_vectors, source_squared, self.tangent_weight)

        if not math.isfinite(source_to_witness) or not math.isfinite(witness_to_source):
            return math.inf

        return 0.5 * (source_to_witness + witness_to_source)

    def _compute_witness(self, request: CostRequest) -> XYArray | None:
        return self._finder.path(*request)

    def clear_cache(self) -> None:
        super().clear_cache()
        self._source_sample_cache.clear()
        self._source_prepared_cache.clear()
