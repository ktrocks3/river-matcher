from __future__ import annotations

import math

import numpy as np
from numba import njit

from river_matcher.costs.base import BaseEdgeCost, CostName, CostRequest, CostResources
from river_matcher.geometry import PreparedPolyline, XYArray, prepare_polyline_segments, sample_polyline_by_arclength
from river_matcher.witnesses import SourceGuidedWitnessFinder

type SampledCurve = tuple[XYArray, XYArray]

_TANGENT_TOLERANCE = 1e-12
_VERTEX_PROJECTION_TOLERANCE = 1e-12
_TANGENT_NORM_TOLERANCE = 1e-12
_DOT_ALIGNMENT_TOLERANCE = 1e-12


def _sampled_unit_tangents(points: XYArray) -> XYArray | None:
    """Estimate orientation-consistent tangents from equally spaced samples.
    Interior tangents bisect the adjacent sampled-segment directions. This avoids choosing an arbitrary incoming or outgoing segment at a vertex."""
    if len(points) < 2:
        return None

    intervals = np.diff(points, axis=0)
    interval_lengths = np.linalg.norm(intervals, axis=1)
    if not np.all(np.isfinite(interval_lengths)) or np.any(interval_lengths <= _TANGENT_TOLERANCE):
        return None

    unit_intervals = intervals / interval_lengths[:, None]
    tangents = np.empty_like(points)
    tangents[0] = unit_intervals[0]
    tangents[-1] = unit_intervals[-1]

    if len(points) > 2:
        interior = unit_intervals[:-1] + unit_intervals[1:]
        interior_lengths = np.linalg.norm(interior, axis=1)
        regular = interior_lengths > _TANGENT_TOLERANCE
        interior[regular] /= interior_lengths[regular, None]

        # At a complete reversal the bisector is undefined. Either adjacent direction is equivalent because the metric uses an absolute dot.
        interior[~regular] = unit_intervals[1:][~regular]
        tangents[1:-1] = interior
    return np.ascontiguousarray(tangents, dtype=np.float64)


@njit(cache=True, fastmath=False)
def _directed_mean_distance_tangent(
    sample_points: XYArray, sample_tangents: XYArray, segment_starts: XYArray, segment_vectors: XYArray, segment_squared_lengths: np.ndarray, tangent_weight: float,
) -> float:
    """
    Compare sampled points and tangents against a target polyline.

    Distances use exact point-to-segment projections. At an interior target
    vertex, the tangent uses the bisector of the adjacent segment directions.
    """
    sample_count = sample_points.shape[0]
    segment_count = segment_starts.shape[0]

    if sample_count == 0 or segment_count == 0:
        return math.inf

    _count = sample_points.shape[0]
    segment_count = segment_starts.shape[0]

    if sample_count == 0 or segment_count == 0:
        return math.inf

    total = 0.0

    for sample_index in range(sample_count):
        px = sample_points[sample_index, 0]
        py = sample_points[sample_index, 1]
        sample_tx = sample_tangents[sample_index, 0]
        sample_ty = sample_tangents[sample_index, 1]

        best_squared_distance = math.inf
        best_segment = -1
        best_projection = 0.0

        for segment_index in range(segment_count):
            sx = segment_starts[segment_index, 0]
            sy = segment_starts[segment_index, 1]
            vx = segment_vectors[segment_index, 0]
            vy = segment_vectors[segment_index, 1]
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
                best_squared_distance = squared_distance
                best_segment = segment_index
                best_projection = projection
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
    current_x = segment_vectors[segment_index, 0] / current_length
    current_y = segment_vectors[segment_index, 1] / current_length

    tangent_x = current_x
    tangent_y = current_y

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
    """
    Symmetric sampled mean distance with local tangent disagreement.

    ``tangent_weight`` converts the dimensionless normalized angular error
    into the same units as the graph coordinates.
    """

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

    def _sample_curve(self, polyline: XYArray) -> SampledCurve | None:
        points = sample_polyline_by_arclength(polyline, self.curve_samples)

        if points is None:
            return None

        tangents = _sampled_unit_tangents(points)

        if tangents is None:
            return None

        return points, tangents

    def _source_samples(self, edge_id: int) -> SampledCurve | None:
        if edge_id not in self._source_sample_cache:
            edge = self._source_edges[edge_id]
            self._source_sample_cache[edge_id] = self._sample_curve(edge.polyline)

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

        source_samples = self._source_samples(edge_id)
        source_prepared = self._source_prepared(edge_id)
        witness_samples = self._sample_curve(witness)
        witness_prepared = prepare_polyline_segments(witness)

        if source_samples is None or source_prepared is None or witness_samples is None or witness_prepared is None:
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
