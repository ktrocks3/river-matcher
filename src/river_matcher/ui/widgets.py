from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QStackedWidget, QVBoxLayout, QWidget

from river_matcher.matcher import MatchedEdge
from river_matcher.models import JunctionGraph

FloatArray = np.ndarray


def _as_points(value: object) -> FloatArray | None:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return None
    return np.ascontiguousarray(points)


def _vertex_positions(graph: JunctionGraph) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}

    for edge in graph.edges:
        points = _as_points(edge.polyline)
        if points is None:
            continue
        positions.setdefault(int(edge.u), (float(points[0, 0]), float(points[0, 1])))
        positions.setdefault(int(edge.v), (float(points[-1, 0]), float(points[-1, 1])))

    vertices = graph.vertices
    items: Iterable[tuple[object, object]]

    if isinstance(vertices, Mapping):
        items = vertices.items()
    else:
        items = ((getattr(vertex, "id", vertex), vertex) for vertex in vertices)

    for raw_id, vertex in items:
        try:
            vertex_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        if vertex_id in positions:
            continue

        x = getattr(vertex, "x", None)
        y = getattr(vertex, "y", None)
        if x is not None and y is not None:
            positions[vertex_id] = (float(x), float(y))
            continue

        for attribute in ("point", "xy", "position", "coordinates"):
            value = getattr(vertex, attribute, None)
            if value is None:
                continue
            point = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(point) >= 2:
                positions[vertex_id] = (float(point[0]), float(point[1]))
                break

    return positions


def _joined_path(polylines: Iterable[FloatArray]) -> tuple[FloatArray, FloatArray]:
    x_parts: list[FloatArray] = []
    y_parts: list[FloatArray] = []

    for points in polylines:
        if len(points) < 2:
            continue
        x_parts.extend((points[:, 0], np.asarray([np.nan])))
        y_parts.extend((points[:, 1], np.asarray([np.nan])))

    if not x_parts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    return np.concatenate(x_parts), np.concatenate(y_parts)


@dataclass(slots=True)
class PolylineIndex:
    starts: FloatArray
    vectors: FloatArray
    squared_lengths: FloatArray
    lengths: FloatArray
    path_ids: np.ndarray
    cumulative_starts: FloatArray
    total_lengths: dict[int, float]
    polylines: dict[int, FloatArray]

    @classmethod
    def build(cls, polylines: Mapping[int, FloatArray]) -> PolylineIndex:
        starts: list[FloatArray] = []
        vectors: list[FloatArray] = []
        squared_lengths: list[FloatArray] = []
        lengths: list[FloatArray] = []
        path_ids: list[np.ndarray] = []
        cumulative_starts: list[FloatArray] = []
        totals: dict[int, float] = {}
        retained: dict[int, FloatArray] = {}

        for path_id, raw_points in polylines.items():
            points = _as_points(raw_points)
            if points is None or len(points) < 2:
                continue

            segment_vectors = np.diff(points, axis=0)
            segment_squared = np.einsum("ij,ij->i", segment_vectors, segment_vectors)
            keep = segment_squared > 0.0
            if not np.any(keep):
                continue

            segment_vectors = segment_vectors[keep]
            segment_starts = points[:-1][keep]
            segment_squared = segment_squared[keep]
            segment_lengths = np.sqrt(segment_squared)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            totals[int(path_id)] = float(cumulative[-1])
            retained[int(path_id)] = points

            starts.append(segment_starts)
            vectors.append(segment_vectors)
            squared_lengths.append(segment_squared)
            lengths.append(segment_lengths)
            path_ids.append(np.full(len(segment_lengths), int(path_id), dtype=np.int64))
            cumulative_starts.append(cumulative[:-1])

        if not starts:
            return cls(np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), {}, {})

        return cls(np.concatenate(starts), np.concatenate(vectors), np.concatenate(squared_lengths), np.concatenate(lengths), np.concatenate(path_ids),
            np.concatenate(cumulative_starts), totals, retained)

    def nearest(self, x: float, y: float) -> tuple[int, float, float] | None:
        if len(self.starts) == 0:
            return None

        point = np.asarray([x, y], dtype=np.float64)
        offsets = point - self.starts
        projections = np.einsum("ij,ij->i", offsets, self.vectors) / self.squared_lengths
        projections = np.clip(projections, 0.0, 1.0)
        closest = self.starts + projections[:, None] * self.vectors
        differences = closest - point
        squared_distances = np.einsum("ij,ij->i", differences, differences)
        index = int(np.argmin(squared_distances))
        path_id = int(self.path_ids[index])
        total = self.total_lengths[path_id]
        fraction = (self.cumulative_starts[index] + projections[index] * self.lengths[index]) / total
        return path_id, float(fraction), math.sqrt(float(squared_distances[index]))

    def point_at(self, path_id: int, fraction: float) -> tuple[float, float] | None:
        points = self.polylines.get(int(path_id))
        if points is None or len(points) == 0:
            return None
        if len(points) == 1:
            return float(points[0, 0]), float(points[0, 1])

        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        total = float(lengths.sum())
        if total <= 0.0:
            return float(points[0, 0]), float(points[0, 1])

        distance = min(max(float(fraction), 0.0), 1.0) * total
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(lengths) - 1)
        local = 0.0 if lengths[segment] == 0.0 else (distance - cumulative[segment]) / lengths[segment]
        point = points[segment] + local * vectors[segment]
        return float(point[0]), float(point[1])


class GraphView(pg.PlotWidget):
    vertexSelected = Signal(int)
    edgePositionSelected = Signal(int, float)

    def __init__(self, role: str, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.role = role
        self.graph: JunctionGraph | None = None
        self._positions: dict[int, tuple[float, float]] = {}
        self._vertex_ids = np.empty(0, dtype=np.int64)
        self._vertex_xy = np.empty((0, 2), dtype=np.float64)
        self._graph_index = PolylineIndex.build({})
        self._witness_index = PolylineIndex.build({})

        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.12)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=True, y=True)

        self._edge_item = pg.PlotDataItem(pen=pg.mkPen("#7b8794", width=1))
        self._witness_item = pg.PlotDataItem(pen=pg.mkPen("#d47b24", width=2))
        self._vertex_item = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#263238"), brush=pg.mkBrush("#f5f5f5"))
        self._selected_vertices = pg.ScatterPlotItem(size=13, pen=pg.mkPen("#c62828", width=2), brush=pg.mkBrush("#ffebee"))
        self._selected_point = pg.ScatterPlotItem(size=12, symbol="x", pen=pg.mkPen("#1565c0", width=3))
        self._selected_path = pg.PlotDataItem(pen=pg.mkPen("#c62828", width=4))

        for item in (self._edge_item, self._witness_item, self._vertex_item, self._selected_path, self._selected_vertices, self._selected_point):
            self.addItem(item)

        self.scene().sigMouseClicked.connect(self._mouse_clicked)

    def set_graph(self, graph: JunctionGraph, *, title: str) -> None:
        self.graph = graph
        self.setTitle(title)
        edge_polylines = {int(edge.id): points for edge in graph.edges if (points := _as_points(edge.polyline)) is not None}
        self._graph_index = PolylineIndex.build(edge_polylines)
        self._positions = _vertex_positions(graph)
        ordered = sorted(self._positions)
        self._vertex_ids = np.asarray(ordered, dtype=np.int64)
        self._vertex_xy = np.asarray([self._positions[vertex] for vertex in ordered], dtype=np.float64)

        edge_x, edge_y = _joined_path(edge_polylines.values())
        self._edge_item.setData(edge_x, edge_y, connect="finite")

        if len(self._vertex_xy):
            self._vertex_item.setData(x=self._vertex_xy[:, 0], y=self._vertex_xy[:, 1], data=[int(vertex) for vertex in self._vertex_ids])
        else:
            self._vertex_item.clear()

        self.clear_result_overlays()
        self.enableAutoRange()

    def set_witnesses(self, edges: Iterable[MatchedEdge]) -> None:
        witnesses = {int(edge.edge_id): points for edge in edges if (points := _as_points(edge.witness)) is not None}
        self._witness_index = PolylineIndex.build(witnesses)
        x, y = _joined_path(witnesses.values())
        self._witness_item.setData(x, y, connect="finite")

    def clear_result_overlays(self) -> None:
        self._witness_index = PolylineIndex.build({})
        self._witness_item.clear()
        self._selected_path.clear()
        self._selected_vertices.clear()
        self._selected_point.clear()

    def highlight_vertices(self, vertex_ids: Iterable[int]) -> None:
        points = [self._positions[int(vertex)] for vertex in vertex_ids if int(vertex) in self._positions]
        if not points:
            self._selected_vertices.clear()
            return
        array = np.asarray(points, dtype=np.float64)
        self._selected_vertices.setData(array[:, 0], array[:, 1])

    def highlight_edge(self, edge_id: int, *, witness: bool = False) -> None:
        index = self._witness_index if witness else self._graph_index
        points = index.polylines.get(int(edge_id))
        if points is None:
            self._selected_path.clear()
            return
        self._selected_path.setData(points[:, 0], points[:, 1])

    def highlight_fraction(self, edge_id: int, fraction: float, *, witness: bool = False) -> None:
        index = self._witness_index if witness else self._graph_index
        point = index.point_at(edge_id, fraction)
        if point is None:
            self._selected_point.clear()
            return
        self._selected_point.setData([point[0]], [point[1]])

    def _data_tolerance(self) -> float:
        pixel_size = self.getViewBox().viewPixelSize()
        if pixel_size is None:
            return 1.0
        return 8.0 * max(abs(float(pixel_size[0])), abs(float(pixel_size[1])))

    def _nearest_vertex(self, x: float, y: float) -> tuple[int, float] | None:
        if len(self._vertex_xy) == 0:
            return None
        differences = self._vertex_xy - np.asarray([x, y], dtype=np.float64)
        squared = np.einsum("ij,ij->i", differences, differences)
        index = int(np.argmin(squared))
        return int(self._vertex_ids[index]), math.sqrt(float(squared[index]))

    def _mouse_clicked(self, event: object) -> None:
        button = getattr(event, "button", lambda: None)()
        if button != Qt.MouseButton.LeftButton:
            return

        scene_position = getattr(event, "scenePos")()
        if not self.sceneRect().contains(scene_position):
            return

        point = self.getViewBox().mapSceneToView(scene_position)
        x, y = float(point.x()), float(point.y())
        tolerance = self._data_tolerance()
        nearest_vertex = self._nearest_vertex(x, y)

        if nearest_vertex is not None and nearest_vertex[1] <= tolerance:
            self.vertexSelected.emit(nearest_vertex[0])
            return

        index = self._graph_index if self.role == "source" else self._witness_index
        nearest = index.nearest(x, y)
        if nearest is not None and nearest[2] <= tolerance:
            self.edgePositionSelected.emit(nearest[0], nearest[1])


@dataclass(frozen=True, slots=True)
class OptionSpec:
    name: str
    label: str
    kind: str
    default: object
    minimum: float = 0.0
    maximum: float = 1_000_000.0
    decimals: int = 3


_COST_OPTIONS: dict[str, tuple[OptionSpec, ...]] = {"relative_length_error": (), "log_length_distortion": (),
    "hausdorff_distance": (OptionSpec("rho", "Witness rho", "float", 10.0), OptionSpec("edge_samples", "Guide samples per edge", "int", 12, 2, 10_000),),
    "mean_distance_tangent": (OptionSpec("rho", "Witness rho", "float", 10.0), OptionSpec("edge_samples", "Guide samples per edge", "int", 12, 2, 10_000),
                              OptionSpec("curve_samples", "Curve samples", "int", 64, 2, 100_000), OptionSpec("tangent_weight", "Tangent weight", "float", 1.0),),
    "symmetric_corridor_exceedance": (OptionSpec("rho", "Witness rho", "float", 10.0), OptionSpec("edge_samples", "Guide samples per edge", "int", 12, 2, 10_000),
                                      OptionSpec("curve_samples", "Curve samples", "int", 64, 2, 100_000),
                                      OptionSpec("corridor_radius", "Exceedance radius", "float", 10.0, 1e-9),),
    "discrete_frechet_distance": (OptionSpec("rho", "Witness rho", "float", 10.0), OptionSpec("edge_samples", "Guide samples per edge", "int", 12, 2, 10_000),
                                  OptionSpec("curve_samples", "Curve samples", "int", 64, 2, 100_000),),
    "dynamic_time_warping_distance": (OptionSpec("rho", "Witness rho", "float", 10.0), OptionSpec("edge_samples", "Guide samples per edge", "int", 12, 2, 10_000),
                                      OptionSpec("curve_samples", "Curve samples", "int", 80, 2, 100_000), OptionSpec("warping_window", "Warping window", "int", 8, 1, 100_000),
                                      OptionSpec("normalize", "Normalize", "bool", True),)}


class CostOptionsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}
        self._fields: dict[str, dict[str, QWidget]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

        for cost_name, specs in _COST_OPTIONS.items():
            page = QWidget()
            form = QFormLayout(page)
            fields: dict[str, QWidget] = {}

            if not specs:
                form.addRow(QLabel("No cost-specific parameters."))

            for spec in specs:
                if spec.kind == "int":
                    spin = QSpinBox()
                    spin.setRange(int(spec.minimum), int(spec.maximum))
                    spin.setValue(int(spec.default))
                    widget: QWidget = spin
                elif spec.kind == "bool":
                    check = QCheckBox()
                    check.setChecked(bool(spec.default))
                    widget = check
                else:
                    double_spin = QDoubleSpinBox()
                    double_spin.setLocale(QLocale.c())
                    double_spin.setDecimals(spec.decimals)
                    double_spin.setRange(float(spec.minimum), float(spec.maximum))
                    double_spin.setValue(float(spec.default))
                    widget = double_spin

                fields[spec.name] = widget
                form.addRow(spec.label, widget)

            self._pages[cost_name] = page
            self._fields[cost_name] = fields
            self._stack.addWidget(page)

    def set_cost(self, cost_name: str) -> None:
        page = self._pages.get(cost_name)
        if page is not None:
            self._stack.setCurrentWidget(page)

    def options_for(self, cost_name: str) -> dict[str, object]:
        options: dict[str, object] = {}

        for name, widget in self._fields.get(cost_name, {}).items():
            if isinstance(widget, QCheckBox):
                options[name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                options[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                options[name] = widget.value()

        return options
