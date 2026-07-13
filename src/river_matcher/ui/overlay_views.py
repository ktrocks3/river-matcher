from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from river_matcher.matcher import MatchedEdge
from river_matcher.models import JunctionGraph
from river_matcher.ui.widgets import PolylineIndex, _as_points, _joined_path, _vertex_positions

try:
    import pyqtgraph.opengl as gl
except Exception as exc:
    gl = None
    _OPENGL_IMPORT_ERROR = str(exc)
else:
    _OPENGL_IMPORT_ERROR = ""

FloatArray = np.ndarray


def _polyline_map(graph: JunctionGraph) -> dict[int, FloatArray]:
    return {int(edge.id): points for edge in graph.edges if (points := _as_points(edge.polyline)) is not None}


def _set_joined_data(item: pg.PlotDataItem, polylines: Iterable[FloatArray]) -> None:
    x, y = _joined_path(polylines)
    item.setData(x, y, connect="finite")


def _toolbar_checkbox(label: str) -> QCheckBox:
    checkbox = QCheckBox(label)
    checkbox.setChecked(True)
    return checkbox


class Overlay2DView(QWidget):
    sourceVertexSelected = Signal(int)
    targetVertexSelected = Signal(int)
    sourceEdgePositionSelected = Signal(int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source: JunctionGraph | None = None
        self.target: JunctionGraph | None = None
        self._mapping: dict[int, int] = {}
        self._matched_edges: tuple[MatchedEdge, ...] = ()
        self._source_positions: dict[int, tuple[float, float]] = {}
        self._target_positions: dict[int, tuple[float, float]] = {}
        self._source_vertex_ids = np.empty(0, dtype=np.int64)
        self._target_vertex_ids = np.empty(0, dtype=np.int64)
        self._source_vertex_xy = np.empty((0, 2), dtype=np.float64)
        self._target_vertex_xy = np.empty((0, 2), dtype=np.float64)
        self._source_index = PolylineIndex.build({})
        self._witness_index = PolylineIndex.build({})
        self._selected_source_vertex: int | None = None
        self._selected_target_vertex: int | None = None
        self._selected_source_edge: int | None = None

        self.source_checkbox = _toolbar_checkbox("Source graph")
        self.target_checkbox = _toolbar_checkbox("Target graph")
        self.connectors_checkbox = _toolbar_checkbox("Assignment connectors")
        self.witnesses_checkbox = _toolbar_checkbox("Witnesses")
        self.vertices_checkbox = _toolbar_checkbox("Vertices")

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        for checkbox in (self.source_checkbox, self.target_checkbox, self.connectors_checkbox, self.witnesses_checkbox, self.vertices_checkbox):
            toolbar.addWidget(checkbox)
            checkbox.toggled.connect(self._update_visibility)
        toolbar.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setAspectLocked(True)
        self.plot.setMouseEnabled(x=True, y=True)

        self._target_item = pg.PlotDataItem(pen=pg.mkPen("#b6bec6", width=1))
        self._connector_item = pg.PlotDataItem(pen=pg.mkPen((91, 103, 112, 95), width=1))
        self._witness_item = pg.PlotDataItem(pen=pg.mkPen("#e07a1f", width=2))
        self._source_item = pg.PlotDataItem(pen=pg.mkPen("#246fa8", width=1.5))
        self._target_vertices_item = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#707b84"), brush=pg.mkBrush("#e3e7ea"))
        self._source_vertices_item = pg.ScatterPlotItem(size=7, pen=pg.mkPen("#174b70"), brush=pg.mkBrush("#4b9bd3"))
        self._selected_source_edge_item = pg.PlotDataItem(pen=pg.mkPen("#c62828", width=4))
        self._selected_witness_item = pg.PlotDataItem(pen=pg.mkPen("#f57c00", width=5))
        self._selected_connector_item = pg.PlotDataItem(pen=pg.mkPen("#c62828", width=3))
        self._selected_source_vertices_item = pg.ScatterPlotItem(size=14, pen=pg.mkPen("#b71c1c", width=2), brush=pg.mkBrush("#ffebee"))
        self._selected_target_vertices_item = pg.ScatterPlotItem(size=14, pen=pg.mkPen("#e65100", width=2), brush=pg.mkBrush("#fff3e0"))

        for item in (
            self._target_item,
            self._connector_item,
            self._witness_item,
            self._source_item,
            self._target_vertices_item,
            self._source_vertices_item,
            self._selected_source_edge_item,
            self._selected_witness_item,
            self._selected_connector_item,
            self._selected_source_vertices_item,
            self._selected_target_vertices_item,
        ):
            self.plot.addItem(item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)

    def set_graphs(self, source: JunctionGraph, target: JunctionGraph) -> None:
        if self.source is source and self.target is target:
            return

        self.source = source
        self.target = target
        source_polylines = _polyline_map(source)
        target_polylines = _polyline_map(target)
        self._source_index = PolylineIndex.build(source_polylines)
        self._source_positions = _vertex_positions(source)
        self._target_positions = _vertex_positions(target)
        self._source_vertex_ids = np.asarray(sorted(self._source_positions), dtype=np.int64)
        self._target_vertex_ids = np.asarray(sorted(self._target_positions), dtype=np.int64)
        self._source_vertex_xy = self._positions_array(self._source_positions, self._source_vertex_ids)
        self._target_vertex_xy = self._positions_array(self._target_positions, self._target_vertex_ids)

        _set_joined_data(self._source_item, source_polylines.values())
        _set_joined_data(self._target_item, target_polylines.values())
        self._set_vertex_item(self._source_vertices_item, self._source_vertex_ids, self._source_vertex_xy)
        self._set_vertex_item(self._target_vertices_item, self._target_vertex_ids, self._target_vertex_xy)
        self.clear_solution()
        self.plot.enableAutoRange()
        self.plot.getViewBox().autoRange()

    def set_solution(self, mapping: Mapping[int, int], edges: Sequence[MatchedEdge]) -> None:
        self._mapping = {int(source): int(target) for source, target in mapping.items()}
        self._matched_edges = tuple(edges)
        self._selected_source_vertex = None
        self._selected_target_vertex = None
        self._selected_source_edge = None
        self._rebuild_solution_layers()
        self._rebuild_highlights()

    def set_mapping(self, mapping: Mapping[int, int]) -> None:
        self.set_solution(mapping, ())

    def clear_solution(self) -> None:
        self._mapping = {}
        self._matched_edges = ()
        self._witness_index = PolylineIndex.build({})
        self._connector_item.clear()
        self._witness_item.clear()
        self._selected_source_vertex = None
        self._selected_target_vertex = None
        self._selected_source_edge = None
        self._clear_highlights()

    def set_selected_source_vertex(self, vertex_id: int | None) -> None:
        self._selected_source_vertex = None if vertex_id is None else int(vertex_id)
        self._rebuild_vertex_highlights()

    def set_selected_target_vertex(self, vertex_id: int | None) -> None:
        self._selected_target_vertex = None if vertex_id is None else int(vertex_id)
        self._rebuild_vertex_highlights()

    def set_selected_source_edge(self, edge_id: int | None) -> None:
        self._selected_source_edge = None if edge_id is None else int(edge_id)
        self._rebuild_edge_highlights()

    @staticmethod
    def _positions_array(positions: Mapping[int, tuple[float, float]], vertex_ids: np.ndarray) -> FloatArray:
        if len(vertex_ids) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray([positions[int(vertex)] for vertex in vertex_ids], dtype=np.float64)

    @staticmethod
    def _set_vertex_item(item: pg.ScatterPlotItem, vertex_ids: np.ndarray, points: FloatArray) -> None:
        if len(points) == 0:
            item.clear()
            return
        item.setData(x=points[:, 0], y=points[:, 1], data=[int(vertex) for vertex in vertex_ids])

    def _rebuild_solution_layers(self) -> None:
        connectors = []
        for source_id, target_id in sorted(self._mapping.items()):
            source_point = self._source_positions.get(source_id)
            target_point = self._target_positions.get(target_id)
            if source_point is not None and target_point is not None:
                connectors.append(np.asarray((source_point, target_point), dtype=np.float64))
        _set_joined_data(self._connector_item, connectors)

        witnesses = {
            int(edge.edge_id): points
            for edge in self._matched_edges
            if (points := _as_points(edge.witness)) is not None
        }
        self._witness_index = PolylineIndex.build(witnesses)
        _set_joined_data(self._witness_item, witnesses.values())
        self._update_visibility()

    def _clear_highlights(self) -> None:
        self._selected_source_edge_item.clear()
        self._selected_witness_item.clear()
        self._selected_connector_item.clear()
        self._selected_source_vertices_item.clear()
        self._selected_target_vertices_item.clear()

    def _rebuild_highlights(self) -> None:
        self._rebuild_vertex_highlights()
        self._rebuild_edge_highlights()

    def _rebuild_vertex_highlights(self) -> None:
        source_ids: set[int] = set()
        target_ids: set[int] = set()

        if self._selected_source_vertex is not None:
            source_ids.add(self._selected_source_vertex)
            mapped = self._mapping.get(self._selected_source_vertex)
            if mapped is not None:
                target_ids.add(mapped)
        elif self._selected_target_vertex is not None:
            source_ids.update(source for source, target in self._mapping.items() if target == self._selected_target_vertex)

        if self._selected_target_vertex is not None:
            target_ids.add(self._selected_target_vertex)

        self._set_highlight_scatter(self._selected_source_vertices_item, source_ids, self._source_positions)
        self._set_highlight_scatter(self._selected_target_vertices_item, target_ids, self._target_positions)

        connector = None
        if self._selected_source_vertex is not None:
            target_id = self._mapping.get(self._selected_source_vertex)
            source_point = self._source_positions.get(self._selected_source_vertex)
            target_point = None if target_id is None else self._target_positions.get(target_id)
            if source_point is not None and target_point is not None:
                connector = np.asarray((source_point, target_point), dtype=np.float64)

        if connector is None:
            self._selected_connector_item.clear()
        else:
            self._selected_connector_item.setData(connector[:, 0], connector[:, 1])
        self._update_visibility()

    @staticmethod
    def _set_highlight_scatter(item: pg.ScatterPlotItem, vertex_ids: Iterable[int], positions: Mapping[int, tuple[float, float]]) -> None:
        points = [positions[vertex] for vertex in sorted(set(vertex_ids)) if vertex in positions]
        if not points:
            item.clear()
            return
        array = np.asarray(points, dtype=np.float64)
        item.setData(array[:, 0], array[:, 1])

    def _rebuild_edge_highlights(self) -> None:
        edge_id = self._selected_source_edge
        source_points = None if edge_id is None else self._source_index.polylines.get(edge_id)
        witness_points = None if edge_id is None else self._witness_index.polylines.get(edge_id)

        if source_points is None:
            self._selected_source_edge_item.clear()
        else:
            self._selected_source_edge_item.setData(source_points[:, 0], source_points[:, 1])

        if witness_points is None:
            self._selected_witness_item.clear()
        else:
            self._selected_witness_item.setData(witness_points[:, 0], witness_points[:, 1])
        self._update_visibility()

    def _update_visibility(self) -> None:
        source_visible = self.source_checkbox.isChecked()
        target_visible = self.target_checkbox.isChecked()
        connectors_visible = self.connectors_checkbox.isChecked()
        witnesses_visible = self.witnesses_checkbox.isChecked()
        vertices_visible = self.vertices_checkbox.isChecked()
        self._source_item.setVisible(source_visible)
        self._target_item.setVisible(target_visible)
        self._connector_item.setVisible(connectors_visible)
        self._witness_item.setVisible(witnesses_visible)
        self._source_vertices_item.setVisible(vertices_visible and source_visible)
        self._target_vertices_item.setVisible(vertices_visible and target_visible)
        self._selected_source_edge_item.setVisible(source_visible)
        self._selected_witness_item.setVisible(witnesses_visible)
        self._selected_connector_item.setVisible(connectors_visible)
        self._selected_source_vertices_item.setVisible(vertices_visible and source_visible)
        self._selected_target_vertices_item.setVisible(vertices_visible and target_visible)

    def _data_tolerance(self) -> float:
        pixel_size = self.plot.getViewBox().viewPixelSize()
        if pixel_size is None:
            return 1.0
        return 8.0 * max(abs(float(pixel_size[0])), abs(float(pixel_size[1])))

    @staticmethod
    def _nearest_vertex(points: FloatArray, vertex_ids: np.ndarray, x: float, y: float) -> tuple[int, float] | None:
        if len(points) == 0:
            return None
        differences = points - np.asarray([x, y], dtype=np.float64)
        squared = np.einsum("ij,ij->i", differences, differences)
        index = int(np.argmin(squared))
        return int(vertex_ids[index]), math.sqrt(float(squared[index]))

    def _mouse_clicked(self, event: object) -> None:
        button = getattr(event, "button", lambda: None)()
        if button != Qt.MouseButton.LeftButton:
            return

        scene_position = getattr(event, "scenePos")()
        if not self.plot.sceneRect().contains(scene_position):
            return

        point = self.plot.getViewBox().mapSceneToView(scene_position)
        x, y = float(point.x()), float(point.y())
        tolerance = self._data_tolerance()

        if self.source_checkbox.isChecked() and self.vertices_checkbox.isChecked():
            nearest_source = self._nearest_vertex(self._source_vertex_xy, self._source_vertex_ids, x, y)
            if nearest_source is not None and nearest_source[1] <= tolerance:
                self.sourceVertexSelected.emit(nearest_source[0])
                return

        if self.target_checkbox.isChecked() and self.vertices_checkbox.isChecked():
            nearest_target = self._nearest_vertex(self._target_vertex_xy, self._target_vertex_ids, x, y)
            if nearest_target is not None and nearest_target[1] <= tolerance:
                self.targetVertexSelected.emit(nearest_target[0])
                return

        if self.source_checkbox.isChecked():
            nearest_edge = self._source_index.nearest(x, y)
            if nearest_edge is not None and nearest_edge[2] <= tolerance:
                self.sourceEdgePositionSelected.emit(nearest_edge[0], nearest_edge[1])


def _segments_xy(polylines: Iterable[FloatArray]) -> FloatArray:
    segments: list[FloatArray] = []
    for points in polylines:
        if len(points) >= 2:
            segments.append(np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 2))
    if not segments:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(segments)


class LayeredGraph3DView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source: JunctionGraph | None = None
        self.target: JunctionGraph | None = None
        self._mapping: dict[int, int] = {}
        self._matched_edges: tuple[MatchedEdge, ...] = ()
        self._source_positions: dict[int, tuple[float, float]] = {}
        self._target_positions: dict[int, tuple[float, float]] = {}
        self._source_polylines: dict[int, FloatArray] = {}
        self._target_polylines: dict[int, FloatArray] = {}
        self._witness_polylines: dict[int, FloatArray] = {}
        self._source_vertex_xy = np.empty((0, 2), dtype=np.float64)
        self._target_vertex_xy = np.empty((0, 2), dtype=np.float64)
        self._source_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._target_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._connector_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._witness_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._selected_source_vertex: int | None = None
        self._selected_target_vertex: int | None = None
        self._selected_source_edge: int | None = None
        self._items: dict[str, Any] = {}
        self._centre = np.zeros(2, dtype=np.float64)
        self._scale = 1.0

        self.source_checkbox = _toolbar_checkbox("Source graph")
        self.target_checkbox = _toolbar_checkbox("Target graph")
        self.connectors_checkbox = _toolbar_checkbox("Assignment connectors")
        self.witnesses_checkbox = _toolbar_checkbox("Witnesses")
        self.vertices_checkbox = _toolbar_checkbox("Vertices")
        self.reset_camera_button = QPushButton("Reset camera")
        self.layer_separation = QDoubleSpinBox()
        self.layer_separation.setRange(0.05, 20.0)
        self.layer_separation.setDecimals(2)
        self.layer_separation.setSingleStep(0.1)
        self.layer_separation.setValue(1.0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        for checkbox in (self.source_checkbox, self.target_checkbox, self.connectors_checkbox, self.witnesses_checkbox, self.vertices_checkbox):
            toolbar.addWidget(checkbox)
            checkbox.toggled.connect(self._update_visibility)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Layer separation"))
        toolbar.addWidget(self.layer_separation)
        toolbar.addWidget(self.reset_camera_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(toolbar)

        self._view: Any | None = None
        view_error = _OPENGL_IMPORT_ERROR
        if gl is not None:
            try:
                self._view = gl.GLViewWidget()
                self._view.setBackgroundColor("w")
            except Exception as exc:
                self._view = None
                view_error = str(exc)

        if self._view is None:
            message = QLabel(f"3D view unavailable. Install or repair PyOpenGL to enable it.\n{view_error}")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setWordWrap(True)
            layout.addWidget(message, 1)
            self.reset_camera_button.setEnabled(False)
            self.layer_separation.setEnabled(False)
        else:
            layout.addWidget(self._view, 1)
            self.reset_camera_button.clicked.connect(self.reset_camera)
            self.layer_separation.valueChanged.connect(self._layer_separation_changed)
            self.reset_camera()

    def set_graphs(self, source: JunctionGraph, target: JunctionGraph) -> None:
        if self.source is source and self.target is target:
            return

        self.source = source
        self.target = target
        self._calculate_normalization(source, target)
        self._source_positions = self._normalize_positions(_vertex_positions(source))
        self._target_positions = self._normalize_positions(_vertex_positions(target))
        self._source_polylines = self._normalize_polylines(_polyline_map(source))
        self._target_polylines = self._normalize_polylines(_polyline_map(target))
        self._source_vertex_xy = self._ordered_position_array(self._source_positions)
        self._target_vertex_xy = self._ordered_position_array(self._target_positions)
        self._source_segments_xy = _segments_xy(self._source_polylines.values())
        self._target_segments_xy = _segments_xy(self._target_polylines.values())
        self.clear_solution()
        self._rebuild_static_layers()
        self.reset_camera()

    def set_solution(self, mapping: Mapping[int, int], edges: Sequence[MatchedEdge]) -> None:
        self._mapping = {int(source): int(target) for source, target in mapping.items()}
        self._matched_edges = tuple(edges)
        self._selected_source_vertex = None
        self._selected_target_vertex = None
        self._selected_source_edge = None
        self._rebuild_dynamic_layers()
        self._rebuild_highlights()

    def set_mapping(self, mapping: Mapping[int, int]) -> None:
        self.set_solution(mapping, ())

    def clear_solution(self) -> None:
        self._mapping = {}
        self._matched_edges = ()
        self._witness_polylines = {}
        self._connector_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._witness_segments_xy = np.empty((0, 2), dtype=np.float64)
        self._selected_source_vertex = None
        self._selected_target_vertex = None
        self._selected_source_edge = None
        for name in ("connectors", "witnesses", "selected_source_edge", "selected_witness", "selected_connector", "selected_source_vertices", "selected_target_vertices"):
            self._remove_item(name)

    def set_selected_source_vertex(self, vertex_id: int | None) -> None:
        self._selected_source_vertex = None if vertex_id is None else int(vertex_id)
        self._rebuild_vertex_highlights()

    def set_selected_target_vertex(self, vertex_id: int | None) -> None:
        self._selected_target_vertex = None if vertex_id is None else int(vertex_id)
        self._rebuild_vertex_highlights()

    def set_selected_source_edge(self, edge_id: int | None) -> None:
        self._selected_source_edge = None if edge_id is None else int(edge_id)
        self._rebuild_edge_highlights()

    def _calculate_normalization(self, source: JunctionGraph, target: JunctionGraph) -> None:
        points = [points for graph in (source, target) for edge in graph.edges if (points := _as_points(edge.polyline)) is not None]
        if not points:
            points = [np.asarray(tuple(source.coordinates.values()) + tuple(target.coordinates.values()), dtype=np.float64)]
        combined = np.concatenate(points)
        minimum = np.min(combined, axis=0)
        maximum = np.max(combined, axis=0)
        self._centre = (minimum + maximum) * 0.5
        self._scale = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]), 1e-12)

    def _normalize_positions(self, positions: Mapping[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
        normalized: dict[int, tuple[float, float]] = {}
        for vertex, point in positions.items():
            transformed = (np.asarray(point, dtype=np.float64) - self._centre) / self._scale
            normalized[vertex] = float(transformed[0]), float(transformed[1])
        return normalized

    def _normalize_polylines(self, polylines: Mapping[int, FloatArray]) -> dict[int, FloatArray]:
        return {path_id: np.ascontiguousarray((points - self._centre) / self._scale) for path_id, points in polylines.items()}

    @staticmethod
    def _ordered_position_array(positions: Mapping[int, tuple[float, float]]) -> FloatArray:
        if not positions:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray([positions[vertex] for vertex in sorted(positions)], dtype=np.float64)

    def _rebuild_static_layers(self) -> None:
        for name in ("target", "source", "target_vertices", "source_vertices"):
            self._remove_item(name)
        self._install_line("target", self._with_z(self._target_segments_xy, 0.0), (0.62, 0.66, 0.69, 1.0), 1.0)
        self._install_line("source", self._with_z(self._source_segments_xy, self.layer_separation.value()), (0.12, 0.40, 0.66, 1.0), 1.5)
        self._install_scatter("target_vertices", self._with_z(self._target_vertex_xy, 0.0), (0.62, 0.66, 0.69, 1.0), 4.0)
        self._install_scatter("source_vertices", self._with_z(self._source_vertex_xy, self.layer_separation.value()), (0.18, 0.51, 0.75, 1.0), 5.0)
        self._update_visibility()

    def _rebuild_dynamic_layers(self) -> None:
        connector_pairs: list[FloatArray] = []
        for source_id, target_id in sorted(self._mapping.items()):
            source_point = self._source_positions.get(source_id)
            target_point = self._target_positions.get(target_id)
            if source_point is not None and target_point is not None:
                connector_pairs.append(np.asarray((source_point, target_point), dtype=np.float64))
        self._connector_segments_xy = np.concatenate(connector_pairs) if connector_pairs else np.empty((0, 2), dtype=np.float64)

        self._witness_polylines = {
            int(edge.edge_id): np.ascontiguousarray((points - self._centre) / self._scale)
            for edge in self._matched_edges
            if (points := _as_points(edge.witness)) is not None
        }
        self._witness_segments_xy = _segments_xy(self._witness_polylines.values())
        self._remove_item("connectors")
        self._remove_item("witnesses")
        self._install_line("connectors", self._connector_positions(), (0.34, 0.39, 0.42, 0.55), 1.0)
        self._install_line("witnesses", self._with_z(self._witness_segments_xy, 0.0), (0.90, 0.43, 0.08, 1.0), 2.0)
        self._update_visibility()

    def _connector_positions(self) -> FloatArray:
        if len(self._connector_segments_xy) == 0:
            return np.empty((0, 3), dtype=np.float64)
        positions = np.empty((len(self._connector_segments_xy), 3), dtype=np.float64)
        positions[:, :2] = self._connector_segments_xy
        positions[:, 2] = np.tile(np.asarray((self.layer_separation.value(), 0.0)), len(positions) // 2)
        return positions

    @staticmethod
    def _with_z(points: FloatArray, z: float) -> FloatArray:
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        positions = np.empty((len(points), 3), dtype=np.float64)
        positions[:, :2] = points
        positions[:, 2] = float(z)
        return positions

    def _install_line(self, name: str, positions: FloatArray, color: tuple[float, float, float, float], width: float) -> None:
        if gl is None or self._view is None or len(positions) == 0:
            return
        item = gl.GLLinePlotItem(pos=positions, color=color, width=width, antialias=True, mode="lines")
        self._items[name] = item
        self._view.addItem(item)

    def _install_scatter(self, name: str, positions: FloatArray, color: tuple[float, float, float, float], size: float) -> None:
        if gl is None or self._view is None or len(positions) == 0:
            return
        item = gl.GLScatterPlotItem(pos=positions, color=color, size=size, pxMode=True)
        self._items[name] = item
        self._view.addItem(item)

    def _remove_item(self, name: str) -> None:
        item = self._items.pop(name, None)
        if item is not None and self._view is not None:
            self._view.removeItem(item)

    def _rebuild_highlights(self) -> None:
        self._rebuild_vertex_highlights()
        self._rebuild_edge_highlights()

    def _rebuild_vertex_highlights(self) -> None:
        for name in ("selected_connector", "selected_source_vertices", "selected_target_vertices"):
            self._remove_item(name)

        source_ids: set[int] = set()
        target_ids: set[int] = set()
        if self._selected_source_vertex is not None:
            source_ids.add(self._selected_source_vertex)
            mapped = self._mapping.get(self._selected_source_vertex)
            if mapped is not None:
                target_ids.add(mapped)
                source_point = self._source_positions.get(self._selected_source_vertex)
                target_point = self._target_positions.get(mapped)
                if source_point is not None and target_point is not None:
                    points = np.asarray((source_point, target_point), dtype=np.float64)
                    positions = self._with_z(points, 0.0)
                    positions[0, 2] = self.layer_separation.value()
                    self._install_line("selected_connector", positions, (0.78, 0.10, 0.10, 1.0), 3.0)
        elif self._selected_target_vertex is not None:
            source_ids.update(source for source, target in self._mapping.items() if target == self._selected_target_vertex)

        if self._selected_target_vertex is not None:
            target_ids.add(self._selected_target_vertex)

        source_points = np.asarray([self._source_positions[vertex] for vertex in sorted(source_ids) if vertex in self._source_positions], dtype=np.float64)
        target_points = np.asarray([self._target_positions[vertex] for vertex in sorted(target_ids) if vertex in self._target_positions], dtype=np.float64)
        if source_points.size:
            self._install_scatter("selected_source_vertices", self._with_z(source_points.reshape(-1, 2), self.layer_separation.value()), (0.75, 0.05, 0.05, 1.0), 11.0)
        if target_points.size:
            self._install_scatter("selected_target_vertices", self._with_z(target_points.reshape(-1, 2), 0.0), (0.95, 0.35, 0.05, 1.0), 11.0)
        self._update_visibility()

    def _rebuild_edge_highlights(self) -> None:
        self._remove_item("selected_source_edge")
        self._remove_item("selected_witness")
        if self._selected_source_edge is not None:
            source_points = self._source_polylines.get(self._selected_source_edge)
            witness_points = self._witness_polylines.get(self._selected_source_edge)
            if source_points is not None:
                self._install_line("selected_source_edge", _segments_with_z(source_points, self.layer_separation.value()), (0.78, 0.10, 0.10, 1.0), 4.0)
            if witness_points is not None:
                self._install_line("selected_witness", _segments_with_z(witness_points, 0.0), (0.96, 0.35, 0.0, 1.0), 5.0)
        self._update_visibility()

    def _layer_separation_changed(self) -> None:
        separation = self.layer_separation.value()
        updates = {
            "source": self._with_z(self._source_segments_xy, separation),
            "source_vertices": self._with_z(self._source_vertex_xy, separation),
            "connectors": self._connector_positions(),
        }

        if self._selected_source_edge is not None:
            source_points = self._source_polylines.get(self._selected_source_edge)
            if source_points is not None:
                updates["selected_source_edge"] = _segments_with_z(source_points, separation)

        source_ids: set[int] = set()
        if self._selected_source_vertex is not None:
            source_ids.add(self._selected_source_vertex)
            target_id = self._mapping.get(self._selected_source_vertex)
            source_point = self._source_positions.get(self._selected_source_vertex)
            target_point = None if target_id is None else self._target_positions.get(target_id)
            if source_point is not None and target_point is not None:
                connector = self._with_z(np.asarray((source_point, target_point), dtype=np.float64), 0.0)
                connector[0, 2] = separation
                updates["selected_connector"] = connector
        elif self._selected_target_vertex is not None:
            source_ids.update(source for source, target in self._mapping.items() if target == self._selected_target_vertex)

        source_points = np.asarray([self._source_positions[vertex] for vertex in sorted(source_ids) if vertex in self._source_positions], dtype=np.float64)
        if source_points.size:
            updates["selected_source_vertices"] = self._with_z(source_points.reshape(-1, 2), separation)

        for name, positions in updates.items():
            item = self._items.get(name)
            if item is not None:
                item.setData(pos=positions)
        if self._view is not None:
            self._view.opts["center"] = QVector3D(0.0, 0.0, separation * 0.5)

    def _update_visibility(self) -> None:
        visibility = {
            "source": self.source_checkbox.isChecked(),
            "target": self.target_checkbox.isChecked(),
            "connectors": self.connectors_checkbox.isChecked(),
            "witnesses": self.witnesses_checkbox.isChecked(),
            "source_vertices": self.source_checkbox.isChecked() and self.vertices_checkbox.isChecked(),
            "target_vertices": self.target_checkbox.isChecked() and self.vertices_checkbox.isChecked(),
            "selected_source_edge": self.source_checkbox.isChecked(),
            "selected_witness": self.witnesses_checkbox.isChecked(),
            "selected_connector": self.connectors_checkbox.isChecked(),
            "selected_source_vertices": self.source_checkbox.isChecked() and self.vertices_checkbox.isChecked(),
            "selected_target_vertices": self.target_checkbox.isChecked() and self.vertices_checkbox.isChecked(),
        }
        for name, visible in visibility.items():
            item = self._items.get(name)
            if item is not None:
                item.setVisible(visible)

    def reset_camera(self) -> None:
        if self._view is None:
            return
        separation = self.layer_separation.value()
        self._view.opts["center"] = QVector3D(0.0, 0.0, separation * 0.5)
        self._view.setCameraPosition(distance=max(2.4, 1.8 + separation), elevation=24.0, azimuth=-45.0)


def _segments_with_z(points: FloatArray, z: float) -> FloatArray:
    return LayeredGraph3DView._with_z(_segments_xy((points,)), z)
