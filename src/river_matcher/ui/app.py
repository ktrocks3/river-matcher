from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLocale, QSettings, Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from river_matcher.cancellation import CancellationToken
from river_matcher.costs import available_costs
from river_matcher.dynamic_programming import Objective
from river_matcher.matcher import BothMatchResult, MappingEvaluation, MatchedEdge, MatchSolution
from river_matcher.preflight import MatchingPreflight
from river_matcher.ui.widgets import CostOptionsWidget, GraphView
from river_matcher.ui.workers import (
    CatalogOutcome,
    CatalogWorker,
    GraphInfo,
    GraphRepository,
    MappingScoreOutcome,
    MappingScoreWorker,
    MatchWorker,
    PairSessionStore,
    PreflightOutcome,
    PreflightWorker,
    PreviewOutcome,
    PreviewWorker,
    RunOutcome,
    normalize_sparse_to_dense,
)

_WARN_STATE_LIMIT = 2_000_000
_BLOCK_STATE_LIMIT = 10_000_000


@dataclass(frozen=True, slots=True)
class ImportedMapping:
    json_path: Path
    source_path: Path
    target_path: Path
    mapping: Mapping[int, int]
    saved_cost_name: str | None
    saved_cost_options: Mapping[str, object]
    saved_objective: str | None
    saved_value: float | None
    saved_edges: tuple[MatchedEdge, ...]
    saved_additive_value: float | None
    saved_bottleneck_value: float | None

    @property
    def label(self) -> str:
        objective = "" if self.saved_objective is None else f" ({self.saved_objective})"
        return f"Imported φ: {self.json_path.name}{objective}"


def _find_graph_directory() -> Path:
    candidates = (Path.cwd() / "GraphExport", Path(__file__).resolve().parents[3] / "GraphExport")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return Path.cwd()


def _solution_payload(solution: MatchSolution | None) -> dict[str, object]:
    if solution is None:
        return {"feasible": False}

    return {
        "feasible": True,
        "objective": solution.objective.value,
        "value": float(solution.value),
        "mapping": [{"source_vertex": int(source), "target_vertex": int(target)} for source, target in sorted(solution.mapping.items())],
        "edges": [
            {
                "edge_id": int(edge.edge_id),
                "source_u": int(edge.source_u),
                "source_v": int(edge.source_v),
                "target_u": int(edge.target_u),
                "target_v": int(edge.target_v),
                "cost": float(edge.cost),
                "witness": edge.witness.tolist(),
            }
            for edge in solution.edges
        ],
    }


def _preflight_payload(preflight: MatchingPreflight | None) -> dict[str, object] | None:
    if preflight is None:
        return None

    return {
        "empty_domains": preflight.empty_domains,
        "total_candidates": preflight.total_candidates,
        "minimum_candidates": preflight.minimum_candidates,
        "maximum_candidates": preflight.maximum_candidates,
        "estimated_state_upper_bound": preflight.estimated_state_upper_bound,
        "largest_candidate_product": preflight.largest_candidate_product,
        "largest_bag": None if preflight.largest_bag is None else sorted(preflight.largest_bag),
    }


def _result_payload(outcome: RunOutcome) -> dict[str, object]:
    result = outcome.result
    candidates = result.candidate_statistics
    effective = result.effective_candidate_statistics or candidates
    decomposition = result.decomposition
    dp = result.dp_statistics
    compatibility = result.compatibility_statistics

    return {
        "schema_version": 2,
        "source": {"path": str(outcome.source_path), "name": outcome.source.name, "vertices": len(outcome.source.vertices), "edges": len(outcome.source.edges)},
        "target": {"path": str(outcome.target_path), "name": outcome.target.name, "vertices": len(outcome.target.vertices), "edges": len(outcome.target.edges)},
        "cost": {"name": outcome.cost_name, "options": dict(outcome.cost_options)},
        "candidate_parameters": {"rho": outcome.candidate_rho, "top_k": outcome.top_k},
        "candidate_statistics": {
            "source_vertices": candidates.source_vertices,
            "empty_domains": candidates.empty_domains,
            "total_candidates": candidates.total_candidates,
            "minimum_candidates": candidates.minimum_candidates,
            "maximum_candidates": candidates.maximum_candidates,
        },
        "effective_candidate_statistics": {
            "source_vertices": effective.source_vertices,
            "empty_domains": effective.empty_domains,
            "total_candidates": effective.total_candidates,
            "minimum_candidates": effective.minimum_candidates,
            "maximum_candidates": effective.maximum_candidates,
        },
        "candidate_sets": {str(vertex): list(values) for vertex, values in result.candidate_sets.items()},
        "effective_candidate_sets": {str(vertex): list(values) for vertex, values in (result.effective_candidate_sets or result.candidate_sets).items()},
        "preflight": _preflight_payload(result.preflight),
        "effective_preflight": _preflight_payload(result.effective_preflight),
        "compatibility": None
        if compatibility is None
        else {
            "initial_candidates": compatibility.initial_candidates,
            "remaining_candidates": compatibility.remaining_candidates,
            "removed_candidates": compatibility.removed_candidates,
            "revised_arcs": compatibility.revised_arcs,
            "empty_domains": compatibility.empty_domains,
        },
        "decomposition": {
            "width": decomposition.width,
            "maximum_bag_size": decomposition.maximum_bag_size,
            "bag_count": decomposition.bag_count,
            "heuristic": decomposition.heuristic.value,
            "minimum_fill_width": decomposition.minimum_fill_width,
            "minimum_degree_width": decomposition.minimum_degree_width,
        },
        "dynamic_programming": {
            "enumerated_states": dp.enumerated_states,
            "feasible_states": dp.feasible_states,
            "message_entries": dp.message_entries,
            "unique_cost_requests": dp.unique_cost_requests,
            "partial_assignments": dp.partial_assignments,
        },
        "timing": None
        if result.timing is None
        else {
            "arc_consistency_seconds": result.timing.arc_consistency_seconds,
            "feasibility_dp_seconds": result.timing.feasibility_dp_seconds,
            "cost_setup_seconds": result.timing.cost_setup_seconds,
            "cost_dp_seconds": result.timing.cost_dp_seconds,
            "materialization_seconds": result.timing.materialization_seconds,
            "uncached_local_cost_seconds": result.timing.uncached_local_cost_seconds,
            "uncached_local_cost_calls": result.timing.uncached_local_cost_calls,
            "local_cost_cache_hits": result.timing.local_cost_cache_hits,
            "witness_adjacency_seconds": result.timing.witness_adjacency_seconds,
            "witness_adjacency_builds": result.timing.witness_adjacency_builds,
            "witness_dijkstra_seconds": result.timing.witness_dijkstra_seconds,
            "witness_dijkstra_runs": result.timing.witness_dijkstra_runs,
            "feasibility_reused": result.timing.feasibility_reused,
        },
        "solutions": {Objective.ADDITIVE.value: _solution_payload(result.additive), Objective.BOTTLENECK.value: _solution_payload(result.bottleneck)},
    }


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("River Graph Matcher")
        self.resize(1500, 900)

        self.settings = QSettings("GraphThesis", "RiverGraphMatcher")
        self.thread_pool = QThreadPool.globalInstance()
        self.repository = GraphRepository()
        self.sessions = PairSessionStore(maximum_sessions=6)
        self._preview_request_id = 0
        self._graph_catalog: dict[Path, GraphInfo] = {}
        self._rebuilding_targets = False
        self._outcomes: dict[tuple[str, str, float, int, str, str], RunOutcome] = {}
        self._current_outcome: RunOutcome | None = None
        self._imported_mapping: ImportedMapping | None = None
        self._mapping_scores: dict[tuple[str, str, str, str, tuple[tuple[int, int], ...]], MappingScoreOutcome] = {}
        self._active_token: CancellationToken | None = None
        # Keep the QRunnable alive until its queued finished signal reaches the UI thread.
        self._active_worker: PreflightWorker | MatchWorker | MappingScoreWorker | None = None
        self._pending_run: tuple[tuple[Path, Path], tuple[tuple[str, Mapping[str, object]], ...]] | None = None
        self._busy = False

        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        self.source_browse = QPushButton("Browse")
        self.target_browse = QPushButton("Browse")
        self.direction_label = QLabel("The graph with fewer vertices is always used as the source.")
        self.direction_label.setWordWrap(True)

        self.cost_combo = QComboBox()

        for cost in available_costs():
            self.cost_combo.addItem(cost.value.replace("_", " "), cost.value)

        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItem("Sum", Objective.ADDITIVE.value)
        self.aggregation_combo.addItem("Max", Objective.BOTTLENECK.value)
        # self.aggregation_combo.addItem("Length-weighted sum", Objective.LENGTH_WEIGHTED_ADDITIVE.value)

        self.mapping_mode_combo = QComboBox()
        self.mapping_mode_combo.addItem("Computed optimum", "computed")
        self.score_label = QLabel("Score: —")
        self.score_label.setStyleSheet("font-weight: 600;")

        self.candidate_rho = QDoubleSpinBox()
        self.candidate_rho.setLocale(QLocale.c())
        self.candidate_rho.setRange(0.0, 1_000_000.0)
        self.candidate_rho.setDecimals(3)
        self.candidate_rho.setValue(float(self.settings.value("candidate_rho", 10.0)))

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100_000)
        self.top_k.setValue(int(self.settings.value("top_k", 25)))

        self.cost_options = CostOptionsWidget()
        self.run_button = QPushButton("Run selected cost")
        self.compute_all_button = QPushButton("Compute all costs")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.save_button = QPushButton("Save result JSON")
        self.save_button.setEnabled(False)
        self.import_button = QPushButton("Import result JSON")
        self.score_mapping_button = QPushButton("Score imported φ")
        self.score_mapping_button.setEnabled(False)
        self.clear_cache_button = QPushButton("Clear caches")

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status_label = QLabel("Ready")

        self.source_view = GraphView("source")
        self.target_view = GraphView("target")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(1000)

        self._build_layout()
        self._connect_signals()
        self.cost_options.set_cost(self.current_cost_name)
        self._start_catalog_load()

    @property
    def current_cost_name(self) -> str:
        return str(self.cost_combo.currentData())

    @staticmethod
    def _outcome_key(outcome: RunOutcome) -> tuple[str, str, float, int, str, str]:
        options = json.dumps(dict(outcome.cost_options), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (str(outcome.source_path.resolve()), str(outcome.target_path.resolve()), float(outcome.candidate_rho), int(outcome.top_k), outcome.cost_name, options)

    def _current_key(self, cost_name: str) -> tuple[str, str, float, int, str, str] | None:
        paths = self._selected_paths()

        if paths is None:
            return None

        options = json.dumps(self.cost_options.options_for(cost_name), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (str(paths[0].resolve()), str(paths[1].resolve()), float(self.candidate_rho.value()), int(self.top_k.value()), cost_name, options)

    @property
    def _display_mode(self) -> str:
        return str(self.mapping_mode_combo.currentData() or "computed")

    def _mapping_score_key(self, mapping: Mapping[int, int], cost_name: str | None = None) -> tuple[str, str, str, str, tuple[tuple[int, int], ...]] | None:
        paths = self._selected_paths()

        if paths is None:
            return None

        resolved_cost = self.current_cost_name if cost_name is None else cost_name
        options = json.dumps(self.cost_options.options_for(resolved_cost), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (str(paths[0].resolve()), str(paths[1].resolve()), resolved_cost, options, tuple(sorted((int(source), int(target)) for source, target in mapping.items())))

    def _imported_pair_is_current(self) -> bool:
        imported = self._imported_mapping
        paths = self._selected_paths()
        return imported is not None and paths is not None and paths[0].resolve() == imported.source_path.resolve() and paths[1].resolve() == imported.target_path.resolve()

    def _current_imported_score(self) -> MappingScoreOutcome | None:
        imported = self._imported_mapping

        if imported is None or not self._imported_pair_is_current():
            return None

        key = self._mapping_score_key(imported.mapping)
        return None if key is None else self._mapping_scores.get(key)

    def _build_layout(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)

        files_group = QGroupBox("Graph pair")
        files = QGridLayout(files_group)
        files.addWidget(QLabel("Sparse source"), 0, 0)
        files.addWidget(self.source_combo, 0, 1)
        files.addWidget(self.source_browse, 0, 2)
        files.addWidget(QLabel("Dense target"), 1, 0)
        files.addWidget(self.target_combo, 1, 1)
        files.addWidget(self.target_browse, 1, 2)
        files.addWidget(self.direction_label, 2, 0, 1, 3)

        parameters_group = QGroupBox("Matching")
        parameters = QGridLayout(parameters_group)
        parameters.addWidget(QLabel("Cost"), 0, 0)
        parameters.addWidget(self.cost_combo, 0, 1)
        parameters.addWidget(QLabel("Displayed aggregation"), 0, 2)
        parameters.addWidget(self.aggregation_combo, 0, 3)
        parameters.addWidget(QLabel("Candidate radius ρ"), 1, 0)
        parameters.addWidget(self.candidate_rho, 1, 1)
        parameters.addWidget(QLabel("Top-k candidates"), 1, 2)
        parameters.addWidget(self.top_k, 1, 3)
        parameters.addWidget(self.cost_options, 2, 0, 1, 4)
        parameters.addWidget(QLabel("Displayed mapping"), 3, 0)
        parameters.addWidget(self.mapping_mode_combo, 3, 1)
        parameters.addWidget(QLabel("Current score"), 3, 2)
        parameters.addWidget(self.score_label, 3, 3)

        controls = QHBoxLayout()
        controls.addWidget(files_group, 2)
        controls.addWidget(parameters_group, 3)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.compute_all_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.score_mapping_button)
        buttons.addStretch(1)
        buttons.addWidget(self.clear_cache_button)

        views = QSplitter(Qt.Orientation.Horizontal)
        views.addWidget(self.source_view)
        views.addWidget(self.target_view)
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)

        outer.addLayout(controls)
        outer.addLayout(buttons)
        outer.addWidget(views, 1)
        outer.addWidget(QLabel("Selection and run details"))
        outer.addWidget(self.details)

        status = QHBoxLayout()
        status.addWidget(self.progress)
        status.addWidget(self.status_label, 1)
        outer.addLayout(status)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.source_browse.clicked.connect(lambda: self._browse_graph(self.source_combo))
        self.target_browse.clicked.connect(lambda: self._browse_graph(self.target_combo))
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.cost_combo.currentIndexChanged.connect(self._cost_changed)
        self.aggregation_combo.currentIndexChanged.connect(self._objective_changed)
        self.mapping_mode_combo.currentIndexChanged.connect(self._mapping_mode_changed)
        self.cost_options.optionsChanged.connect(self._cost_options_changed)
        self.run_button.clicked.connect(self._run_selected)
        self.compute_all_button.clicked.connect(self._compute_all)
        self.cancel_button.clicked.connect(self._cancel_active)
        self.save_button.clicked.connect(self._save_result)
        self.import_button.clicked.connect(self._import_result)
        self.score_mapping_button.clicked.connect(self._score_imported_mapping)
        self.clear_cache_button.clicked.connect(self._clear_caches)
        self.source_view.vertexSelected.connect(self._source_vertex_selected)
        self.target_view.vertexSelected.connect(self._target_vertex_selected)
        self.source_view.edgePositionSelected.connect(self._source_edge_selected)
        self.target_view.edgePositionSelected.connect(self._target_witness_selected)

    @staticmethod
    def _graph_group(path: Path) -> str:
        prefix, separator, _ = path.stem.partition("e")
        return prefix if separator else path.stem

    @staticmethod
    def _graph_label(info: GraphInfo) -> str:
        return f"{info.path.name} — {info.vertices} V / {info.edges} E"

    def _start_catalog_load(self) -> None:
        paths = sorted(_find_graph_directory().glob("*.txt"))
        self.source_combo.setEnabled(False)
        self.target_combo.setEnabled(False)
        self.run_button.setEnabled(False)
        self.compute_all_button.setEnabled(False)

        if not paths:
            self.status_label.setText(f"No .txt graph files found in {_find_graph_directory()}")
            return

        worker = CatalogWorker(self.repository, paths)
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.result.connect(self._catalog_ready)
        worker.signals.failed.connect(self._worker_failed)
        self.thread_pool.start(worker)

    def _catalog_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, CatalogOutcome):
            return

        self._graph_catalog = {graph.path.resolve(): graph for graph in raw_outcome.graphs}
        self._populate_source_choices()
        self._restore_pair_selection()
        self.source_combo.setEnabled(True)
        self.target_combo.setEnabled(True)
        self.run_button.setEnabled(True)
        self.compute_all_button.setEnabled(True)
        self.status_label.setText(f"Loaded {len(self._graph_catalog)} graph files")
        self._schedule_preview()

    def _populate_source_choices(self, preferred_path: Path | None = None) -> None:
        previous = preferred_path

        if previous is None and self.source_combo.currentData():
            previous = Path(str(self.source_combo.currentData())).resolve()

        graphs = sorted(
            (graph for graph in self._graph_catalog.values() if any(other.vertices > graph.vertices for other in self._graph_catalog.values())),
            key=lambda graph: (graph.vertices, graph.path.name.lower()),
        )
        self.source_combo.blockSignals(True)
        self.source_combo.clear()

        for graph in graphs:
            self.source_combo.addItem(self._graph_label(graph), str(graph.path))

        self.source_combo.blockSignals(False)

        if previous is not None:
            self._select_path(self.source_combo, previous)

    def _restore_pair_selection(self) -> None:
        saved_source = str(self.settings.value("source_path", ""))
        saved_target = str(self.settings.value("target_path", ""))
        source_path = Path(saved_source).resolve() if saved_source else None
        target_path = Path(saved_target).resolve() if saved_target else None

        if source_path is None or source_path not in self._graph_catalog:
            source_path = next((path for path in self._graph_catalog if path.name.lower() == "1955e5.txt"), None)

        if source_path is not None:
            self._select_path(self.source_combo, source_path)

        self._rebuild_target_choices(preferred_path=target_path)

        if not saved_target:
            default_target = next((path for path in self._graph_catalog if path.name.lower() == "1955e3.txt"), None)

            if default_target is not None:
                self._select_path(self.target_combo, default_target)

        self._select_data(self.cost_combo, str(self.settings.value("cost", "relative_length_error")))
        self._select_data(self.aggregation_combo, str(self.settings.value("aggregation", Objective.ADDITIVE.value)))

    def _source_changed(self) -> None:
        if self._rebuilding_targets:
            return

        self._rebuild_target_choices()
        self._pair_changed()

    def _target_changed(self) -> None:
        if not self._rebuilding_targets:
            self._pair_changed()

    def _pair_changed(self) -> None:
        if self._busy:
            self._cancel_active()

        self._current_outcome = None
        self.save_button.setEnabled(False)

        if self._display_mode == "imported" and not self._imported_pair_is_current():
            self._select_data(self.mapping_mode_combo, "computed")

        self.score_mapping_button.setEnabled(not self._busy and self._display_mode == "imported" and self._imported_pair_is_current())
        self.score_label.setText("Score: —")
        self._schedule_preview()

    def _rebuild_target_choices(self, preferred_path: Path | None = None) -> None:
        source_data = self.source_combo.currentData()

        if not source_data:
            return

        source_path = Path(str(source_data)).resolve()
        source = self._graph_catalog.get(source_path)

        if source is None:
            return

        current_path = preferred_path

        if current_path is None and self.target_combo.currentData():
            current_path = Path(str(self.target_combo.currentData())).resolve()

        source_group = self._graph_group(source.path)
        valid = [graph for graph in self._graph_catalog.values() if graph.vertices > source.vertices]
        valid.sort(key=lambda graph: (self._graph_group(graph.path) != source_group, graph.vertices - source.vertices, graph.path.name.lower()))
        self._rebuilding_targets = True
        self.target_combo.blockSignals(True)
        self.target_combo.clear()

        for graph in valid:
            self.target_combo.addItem(self._graph_label(graph), str(graph.path))

        selected = current_path is not None and self._select_path(self.target_combo, current_path)

        if not selected and valid:
            self.target_combo.setCurrentIndex(0)

        self.target_combo.blockSignals(False)
        self._rebuilding_targets = False

        if valid:
            target = self._graph_catalog[Path(str(self.target_combo.currentData())).resolve()]
            self.direction_label.setText(f"Sparse-to-dense direction: {source.path.name} ({source.vertices} V) → {target.path.name} ({target.vertices} V)")
        else:
            self.direction_label.setText("No denser target is available for this source.")

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> bool:
        index = combo.findData(value)

        if index < 0:
            return False

        combo.setCurrentIndex(index)
        return True

    @staticmethod
    def _select_path(combo: QComboBox, path: Path) -> bool:
        resolved = path.expanduser().resolve()

        for index in range(combo.count()):
            data = combo.itemData(index)

            if data and Path(str(data)).resolve() == resolved:
                combo.setCurrentIndex(index)
                return True

        return False

    def _selected_paths(self) -> tuple[Path, Path] | None:
        source = self.source_combo.currentData()
        target = self.target_combo.currentData()

        if not source or not target:
            return None

        return Path(str(source)), Path(str(target))

    def _browse_graph(self, combo: QComboBox) -> None:
        current = combo.currentData()
        directory = str(Path(str(current)).parent) if current else str(_find_graph_directory())
        filename, _ = QFileDialog.getOpenFileName(self, "Select TopoTide graph", directory, "Graph exports (*.txt);;All files (*)")

        if not filename:
            return

        requested = Path(filename).resolve()
        worker = CatalogWorker(self.repository, (requested,))
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.failed.connect(self._worker_failed)

        def loaded(raw_outcome: object) -> None:
            if not isinstance(raw_outcome, CatalogOutcome) or not raw_outcome.graphs:
                return

            info = raw_outcome.graphs[0]
            self._graph_catalog[info.path.resolve()] = info
            self._populate_source_choices(preferred_path=info.path if combo is self.source_combo else None)

            if combo is self.source_combo:
                if not self._select_path(self.source_combo, info.path):
                    self.status_label.setText(f"{info.path.name} cannot be a source because no denser graph is loaded.")
                    return
                self._rebuild_target_choices()
            else:
                self._rebuild_target_choices(preferred_path=info.path)

                if not self._select_path(self.target_combo, info.path):
                    selected = self.target_combo.currentData()
                    selected_name = Path(str(selected)).name if selected else "none"
                    self.status_label.setText(f"{info.path.name} is not denser than the selected source; using {selected_name}.")

            self._pair_changed()

        worker.signals.result.connect(loaded)
        self.thread_pool.start(worker)

    def _schedule_preview(self) -> None:
        paths = self._selected_paths()

        if paths is None:
            return

        self._preview_request_id += 1
        request_id = self._preview_request_id

        def start() -> None:
            if request_id != self._preview_request_id:
                return

            worker = PreviewWorker(self.repository, paths[0], paths[1], request_id=request_id)
            worker.signals.result.connect(self._preview_ready)
            worker.signals.failed.connect(self._worker_failed)
            self.thread_pool.start(worker)

        QTimer.singleShot(150, start)

    def _preview_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, PreviewOutcome) or raw_outcome.request_id != self._preview_request_id:
            return

        outcome = raw_outcome
        self.source_view.set_graph(outcome.source, title=f"Source: {outcome.source.name} — {len(outcome.source.vertices)} V / {len(outcome.source.edges)} E")
        self.target_view.set_graph(outcome.target, title=f"Target: {outcome.target.name} — {len(outcome.target.vertices)} V / {len(outcome.target.edges)} E")
        if self._display_mode == "imported" and self._imported_pair_is_current():
            self._display_imported_mapping()
        else:
            self.details.setPlainText("Graphs loaded. Run a cost to compute a mapping.")
            self.score_label.setText("Score: —")

    def _cost_changed(self) -> None:
        self.cost_options.set_cost(self.current_cost_name)
        self.settings.setValue("cost", self.current_cost_name)

        if self._display_mode == "imported":
            self._display_imported_mapping()
            self.status_label.setText("Choose the cost options, then score the imported mapping.")
            return

        key = self._current_key(self.current_cost_name)
        cached = None if key is None else self._outcomes.get(key)

        if cached is not None:
            self._display_outcome(cached)
        else:
            self.score_label.setText("Score: —")
            self.status_label.setText("This cost has not been computed for the current pair and parameters.")

    def _objective_changed(self) -> None:
        self.settings.setValue("aggregation", str(self.aggregation_combo.currentData()))

        if self._display_mode == "imported":
            self._display_imported_mapping()
        elif self._current_outcome is not None:
            self._display_outcome(self._current_outcome)

    def _cost_options_changed(self) -> None:
        if self._display_mode == "imported":
            self._display_imported_mapping()

    def _mapping_mode_changed(self) -> None:
        imported_mode = self._display_mode == "imported"
        self.score_mapping_button.setEnabled(not self._busy and imported_mode and self._imported_pair_is_current())

        if imported_mode:
            self._display_imported_mapping()
            return

        key = self._current_key(self.current_cost_name)
        cached = None if key is None else self._outcomes.get(key)

        if cached is not None:
            self._display_outcome(cached)
        else:
            self.score_label.setText("Score: —")
            self.save_button.setEnabled(False)
            self.target_view.clear_result_overlays()
            self.details.setPlainText("Computed optimum selected. Run the current cost to compute a mapping.")

    def _run_selected(self) -> None:
        paths = self._selected_paths()

        if paths is None:
            return

        self._select_data(self.mapping_mode_combo, "computed")
        cost_name = self.current_cost_name
        self._begin_run(paths, ((cost_name, self.cost_options.options_for(cost_name)),))

    def _compute_all(self) -> None:
        paths = self._selected_paths()

        if paths is None:
            return

        self._select_data(self.mapping_mode_combo, "computed")
        selected = self.current_cost_name
        names = [selected] + [str(self.cost_combo.itemData(index)) for index in range(self.cost_combo.count()) if str(self.cost_combo.itemData(index)) != selected]
        costs = tuple((name, self.cost_options.options_for(name)) for name in names)
        self._begin_run(paths, costs)

    def _begin_run(self, paths: tuple[Path, Path], costs: Sequence[tuple[str, Mapping[str, object]]]) -> None:
        if self._busy:
            return

        token = CancellationToken()
        self._active_token = token
        self._pending_run = (paths, tuple(costs))
        self._set_busy(True)
        self.settings.setValue("candidate_rho", self.candidate_rho.value())
        self.settings.setValue("top_k", self.top_k.value())

        worker = PreflightWorker(self.repository, self.sessions, paths[0], paths[1], candidate_rho=self.candidate_rho.value(), top_k=self.top_k.value(), cancellation_token=token)
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.result.connect(self._preflight_ready)
        worker.signals.failed.connect(self._worker_failed_and_finish)
        worker.signals.cancelled.connect(self._worker_cancelled_and_finish)
        self._active_worker = worker
        self.thread_pool.start(worker)

    def _preflight_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, PreflightOutcome) or self._pending_run is None:
            return

        preflight = raw_outcome.preflight
        self.details.setPlainText(self._format_preflight(preflight))

        if preflight.empty_domains:
            QMessageBox.information(
                self, "No complete candidate mapping", f"{preflight.empty_domains} source vertices have no candidates. Increase Candidate rho or Top-k before running a cost."
            )
            self._finish_job("Preflight stopped: empty candidate domains")
            return

        if preflight.estimated_state_upper_bound > _BLOCK_STATE_LIMIT:
            QMessageBox.warning(
                self,
                "Run blocked",
                f"The current parameters estimate {preflight.estimated_state_upper_bound:,} bag states.\n\n"
                f"The default safety limit is {_BLOCK_STATE_LIMIT:,}. Reduce Candidate rho or Top-k.",
            )
            self._finish_job("Preflight blocked an oversized exact-DP run")
            return

        if preflight.estimated_state_upper_bound > _WARN_STATE_LIMIT:
            answer = QMessageBox.question(
                self,
                "Large exact-DP run",
                f"The current parameters estimate {preflight.estimated_state_upper_bound:,} bag states "
                f"and a largest bag product of {preflight.largest_candidate_product:,}.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer != QMessageBox.StandardButton.Yes:
                self._finish_job("Run cancelled after preflight warning")
                return

        self._launch_match_worker()

    @staticmethod
    def _format_preflight(preflight: MatchingPreflight) -> str:
        largest = "none" if preflight.largest_bag is None else str(tuple(sorted(preflight.largest_bag)))
        return (
            "preflight\n"
            f"candidate domains: {preflight.total_candidates} total, {preflight.empty_domains} empty\n"
            f"candidate range: {preflight.minimum_candidates}–{preflight.maximum_candidates}\n"
            f"estimated bag states: {preflight.estimated_state_upper_bound:,}\n"
            f"largest bag product: {preflight.largest_candidate_product:,}\n"
            f"largest bag: {largest}"
        )

    def _launch_match_worker(self) -> None:
        if self._pending_run is None or self._active_token is None:
            self._finish_job("No pending run")
            return

        paths, costs = self._pending_run
        worker = MatchWorker(
            self.repository,
            self.sessions,
            paths[0],
            paths[1],
            candidate_rho=self.candidate_rho.value(),
            top_k=self.top_k.value(),
            costs=costs,
            cancellation_token=self._active_token,
        )
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.result.connect(self._match_ready)
        worker.signals.failed.connect(self._worker_failed_and_finish)
        worker.signals.cancelled.connect(self._worker_cancelled_and_finish)
        worker.signals.finished.connect(self._match_worker_finished)
        self._active_worker = worker
        self.thread_pool.start(worker)

    def _cancel_active(self) -> None:
        if self._active_token is None:
            return

        self._active_token.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancellation requested; stopping at the next safe checkpoint")

    def _worker_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _match_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, RunOutcome):
            return

        outcome = raw_outcome
        self._outcomes[self._outcome_key(outcome)] = outcome

        if outcome.cost_name == self.current_cost_name and self._display_mode == "computed":
            self._display_outcome(outcome)

        timing = "cached" if outcome.from_cache else f"{outcome.elapsed_seconds:.3f} s"
        self.status_label.setText(f"{outcome.cost_name.replace('_', ' ')} ready ({timing})")

    def _worker_failed(self, traceback_text: str) -> None:
        self.details.setPlainText(traceback_text)
        QMessageBox.critical(self, "River matcher failed", traceback_text.splitlines()[-1] if traceback_text else "Unknown error")

    def _worker_failed_and_finish(self, traceback_text: str) -> None:
        self._worker_failed(traceback_text)
        self._finish_job("Run failed")

    def _worker_cancelled_and_finish(self, message: str) -> None:
        self._finish_job(message or "Matching cancelled")

    def _match_worker_finished(self) -> None:
        # The result signal is emitted before finished, so preserve the useful
        # '<cost> ready (<seconds>)' status while re-enabling the controls.
        status = self.status_label.text()
        self._active_worker = None

        if self._busy:
            self._finish_job(status)

    def _finish_job(self, status: str) -> None:
        if not self._busy:
            return

        self._pending_run = None
        self._active_token = None
        self._active_worker = None
        self._set_busy(False)
        self.status_label.setText(status)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.run_button.setEnabled(not busy)
        self.compute_all_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.source_combo.setEnabled(not busy)
        self.target_combo.setEnabled(not busy)
        self.source_browse.setEnabled(not busy)
        self.target_browse.setEnabled(not busy)
        self.cost_combo.setEnabled(not busy)
        self.aggregation_combo.setEnabled(not busy)
        self.mapping_mode_combo.setEnabled(not busy)
        self.cost_options.setEnabled(not busy)
        self.candidate_rho.setEnabled(not busy)
        self.top_k.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
        self.score_mapping_button.setEnabled(not busy and self._display_mode == "imported" and self._imported_pair_is_current())
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0 if busy else 1)

    def _selected_solution(self, result: BothMatchResult) -> MatchSolution | None:
        return result.bottleneck if str(self.aggregation_combo.currentData()) == Objective.BOTTLENECK.value else result.additive

    @staticmethod
    def _format_timing(result: BothMatchResult) -> str:
        timing = result.timing

        if timing is None:
            return ""

        feasibility = "cached" if timing.feasibility_reused else f"{timing.feasibility_dp_seconds:.3f} s"
        return (
            f"\ntiming: arc consistency {timing.arc_consistency_seconds:.3f} s, feasibility DP {feasibility}, "
            f"cost setup {timing.cost_setup_seconds:.3f} s, cost DP {timing.cost_dp_seconds:.3f} s, "
            f"materialization {timing.materialization_seconds:.3f} s\n"
            f"local costs: {timing.uncached_local_cost_calls:,} uncached in {timing.uncached_local_cost_seconds:.3f} s; "
            f"guided adjacency {timing.witness_adjacency_seconds:.3f} s / {timing.witness_adjacency_builds:,} builds; "
            f"Dijkstra {timing.witness_dijkstra_seconds:.3f} s / {timing.witness_dijkstra_runs:,} runs"
        )

    def _display_outcome(self, outcome: RunOutcome) -> None:
        self._current_outcome = outcome
        result = outcome.result
        solution = self._selected_solution(result)
        self.source_view.set_graph(outcome.source, title=f"Source: {outcome.source.name} — {len(outcome.source.vertices)} V / {len(outcome.source.edges)} E")
        self.target_view.set_graph(outcome.target, title=f"Target: {outcome.target.name} — {len(outcome.target.vertices)} V / {len(outcome.target.edges)} E")
        effective = result.effective_candidate_statistics or result.candidate_statistics
        compatibility = result.compatibility_statistics
        pruning = 0 if compatibility is None else compatibility.removed_candidates
        preflight = result.effective_preflight or result.preflight
        estimate = 0 if preflight is None else preflight.estimated_state_upper_bound

        if solution is None:
            self.target_view.clear_result_overlays()
            self.score_label.setText("Score: infeasible")
            self.details.setPlainText(
                f"cost: {outcome.cost_name}\n"
                f"aggregation: {self.aggregation_combo.currentData()}\n"
                "No globally feasible mapping exists for these candidate domains.\n"
                f"arc-consistency removed: {pruning} candidates\n"
                f"effective candidates: {effective.total_candidates}, empty domains: {effective.empty_domains}\n"
                f"effective state estimate: {estimate:,}"
                f"{self._format_timing(result)}"
            )
            self.save_button.setEnabled(True)
            return

        self.target_view.set_witnesses(solution.edges)
        self.score_label.setText(f"Optimal {solution.objective.value}: {solution.value:.12g}")
        dp = result.dp_statistics
        self.details.setPlainText(
            f"cost: {outcome.cost_name}\n"
            f"aggregation: {solution.objective.value}\n"
            f"optimal score: {solution.value:.12g}\n"
            f"mapping: {len(solution.mapping)} source vertices\n"
            f"witnesses: {len(solution.edges)} source edges\n"
            f"candidate pruning: {result.candidate_statistics.total_candidates} → {effective.total_candidates} "
            f"({pruning} removed)\n"
            f"effective state estimate: {estimate:,}\n"
            f"DP: {dp.enumerated_states:,} complete states, {dp.partial_assignments:,} partial assignments, "
            f"{dp.message_entries:,} messages, {dp.unique_cost_requests:,} unique local costs\n"
            f"computation: {'cache hit' if outcome.from_cache else f'{outcome.elapsed_seconds:.3f} s'}"
            f"{self._format_timing(result)}"
        )
        self.save_button.setEnabled(True)

    def _saved_import_matches_current_cost(self) -> bool:
        imported = self._imported_mapping

        if imported is None or imported.saved_cost_name != self.current_cost_name:
            return False

        current = json.dumps(self.cost_options.options_for(self.current_cost_name), sort_keys=True, separators=(",", ":"), allow_nan=False)
        saved = json.dumps(dict(imported.saved_cost_options), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return current == saved

    def _display_imported_mapping(self) -> None:
        imported = self._imported_mapping

        if imported is None:
            self.score_label.setText("Score: —")
            self.details.setPlainText("No imported mapping is loaded.")
            self.score_mapping_button.setEnabled(False)
            return

        if not self._imported_pair_is_current():
            self.score_label.setText("Score: —")
            self.details.setPlainText("The imported mapping belongs to a different graph pair.")
            self.score_mapping_button.setEnabled(False)
            return

        source = self.repository.load(imported.source_path).graph
        target = self.repository.load(imported.target_path).graph

        if self.source_view.graph is not source:
            self.source_view.set_graph(source, title=f"Source: {source.name} — {len(source.vertices)} V / {len(source.edges)} E")
        if self.target_view.graph is not target:
            self.target_view.set_graph(target, title=f"Target: {target.name} — {len(target.vertices)} V / {len(target.edges)} E")

        outcome = self._current_imported_score()
        objective = str(self.aggregation_combo.currentData())

        if outcome is not None:
            evaluation = outcome.evaluation
            self.target_view.set_witnesses(evaluation.edges)
            value = evaluation.bottleneck_value if objective == Objective.BOTTLENECK.value else evaluation.additive_value
            value_text = "infeasible" if not math.isfinite(value) else f"{value:.12g}"
            self.score_label.setText(f"Imported φ {objective}: {value_text}")
            invalid = "none" if not evaluation.invalid_edge_ids else ", ".join(f"e{edge}" for edge in evaluation.invalid_edge_ids[:20])
            if len(evaluation.invalid_edge_ids) > 20:
                invalid += f", … ({len(evaluation.invalid_edge_ids)} total)"
            candidate_note = (
                f"{len(evaluation.candidate_violations)} source vertices lie outside the current candidate domains"
                if evaluation.candidate_violations
                else "all mapped vertices lie inside the current candidate domains"
            )
            timing = evaluation.timing
            timing_text = ""
            if timing is not None:
                timing_text = (
                    f"\nevaluation timing: {outcome.elapsed_seconds:.3f} s total; "
                    f"{timing.uncached_local_cost_calls:,} uncached local costs in "
                    f"{timing.uncached_local_cost_seconds:.3f} s; "
                    f"guided adjacency {timing.witness_adjacency_seconds:.3f} s; "
                    f"Dijkstra {timing.witness_dijkstra_seconds:.3f} s"
                )
            self.details.setPlainText(
                f"displayed mapping: imported φ from {imported.json_path.name}\n"
                f"mapping: {len(imported.mapping)} source vertices\n"
                f"scored with: {outcome.cost_name}\n"
                f"additive score: {evaluation.additive_value:.12g}\n"
                f"bottleneck score: {evaluation.bottleneck_value:.12g}\n"
                f"valid witness for every source edge: {'yes' if evaluation.feasible else 'no'}\n"
                f"invalid edges: {invalid}\n"
                f"candidate-domain check: {candidate_note}"
                f"{timing_text}"
            )
        elif self._saved_import_matches_current_cost() and imported.saved_edges:
            self.target_view.set_witnesses(imported.saved_edges)
            value = imported.saved_bottleneck_value if objective == Objective.BOTTLENECK.value else imported.saved_additive_value
            value_text = "—" if value is None else f"{value:.12g}"
            self.score_label.setText(f"Imported φ {objective}: {value_text}")
            self.details.setPlainText(
                f"displayed mapping: imported φ from {imported.json_path.name}\n"
                f"mapping: {len(imported.mapping)} source vertices\n"
                f"saved cost: {imported.saved_cost_name}\n"
                f"saved mapping aggregation: {imported.saved_objective or 'unknown'}\n"
                f"additive score from saved local costs: "
                f"{imported.saved_additive_value if imported.saved_additive_value is not None else 'unknown'}\n"
                f"bottleneck score from saved local costs: "
                f"{imported.saved_bottleneck_value if imported.saved_bottleneck_value is not None else 'unknown'}\n"
                "The displayed witnesses are the witnesses stored in the JSON. "
                "Click 'Score imported φ' to recompute them under the selected cost and options."
            )
        else:
            self.target_view.clear_result_overlays()
            self.score_label.setText("Imported φ: not scored for this cost")
            self.details.setPlainText(
                f"displayed mapping: imported φ from {imported.json_path.name}\n"
                f"mapping: {len(imported.mapping)} source vertices\n"
                f"selected cost: {self.current_cost_name}\n"
                "Click 'Score imported φ' to evaluate this fixed mapping. "
                "The optimizer will not run and the mapping will not change."
            )

        self.score_mapping_button.setEnabled(not self._busy)
        self.save_button.setEnabled(False)

    def _displayed_mapping_and_edges(self) -> tuple[Mapping[int, int], tuple[MatchedEdge, ...]] | None:
        if self._display_mode == "imported":
            imported = self._imported_mapping
            if imported is None or not self._imported_pair_is_current():
                return None
            outcome = self._current_imported_score()
            if outcome is not None:
                return imported.mapping, outcome.evaluation.edges
            edges = imported.saved_edges if self._saved_import_matches_current_cost() else ()
            return imported.mapping, edges

        if self._current_outcome is None:
            return None
        solution = self._selected_solution(self._current_outcome.result)
        if solution is None:
            return None
        return solution.mapping, solution.edges

    def _source_vertex_selected(self, source_vertex: int) -> None:
        displayed = self._displayed_mapping_and_edges()

        if displayed is None:
            return

        mapping, edges = displayed
        target_vertex = mapping.get(source_vertex)

        if target_vertex is None:
            return

        self.source_view.highlight_vertices([source_vertex])
        self.target_view.highlight_vertices([target_vertex])
        incident = [edge for edge in edges if edge.source_u == source_vertex or edge.source_v == source_vertex]

        if incident:
            self.source_view.highlight_edge(incident[0].edge_id)
            self.target_view.highlight_edge(incident[0].edge_id, witness=True)

        self.details.appendPlainText(f"\nsource vertex {source_vertex} maps to target vertex {target_vertex}")

    def _target_vertex_selected(self, target_vertex: int) -> None:
        displayed = self._displayed_mapping_and_edges()

        if displayed is None:
            return

        mapping, _ = displayed
        sources = sorted(source for source, target in mapping.items() if target == target_vertex)
        self.target_view.highlight_vertices([target_vertex])
        self.source_view.highlight_vertices(sources)
        self.details.appendPlainText(f"\ntarget vertex {target_vertex} receives source vertices {sources or 'none'}")

    def _source_edge_selected(self, edge_id: int, fraction: float) -> None:
        displayed = self._displayed_mapping_and_edges()

        if displayed is None:
            return

        _, edges = displayed
        edge = next((item for item in edges if item.edge_id == edge_id), None)

        if edge is None:
            self.details.appendPlainText(f"\nsource edge e{edge_id}: no witness is currently loaded for the displayed mapping and cost")
            return

        self.source_view.highlight_edge(edge_id)
        self.source_view.highlight_fraction(edge_id, fraction)
        self.target_view.highlight_edge(edge_id, witness=True)
        self.target_view.highlight_fraction(edge_id, fraction, witness=True)
        self.details.appendPlainText(f"\nsource edge e{edge_id} at {fraction:.1%} maps along its witness at {fraction:.1%}; local cost={edge.cost:.12g}")

    def _target_witness_selected(self, edge_id: int, fraction: float) -> None:
        displayed = self._displayed_mapping_and_edges()

        if displayed is None:
            return

        _, edges = displayed
        edge = next((item for item in edges if item.edge_id == edge_id), None)

        if edge is None:
            return

        self.target_view.highlight_edge(edge_id, witness=True)
        self.target_view.highlight_fraction(edge_id, fraction, witness=True)
        self.source_view.highlight_edge(edge_id)
        self.source_view.highlight_fraction(edge_id, fraction)
        self.details.appendPlainText(f"\nwitness for source edge e{edge_id} selected at {fraction:.1%}")

    def _save_result(self) -> None:
        if self._current_outcome is None:
            return

        outcome = self._current_outcome
        default = Path.cwd() / "results" / f"{outcome.source_path.stem}_to_{outcome.target_path.stem}_{outcome.cost_name}.json"
        filename, _ = QFileDialog.getSaveFileName(self, "Save match result", str(default), "JSON files (*.json)")

        if not filename:
            return

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_result_payload(outcome), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.status_label.setText(f"Saved {path}")

    @staticmethod
    def _parse_mapping(raw_mapping: object) -> dict[int, int]:
        mapping: dict[int, int] = {}

        if isinstance(raw_mapping, Mapping):
            entries = raw_mapping.items()
        elif isinstance(raw_mapping, Sequence) and not isinstance(raw_mapping, (str, bytes, bytearray)):
            entries = []
            for entry in raw_mapping:
                if not isinstance(entry, Mapping):
                    raise ValueError("Every mapping entry must be an object.")
                entries.append((entry.get("source_vertex"), entry.get("target_vertex")))
        else:
            raise ValueError("The JSON does not contain a mapping object or mapping-entry list.")

        for raw_source, raw_target in entries:
            source = int(raw_source)
            target = int(raw_target)
            if source in mapping:
                raise ValueError(f"Duplicate source vertex {source} in imported mapping.")
            mapping[source] = target

        if not mapping:
            raise ValueError("The imported mapping is empty.")

        return {source: mapping[source] for source in sorted(mapping)}

    @staticmethod
    def _parse_saved_edges(raw_edges: object) -> tuple[MatchedEdge, ...]:
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes, bytearray)):
            return ()

        edges: list[MatchedEdge] = []
        for raw in raw_edges:
            if not isinstance(raw, Mapping):
                continue
            witness = np.asarray(raw.get("witness", ()), dtype=np.float64)
            if witness.ndim != 2 or witness.shape[1:] != (2,) or len(witness) < 2:
                continue
            edges.append(
                MatchedEdge(
                    edge_id=int(raw["edge_id"]),
                    source_u=int(raw["source_u"]),
                    source_v=int(raw["source_v"]),
                    target_u=int(raw["target_u"]),
                    target_v=int(raw["target_v"]),
                    cost=float(raw["cost"]),
                    witness=np.ascontiguousarray(witness),
                )
            )
        return tuple(sorted(edges, key=lambda edge: edge.edge_id))

    def _resolve_import_graph(self, metadata: Mapping[str, object], role: str, json_path: Path) -> Path:
        raw_path = metadata.get("path")
        if raw_path:
            path = Path(str(raw_path)).expanduser()
            if path.is_file():
                return path.resolve()

        expected_name = str(metadata.get("name", ""))
        expected_file = Path(str(raw_path)).name if raw_path else ""
        expected_vertices = metadata.get("vertices")
        expected_edges = metadata.get("edges")
        matches: list[Path] = []

        for info in self._graph_catalog.values():
            name_match = (expected_file and info.path.name == expected_file) or (expected_name and info.name == expected_name)
            count_match = (expected_vertices is None or info.vertices == int(expected_vertices)) and (expected_edges is None or info.edges == int(expected_edges))
            if name_match and count_match:
                matches.append(info.path)

        if len(matches) == 1:
            return matches[0].resolve()

        filename, _ = QFileDialog.getOpenFileName(self, f"Locate imported {role} graph", str(json_path.parent), "Graph exports (*.txt);;All files (*)")
        if not filename:
            raise ValueError(f"Could not locate the imported {role} graph.")
        return Path(filename).resolve()

    def _import_result(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import match result", str(Path.cwd() / "results"), "JSON files (*.json);;All files (*)")
        if not filename:
            return

        json_path = Path(filename).resolve()
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("The JSON root must be an object.")

            source_meta = payload.get("source")
            target_meta = payload.get("target")
            if not isinstance(source_meta, Mapping) or not isinstance(target_meta, Mapping):
                raise ValueError("The JSON must contain source and target graph metadata.")

            solutions = payload.get("solutions")
            chosen_objective: str | None = None
            chosen_solution: Mapping[str, object] | None = None

            if isinstance(solutions, Mapping):
                available: list[tuple[str, Mapping[str, object]]] = []
                for objective in (Objective.ADDITIVE.value, Objective.BOTTLENECK.value):
                    raw_solution = solutions.get(objective)
                    if isinstance(raw_solution, Mapping) and bool(raw_solution.get("feasible", False)):
                        available.append((objective, raw_solution))

                if not available:
                    raise ValueError("The result JSON contains no feasible mapping.")

                if len(available) == 1:
                    chosen_objective, chosen_solution = available[0]
                else:
                    labels = [f"{objective} — score {float(solution.get('value', math.nan)):.12g}" for objective, solution in available]
                    preferred = str(self.aggregation_combo.currentData())
                    default_index = next((i for i, item in enumerate(available) if item[0] == preferred), 0)
                    selected, accepted = QInputDialog.getItem(
                        self, "Choose mapping", "The JSON contains two optimized mappings. Which φ should be imported?", labels, default_index, False
                    )
                    if not accepted:
                        return
                    chosen_objective, chosen_solution = available[labels.index(selected)]

                raw_mapping = chosen_solution.get("mapping")
                raw_edges = chosen_solution.get("edges", ())
                saved_value = float(chosen_solution.get("value", math.nan))
            else:
                raw_mapping = payload.get("mapping")
                raw_edges = payload.get("edges", ())
                saved_value = None

            mapping = self._parse_mapping(raw_mapping)
            saved_edges = self._parse_saved_edges(raw_edges)
            source_path = self._resolve_import_graph(source_meta, "source", json_path)
            target_path = self._resolve_import_graph(target_meta, "target", json_path)
            source_loaded = self.repository.load(source_path)
            target_loaded = self.repository.load(target_path)
            normalized_source, normalized_target, swapped = normalize_sparse_to_dense(source_loaded, target_loaded)

            if swapped:
                raise ValueError("The imported JSON identifies the denser graph as the source; this UI only supports sparse-to-dense mappings.")

            source_vertices = set(normalized_source.graph.vertices)
            target_vertices = set(normalized_target.graph.vertices)
            if set(mapping) != source_vertices:
                missing = sorted(source_vertices - set(mapping))
                extra = sorted(set(mapping) - source_vertices)
                raise ValueError(f"Imported mapping does not match the source graph; missing={missing}, extra={extra}.")
            unknown_targets = sorted(set(mapping.values()) - target_vertices)
            if unknown_targets:
                raise ValueError(f"Imported mapping contains unknown target vertices {unknown_targets}.")

            self._graph_catalog[source_path] = GraphInfo(source_path, normalized_source.graph.name, len(normalized_source.graph.vertices), len(normalized_source.graph.edges))
            self._graph_catalog[target_path] = GraphInfo(target_path, normalized_target.graph.name, len(normalized_target.graph.vertices), len(normalized_target.graph.edges))

            cost_payload = payload.get("cost")
            saved_cost_name = None
            saved_cost_options: Mapping[str, object] = {}
            if isinstance(cost_payload, Mapping):
                raw_name = cost_payload.get("name")
                saved_cost_name = None if raw_name is None else str(raw_name)
                raw_options = cost_payload.get("options", {})
                if isinstance(raw_options, Mapping):
                    saved_cost_options = dict(raw_options)

            complete_saved_edges = len(saved_edges) == len(normalized_source.graph.edges)
            saved_additive = sum(edge.cost for edge in saved_edges) if complete_saved_edges else None
            saved_bottleneck = max((edge.cost for edge in saved_edges), default=0.0) if complete_saved_edges else None
            imported = ImportedMapping(
                json_path=json_path,
                source_path=source_path,
                target_path=target_path,
                mapping=mapping,
                saved_cost_name=saved_cost_name,
                saved_cost_options=saved_cost_options,
                saved_objective=chosen_objective,
                saved_value=saved_value,
                saved_edges=saved_edges,
                saved_additive_value=saved_additive,
                saved_bottleneck_value=saved_bottleneck,
            )
            self._imported_mapping = imported

            candidate_parameters = payload.get("candidate_parameters")
            if isinstance(candidate_parameters, Mapping):
                if "rho" in candidate_parameters:
                    self.candidate_rho.setValue(float(candidate_parameters["rho"]))
                if "top_k" in candidate_parameters:
                    self.top_k.setValue(int(candidate_parameters["top_k"]))

            self._populate_source_choices(preferred_path=source_path)
            if not self._select_path(self.source_combo, source_path):
                raise ValueError("The imported source graph cannot be selected as a sparse source.")
            self._rebuild_target_choices(preferred_path=target_path)
            if not self._select_path(self.target_combo, target_path):
                raise ValueError("The imported target graph is not denser than the imported source graph.")

            if saved_cost_name is not None and self._select_data(self.cost_combo, saved_cost_name):
                self.cost_options.set_cost(saved_cost_name)
                self.cost_options.set_options(saved_cost_name, saved_cost_options)

            index = self.mapping_mode_combo.findData("imported")
            if index < 0:
                self.mapping_mode_combo.addItem(imported.label, "imported")
            else:
                self.mapping_mode_combo.setItemText(index, imported.label)
            self._select_data(self.mapping_mode_combo, "imported")
            self._display_imported_mapping()
            self.status_label.setText(f"Imported mapping from {json_path.name}")
        except Exception as error:
            QMessageBox.critical(self, "Could not import result", str(error))

    def _score_imported_mapping(self) -> None:
        imported = self._imported_mapping
        paths = self._selected_paths()
        if imported is None or paths is None or not self._imported_pair_is_current() or self._busy:
            return

        token = CancellationToken()
        self._active_token = token
        self._pending_run = None
        self._set_busy(True)
        worker = MappingScoreWorker(
            self.repository,
            self.sessions,
            paths[0],
            paths[1],
            candidate_rho=self.candidate_rho.value(),
            top_k=self.top_k.value(),
            mapping=imported.mapping,
            cost_name=self.current_cost_name,
            cost_options=self.cost_options.options_for(self.current_cost_name),
            cancellation_token=token,
        )
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.result.connect(self._mapping_score_ready)
        worker.signals.failed.connect(self._worker_failed_and_finish)
        worker.signals.cancelled.connect(self._worker_cancelled_and_finish)
        worker.signals.finished.connect(self._match_worker_finished)
        self._active_worker = worker
        self.thread_pool.start(worker)

    def _mapping_score_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, MappingScoreOutcome):
            return

        options = json.dumps(dict(raw_outcome.cost_options), sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = (
            str(raw_outcome.source_path.resolve()),
            str(raw_outcome.target_path.resolve()),
            raw_outcome.cost_name,
            options,
            tuple(sorted((int(source), int(target)) for source, target in raw_outcome.mapping.items())),
        )
        self._mapping_scores[key] = raw_outcome
        if self._display_mode == "imported":
            self._display_imported_mapping()
        timing = "cached" if raw_outcome.from_cache else f"{raw_outcome.elapsed_seconds:.3f} s"
        self.status_label.setText(f"Imported mapping scored with {raw_outcome.cost_name.replace('_', ' ')} ({timing})")

    def _clear_caches(self) -> None:
        if self._busy:
            self._cancel_active()
            return

        self.repository.clear()
        self.sessions.clear()
        self._outcomes.clear()
        self._mapping_scores.clear()
        self._current_outcome = None
        self.save_button.setEnabled(False)
        self.score_label.setText("Score: —")
        self.status_label.setText("Caches cleared")
        self._schedule_preview()

    def closeEvent(self, event: object) -> None:
        if self._active_token is not None:
            self._active_token.cancel()

        self.settings.setValue("source_path", str(self.source_combo.currentData() or ""))
        self.settings.setValue("target_path", str(self.target_combo.currentData() or ""))
        self.settings.setValue("candidate_rho", self.candidate_rho.value())
        self.settings.setValue("top_k", self.top_k.value())
        self.settings.setValue("cost", self.current_cost_name)
        self.settings.setValue("aggregation", str(self.aggregation_combo.currentData()))
        getattr(event, "accept")()


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("River Graph Matcher")
    application.setOrganizationName("GraphThesis")
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
