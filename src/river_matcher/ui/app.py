from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import QLocale, QSettings, QThreadPool, QTimer, Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QPlainTextEdit, QProgressBar, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from river_matcher.cancellation import CancellationToken
from river_matcher.costs import available_costs
from river_matcher.dynamic_programming import Objective
from river_matcher.matcher import BothMatchResult, MatchSolution
from river_matcher.preflight import MatchingPreflight
from river_matcher.ui.widgets import CostOptionsWidget, GraphView
from river_matcher.ui.workers import (CatalogOutcome, CatalogWorker, GraphInfo, GraphRepository, MatchWorker, PairSessionStore, PreflightOutcome, PreflightWorker, PreviewOutcome,
                                      PreviewWorker, RunOutcome)

_WARN_STATE_LIMIT = 2_000_000
_BLOCK_STATE_LIMIT = 10_000_000


def _find_graph_directory() -> Path:
    candidates = (Path.cwd() / "GraphExport", Path(__file__).resolve().parents[3] / "GraphExport",)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return Path.cwd()


def _solution_payload(solution: MatchSolution | None) -> dict[str, object]:
    if solution is None:
        return {"feasible": False}

    return {"feasible": True, "objective": solution.objective.value, "value": float(solution.value),
            "mapping": [{"source_vertex": int(source), "target_vertex": int(target)} for source, target in sorted(solution.mapping.items())], "edges": [
            {"edge_id": int(edge.edge_id), "source_u": int(edge.source_u), "source_v": int(edge.source_v), "target_u": int(edge.target_u), "target_v": int(edge.target_v),
             "cost": float(edge.cost), "witness": edge.witness.tolist()} for edge in solution.edges]}


def _preflight_payload(preflight: MatchingPreflight | None) -> dict[str, object] | None:
    if preflight is None:
        return None

    return {"empty_domains": preflight.empty_domains, "total_candidates": preflight.total_candidates, "minimum_candidates": preflight.minimum_candidates,
            "maximum_candidates": preflight.maximum_candidates, "estimated_state_upper_bound": preflight.estimated_state_upper_bound,
            "largest_candidate_product": preflight.largest_candidate_product, "largest_bag": None if preflight.largest_bag is None else sorted(preflight.largest_bag)}


def _result_payload(outcome: RunOutcome) -> dict[str, object]:
    result = outcome.result
    candidates = result.candidate_statistics
    effective = result.effective_candidate_statistics or candidates
    decomposition = result.decomposition
    dp = result.dp_statistics
    compatibility = result.compatibility_statistics

    return {"schema_version": 2,
            "source": {"path": str(outcome.source_path), "name": outcome.source.name, "vertices": len(outcome.source.vertices), "edges": len(outcome.source.edges)},
            "target": {"path": str(outcome.target_path), "name": outcome.target.name, "vertices": len(outcome.target.vertices), "edges": len(outcome.target.edges)},
            "cost": {"name": outcome.cost_name, "options": dict(outcome.cost_options)}, "candidate_parameters": {"rho": outcome.candidate_rho, "top_k": outcome.top_k},
            "candidate_statistics": {"source_vertices": candidates.source_vertices, "empty_domains": candidates.empty_domains, "total_candidates": candidates.total_candidates,
                                     "minimum_candidates": candidates.minimum_candidates, "maximum_candidates": candidates.maximum_candidates},
            "effective_candidate_statistics": {"source_vertices": effective.source_vertices, "empty_domains": effective.empty_domains,
                                               "total_candidates": effective.total_candidates, "minimum_candidates": effective.minimum_candidates,
                                               "maximum_candidates": effective.maximum_candidates},
            "candidate_sets": {str(vertex): list(values) for vertex, values in result.candidate_sets.items()},
            "effective_candidate_sets": {str(vertex): list(values) for vertex, values in (result.effective_candidate_sets or result.candidate_sets).items()},
            "preflight": _preflight_payload(result.preflight), "effective_preflight": _preflight_payload(result.effective_preflight),
            "compatibility": None if compatibility is None else {"initial_candidates": compatibility.initial_candidates, "remaining_candidates": compatibility.remaining_candidates,
                                                                 "removed_candidates": compatibility.removed_candidates, "revised_arcs": compatibility.revised_arcs,
                                                                 "empty_domains": compatibility.empty_domains},
            "decomposition": {"width": decomposition.width, "maximum_bag_size": decomposition.maximum_bag_size, "bag_count": decomposition.bag_count,
                              "heuristic": decomposition.heuristic.value, "minimum_fill_width": decomposition.minimum_fill_width,
                              "minimum_degree_width": decomposition.minimum_degree_width},
            "dynamic_programming": {"enumerated_states": dp.enumerated_states, "feasible_states": dp.feasible_states, "message_entries": dp.message_entries,
                                    "unique_cost_requests": dp.unique_cost_requests, "partial_assignments": dp.partial_assignments},
            "solutions": {Objective.ADDITIVE.value: _solution_payload(result.additive), Objective.BOTTLENECK.value: _solution_payload(result.bottleneck)}}


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
        self._active_token: CancellationToken | None = None
        # Keep the QRunnable alive until its queued finished signal reaches the UI thread.
        self._active_worker: PreflightWorker | MatchWorker | None = None
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

        self.objective_combo = QComboBox()
        self.objective_combo.addItem("Additive", Objective.ADDITIVE.value)
        self.objective_combo.addItem("Bottleneck", Objective.BOTTLENECK.value)

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
        return (str(outcome.source_path.resolve()), str(outcome.target_path.resolve()), float(outcome.candidate_rho), int(outcome.top_k), outcome.cost_name, options,)

    def _current_key(self, cost_name: str) -> tuple[str, str, float, int, str, str] | None:
        paths = self._selected_paths()

        if paths is None:
            return None

        options = json.dumps(self.cost_options.options_for(cost_name), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (str(paths[0].resolve()), str(paths[1].resolve()), float(self.candidate_rho.value()), int(self.top_k.value()), cost_name, options,)

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
        parameters.addWidget(QLabel("Displayed objective"), 0, 2)
        parameters.addWidget(self.objective_combo, 0, 3)
        parameters.addWidget(QLabel("Candidate rho"), 1, 0)
        parameters.addWidget(self.candidate_rho, 1, 1)
        parameters.addWidget(QLabel("Top-k candidates"), 1, 2)
        parameters.addWidget(self.top_k, 1, 3)
        parameters.addWidget(self.cost_options, 2, 0, 1, 4)

        controls = QHBoxLayout()
        controls.addWidget(files_group, 2)
        controls.addWidget(parameters_group, 3)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.compute_all_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)
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
        self.objective_combo.currentIndexChanged.connect(self._objective_changed)
        self.run_button.clicked.connect(self._run_selected)
        self.compute_all_button.clicked.connect(self._compute_all)
        self.cancel_button.clicked.connect(self._cancel_active)
        self.save_button.clicked.connect(self._save_result)
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

        graphs = sorted((graph for graph in self._graph_catalog.values() if any(other.vertices > graph.vertices for other in self._graph_catalog.values())),
                        key=lambda graph: (graph.vertices, graph.path.name.lower()))
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
        self._select_data(self.objective_combo, str(self.settings.value("objective", Objective.ADDITIVE.value)))

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
        valid.sort(key=lambda graph: (self._graph_group(graph.path) != source_group, graph.vertices - source.vertices, graph.path.name.lower(),))
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
            self.direction_label.setText(f"Sparse-to-dense direction: {source.path.name} ({source.vertices} V) → "
                                         f"{target.path.name} ({target.vertices} V)")
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
        self.details.setPlainText("Graphs loaded. Run a cost to compute a mapping.")

    def _cost_changed(self) -> None:
        self.cost_options.set_cost(self.current_cost_name)
        self.settings.setValue("cost", self.current_cost_name)
        key = self._current_key(self.current_cost_name)
        cached = None if key is None else self._outcomes.get(key)

        if cached is not None:
            self._display_outcome(cached)
        else:
            self.status_label.setText("This cost has not been computed for the current pair and parameters.")

    def _objective_changed(self) -> None:
        self.settings.setValue("objective", str(self.objective_combo.currentData()))

        if self._current_outcome is not None:
            self._display_outcome(self._current_outcome)

    def _run_selected(self) -> None:
        paths = self._selected_paths()

        if paths is None:
            return

        cost_name = self.current_cost_name
        self._begin_run(paths, ((cost_name, self.cost_options.options_for(cost_name)),))

    def _compute_all(self) -> None:
        paths = self._selected_paths()

        if paths is None:
            return

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
            QMessageBox.information(self, "No complete candidate mapping", f"{preflight.empty_domains} source vertices have no candidates. "
                                                                           "Increase Candidate rho or Top-k before running a cost.")
            self._finish_job("Preflight stopped: empty candidate domains")
            return

        if preflight.estimated_state_upper_bound > _BLOCK_STATE_LIMIT:
            QMessageBox.warning(self, "Run blocked", f"The current parameters estimate {preflight.estimated_state_upper_bound:,} bag states.\n\n"
                                                     f"The default safety limit is {_BLOCK_STATE_LIMIT:,}. Reduce Candidate rho or Top-k.")
            self._finish_job("Preflight blocked an oversized exact-DP run")
            return

        if preflight.estimated_state_upper_bound > _WARN_STATE_LIMIT:
            answer = QMessageBox.question(self, "Large exact-DP run", f"The current parameters estimate {preflight.estimated_state_upper_bound:,} bag states "
                                                                      f"and a largest bag product of {preflight.largest_candidate_product:,}.\n\nContinue?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

            if answer != QMessageBox.StandardButton.Yes:
                self._finish_job("Run cancelled after preflight warning")
                return

        self._launch_match_worker()

    @staticmethod
    def _format_preflight(preflight: MatchingPreflight) -> str:
        largest = "none" if preflight.largest_bag is None else str(tuple(sorted(preflight.largest_bag)))
        return ("preflight\n"
                f"candidate domains: {preflight.total_candidates} total, {preflight.empty_domains} empty\n"
                f"candidate range: {preflight.minimum_candidates}–{preflight.maximum_candidates}\n"
                f"estimated bag states: {preflight.estimated_state_upper_bound:,}\n"
                f"largest bag product: {preflight.largest_candidate_product:,}\n"
                f"largest bag: {largest}")

    def _launch_match_worker(self) -> None:
        if self._pending_run is None or self._active_token is None:
            self._finish_job("No pending run")
            return

        paths, costs = self._pending_run
        worker = MatchWorker(self.repository, self.sessions, paths[0], paths[1], candidate_rho=self.candidate_rho.value(), top_k=self.top_k.value(), costs=costs,
                             cancellation_token=self._active_token)
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

        if outcome.cost_name == self.current_cost_name:
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
        self.candidate_rho.setEnabled(not busy)
        self.top_k.setEnabled(not busy)
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0 if busy else 1)

    def _selected_solution(self, result: BothMatchResult) -> MatchSolution | None:
        return result.bottleneck if str(self.objective_combo.currentData()) == Objective.BOTTLENECK.value else result.additive

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
            self.details.setPlainText(f"cost: {outcome.cost_name}\n"
                                      f"objective: {self.objective_combo.currentData()}\n"
                                      "No globally feasible mapping exists for these candidate domains.\n"
                                      f"arc-consistency removed: {pruning} candidates\n"
                                      f"effective candidates: {effective.total_candidates}, empty domains: {effective.empty_domains}\n"
                                      f"effective state estimate: {estimate:,}")
            self.save_button.setEnabled(True)
            return

        self.target_view.set_witnesses(solution.edges)
        dp = result.dp_statistics
        self.details.setPlainText(f"cost: {outcome.cost_name}\n"
                                  f"objective: {solution.objective.value}\n"
                                  f"value: {solution.value:.12g}\n"
                                  f"mapping: {len(solution.mapping)} source vertices\n"
                                  f"witnesses: {len(solution.edges)} source edges\n"
                                  f"candidate pruning: {result.candidate_statistics.total_candidates} → {effective.total_candidates} "
                                  f"({pruning} removed)\n"
                                  f"effective state estimate: {estimate:,}\n"
                                  f"DP: {dp.enumerated_states:,} complete states, {dp.partial_assignments:,} partial assignments, "
                                  f"{dp.message_entries:,} messages, {dp.unique_cost_requests:,} unique local costs\n"
                                  f"computation: {'cache hit' if outcome.from_cache else f'{outcome.elapsed_seconds:.3f} s'}")
        self.save_button.setEnabled(True)

    def _source_vertex_selected(self, source_vertex: int) -> None:
        if self._current_outcome is None:
            return

        solution = self._selected_solution(self._current_outcome.result)

        if solution is None:
            return

        target_vertex = solution.mapping.get(source_vertex)

        if target_vertex is None:
            return

        self.source_view.highlight_vertices([source_vertex])
        self.target_view.highlight_vertices([target_vertex])
        incident = [edge for edge in solution.edges if edge.source_u == source_vertex or edge.source_v == source_vertex]

        if incident:
            self.source_view.highlight_edge(incident[0].edge_id)
            self.target_view.highlight_edge(incident[0].edge_id, witness=True)

        self.details.appendPlainText(f"\nsource vertex {source_vertex} maps to target vertex {target_vertex}")

    def _target_vertex_selected(self, target_vertex: int) -> None:
        if self._current_outcome is None:
            return

        solution = self._selected_solution(self._current_outcome.result)

        if solution is None:
            return

        sources = sorted(source for source, target in solution.mapping.items() if target == target_vertex)
        self.target_view.highlight_vertices([target_vertex])
        self.source_view.highlight_vertices(sources)
        self.details.appendPlainText(f"\ntarget vertex {target_vertex} receives source vertices {sources or 'none'}")

    def _source_edge_selected(self, edge_id: int, fraction: float) -> None:
        if self._current_outcome is None:
            return

        solution = self._selected_solution(self._current_outcome.result)

        if solution is None:
            return

        edge = next((item for item in solution.edges if item.edge_id == edge_id), None)

        if edge is None:
            return

        self.source_view.highlight_edge(edge_id)
        self.source_view.highlight_fraction(edge_id, fraction)
        self.target_view.highlight_edge(edge_id, witness=True)
        self.target_view.highlight_fraction(edge_id, fraction, witness=True)
        self.details.appendPlainText(f"\nsource edge e{edge_id} at {fraction:.1%} maps along its witness at {fraction:.1%}; "
                                     f"local cost={edge.cost:.12g}")

    def _target_witness_selected(self, edge_id: int, fraction: float) -> None:
        if self._current_outcome is None:
            return

        solution = self._selected_solution(self._current_outcome.result)

        if solution is None:
            return

        edge = next((item for item in solution.edges if item.edge_id == edge_id), None)

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

    def _clear_caches(self) -> None:
        if self._busy:
            self._cancel_active()
            return

        self.repository.clear()
        self.sessions.clear()
        self._outcomes.clear()
        self._current_outcome = None
        self.save_button.setEnabled(False)
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
        self.settings.setValue("objective", str(self.objective_combo.currentData()))
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
