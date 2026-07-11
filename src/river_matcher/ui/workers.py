from __future__ import annotations

import json
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from river_matcher.cancellation import CancellationToken, OperationCancelled
from river_matcher.candidates import CandidateMode, prepare_candidate_target
from river_matcher.costs.base import CostName
from river_matcher.matcher import BothMatchResult, MappingEvaluation, RiverGraphMatcher
from river_matcher.models import JunctionGraph
from river_matcher.preflight import MatchingPreflight
from river_matcher.preprocessing import load_embedded_graph, load_junction_graph


@dataclass(frozen=True, slots=True)
class GraphKey:
    path: Path
    size: int
    modified_time_ns: int
    variant: str


@dataclass(frozen=True, slots=True)
class LoadedGraph:
    key: GraphKey
    graph: JunctionGraph


@dataclass(frozen=True, slots=True)
class GraphInfo:
    path: Path
    name: str
    vertices: int
    edges: int


@dataclass(frozen=True, slots=True)
class CatalogOutcome:
    graphs: tuple[GraphInfo, ...]


@dataclass(frozen=True, slots=True)
class PairKey:
    source: GraphKey
    target: GraphKey
    candidate_rho: float
    top_k: int
    candidate_mode: CandidateMode
    subdivision_points: int
    adaptive_min_separation: float


@dataclass(frozen=True, slots=True)
class CostKey:
    name: str
    options_json: str


@dataclass(frozen=True, slots=True)
class PreviewOutcome:
    request_id: int
    source_path: Path
    target_path: Path
    source: JunctionGraph
    target: JunctionGraph
    swapped: bool


@dataclass(frozen=True, slots=True)
class PreflightOutcome:
    source_path: Path
    target_path: Path
    source: JunctionGraph
    target: JunctionGraph
    candidate_rho: float
    top_k: int
    candidate_mode: CandidateMode
    subdivision_points: int
    adaptive_min_separation: float
    preflight: MatchingPreflight


@dataclass(frozen=True, slots=True)
class RunOutcome:
    source_path: Path
    target_path: Path
    source: JunctionGraph
    target: JunctionGraph
    cost_name: str
    cost_options: Mapping[str, object]
    candidate_rho: float
    top_k: int
    candidate_mode: CandidateMode
    subdivision_points: int
    adaptive_min_separation: float
    result: BothMatchResult
    swapped: bool
    from_cache: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MappingScoreOutcome:
    source_path: Path
    target_path: Path
    source: JunctionGraph
    target: JunctionGraph
    cost_name: str
    cost_options: Mapping[str, object]
    candidate_mode: CandidateMode
    subdivision_points: int
    adaptive_min_separation: float
    mapping: Mapping[int, int]
    evaluation: MappingEvaluation
    from_cache: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MappingEvaluationKey:
    cost: CostKey
    mapping: tuple[tuple[int, int], ...]


class WorkerSignals(QObject):
    progress = Signal(str)
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    finished = Signal()


class GraphRepository:
    """Thread-safe file cache invalidated by size or modification time."""

    def __init__(self) -> None:
        self._cache: dict[GraphKey, LoadedGraph] = {}
        self._latest_by_path: dict[tuple[Path, str], GraphKey] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(path: Path, variant: str) -> GraphKey:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        return GraphKey(resolved, stat.st_size, stat.st_mtime_ns, variant)

    def load(self, path: Path, *, variant: str = "junction") -> LoadedGraph:
        if variant not in {"junction", "original"}:
            raise ValueError(f"Unknown graph variant {variant!r}")
        key = self._key(path, variant)

        with self._lock:
            cached = self._cache.get(key)

            if cached is not None:
                return cached

        graph = load_junction_graph(key.path) if variant == "junction" else load_embedded_graph(key.path)
        loaded = LoadedGraph(key, graph)

        with self._lock:
            latest_key = (key.path, variant)
            previous = self._latest_by_path.get(latest_key)

            if previous is not None and previous != key:
                self._cache.pop(previous, None)

            self._cache[key] = loaded
            self._latest_by_path[latest_key] = key

        return loaded

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._latest_by_path.clear()


def normalize_sparse_to_dense(first: LoadedGraph, second: LoadedGraph) -> tuple[LoadedGraph, LoadedGraph, bool]:
    first_vertices = len(first.graph.vertices)
    second_vertices = len(second.graph.vertices)

    if first_vertices == second_vertices:
        raise ValueError("The selected graphs have the same vertex count, so sparse-to-dense direction cannot be inferred.")

    if first_vertices < second_vertices:
        return first, second, False

    return second, first, True


def load_matching_pair(repository: GraphRepository, first_path: Path, second_path: Path, candidate_mode: CandidateMode | str) -> tuple[LoadedGraph, LoadedGraph, bool]:
    """Resolve direction using junction graphs, then load the requested target representation."""
    mode = CandidateMode(candidate_mode)
    first = repository.load(first_path)
    second = repository.load(second_path)
    source, target, swapped = normalize_sparse_to_dense(first, second)
    if mode is CandidateMode.ORIGINAL_TARGET_VERTICES:
        target = repository.load(target.key.path, variant="original")
    return source, target, swapped


def _canonical_options(options: Mapping[str, object]) -> str:
    return json.dumps(dict(options), sort_keys=True, separators=(",", ":"), allow_nan=False)


class PairSession:
    """Reusable matcher and result cache for one graph pair and candidate configuration."""

    def __init__(
        self,
        source: LoadedGraph,
        target: LoadedGraph,
        *,
        candidate_rho: float,
        top_k: int,
        candidate_mode: CandidateMode | str,
        subdivision_points: int,
        adaptive_min_separation: float,
    ) -> None:
        mode = CandidateMode(candidate_mode)
        matching_target = prepare_candidate_target(
            source.graph, target.graph, candidate_mode=mode, rho=candidate_rho, subdivision_points=subdivision_points, adaptive_min_separation=adaptive_min_separation,
        )
        self.source = source
        self.target = LoadedGraph(target.key, matching_target)
        self.candidate_rho = candidate_rho
        self.top_k = top_k
        self.candidate_mode = mode
        self.subdivision_points = int(subdivision_points)
        self.adaptive_min_separation = float(adaptive_min_separation)
        self.matcher = RiverGraphMatcher(source.graph, matching_target, candidate_rho=candidate_rho, top_k=top_k)
        self._results: dict[CostKey, BothMatchResult] = {}
        self._mapping_evaluations: dict[MappingEvaluationKey, MappingEvaluation] = {}
        self._lock = threading.RLock()

    @property
    def preflight(self) -> MatchingPreflight:
        return self.matcher.preflight

    def match(
        self, cost_name: str, options: Mapping[str, object], *, cancellation_token: CancellationToken, progress: Callable[[str], None] | None = None,
    ) -> tuple[BothMatchResult, bool, float]:
        key = CostKey(cost_name, _canonical_options(options))

        with self._lock:
            cached = self._results.get(key)

            if cached is not None:
                return cached, True, 0.0

            cancellation_token.check()
            started = time.perf_counter()
            result = self.matcher.match_both(cost_name, cancellation_token=cancellation_token, progress=progress, **dict(options))
            elapsed = time.perf_counter() - started
            cancellation_token.check()
            self._results[key] = result
            return result, False, elapsed

    def evaluate_mapping(
        self, mapping: Mapping[int, int], cost_name: str, options: Mapping[str, object], *, cancellation_token: CancellationToken, progress: Callable[[str], None] | None = None,
    ) -> tuple[MappingEvaluation, bool, float]:
        normalized_mapping = tuple(sorted((int(source), int(target)) for source, target in mapping.items()))
        key = MappingEvaluationKey(CostKey(cost_name, _canonical_options(options)), normalized_mapping)

        with self._lock:
            cached = self._mapping_evaluations.get(key)

            if cached is not None:
                return cached, True, 0.0

            cancellation_token.check()
            started = time.perf_counter()
            evaluation = self.matcher.evaluate_mapping(dict(normalized_mapping), cost_name, cancellation_token=cancellation_token, progress=progress, **dict(options))
            elapsed = time.perf_counter() - started
            cancellation_token.check()
            self._mapping_evaluations[key] = evaluation
            return evaluation, False, elapsed


class PairSessionStore:
    """Small LRU so switching back to a previously used graph pair is immediate."""

    def __init__(self, *, maximum_sessions: int = 6) -> None:
        self.maximum_sessions = maximum_sessions
        self._sessions: OrderedDict[PairKey, PairSession] = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create(
        self,
        source: LoadedGraph,
        target: LoadedGraph,
        *,
        candidate_rho: float,
        top_k: int,
        candidate_mode: CandidateMode | str,
        subdivision_points: int,
        adaptive_min_separation: float,
    ) -> PairSession:
        mode = CandidateMode(candidate_mode)
        key = PairKey(source.key, target.key, float(candidate_rho), int(top_k), mode, int(subdivision_points), float(adaptive_min_separation))

        with self._lock:
            existing = self._sessions.get(key)

            if existing is not None:
                self._sessions.move_to_end(key)
                return existing

        session = PairSession(
            source, target, candidate_rho=candidate_rho, top_k=top_k, candidate_mode=mode, subdivision_points=subdivision_points, adaptive_min_separation=adaptive_min_separation,
        )

        with self._lock:
            existing = self._sessions.get(key)

            if existing is not None:
                self._sessions.move_to_end(key)
                return existing

            self._sessions[key] = session
            self._sessions.move_to_end(key)

            while len(self._sessions) > self.maximum_sessions:
                self._sessions.popitem(last=False)

            return session

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


class CatalogWorker(QRunnable):
    def __init__(self, repository: GraphRepository, paths: Sequence[Path]) -> None:
        super().__init__()
        self.repository = repository
        self.paths = tuple(paths)
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            graphs: list[GraphInfo] = []

            for index, path in enumerate(self.paths, start=1):
                self.signals.progress.emit(f"Loading graph catalogue ({index}/{len(self.paths)}): {path.name}")
                loaded = self.repository.load(path)
                graphs.append(GraphInfo(path=loaded.key.path, name=loaded.graph.name, vertices=len(loaded.graph.vertices), edges=len(loaded.graph.edges)))

            self.signals.result.emit(CatalogOutcome(tuple(graphs)))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class PreviewWorker(QRunnable):
    def __init__(self, repository: GraphRepository, first_path: Path, second_path: Path, *, request_id: int) -> None:
        super().__init__()
        self.repository = repository
        self.first_path = first_path
        self.second_path = second_path
        self.request_id = request_id
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            first = self.repository.load(self.first_path)
            second = self.repository.load(self.second_path)
            source, target, swapped = normalize_sparse_to_dense(first, second)
            self.signals.result.emit(
                PreviewOutcome(request_id=self.request_id, source_path=source.key.path, target_path=target.key.path, source=source.graph, target=target.graph, swapped=swapped),
            )
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class PreflightWorker(QRunnable):
    def __init__(
        self,
        repository: GraphRepository,
        sessions: PairSessionStore,
        source_path: Path,
        target_path: Path,
        *,
        candidate_rho: float,
        top_k: int,
        candidate_mode: CandidateMode | str,
        subdivision_points: int,
        adaptive_min_separation: float,
        cancellation_token: CancellationToken,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.sessions = sessions
        self.source_path = source_path
        self.target_path = target_path
        self.candidate_rho = candidate_rho
        self.top_k = top_k
        self.candidate_mode = CandidateMode(candidate_mode)
        self.subdivision_points = subdivision_points
        self.adaptive_min_separation = adaptive_min_separation
        self.cancellation_token = cancellation_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.cancellation_token.check()
            self.signals.progress.emit("Preparing candidates and tree decomposition")
            source, target, _ = load_matching_pair(self.repository, self.source_path, self.target_path, self.candidate_mode)
            self.cancellation_token.check()
            session = self.sessions.get_or_create(
                source,
                target,
                candidate_rho=self.candidate_rho,
                top_k=self.top_k,
                candidate_mode=self.candidate_mode,
                subdivision_points=self.subdivision_points,
                adaptive_min_separation=self.adaptive_min_separation,
            )
            self.cancellation_token.check()
            self.signals.result.emit(
                PreflightOutcome(
                    source_path=source.key.path,
                    target_path=target.key.path,
                    source=session.source.graph,
                    target=session.target.graph,
                    candidate_rho=self.candidate_rho,
                    top_k=self.top_k,
                    candidate_mode=self.candidate_mode,
                    subdivision_points=self.subdivision_points,
                    adaptive_min_separation=self.adaptive_min_separation,
                    preflight=session.preflight,
                ),
            )
        except OperationCancelled as error:
            self.signals.cancelled.emit(str(error))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class MatchWorker(QRunnable):
    def __init__(
        self,
        repository: GraphRepository,
        sessions: PairSessionStore,
        source_path: Path,
        target_path: Path,
        *,
        candidate_rho: float,
        top_k: int,
        candidate_mode: CandidateMode | str,
        subdivision_points: int,
        adaptive_min_separation: float,
        costs: Sequence[tuple[str, Mapping[str, object]]],
        cancellation_token: CancellationToken,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.sessions = sessions
        self.source_path = source_path
        self.target_path = target_path
        self.candidate_rho = candidate_rho
        self.top_k = top_k
        self.candidate_mode = CandidateMode(candidate_mode)
        self.subdivision_points = subdivision_points
        self.adaptive_min_separation = adaptive_min_separation
        self.costs = tuple(costs)
        self.cancellation_token = cancellation_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.cancellation_token.check()
            source, target, swapped = load_matching_pair(self.repository, self.source_path, self.target_path, self.candidate_mode)
            session = self.sessions.get_or_create(
                source,
                target,
                candidate_rho=self.candidate_rho,
                top_k=self.top_k,
                candidate_mode=self.candidate_mode,
                subdivision_points=self.subdivision_points,
                adaptive_min_separation=self.adaptive_min_separation,
            )

            for index, (cost_name, options) in enumerate(self.costs, start=1):
                self.cancellation_token.check()
                prefix = f"{cost_name.replace('_', ' ')} ({index}/{len(self.costs)})"

                def progress(message: str, prefix: str = prefix) -> None:
                    self.signals.progress.emit(f"{prefix}: {message}")

                result, from_cache, elapsed = session.match(cost_name, options, cancellation_token=self.cancellation_token, progress=progress)
                self.signals.result.emit(
                    RunOutcome(
                        source_path=source.key.path,
                        target_path=target.key.path,
                        source=session.source.graph,
                        target=session.target.graph,
                        cost_name=cost_name,
                        cost_options=dict(options),
                        candidate_rho=self.candidate_rho,
                        top_k=self.top_k,
                        candidate_mode=self.candidate_mode,
                        subdivision_points=self.subdivision_points,
                        adaptive_min_separation=self.adaptive_min_separation,
                        result=result,
                        swapped=swapped,
                        from_cache=from_cache,
                        elapsed_seconds=elapsed,
                    ),
                )
        except OperationCancelled as error:
            self.signals.cancelled.emit(str(error))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class MappingScoreWorker(QRunnable):
    def __init__(
        self,
        repository: GraphRepository,
        sessions: PairSessionStore,
        source_path: Path,
        target_path: Path,
        *,
        candidate_rho: float,
        top_k: int,
        candidate_mode: CandidateMode | str,
        subdivision_points: int,
        adaptive_min_separation: float,
        mapping: Mapping[int, int],
        cost_name: str,
        cost_options: Mapping[str, object],
        cancellation_token: CancellationToken,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.sessions = sessions
        self.source_path = source_path
        self.target_path = target_path
        self.candidate_rho = candidate_rho
        self.top_k = top_k
        self.candidate_mode = CandidateMode(candidate_mode)
        self.subdivision_points = subdivision_points
        self.adaptive_min_separation = adaptive_min_separation
        self.mapping = dict(mapping)
        self.cost_name = cost_name
        self.cost_options = dict(cost_options)
        self.cancellation_token = cancellation_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.cancellation_token.check()
            source, target, _ = load_matching_pair(self.repository, self.source_path, self.target_path, self.candidate_mode)
            session = self.sessions.get_or_create(
                source,
                target,
                candidate_rho=self.candidate_rho,
                top_k=self.top_k,
                candidate_mode=self.candidate_mode,
                subdivision_points=self.subdivision_points,
                adaptive_min_separation=self.adaptive_min_separation,
            )

            def progress(message: str) -> None:
                self.signals.progress.emit(message)

            evaluation, from_cache, elapsed = session.evaluate_mapping(
                self.mapping, self.cost_name, self.cost_options, cancellation_token=self.cancellation_token, progress=progress,
            )
            self.signals.result.emit(
                MappingScoreOutcome(
                    source_path=source.key.path,
                    target_path=target.key.path,
                    source=session.source.graph,
                    target=session.target.graph,
                    cost_name=self.cost_name,
                    cost_options=self.cost_options,
                    candidate_mode=self.candidate_mode,
                    subdivision_points=self.subdivision_points,
                    adaptive_min_separation=self.adaptive_min_separation,
                    mapping=self.mapping,
                    evaluation=evaluation,
                    from_cache=from_cache,
                    elapsed_seconds=elapsed,
                ),
            )
        except OperationCancelled as error:
            self.signals.cancelled.emit(str(error))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


def cost_names() -> tuple[str, ...]:
    return tuple(cost.value for cost in CostName)
