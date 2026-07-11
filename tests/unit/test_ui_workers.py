from __future__ import annotations

from pathlib import Path

import numpy as np

from river_matcher.candidates import CandidateMode
from river_matcher.models import JunctionEdge, JunctionGraph
from river_matcher.ui.workers import GraphKey, GraphRepository, LoadedGraph, PairSessionStore


def loaded(name: str, graph: JunctionGraph) -> LoadedGraph:
    return LoadedGraph(GraphKey(Path(f"{name}.txt"), 0, 0, "junction"), graph)


def test_pair_session_cache_separates_candidate_modes_and_settings() -> None:
    source = loaded("source", JunctionGraph("source", {1: (2.0, 1.0), 2: (8.0, 1.0)}, (JunctionEdge(0, 1, 2, np.asarray([(2.0, 1.0), (8.0, 1.0)])),)))
    target = loaded("target", JunctionGraph("target", {10: (0.0, 0.0), 20: (10.0, 0.0)}, (JunctionEdge(0, 10, 20, np.asarray([(0.0, 0.0), (10.0, 0.0)])),)))
    store = PairSessionStore()

    baseline = store.get_or_create(source, target, candidate_rho=2.0, top_k=10, candidate_mode=CandidateMode.TARGET_JUNCTIONS, subdivision_points=2, adaptive_min_separation=1.0)
    uniform = store.get_or_create(
        source, target, candidate_rho=2.0, top_k=10, candidate_mode=CandidateMode.UNIFORM_TARGET_SUBDIVISION, subdivision_points=2, adaptive_min_separation=1.0,
    )
    uniform_three = store.get_or_create(
        source, target, candidate_rho=2.0, top_k=10, candidate_mode=CandidateMode.UNIFORM_TARGET_SUBDIVISION, subdivision_points=3, adaptive_min_separation=1.0,
    )

    assert baseline is not uniform
    assert uniform is not uniform_three
    assert len(baseline.target.graph.vertices) == 2
    assert len(uniform.target.graph.vertices) == 4
    assert len(uniform_three.target.graph.vertices) == 5
    assert set(uniform.matcher.candidate_sets[1]).issubset(uniform.target.graph.vertices)


def test_graph_repository_caches_original_and_junction_variants_separately(tmp_path: Path) -> None:
    path = tmp_path / "river.txt"
    path.write_text("\n".join(["3", "1 0.0 0.0", "2 1.0 0.0", "3 2.0 0.0", "2", "10 1 2 1.0 0.0 0.0 1.0 0.0", "11 2 3 1.0 1.0 0.0 2.0 0.0"]), encoding="utf-8")
    repository = GraphRepository()

    junction = repository.load(path)
    original = repository.load(path, variant="original")

    assert junction.key != original.key
    assert junction.graph.vertices == (1, 3)
    assert original.graph.vertices == (1, 2, 3)
