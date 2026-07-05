from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import river_matcher.cli as cli
from river_matcher.dynamic_programming import Objective


def _graph(name: str, vertices: int, edges: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, vertices=tuple(range(vertices)), edges=tuple(range(edges)))


def _solution(objective: Objective, value: float) -> SimpleNamespace:
    edge = SimpleNamespace(edge_id=7, source_u=1, source_v=2, target_u=10, target_v=20, cost=value, witness=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64))
    return SimpleNamespace(objective=objective, value=value, mapping={1: 10, 2: 20}, edges=(edge,))


def _result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {"cost_name": SimpleNamespace(value="relative_length_error"),
                                 "candidate_statistics": SimpleNamespace(source_vertices=2, empty_domains=0, total_candidates=3, minimum_candidates=1, maximum_candidates=2),
                                 "candidate_sets": {1: (10, 11), 2: (20,)},
                                 "decomposition": SimpleNamespace(width=1, maximum_bag_size=2, bag_count=2, heuristic=SimpleNamespace(value="minimum_fill"), minimum_fill_width=1,
                                                                  minimum_degree_width=1),
                                 "dp_statistics": SimpleNamespace(enumerated_states=6, feasible_states=5, message_entries=3, unique_cost_requests=4)}
    values.update(overrides)

    return SimpleNamespace(**values)


def _install_graph_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, SimpleNamespace]:
    source = _graph("source", vertices=2, edges=1)
    target = _graph("target", vertices=3, edges=2)
    graphs = iter((source, target))

    monkeypatch.setattr(cli, "load_junction_graph", lambda _: next(graphs))

    return source, target


def test_parse_cost_options_uses_json_scalars() -> None:
    options = cli._parse_cost_options(["rho=10.0", "samples=64", "normalize=true", "label=plain-text"])
    assert options == {"rho": 10.0, "samples": 64, "normalize": True, "label": "plain-text"}


@pytest.mark.parametrize("options", [["missing-separator"], ["rho=10", "rho=20"]])
def test_parse_cost_options_rejects_invalid_entries(options: list[str]) -> None:
    with pytest.raises(ValueError):
        cli._parse_cost_options(options)


def test_main_writes_both_objectives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = _install_graph_loader(monkeypatch)
    additive = _solution(Objective.ADDITIVE, 1.25)
    bottleneck = _solution(Objective.BOTTLENECK, 0.75)
    result = _result(additive=additive, bottleneck=bottleneck)

    matcher = Mock()
    matcher.match_both.return_value = result
    matcher_factory = Mock(return_value=matcher)
    monkeypatch.setattr(cli, "RiverGraphMatcher", matcher_factory)

    output = tmp_path / "both.json"
    exit_code = cli.main(
        ["source.txt", "target.txt", "--cost", "relative_length_error", "--objective", "both", "--candidate-rho", "10", "--top-k", "25", "--cost-option", "rho=10.0", "--output",
         str(output)])

    assert exit_code == 0
    matcher_factory.assert_called_once_with(source, target, candidate_rho=10.0, top_k=25)
    matcher.match_both.assert_called_once_with("relative_length_error", rho=10.0)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["source"]["vertices"] == 2
    assert payload["target"]["edges"] == 2
    assert payload["solutions"]["additive"]["value"] == 1.25
    assert payload["solutions"]["bottleneck"]["value"] == 0.75
    assert payload["solutions"]["additive"]["mapping"] == [{"source_vertex": 1, "target_vertex": 10}, {"source_vertex": 2, "target_vertex": 20}]
    assert payload["solutions"]["additive"]["edges"][0]["witness"] == [[0.0, 0.0], [1.0, 1.0]]


def test_main_returns_two_for_infeasible_single_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_graph_loader(monkeypatch)
    result = _result(solution=None)

    matcher = Mock()
    matcher.match.return_value = result
    monkeypatch.setattr(cli, "RiverGraphMatcher", Mock(return_value=matcher))

    output = tmp_path / "infeasible.json"
    exit_code = cli.main(["source.txt", "target.txt", "--cost", "relative_length_error", "--objective", "bottleneck", "--output", str(output)])

    assert exit_code == 2
    matcher.match.assert_called_once_with("relative_length_error", Objective.BOTTLENECK)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["solutions"] == {"bottleneck": {"feasible": False}}


def test_main_rejects_nonpositive_top_k() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["source.txt", "target.txt", "--cost", "relative_length_error", "--top-k", "0"])

    assert error.value.code == 2

def test_module_entry_point_imports_cli_main() -> None:
    from river_matcher import __main__
    assert __main__.main is cli.main