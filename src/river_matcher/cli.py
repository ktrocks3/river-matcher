from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from river_matcher.costs import available_costs
from river_matcher.dynamic_programming import Objective
from river_matcher.matcher import BothMatchResult, MatchResult, MatchSolution, RiverGraphMatcher
from river_matcher.preprocessing import load_junction_graph


def _parse_scalar(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_cost_options(values: Sequence[str]) -> dict[str, object]:
    options: dict[str, object] = {}

    for value in values:
        key, separator, raw_value = value.partition("=")

        if not separator or not key:
            raise ValueError(f"Invalid cost option {value!r}; expected key=value.")

        if key in options:
            raise ValueError(f"Cost option {key!r} was provided more than once.")

        options[key] = _parse_scalar(raw_value)

    return options


def _solution_payload(solution: MatchSolution | None) -> dict[str, object]:
    if solution is None:
        return {"feasible": False}

    mapping = [{"source_vertex": int(source_vertex), "target_vertex": int(target_vertex)} for source_vertex, target_vertex in sorted(solution.mapping.items())]
    edges = [{"edge_id": edge.edge_id, "source_u": edge.source_u, "source_v": edge.source_v, "target_u": edge.target_u, "target_v": edge.target_v, "cost": edge.cost,
              "witness": edge.witness.tolist()} for edge in solution.edges]

    return {"feasible": True, "objective": solution.objective.value, "value": solution.value, "mapping": mapping, "edges": edges}


def _common_result_payload(result: MatchResult | BothMatchResult) -> dict[str, object]:
    candidate_statistics = result.candidate_statistics
    decomposition = result.decomposition
    dp_statistics = result.dp_statistics

    return {"cost": result.cost_name.value, "candidate_statistics": {"source_vertices": candidate_statistics.source_vertices, "empty_domains": candidate_statistics.empty_domains,
                                                                     "total_candidates": candidate_statistics.total_candidates,
                                                                     "minimum_candidates": candidate_statistics.minimum_candidates,
                                                                     "maximum_candidates": candidate_statistics.maximum_candidates},
            "candidate_sets": {str(source_vertex): list(candidates) for source_vertex, candidates in result.candidate_sets.items()},
            "decomposition": {"width": decomposition.width, "maximum_bag_size": decomposition.maximum_bag_size, "bag_count": decomposition.bag_count,
                              "heuristic": decomposition.heuristic.value, "minimum_fill_width": decomposition.minimum_fill_width,
                              "minimum_degree_width": decomposition.minimum_degree_width},
            "dynamic_programming": {"enumerated_states": dp_statistics.enumerated_states, "feasible_states": dp_statistics.feasible_states,
                                    "message_entries": dp_statistics.message_entries, "unique_cost_requests": dp_statistics.unique_cost_requests}}


def _default_output(source: Path, target: Path, cost: str, objective: str) -> Path:
    return Path(f"match_{source.stem}_to_{target.stem}_{cost}_{objective}.json")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="river-match", description=("Match a source river graph to a target graph using the exact "
                                                                      "tree-decomposition dynamic program."))
    parser.add_argument("source", type=Path, help="Source graph export file.")
    parser.add_argument("target", type=Path, help="Target graph export file.")
    parser.add_argument("--cost", required=True, choices=tuple(cost_name.value for cost_name in available_costs()), help="Local source-edge cost method.")
    parser.add_argument("--objective", choices=(Objective.ADDITIVE.value, Objective.BOTTLENECK.value, "both",), default="both", help="Global optimization objective.")
    parser.add_argument("--candidate-rho", type=float, default=10.0, help="Candidate-generation search radius.")
    parser.add_argument("--top-k", type=int, default=25, help="Maximum candidates retained per source vertex.")
    parser.add_argument("--cost-option", action="append", default=[], metavar="KEY=VALUE", help=("Cost-specific option. May be repeated. Values use JSON syntax "
                                                                                                 "where possible, for example normalize=true."))
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if not math.isfinite(arguments.candidate_rho):
        parser.error("--candidate-rho must be finite.")

    if arguments.candidate_rho <= 0.0:
        parser.error("--candidate-rho must be positive.")

    if arguments.top_k <= 0:
        parser.error("--top-k must be positive.")

    try:
        cost_options = _parse_cost_options(arguments.cost_option)
        source = load_junction_graph(arguments.source)
        target = load_junction_graph(arguments.target)
        matcher = RiverGraphMatcher(source, target, candidate_rho=arguments.candidate_rho, top_k=arguments.top_k)
        common_result: MatchResult | BothMatchResult

        if arguments.objective == "both":
            both_result = matcher.match_both(arguments.cost, **cost_options, )
            common_result = both_result
            solutions = {Objective.ADDITIVE.value: _solution_payload(both_result.additive), Objective.BOTTLENECK.value: _solution_payload(both_result.bottleneck), }
            feasible = (both_result.additive is not None and both_result.bottleneck is not None)
        else:
            objective = Objective(arguments.objective)
            match_result = matcher.match(arguments.cost, objective, **cost_options, )
            common_result = match_result
            solutions = {objective.value: _solution_payload(match_result.solution), }
            feasible = match_result.solution is not None
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = (arguments.output if arguments.output is not None else _default_output(arguments.source, arguments.target, arguments.cost, arguments.objective, ))
    payload = {"schema_version": 1, "source": {"name": source.name, "path": str(arguments.source.resolve()), "vertices": len(source.vertices), "edges": len(source.edges)},
               "target": {"name": target.name, "path": str(arguments.target.resolve()), "vertices": len(target.vertices), "edges": len(target.edges)},
               "parameters": {"cost": arguments.cost, "objective": arguments.objective, "candidate_rho": arguments.candidate_rho, "top_k": arguments.top_k,
                              "cost_options": cost_options}, **_common_result_payload(common_result), "solutions": solutions}

    _write_json(output, payload)

    print(f"{source.name} ({len(source.vertices)} V, {len(source.edges)} E) -> {target.name} ({len(target.vertices)} V, {len(target.edges)} E)")
    print(f"cost={arguments.cost}, objective={arguments.objective}, feasible={feasible}")
    print(f"states={common_result.dp_statistics.enumerated_states}, messages={common_result.dp_statistics.message_entries}, "
          f"unique_costs={common_result.dp_statistics.unique_cost_requests}")
    print(f"output: {output.resolve()}")

    return 0 if feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
