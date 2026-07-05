from __future__ import annotations

import river_matcher


def test_public_api_exports_stable_entry_points() -> None:
    expected = {"BothMatchResult", "CandidateStatistics", "CostName", "JunctionEdge", "JunctionGraph", "MatchedEdge", "MatchResult", "MatchSolution", "Objective",
        "RiverGraphMatcher", "SourceDecomposition", "available_costs", "build_source_decomposition", "create_cost", "load_junction_graph", "match_graphs", "match_graphs_both", }

    assert set(river_matcher.__all__) == expected

    for name in expected:
        assert getattr(river_matcher, name) is not None
