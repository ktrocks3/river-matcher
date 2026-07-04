from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from river_matcher.candidates import compute_candidate_sets
from river_matcher.costs.factory import CostFactory
from river_matcher.decomposition import build_source_decomposition
from river_matcher.dynamic_programming import solve_tree_dp_both
from river_matcher.preprocessing import load_junction_graph

type FloatArray = NDArray[np.float64]


def polyline_length(points: FloatArray) -> float:
    if len(points) < 2:
        return 0.0

    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    source = load_junction_graph(args.source)
    target = load_junction_graph(args.target)
    candidate_sets = compute_candidate_sets(source, target, rho=10.0, top_k=25)
    decomposition = build_source_decomposition(source)
    cost = CostFactory(source, target).create("mean_distance_tangent", rho=10.0, edge_samples=12, curve_samples=64, tangent_weight=1.0)
    from river_matcher.costs.mean_distance_tangent import MeanDistanceTangent
    from river_matcher.geometry import prepare_polyline_segments

    assert isinstance(cost, MeanDistanceTangent)

    problem_edge_ids = (3, 6, 11, 17, 27, 30, 33, 37, 38, 39, 55, 65, 75, 85, 86, 97, 102,)

    print("\nSource preprocessing")

    for edge_id in problem_edge_ids:
        edge = next(edge for edge in source.edges if edge.id == edge_id)
        raw_vectors = np.diff(edge.polyline, axis=0)
        raw_lengths = np.linalg.norm(raw_vectors, axis=1)

        source_samples = cost._source_samples(edge_id)
        source_prepared = cost._source_prepared(edge_id)

        print(f"e{edge_id} ({edge.u}, {edge.v}): "
              f"raw_points={len(edge.polyline)}, "
              f"raw_segments={len(raw_lengths)}, "
              f"zero_raw={int(np.count_nonzero(raw_lengths <= 1e-12))}, "
              f"min_raw={float(raw_lengths.min()) if len(raw_lengths) else None}, "
              f"samples_none={source_samples is None}, "
              f"prepared_none={source_prepared is None}")

        if source_samples is not None:
            sampled_points, sampled_tangents = source_samples
            sampled_lengths = np.linalg.norm(np.diff(sampled_points, axis=0), axis=1, )

            print(f"  sampled_min={float(sampled_lengths.min())}, "
                  f"sampled_zero={int(np.count_nonzero(sampled_lengths <= 1e-12))}, "
                  f"points_finite={bool(np.all(np.isfinite(sampled_points)))}, "
                  f"tangents_finite={bool(np.all(np.isfinite(sampled_tangents)))}")

        if source_prepared is not None:
            _, _, source_squared = source_prepared

            print(f"  prepared_segments={len(source_squared)}, "
                  f"prepared_zero={int(np.count_nonzero(source_squared <= 1e-24))}, "
                  f"prepared_min_squared={float(source_squared.min())}")

    print("\nFirst direct witness per problem edge")

    for edge_id in problem_edge_ids:
        edge = next(edge for edge in source.edges if edge.id == edge_id)
        found = False

        for target_u in candidate_sets[edge.u]:
            for target_v in candidate_sets[edge.v]:
                witness = cost._finder.path(edge.id, edge.u, edge.v, target_u, target_v, )

                if witness is None:
                    continue

                witness_samples = cost._sample_curve(witness)
                witness_prepared = prepare_polyline_segments(witness)

                print(f"e{edge_id}: pair=({target_u}, {target_v}), "
                      f"witness_points={len(witness)}, "
                      f"witness_samples_none={witness_samples is None}, "
                      f"witness_prepared_none={witness_prepared is None}")
                found = True
                break

            if found:
                break

        if not found:
            print(f"e{edge_id}: finder returned no path for every candidate pair")

    result = solve_tree_dp_both(decomposition, candidate_sets, cost)

    print("\nEmpty DP message tables")

    empty_statistics = [statistics for statistics in result.statistics.bags if statistics.message_entries == 0]

    for statistics in empty_statistics:
        plan = decomposition.bag_plans[statistics.bag]

        print(f"bag={plan.variables}, "
              f"enumerated={statistics.enumerated_states}, "
              f"feasible={statistics.feasible_states}, "
              f"owned_edges={decomposition.owned_edges[statistics.bag]}, "
              f"children={len(plan.child_positions)}")

    print("\nNonfinite edge-cost requests")

    total_requests = 0
    total_finite = 0
    total_nonfinite = 0
    total_no_witness = 0
    total_zero_length_witness = 0
    total_nondegenerate_witness = 0
    total_same_target = 0

    for edge in sorted(source.edges, key=lambda item: item.id):
        requests = 0
        finite = 0
        no_witness = 0
        zero_length_witness = 0
        nondegenerate_witness = 0
        same_target = 0
        examples: list[str] = []

        for target_u in candidate_sets[edge.u]:
            for target_v in candidate_sets[edge.v]:
                requests += 1
                value = float(cost(edge.id, edge.u, edge.v, target_u, target_v))

                if math.isfinite(value):
                    finite += 1
                    continue

                if target_u == target_v:
                    same_target += 1

                witness = cost.witness(edge.id, edge.u, edge.v, target_u, target_v)

                if witness is None:
                    no_witness += 1
                    category = "no witness"
                    length = None
                else:
                    points = np.asarray(witness, dtype=np.float64)
                    length = polyline_length(points)

                    if length <= 1e-12:
                        zero_length_witness += 1
                        category = "zero-length witness"
                    else:
                        nondegenerate_witness += 1
                        category = "nondegenerate witness"

                if len(examples) < 3:
                    examples.append(f"({target_u}, {target_v}): "
                                    f"{category}, length={length}")

        nonfinite = requests - finite
        total_requests += requests
        total_finite += finite
        total_nonfinite += nonfinite
        total_no_witness += no_witness
        total_zero_length_witness += zero_length_witness
        total_nondegenerate_witness += nondegenerate_witness
        total_same_target += same_target

        if finite == 0 or nondegenerate_witness or zero_length_witness:
            print(f"e{edge.id} ({edge.u}, {edge.v}): "
                  f"requests={requests}, finite={finite}, "
                  f"no_witness={no_witness}, "
                  f"zero_length={zero_length_witness}, "
                  f"nondegenerate={nondegenerate_witness}, "
                  f"same_target={same_target}")

            for example in examples:
                print(f"  {example}")

    print("\nTotals")
    print(f"requests:               {total_requests}")
    print(f"finite:                 {total_finite}")
    print(f"nonfinite:              {total_nonfinite}")
    print(f"no witness:             {total_no_witness}")
    print(f"zero-length witness:    {total_zero_length_witness}")
    print(f"nondegenerate witness:  {total_nondegenerate_witness}")
    print(f"same-target nonfinite:  {total_same_target}")


if __name__ == "__main__":
    main()
