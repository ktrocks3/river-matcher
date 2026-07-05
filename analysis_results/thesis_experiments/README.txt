Generated analysis tables

runs.csv
    One row per graph pair and cost. Shared candidate, decomposition, DP,
    and one-off experiment timing fields occur only once.

solutions.csv
    One row per graph pair, cost, and objective.

graph_pairs.csv
    One compact row per selected graph pair.

objective_tradeoffs.csv
    Additive-versus-bottleneck comparison within each pair and cost,
    including local-cost sums, worst-edge reduction, and mapping agreement.

zero_cost_edges.csv
    Counts and fractions of zero-cost edges. A zero value is specific to
    the selected cost and must not automatically be interpreted as identical
    geometry.

cost_mapping_agreement.csv
    Pairwise mapping agreement between cost functions for each objective.

runtime_benchmark.csv
    Present only after benchmark_summary.csv is supplied in the analysis
    configuration. Use these repeated timings rather than one-off experiment
    timings in the thesis runtime table.
