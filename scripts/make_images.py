from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from river_matcher import load_junction_graph

ROOT = Path(r"C:\Users\Kishan\Documents\University\Thesis\River-Matcher")

SOURCE_PATH = ROOT / "GraphExport" / "2014e5.txt"
TARGET_PATH = ROOT / "GraphExport" / "1955e5.txt"

PAIR_DIR = (ROOT / "experiment_results" / "cross_year_same_scale" / "2014e5_to_1955e5")

OUTPUT_DIR = ROOT / "thesis_figures" / "chapter5"

BBOX = (200.0, 500.0, 100.0, 320.0)

COSTS = [("relative_length_error", "Relative length error"), ("mean_distance_tangent", "Mean distance and tangent"), ("hausdorff_distance", "Hausdorff distance"),
    ("discrete_frechet_distance", "Discrete Fréchet distance"), ("dynamic_time_warping_distance", "Dynamic time warping")]

SOURCE = "#2f6f9f"
TARGET = "#d95f02"
TARGET_LIGHT = "#d9d9d9"
MAPPING_LINE = "#707070"


def save_figure(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def normalized_mapping(solution):
    return {int(source_vertex): int(target_vertex) for source_vertex, target_vertex in solution["mapping"].items()}


def draw_solution(ax, solution, title, source, target, source_lines, target_lines, changed_vertices=None):
    ax.add_collection(LineCollection(target_lines, colors=TARGET_LIGHT, linewidths=0.7, alpha=0.65, zorder=1))

    ax.add_collection(LineCollection(source_lines, colors=SOURCE, linewidths=1.35, alpha=0.82, zorder=2))

    witnesses = []

    for edge in solution["edges"]:
        witness = np.asarray(edge["witness"], dtype=float)

        if witness.ndim == 2 and witness.shape[0] >= 2:
            witnesses.append(witness)

    ax.add_collection(LineCollection(witnesses, colors=TARGET, linewidths=1.45, linestyles="dashed", alpha=0.82, zorder=3))

    if changed_vertices:
        vertex_mapping = normalized_mapping(solution)

        for source_vertex in sorted(changed_vertices):
            if source_vertex not in vertex_mapping:
                continue

            source_point = np.asarray(source.coordinates[source_vertex], dtype=float)

            target_vertex = vertex_mapping[source_vertex]

            target_point = np.asarray(target.coordinates[target_vertex], dtype=float)

            ax.plot([source_point[0], target_point[0]], [source_point[1], target_point[1]], color=MAPPING_LINE, linewidth=0.7, linestyle=":", alpha=0.55, zorder=4)

            ax.scatter(source_point[0], source_point[1], s=20, marker="o", facecolor="white", edgecolor="black", linewidth=0.9, zorder=5)

            ax.scatter(target_point[0], target_point[1], s=22, marker="s", facecolor="white", edgecolor=TARGET, linewidth=1.0, zorder=6)

    ax.set_xlim(BBOX[0], BBOX[1])
    ax.set_ylim(BBOX[2], BBOX[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


source = load_junction_graph(SOURCE_PATH)
target = load_junction_graph(TARGET_PATH)

source_lines = [np.asarray(edge.polyline, dtype=float) for edge in source.edges if len(edge.polyline) >= 2]

target_lines = [np.asarray(edge.polyline, dtype=float) for edge in target.edges if len(edge.polyline) >= 2]

reports = {}

for cost_name, _ in COSTS:
    report_path = (PAIR_DIR / cost_name / "report.json")

    with report_path.open(encoding="utf-8") as file:
        reports[cost_name] = json.load(file)

fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))

for ax, (cost_name, label) in zip(axes.flat, COSTS):
    draw_solution(ax, reports[cost_name]["solutions"]["additive"], label, source, target, source_lines, target_lines)

axes[1, 2].set_axis_off()

axes[1, 2].legend(
    handles=[Line2D([0], [0], color=SOURCE, linewidth=2.0, label="Source graph"), Line2D([0], [0], color=TARGET, linewidth=2.0, linestyle="--", label="Selected target witnesses"),
        Line2D([0], [0], color=TARGET_LIGHT, linewidth=2.0, label="Other target edges")], loc="center", frameon=False)

axes[1, 2].text(0.5, 0.25, "Additive objective", ha="center", va="center", transform=axes[1, 2].transAxes)

fig.tight_layout()

save_figure(fig, "cost_comparison_dense_region")

bbox_vertices = {vertex for vertex, (x, y) in source.coordinates.items() if (BBOX[0] <= x <= BBOX[1] and BBOX[2] <= y <= BBOX[3])}

objective_comparisons = []

for cost_name, label in COSTS:
    additive = reports[cost_name]["solutions"]["additive"]

    bottleneck = reports[cost_name]["solutions"]["bottleneck"]

    additive_mapping = normalized_mapping(additive)

    bottleneck_mapping = normalized_mapping(bottleneck)

    common = (bbox_vertices & additive_mapping.keys() & bottleneck_mapping.keys())

    if not common:
        continue

    agreement = (sum(additive_mapping[vertex] == bottleneck_mapping[vertex] for vertex in common) / len(common))

    objective_comparisons.append((agreement, cost_name, label, additive, bottleneck, common))

if not objective_comparisons:
    raise RuntimeError("No mapped source vertices lie inside BBOX.")

(agreement, cost_name, label, additive, bottleneck, common) = min(objective_comparisons, key=lambda item: item[0])

additive_mapping = normalized_mapping(additive)

bottleneck_mapping = normalized_mapping(bottleneck)

changed_vertices = {vertex for vertex in common if (additive_mapping[vertex] != bottleneck_mapping[vertex])}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

draw_solution(axes[0], additive, f"{label} — additive", source, target, source_lines, target_lines, changed_vertices)

draw_solution(axes[1], bottleneck, f"{label} — bottleneck", source, target, source_lines, target_lines, changed_vertices)

fig.legend(
    handles=[Line2D([0], [0], color=SOURCE, linewidth=2.0, label="Source graph"), Line2D([0], [0], color=TARGET, linewidth=2.0, linestyle="--", label="Selected target witnesses"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="", label="Changed source junction"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor=TARGET, linestyle="", label="Its mapped target junction")], loc="lower center", ncol=4, frameon=False,
    bbox_to_anchor=(0.5, 0.0))

fig.tight_layout(rect=(0, 0.12, 1, 1))

save_figure(fig, "additive_vs_bottleneck_dense_region")

print(f"Figures written to {OUTPUT_DIR}")

print(f"Objective comparison: {label}")

print(f"Agreement inside crop: "
      f"{100.0 * agreement:.1f}% "
      f"({len(changed_vertices)} changed source junctions)")
