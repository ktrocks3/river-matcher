import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from river_matcher import load_junction_graph
from river_matcher.visualization import display_coordinates


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "GraphExport" / "2014e5.txt"
TARGET_PATH = ROOT / "GraphExport" / "1955e3.txt"
PAIR_DIR = ROOT / "experiment_results" / "confirmed_pairs" / "2014e5_to_1955e3"
OUTPUT_DIR = ROOT / "thesis_figures" / "chapter5"

EXAMPLES = {
    "relative_length_error": ("Relative length error", 1, 145),
    "mean_distance_tangent": ("Mean distance + tangent", 69, 155),
    "hausdorff_distance": ("Hausdorff distance", 58, 155),
    "discrete_frechet_distance": ("Discrete Fréchet distance", 59, 170),
    "dynamic_time_warping_distance": ("Dynamic time warping", 42, 145),
}

SOURCE_CONTEXT = "#b7c9d6"
TARGET_CONTEXT = "#e6d5c3"
SOURCE_HIGHLIGHT = "#174a73"
TARGET_HIGHLIGHT = "#d95f02"
CHANGED = "#111111"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.titlesize": 13})


def graph_lines(graph):
    return [display_coordinates(edge.polyline) for edge in graph.edges if len(edge.polyline) >= 2]


def save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def normalized_mapping(solution):
    return {int(source_vertex): int(target_vertex) for source_vertex, target_vertex in solution["mapping"].items()}


def agreement(left, right):
    left_mapping = normalized_mapping(left)
    right_mapping = normalized_mapping(right)
    common = set(left_mapping) & set(right_mapping)

    return sum(left_mapping[v] == right_mapping[v] for v in common) / len(common)


def changed_vertices(left, right):
    left_mapping = normalized_mapping(left)
    right_mapping = normalized_mapping(right)

    return {v for v in set(left_mapping) & set(right_mapping) if left_mapping[v] != right_mapping[v]}


def choose_regions(vertices, source, count=2):
    vertices = sorted(vertices)

    if len(vertices) <= count:
        return vertices

    coordinates = {vertex: np.asarray(source.coordinates[vertex], dtype=float) for vertex in vertices}

    centroid = np.mean(list(coordinates.values()), axis=0)

    first = max(vertices, key=lambda vertex: np.linalg.norm(coordinates[vertex] - centroid))

    selected = [first]

    while len(selected) < count:
        next_vertex = max(
            (vertex for vertex in vertices if vertex not in selected), key=lambda vertex: min(np.linalg.norm(coordinates[vertex] - coordinates[chosen]) for chosen in selected)
        )

        selected.append(next_vertex)

    return selected


def incident_payloads(solution, vertex):
    return [edge for edge in solution["edges"] if int(edge["source_u"]) == vertex or int(edge["source_v"]) == vertex]


def region_bounds(source, left, right, vertex, minimum_window=155):
    points = [display_coordinates(source.coordinates[vertex])]

    edge_ids = {int(edge["edge_id"]) for edge in incident_payloads(left, vertex) + incident_payloads(right, vertex)}

    for edge_id in edge_ids:
        points.extend(display_coordinates(source.edge_by_id[edge_id].polyline))

    for solution in (left, right):
        for edge in incident_payloads(solution, vertex):
            points.extend(display_coordinates(edge["witness"]))

    points = np.asarray(points, dtype=float)
    low = points.min(axis=0)
    high = points.max(axis=0)
    centre = 0.5 * (low + high)

    side = max(minimum_window, float(np.max(high - low) + 28))

    half = side / 2

    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


def draw_context(ax, source_lines, target_lines):
    ax.add_collection(LineCollection(target_lines, colors=TARGET_CONTEXT, linewidths=1, alpha=0.58, zorder=1))

    ax.add_collection(LineCollection(source_lines, colors=SOURCE_CONTEXT, linewidths=1.1, alpha=0.65, zorder=2))


def draw_region(ax, source, target, source_lines, target_lines, solution, vertex, bounds, title):
    draw_context(ax, source_lines, target_lines)

    payloads = incident_payloads(solution, vertex)

    for payload in payloads:
        edge_id = int(payload["edge_id"])

        source_polyline = display_coordinates(source.edge_by_id[edge_id].polyline)

        witness = display_coordinates(payload["witness"])

        witness_length = float(np.linalg.norm(np.diff(witness, axis=0), axis=1).sum())

        witness_style = "-" if witness_length < 12.0 else "--"

        ax.plot(source_polyline[:, 0], source_polyline[:, 1], color=SOURCE_HIGHLIGHT, linewidth=3.2, solid_capstyle="round", zorder=5)

        ax.plot(witness[:, 0], witness[:, 1], color=TARGET_HIGHLIGHT, linewidth=3.2, linestyle=witness_style, dash_capstyle="round", solid_capstyle="round", zorder=6)

    source_point = display_coordinates(source.coordinates[vertex])

    mapped_vertex = normalized_mapping(solution)[vertex]

    target_point = display_coordinates(target.coordinates[mapped_vertex])

    ax.scatter(source_point[0], source_point[1], s=45, marker="o", facecolor="white", edgecolor=CHANGED, linewidth=1.8, zorder=8)

    ax.scatter(target_point[0], target_point[1], s=55, marker="s", facecolor="white", edgecolor=TARGET_HIGHLIGHT, linewidth=1.8, zorder=8)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_axis_off()


def draw_difference_overview(ax, source, source_lines, target_lines, changed, boxes):
    ax.add_collection(LineCollection(target_lines, colors="#e2e2e2", linewidths=0.8, alpha=0.6))

    ax.add_collection(LineCollection(source_lines, colors="#8799a5", linewidths=1.25, alpha=0.8))

    changed_points = display_coordinates([source.coordinates[vertex] for vertex in sorted(changed)])

    ax.scatter(changed_points[:, 0], changed_points[:, 1], s=18, color=CHANGED, zorder=5)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for index, bounds in enumerate(boxes):
        x_min, x_max, y_min, y_max = bounds

        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor=TARGET_HIGHLIGHT, linewidth=1.5, zorder=6))

        ax.text(
            x_min + 0.02 * (x_max - x_min), y_max - 0.05 * (y_max - y_min), labels[index], fontsize=11, fontweight="bold", color=TARGET_HIGHLIGHT, verticalalignment="top", zorder=7
        )

    box_corners = [np.asarray(((x_min, y_min), (x_max, y_max)), dtype=float) for x_min, x_max, y_min, y_max in boxes]
    all_points = np.vstack(source_lines + target_lines + box_corners)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    padding = 0.025 * np.maximum(high - low, 1)

    ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
    ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Locations of changed source-junction assignments")
    ax.set_axis_off()


if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

source = load_junction_graph(SOURCE_PATH)
target = load_junction_graph(TARGET_PATH)

source_lines = graph_lines(source)
target_lines = graph_lines(target)

reports = {}

for cost_name in EXAMPLES:
    with (PAIR_DIR / cost_name / "report.json").open(encoding="utf-8") as file:
        reports[cost_name] = json.load(file)


fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)

axes = axes.ravel()

for axis, (cost_name, (label, edge_id, minimum_window)) in zip(axes, EXAMPLES.items()):
    solution = reports[cost_name]["solutions"]["additive"]

    payload = next(edge for edge in solution["edges"] if int(edge["edge_id"]) == edge_id)

    source_polyline = display_coordinates(source.edge_by_id[edge_id].polyline)

    witness = display_coordinates(payload["witness"])

    points = np.vstack((source_polyline, witness))
    low = points.min(axis=0)
    high = points.max(axis=0)
    centre = 0.5 * (low + high)

    side = max(minimum_window, float(np.max(high - low) + 28))

    half = side / 2

    draw_context(axis, source_lines, target_lines)

    axis.plot(source_polyline[:, 0], source_polyline[:, 1], color=SOURCE_HIGHLIGHT, linewidth=3.3, solid_capstyle="round", zorder=5)

    axis.plot(witness[:, 0], witness[:, 1], color=TARGET_HIGHLIGHT, linewidth=3.3, linestyle="--", dash_capstyle="round", zorder=6)

    source_u = display_coordinates(source.coordinates[int(payload["source_u"])])

    source_v = display_coordinates(source.coordinates[int(payload["source_v"])])

    target_u = display_coordinates(target.coordinates[int(payload["target_u"])])

    target_v = display_coordinates(target.coordinates[int(payload["target_v"])])

    axis.scatter([source_u[0], source_v[0]], [source_u[1], source_v[1]], marker="o", s=38, facecolor="white", edgecolor=SOURCE_HIGHLIGHT, linewidth=1.5, zorder=8)

    axis.scatter([target_u[0], target_v[0]], [target_u[1], target_v[1]], marker="s", s=42, facecolor="white", edgecolor=TARGET_HIGHLIGHT, linewidth=1.5, zorder=8)

    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    axis.set_title(f"{label}\nsource edge e{edge_id}, cost = {float(payload['cost']):.3g}")

axes[-1].set_visible(False)

fig.suptitle("Examples of the selected local edge costs")

fig.legend(
    handles=[
        Line2D([0], [0], color=SOURCE_CONTEXT, linewidth=1.5, label="Source graph"),
        Line2D([0], [0], color=TARGET_CONTEXT, linewidth=1.5, label="Target graph"),
        Line2D([0], [0], color=SOURCE_HIGHLIGHT, linewidth=3, label="Selected source edge"),
        Line2D([0], [0], color=TARGET_HIGHLIGHT, linewidth=3, linestyle="--", label="Selected target witness"),
    ],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.01),
)

save(fig, "selected_cost_examples")


length_solution = reports["relative_length_error"]["solutions"]["additive"]

tangent_solution = reports["mean_distance_tangent"]["solutions"]["additive"]

cost_changed = changed_vertices(length_solution, tangent_solution)

cost_regions = choose_regions(cost_changed, source, count=2)

cost_boxes = [region_bounds(source, length_solution, tangent_solution, vertex) for vertex in cost_regions]

fig = plt.figure(figsize=(12.5, 8.5), constrained_layout=True)

grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])

overview = fig.add_subplot(grid[0, :])

draw_difference_overview(overview, source, source_lines, target_lines, cost_changed, cost_boxes)

for index, (vertex, bounds) in enumerate(zip(cost_regions, cost_boxes)):
    label = chr(ord("A") + index)

    left_axis = fig.add_subplot(grid[1, 2 * index])

    right_axis = fig.add_subplot(grid[1, 2 * index + 1])

    draw_region(left_axis, source, target, source_lines, target_lines, length_solution, vertex, bounds, f"{label} — Relative length")

    draw_region(right_axis, source, target, source_lines, target_lines, tangent_solution, vertex, bounds, f"{label} — Mean distance + tangent")

fig.suptitle("Length preservation and geometric alignment")

fig.legend(
    handles=[
        Line2D([0], [0], color=SOURCE_HIGHLIGHT, linewidth=3, label="Affected source edges"),
        Line2D([0], [0], color=TARGET_HIGHLIGHT, linewidth=3, linestyle="--", label="Selected target witnesses"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=CHANGED, linestyle="", label="Changed source junction"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor=TARGET_HIGHLIGHT, linestyle="", label="Mapped target junction"),
    ],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.01),
)

save(fig, "length_vs_geometry")


objective_candidates = []

for cost_name, (label, _, _) in EXAMPLES.items():
    additive = reports[cost_name]["solutions"].get("additive")

    bottleneck = reports[cost_name]["solutions"].get("bottleneck")

    if additive is None or bottleneck is None:
        continue

    objective_candidates.append((agreement(additive, bottleneck), cost_name, label, additive, bottleneck))

(objective_agreement, objective_cost, objective_label, additive_solution, bottleneck_solution) = min(objective_candidates, key=lambda item: item[0])

objective_changed = changed_vertices(additive_solution, bottleneck_solution)

objective_regions = choose_regions(objective_changed, source, count=2)

objective_boxes = [region_bounds(source, additive_solution, bottleneck_solution, vertex) for vertex in objective_regions]

fig = plt.figure(figsize=(12.5, 8.5), constrained_layout=True)

grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])

overview = fig.add_subplot(grid[0, :])

draw_difference_overview(overview, source, source_lines, target_lines, objective_changed, objective_boxes)

for index, (vertex, bounds) in enumerate(zip(objective_regions, objective_boxes)):
    label = chr(ord("A") + index)

    additive_axis = fig.add_subplot(grid[1, 2 * index])

    bottleneck_axis = fig.add_subplot(grid[1, 2 * index + 1])

    draw_region(additive_axis, source, target, source_lines, target_lines, additive_solution, vertex, bounds, f"{label} — Additive")

    draw_region(bottleneck_axis, source, target, source_lines, target_lines, bottleneck_solution, vertex, bounds, f"{label} — Bottleneck")

fig.suptitle(f"{objective_label}: additive and bottleneck objectives")

fig.legend(
    handles=[
        Line2D([0], [0], color=SOURCE_HIGHLIGHT, linewidth=3, label="Affected source edges"),
        Line2D([0], [0], color=TARGET_HIGHLIGHT, linewidth=3, linestyle="--", label="Selected target witnesses"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=CHANGED, linestyle="", label="Changed source junction"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor=TARGET_HIGHLIGHT, linestyle="", label="Mapped target junction"),
    ],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.01),
)

save(fig, "additive_vs_bottleneck")

print(f"Figures written to {OUTPUT_DIR}")
print(f"Relative length versus mean distance and tangent agreement: {100 * agreement(length_solution, tangent_solution):.1f}%")
print(f"Largest additive-versus-bottleneck difference: {objective_label}, {100 * objective_agreement:.1f}% agreement")
