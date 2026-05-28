from __future__ import annotations

from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Rectangle


TYPE_COLORS = {
    "object": "#FFD166",
    "location": "#7BDFF2",
    "subtask": "#B8F2E6",
    "action": "#A0C4FF",
    "state": "#E4C1F9",
    "result": "#95D5B2",
}

ROLE_X = {
    "object": 0,
    "target": 0,
    "start": 0,
    "search": 1,
    "detect": 2,
    "move_to_object": 3,
    "pick": 4,
    "grasp_failed": 4.6,
    "retry_pick": 5,
    "in_hand": 5.6,
    "move_to_target": 6.7,
    "path_blocked": 7.2,
    "reroute": 7.7,
    "place": 8.2,
    "success": 9.2,
}

ROLE_Y = {
    "object": 1.1,
    "target": -1.1,
    "start": 0,
    "search": 0.9,
    "detect": 0,
    "move_to_object": 0,
    "pick": 0,
    "grasp_failed": 1.25,
    "retry_pick": 1.25,
    "in_hand": 0,
    "move_to_target": 0,
    "path_blocked": -1.25,
    "reroute": -1.25,
    "place": 0,
    "success": 0,
}


def _layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    counts: dict[str, int] = {}
    positions: dict[str, tuple[float, float]] = {}
    for node_id, attrs in graph.nodes(data=True):
        role = attrs.get("role", node_id)
        x = ROLE_X.get(role, len(positions))
        y = ROLE_Y.get(role, 0)
        counts[role] = counts.get(role, 0) + 1
        positions[node_id] = (x, y + 0.25 * (counts[role] - 1))
    return positions


def draw_task_graph(
    graph: nx.DiGraph,
    highlighted_nodes: set[str] | None = None,
    highlighted_edges: set[tuple[str, str]] | None = None,
    matched_nodes: set[str] | None = None,
    active_node: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    highlighted_nodes = highlighted_nodes or set()
    highlighted_edges = highlighted_edges or set()
    matched_nodes = matched_nodes or set()
    pos = _layout(graph)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_title(title or graph.graph.get("title", "Task graph"), fontsize=12, pad=10)

    node_colors = []
    edgecolors = []
    linewidths = []
    for node_id, attrs in graph.nodes(data=True):
        if node_id == active_node:
            node_colors.append("#F94144")
            edgecolors.append("#7A0404")
            linewidths.append(2.6)
        elif node_id in highlighted_nodes:
            node_colors.append("#F8961E")
            edgecolors.append("#7A4A00")
            linewidths.append(2.2)
        else:
            node_colors.append(TYPE_COLORS.get(attrs.get("type"), "#DDDDDD"))
            edgecolors.append("#586069" if node_id in matched_nodes else "#FFFFFF")
            linewidths.append(2.0 if node_id in matched_nodes else 1.0)

    normal_edges = [edge for edge in graph.edges() if edge not in highlighted_edges]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=normal_edges,
        edge_color="#AAB2BD",
        arrows=True,
        arrowsize=12,
        width=1.4,
        ax=ax,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=list(highlighted_edges),
        edge_color="#E63946",
        arrows=True,
        arrowsize=16,
        width=2.8,
        ax=ax,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        node_size=1750,
        ax=ax,
    )

    labels = {
        node_id: fill(str(attrs.get("label", node_id)), width=13)
        for node_id, attrs in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8.5, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_execution_frame(frame: dict[str, Any]) -> plt.Figure:
    width, height = frame["grid_size"]
    robot_x, robot_y = frame["robot_pos"]
    object_x, object_y = frame["object_pos"]
    target_x, target_y = frame["target_pos"]

    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, color="#D0D7DE", linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_title(frame["message"], fontsize=12, pad=10)

    for x, y in frame.get("obstacles", []):
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color="#ADB5BD"))
        ax.text(x, y, "X", ha="center", va="center", fontsize=12, weight="bold", color="#343A40")

    ax.add_patch(Rectangle((target_x - 0.42, target_y - 0.42), 0.84, 0.84, color="#95D5B2", alpha=0.8))
    ax.text(target_x, target_y + 0.33, "TARGET", ha="center", va="center", fontsize=8, weight="bold")

    if not frame["carrying"]:
        ax.add_patch(Circle((object_x, object_y), 0.28, color="#FFD166"))
        ax.text(object_x, object_y, "OBJ", ha="center", va="center", fontsize=8, weight="bold")

    ax.add_patch(Circle((robot_x, robot_y), 0.34, color="#4D96FF"))
    ax.text(robot_x, robot_y, "BOT", ha="center", va="center", fontsize=8, weight="bold", color="white")

    if frame["carrying"]:
        ax.add_patch(Circle((robot_x + 0.28, robot_y - 0.28), 0.18, color="#FFD166"))
        ax.text(robot_x + 0.28, robot_y - 0.28, "OBJ", ha="center", va="center", fontsize=6, weight="bold")

    ax.set_xlabel(f"Object: {frame['object_label']}    Target: {frame['target_label']}")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

