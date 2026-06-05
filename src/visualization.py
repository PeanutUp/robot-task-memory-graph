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

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    
    # Remove standard grid and axes for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect("equal")
    
    # Title showing the current message
    ax.set_title(frame["message"], fontsize=13, pad=15, weight="bold", color="#343A40", loc="left")

    # Draw rounded tiles instead of grid lines
    from matplotlib.patches import FancyBboxPatch, Wedge, Polygon
    for x in range(width):
        for y in range(height):
            # Checkerboard styling
            color = "#E9ECEF" if (x + y) % 2 == 0 else "#DEE2E6"
            tile = FancyBboxPatch(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                ec="none", fc=color, zorder=0
            )
            ax.add_patch(tile)

    # Draw obstacles (boxes)
    for x, y in frame.get("obstacles", []):
        base = FancyBboxPatch(
            (x - 0.4, y - 0.4), 0.8, 0.8,
            boxstyle="round,pad=0.0,rounding_size=0.1",
            ec="#495057", fc="#6C757D", lw=1.5, zorder=1
        )
        ax.add_patch(base)
        ax.plot([x - 0.4, x + 0.4], [y - 0.4, y + 0.4], color="#495057", lw=1.5, zorder=1)
        ax.plot([x - 0.4, x + 0.4], [y + 0.4, y - 0.4], color="#495057", lw=1.5, zorder=1)

    # Draw target zone (Goal area)
    zone = FancyBboxPatch(
        (target_x - 0.42, target_y - 0.42), 0.84, 0.84,
        boxstyle="round,pad=0.0,rounding_size=0.2",
        ec="#2B8A3E", fc="#D3F9D8", lw=2.5, linestyle="--", zorder=1
    )
    ax.add_patch(zone)
    ax.text(target_x, target_y, "TARGET\nZONE", ha="center", va="center", fontsize=9, weight="bold", color="#2B8A3E", zorder=2)

    # Draw the object if it's not carried
    if not frame["carrying"]:
        # Draw a little package
        obj_base = FancyBboxPatch(
            (object_x - 0.25, object_y - 0.25), 0.5, 0.5,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            ec="#E67700", fc="#FCC419", lw=2, zorder=3
        )
        ax.add_patch(obj_base)
        # Add a bow/tape line to logic object
        ax.plot([object_x - 0.25, object_x + 0.25], [object_y, object_y], color="#E67700", lw=2, zorder=3)
        ax.plot([object_x, object_x], [object_y - 0.25, object_y + 0.25], color="#E67700", lw=2, zorder=3)

    # Draw the Robot
    # Shadow
    ax.add_patch(Circle((robot_x, robot_y + 0.3), 0.3, fc='black', alpha=0.15, zorder=4))
    
    # Tracks / Wheels
    t_left = FancyBboxPatch((robot_x - 0.35, robot_y - 0.15), 0.15, 0.4, boxstyle="round,pad=0,rounding_size=0.1", fc="#212529", zorder=5)
    t_right = FancyBboxPatch((robot_x + 0.2, robot_y - 0.15), 0.15, 0.4, boxstyle="round,pad=0,rounding_size=0.1", fc="#212529", zorder=5)
    ax.add_patch(t_left)
    ax.add_patch(t_right)

    # Main Body
    body = FancyBboxPatch(
        (robot_x - 0.3, robot_y - 0.25), 0.6, 0.55,
        boxstyle="round,pad=0.0,rounding_size=0.1",
        ec="#1864AB", fc="#339AF0", lw=2, zorder=6
    )
    ax.add_patch(body)

    # Head / Screen
    screen = FancyBboxPatch(
        (robot_x - 0.2, robot_y - 0.15), 0.4, 0.25,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        ec="#1864AB", fc="#E3FAFC", lw=1.5, zorder=7
    )
    ax.add_patch(screen)

    # Eyes (Cute robot eyes)
    ax.add_patch(Circle((robot_x - 0.08, robot_y - 0.05), 0.04, color="#1864AB", zorder=8))
    ax.add_patch(Circle((robot_x + 0.08, robot_y - 0.05), 0.04, color="#1864AB", zorder=8))
    
    # Antenna
    ax.plot([robot_x, robot_x], [robot_y - 0.25, robot_y - 0.4], color="#1864AB", lw=2, zorder=5)
    ax.add_patch(Circle((robot_x, robot_y - 0.42), 0.06, color="#FA5252", zorder=6))

    if frame["carrying"]:
        # Draw the object being carried above the robot
        carry_y = robot_y - 0.45
        obj_base = FancyBboxPatch(
            (robot_x - 0.15, carry_y - 0.15), 0.3, 0.3,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            ec="#E67700", fc="#FCC419", lw=1.5, zorder=9
        )
        ax.add_patch(obj_base)
        ax.plot([robot_x - 0.15, robot_x + 0.15], [carry_y, carry_y], color="#E67700", lw=1.5, zorder=9)
        ax.plot([robot_x, robot_x], [carry_y - 0.15, carry_y + 0.15], color="#E67700", lw=1.5, zorder=9)

    # Legend text conceptually
    ax.text(0.5, -0.05, f"Object: {frame['object_label']}    |    Target: {frame['target_label']}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#495057", weight="bold")

    fig.tight_layout()
    return fig
