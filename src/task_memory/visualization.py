from __future__ import annotations

import io
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/task-graph-memory-matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from .graph import build_task_graph
from .schema import RolloutFrame, TaskSpec


def draw_task_graph(task: TaskSpec) -> plt.Figure:
    graph = build_task_graph(task)
    node_colors = {
        "object": "#F59E0B",
        "location": "#64748B",
        "action": "#2563EB",
        "state": "#8B5CF6",
        "result": "#10B981",
    }
    colors = [node_colors.get(graph.nodes[node]["type"], "#94A3B8") for node in graph.nodes]
    labels = {node: graph.nodes[node]["label"].replace("_", "\n") for node in graph.nodes}
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    pos = nx.spring_layout(graph, seed=7, k=0.7)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=900, linewidths=1.5, edgecolors="white", ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_color="#111827", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.1, edge_color="#94A3B8", ax=ax)
    edge_labels = {(u, v): data["type"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.set_title(f"{task.task_id}: {task.object_name} -> {task.target_name}", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_rollout_timeline(frames: list[RolloutFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 2.7))
    actions = [frame.predicted_action for frame in frames]
    colors = ["#10B981" if not frame.failed else "#EF4444" for frame in frames]
    ax.bar(range(len(actions)), [1] * len(actions), color=colors)
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels([action.replace("_", "\n") for action in actions], fontsize=8)
    ax.set_yticks([])
    ax.set_title("Closed-loop predicted action sequence")
    for index, frame in enumerate(frames):
        ax.text(index, 0.5, str(frame.step), ha="center", va="center", color="white", fontsize=8, weight="bold")
    fig.tight_layout()
    return fig


def rollout_gif(frames: list[RolloutFrame], duration_ms: int = 420) -> bytes:
    images: list[Image.Image] = []
    for index in range(1, len(frames) + 1):
        fig = draw_rollout_timeline(frames[:index])
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("P"))
    output = io.BytesIO()
    images[0].save(output, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return output.getvalue()
