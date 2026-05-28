from __future__ import annotations

from typing import Any

import networkx as nx

from .graph_builder import Task, get_node_by_role, get_task_context


def shortest_success_path(graph: nx.DiGraph) -> tuple[list[str], float]:
    start = get_node_by_role(graph, "start") or "start"
    goal = get_node_by_role(graph, "success") or "success"
    path = nx.shortest_path(graph, start, goal, weight="cost")
    cost = nx.path_weight(graph, path, weight="cost")
    return path, float(cost)


def _format_node_label(attrs: dict[str, Any], context: dict[str, str]) -> str:
    template = attrs.get("template") or attrs.get("label") or attrs.get("role") or "step"
    try:
        return str(template).format(**context)
    except KeyError:
        return str(template)


def generate_plan(history_graph: nx.DiGraph, query_task: Task) -> dict[str, Any]:
    path, cost = shortest_success_path(history_graph)
    context = get_task_context(query_task)

    steps = []
    for index, node_id in enumerate(path):
        attrs = dict(history_graph.nodes[node_id])
        steps.append(
            {
                "index": index,
                "node_id": node_id,
                "role": attrs.get("role", node_id),
                "type": attrs.get("type", "unknown"),
                "label": _format_node_label(attrs, context),
            }
        )

    edges = list(zip(path, path[1:]))
    return {
        "path": path,
        "edges": edges,
        "cost": cost,
        "steps": steps,
        "context": context,
    }

