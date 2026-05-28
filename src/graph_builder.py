from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


Task = dict[str, Any]


def load_task_file(path: str | Path) -> list[Task]:
    """Load one JSON file that contains either one task or a list of tasks."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def load_tasks(directory: str | Path) -> list[Task]:
    tasks: list[Task] = []
    for path in sorted(Path(directory).glob("*.json")):
        tasks.extend(load_task_file(path))
    return tasks


def build_task_graph(task: Task) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph.update(
        task_id=task.get("task_id"),
        title=task.get("title"),
        goal=task.get("goal"),
        category=task.get("category"),
        tags=task.get("tags", []),
        metadata=task.get("metadata", {}),
        scene=task.get("scene", {}),
    )

    for node in task.get("nodes", []):
        node_id = node["id"]
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)

    for edge in task.get("edges", []):
        attrs = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target"}
        }
        cost = float(attrs.get("cost", 1.0))
        attrs["cost"] = cost
        attrs["weight"] = cost
        graph.add_edge(edge["source"], edge["target"], **attrs)

    return graph


def build_graphs(tasks: list[Task]) -> dict[str, nx.DiGraph]:
    return {task["task_id"]: build_task_graph(task) for task in tasks}


def get_node_by_role(graph: nx.DiGraph, role: str) -> str | None:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("role") == role:
            return node_id
    return None


def get_task_context(task: Task) -> dict[str, str]:
    metadata = task.get("metadata", {})
    context = {
        "object": metadata.get("object_label", "object"),
        "target": metadata.get("target_label", "target"),
        "relation": metadata.get("relation", "into"),
    }

    for node in task.get("nodes", []):
        role = node.get("role")
        if role == "object":
            context["object"] = metadata.get("object_label", node.get("label", "object"))
        elif role == "target":
            context["target"] = metadata.get("target_label", node.get("label", "target"))

    return context

