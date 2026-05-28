from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism

from .graph_builder import Task, build_task_graph


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 0.0


def _tokenize(*values: Any) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            tokens.update(_tokenize(*value))
            continue
        for token in TOKEN_RE.findall(str(value).lower()):
            tokens[token] += 1
    return tokens


def _node_attr_counter(graph: nx.DiGraph, attr: str) -> Counter[str]:
    return Counter(str(data.get(attr, "")) for _, data in graph.nodes(data=True))


def _edge_attr_counter(graph: nx.DiGraph, attr: str) -> Counter[str]:
    return Counter(str(data.get(attr, "")) for _, _, data in graph.edges(data=True))


def _semantic_counter(task: Task) -> Counter[str]:
    metadata = task.get("metadata", {})
    values = [
        task.get("title"),
        task.get("goal"),
        task.get("category"),
        task.get("tags", []),
        metadata.get("object_label"),
        metadata.get("target_label"),
        metadata.get("object_family"),
        metadata.get("target_family"),
        metadata.get("relation"),
    ]
    return _tokenize(*values)


def graph_edit_similarity(
    query_graph: nx.DiGraph,
    candidate_graph: nx.DiGraph,
    timeout: float = 0.25,
) -> float:
    node_match = isomorphism.categorical_node_match(["type", "role"], [None, None])
    edge_match = isomorphism.categorical_edge_match("type", None)
    try:
        distance = nx.graph_edit_distance(
            query_graph,
            candidate_graph,
            node_match=node_match,
            edge_match=edge_match,
            timeout=timeout,
        )
    except Exception:
        return 0.0
    if distance is None or math.isinf(distance):
        return 0.0
    return 1.0 / (1.0 + float(distance))


def task_similarity(
    query_task: Task,
    candidate_task: Task,
    include_ged: bool = False,
) -> dict[str, float]:
    query_graph = build_task_graph(query_task)
    candidate_graph = build_task_graph(candidate_task)

    node_type = _multiset_jaccard(
        _node_attr_counter(query_graph, "type"),
        _node_attr_counter(candidate_graph, "type"),
    )
    node_role = _multiset_jaccard(
        _node_attr_counter(query_graph, "role"),
        _node_attr_counter(candidate_graph, "role"),
    )
    edge_type = _multiset_jaccard(
        _edge_attr_counter(query_graph, "type"),
        _edge_attr_counter(candidate_graph, "type"),
    )
    semantic = _multiset_jaccard(
        _semantic_counter(query_task),
        _semantic_counter(candidate_task),
    )
    category = 1.0 if query_task.get("category") == candidate_task.get("category") else 0.0

    score = (
        0.15 * node_type
        + 0.20 * node_role
        + 0.20 * edge_type
        + 0.30 * semantic
        + 0.15 * category
    )

    breakdown = {
        "score": round(score, 4),
        "node_type": round(node_type, 4),
        "node_role": round(node_role, 4),
        "edge_type": round(edge_type, 4),
        "semantic": round(semantic, 4),
        "category": round(category, 4),
    }

    if include_ged:
        ged = graph_edit_similarity(query_graph, candidate_graph)
        breakdown["graph_edit"] = round(ged, 4)
        breakdown["score"] = round(0.85 * score + 0.15 * ged, 4)

    return breakdown


def rank_tasks(
    query_task: Task,
    candidate_tasks: list[Task],
    top_k: int | None = None,
    include_ged: bool = False,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for task in candidate_tasks:
        breakdown = task_similarity(query_task, task, include_ged=include_ged)
        rankings.append(
            {
                "task": task,
                "score": breakdown["score"],
                "breakdown": breakdown,
                "graph": build_task_graph(task),
            }
        )
    rankings.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        return rankings[:top_k]
    return rankings

