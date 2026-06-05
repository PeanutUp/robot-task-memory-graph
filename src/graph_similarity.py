from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism

from .graph_builder import Task, build_task_graph


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
WL_ITERATIONS = 2
LEARNED_FEATURES = [
    "wl_kernel",
    "semantic_cosine",
    "category",
    "edge_type",
    "node_role",
    "object_family_match",
    "target_family_match",
    "relation_match",
    "target_label_cosine",
    "object_label_cosine",
    "success_cost_similarity",
]


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


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _document_frequency(documents: list[Counter[str]]) -> Counter[str]:
    df: Counter[str] = Counter()
    for document in documents:
        df.update(document.keys())
    return df


def _tfidf(document: Counter[str], df: Counter[str], doc_count: int) -> Counter[str]:
    vector: Counter[str] = Counter()
    for feature, count in document.items():
        idf = math.log((doc_count + 1) / (df[feature] + 1)) + 1.0
        vector[feature] = float(count) * idf
    return vector


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


def _semantic_feature_counter(task: Task) -> Counter[str]:
    metadata = task.get("metadata", {})
    features: Counter[str] = Counter()

    def add_prefixed(prefix: str, values: Any) -> None:
        for token, count in _tokenize(values).items():
            features[f"{prefix}:{token}"] += count

    add_prefixed("title", task.get("title"))
    add_prefixed("goal", task.get("goal"))
    add_prefixed("tag", task.get("tags", []))
    add_prefixed("category", task.get("category"))
    add_prefixed("object", metadata.get("object_label"))
    add_prefixed("target", metadata.get("target_label"))
    add_prefixed("object_family", metadata.get("object_family"))
    add_prefixed("target_family", metadata.get("target_family"))
    add_prefixed("relation", metadata.get("relation"))
    return features


def _metadata(task: Task) -> dict[str, Any]:
    return task.get("metadata", {})


def _token_cosine(left: Any, right: Any) -> float:
    return _cosine(_tokenize(left), _tokenize(right))


def _same_metadata(task_a: Task, task_b: Task, key: str) -> float:
    value_a = _metadata(task_a).get(key)
    value_b = _metadata(task_b).get(key)
    return 1.0 if value_a and value_a == value_b else 0.0


def _initial_wl_labels(graph: nx.DiGraph) -> dict[str, str]:
    return {
        node_id: f"{attrs.get('type', 'unknown')}|{attrs.get('role', 'unknown')}"
        for node_id, attrs in graph.nodes(data=True)
    }


def _node_by_role(graph: nx.DiGraph, role: str) -> str | None:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("role") == role:
            return node_id
    return None


def _success_path_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
    start = _node_by_role(graph, "start") or "start"
    success = _node_by_role(graph, "success") or "success"
    try:
        path = nx.shortest_path(graph, start, success, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return graph
    return graph.subgraph(path).copy()


def _success_path_cost(graph: nx.DiGraph) -> float:
    start = _node_by_role(graph, "start") or "start"
    success = _node_by_role(graph, "success") or "success"
    try:
        path = nx.shortest_path(graph, start, success, weight="cost")
        return float(nx.path_weight(graph, path, weight="cost"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def wl_graph_features(graph: nx.DiGraph, iterations: int = WL_ITERATIONS) -> Counter[str]:
    """Extract Weisfeiler-Lehman subtree features for a directed task graph.

    The feature vector is a graph-kernel style representation: nodes are
    iteratively relabeled by their incoming/outgoing typed neighborhoods, then
    label histograms are compared with cosine similarity in retrieval.
    """
    labels = _initial_wl_labels(graph)
    features: Counter[str] = Counter(f"wl0:{label}" for label in labels.values())

    for source, target, attrs in graph.edges(data=True):
        source_role = graph.nodes[source].get("role", "unknown")
        target_role = graph.nodes[target].get("role", "unknown")
        edge_type = attrs.get("type", "unknown")
        features[f"edge:{source_role}>{edge_type}>{target_role}"] += 1

    for iteration in range(1, iterations + 1):
        next_labels: dict[str, str] = {}
        for node_id in graph.nodes:
            incoming = sorted(
                f"{graph.edges[pred, node_id].get('type', 'unknown')}:{labels[pred]}"
                for pred in graph.predecessors(node_id)
            )
            outgoing = sorted(
                f"{graph.edges[node_id, succ].get('type', 'unknown')}:{labels[succ]}"
                for succ in graph.successors(node_id)
            )
            next_labels[node_id] = (
                labels[node_id]
                + "|in="
                + ",".join(incoming)
                + "|out="
                + ",".join(outgoing)
            )
        labels = next_labels
        features.update(f"wl{iteration}:{label}" for label in labels.values())

    return features


def plan_aware_wl_features(graph: nx.DiGraph, iterations: int = WL_ITERATIONS) -> Counter[str]:
    """Represent reusable task structure with emphasis on the success path."""
    success_path = _success_path_subgraph(graph)
    features: Counter[str] = Counter()

    for feature, count in wl_graph_features(success_path, iterations).items():
        features[f"path:{feature}"] += 2 * count

    for feature, count in wl_graph_features(graph, iterations).items():
        features[f"full:{feature}"] += count

    return features


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


def _fusion_score(features: dict[str, float]) -> float:
    return (
        0.50 * features["wl_kernel"]
        + 0.35 * features["semantic_cosine"]
        + 0.10 * features["category"]
        + 0.05 * features["edge_type"]
    )


def _is_reusable_pair(query_task: Task, candidate_task: Task) -> bool:
    """Weak supervision label for learning retrieval weights from task pairs."""
    if query_task.get("task_id") == candidate_task.get("task_id"):
        return True

    query_meta = _metadata(query_task)
    candidate_meta = _metadata(candidate_task)
    same_relation = query_meta.get("relation") == candidate_meta.get("relation")
    same_target_family = query_meta.get("target_family") == candidate_meta.get("target_family")
    same_object_family = query_meta.get("object_family") == candidate_meta.get("object_family")
    same_category = query_task.get("category") == candidate_task.get("category")

    return bool(
        (same_target_family and same_relation)
        or (same_object_family and same_target_family)
        or (same_category and same_object_family and same_relation)
    )


def _build_similarity_rows(
    query_task: Task,
    candidate_tasks: list[Task],
) -> list[dict[str, Any]]:
    query_graph = build_task_graph(query_task)
    candidate_graphs = [build_task_graph(task) for task in candidate_tasks]

    structural_raw = [plan_aware_wl_features(query_graph)] + [
        plan_aware_wl_features(graph) for graph in candidate_graphs
    ]
    semantic_raw = [_semantic_feature_counter(query_task)] + [
        _semantic_feature_counter(task) for task in candidate_tasks
    ]
    structural_df = _document_frequency(structural_raw)
    semantic_df = _document_frequency(semantic_raw)
    doc_count = len(structural_raw)

    query_structural = _tfidf(structural_raw[0], structural_df, doc_count)
    query_semantic = _tfidf(semantic_raw[0], semantic_df, doc_count)
    query_cost = _success_path_cost(query_graph)

    rows: list[dict[str, Any]] = []
    for index, task in enumerate(candidate_tasks):
        candidate_graph = candidate_graphs[index]
        candidate_cost = _success_path_cost(candidate_graph)
        cost_gap = abs(query_cost - candidate_cost)
        if math.isinf(cost_gap):
            success_cost_similarity = 0.0
        else:
            success_cost_similarity = 1.0 / (1.0 + cost_gap)

        features = {
            "wl_kernel": _cosine(
                query_structural,
                _tfidf(structural_raw[index + 1], structural_df, doc_count),
            ),
            "semantic_cosine": _cosine(
                query_semantic,
                _tfidf(semantic_raw[index + 1], semantic_df, doc_count),
            ),
            "category": 1.0 if query_task.get("category") == task.get("category") else 0.0,
            "edge_type": _multiset_jaccard(
                _edge_attr_counter(query_graph, "type"),
                _edge_attr_counter(candidate_graph, "type"),
            ),
            "node_type": _multiset_jaccard(
                _node_attr_counter(query_graph, "type"),
                _node_attr_counter(candidate_graph, "type"),
            ),
            "node_role": _multiset_jaccard(
                _node_attr_counter(query_graph, "role"),
                _node_attr_counter(candidate_graph, "role"),
            ),
            "semantic": _multiset_jaccard(
                _semantic_counter(query_task),
                _semantic_counter(task),
            ),
            "object_family_match": _same_metadata(query_task, task, "object_family"),
            "target_family_match": _same_metadata(query_task, task, "target_family"),
            "relation_match": _same_metadata(query_task, task, "relation"),
            "target_label_cosine": _token_cosine(
                _metadata(query_task).get("target_label"),
                _metadata(task).get("target_label"),
            ),
            "object_label_cosine": _token_cosine(
                _metadata(query_task).get("object_label"),
                _metadata(task).get("object_label"),
            ),
            "success_cost_similarity": success_cost_similarity,
        }
        rows.append(
            {
                "task": task,
                "graph": candidate_graph,
                "features": features,
                "fusion_score": _fusion_score(features),
            }
        )
    return rows


def train_retrieval_weights(
    training_tasks: list[Task],
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.015,
) -> dict[str, Any]:
    examples: list[tuple[list[float], float]] = []
    for query_task in training_tasks:
        for row in _build_similarity_rows(query_task, training_tasks):
            label = 1.0 if _is_reusable_pair(query_task, row["task"]) else 0.0
            vector = [row["features"][feature] for feature in LEARNED_FEATURES]
            examples.append((vector, label))

    positive_count = sum(label for _, label in examples)
    negative_count = len(examples) - positive_count
    if not positive_count or not negative_count:
        weights = {feature: 1.0 / len(LEARNED_FEATURES) for feature in LEARNED_FEATURES}
        return {
            "algorithm": "fallback_uniform_ranker",
            "weights": weights,
            "bias": 0.0,
            "positive_pairs": int(positive_count),
            "negative_pairs": int(negative_count),
        }

    weights = {feature: 0.0 for feature in LEARNED_FEATURES}
    bias = math.log(positive_count / negative_count)
    pos_weight = negative_count / positive_count

    for _ in range(epochs):
        for vector, label in examples:
            score = bias + sum(weights[name] * value for name, value in zip(LEARNED_FEATURES, vector))
            prediction = _sigmoid(score)
            sample_weight = pos_weight if label == 1.0 else 1.0
            error = (prediction - label) * sample_weight

            for name, value in zip(LEARNED_FEATURES, vector):
                weights[name] -= learning_rate * (error * value + l2 * weights[name])
            bias -= learning_rate * error

    return {
        "algorithm": "weakly_supervised_logistic_ranker",
        "weights": weights,
        "bias": bias,
        "positive_pairs": int(positive_count),
        "negative_pairs": int(negative_count),
        "features": LEARNED_FEATURES,
    }


def _predict_reuse_probability(features: dict[str, float], model: dict[str, Any]) -> float:
    weights = model["weights"]
    score = model["bias"] + sum(weights[name] * features[name] for name in LEARNED_FEATURES)
    return _sigmoid(score)


def task_similarity(
    query_task: Task,
    candidate_task: Task,
    include_ged: bool = False,
) -> dict[str, Any]:
    row = _build_similarity_rows(query_task, [candidate_task])[0]
    features = row["features"]

    breakdown = {
        "score": round(row["fusion_score"], 4),
        "algorithm": "manual_wl_tfidf_fusion",
        "fusion_score": round(row["fusion_score"], 4),
        "wl_kernel": round(features["wl_kernel"], 4),
        "semantic_cosine": round(features["semantic_cosine"], 4),
        "node_type": round(features["node_type"], 4),
        "node_role": round(features["node_role"], 4),
        "edge_type": round(features["edge_type"], 4),
        "semantic": round(features["semantic"], 4),
        "category": round(features["category"], 4),
        "object_family_match": round(features["object_family_match"], 4),
        "target_family_match": round(features["target_family_match"], 4),
        "relation_match": round(features["relation_match"], 4),
        "target_label_cosine": round(features["target_label_cosine"], 4),
        "object_label_cosine": round(features["object_label_cosine"], 4),
        "success_cost_similarity": round(features["success_cost_similarity"], 4),
    }

    if include_ged:
        ged = graph_edit_similarity(build_task_graph(query_task), row["graph"])
        breakdown["graph_edit"] = round(ged, 4)

    return breakdown


def rank_tasks(
    query_task: Task,
    candidate_tasks: list[Task],
    top_k: int | None = None,
    include_ged: bool = False,
    use_learned_ranker: bool = True,
) -> list[dict[str, Any]]:
    rows = _build_similarity_rows(query_task, candidate_tasks)
    model = train_retrieval_weights(candidate_tasks)

    rankings: list[dict[str, Any]] = []
    for row in rows:
        task = row["task"]
        candidate_graph = row["graph"]
        features = row["features"]
        learned_score = _predict_reuse_probability(features, model)
        score = learned_score if use_learned_ranker else row["fusion_score"]
        breakdown: dict[str, Any] = {
            "score": round(score, 4),
            "algorithm": model["algorithm"] if use_learned_ranker else "manual_wl_tfidf_fusion",
            "learned_score": round(learned_score, 4),
            "fusion_score": round(row["fusion_score"], 4),
            "wl_kernel": round(features["wl_kernel"], 4),
            "semantic_cosine": round(features["semantic_cosine"], 4),
            "node_type": round(features["node_type"], 4),
            "node_role": round(features["node_role"], 4),
            "edge_type": round(features["edge_type"], 4),
            "semantic": round(features["semantic"], 4),
            "category": round(features["category"], 4),
            "object_family_match": round(features["object_family_match"], 4),
            "target_family_match": round(features["target_family_match"], 4),
            "relation_match": round(features["relation_match"], 4),
            "target_label_cosine": round(features["target_label_cosine"], 4),
            "object_label_cosine": round(features["object_label_cosine"], 4),
            "success_cost_similarity": round(features["success_cost_similarity"], 4),
            "learned_weights": {
                name: round(value, 4)
                for name, value in model["weights"].items()
            },
            "training_pairs": {
                "positive": model["positive_pairs"],
                "negative": model["negative_pairs"],
            },
        }
        if include_ged:
            ged = graph_edit_similarity(build_task_graph(query_task), candidate_graph)
            breakdown["graph_edit"] = round(ged, 4)
        rankings.append(
            {
                "task": task,
                "score": breakdown["score"],
                "breakdown": breakdown,
                "graph": candidate_graph,
            }
        )
    rankings.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        return rankings[:top_k]
    return rankings
