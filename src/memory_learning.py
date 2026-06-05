from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .graph_builder import Task, build_task_graph
from .graph_similarity import (
    _cosine,
    _semantic_feature_counter,
    _success_path_cost,
    _token_cosine,
    plan_aware_wl_features,
)


MODEL_FEATURES = [
    "wl_kernel",
    "semantic_cosine",
    "same_category",
    "same_object_family",
    "same_target_family",
    "same_relation",
    "same_risk_type",
    "same_strategy",
    "target_label_cosine",
    "object_label_cosine",
    "success_cost_similarity",
]


@dataclass(frozen=True)
class TaskView:
    task: Task
    graph_features: dict[str, float]
    semantic_features: dict[str, float]
    success_cost: float


class MemoryRetrievalModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        )
        self.training_report: dict[str, Any] = {}

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "MemoryRetrievalModel":
        x_scaled = self.scaler.fit_transform(x_train)
        self.classifier.fit(x_scaled, y_train)
        prediction = self.classifier.predict(x_scaled)
        probability = self.classifier.predict_proba(x_scaled)[:, 1]
        self.training_report = {
            "train_accuracy": float(accuracy_score(y_train, prediction)),
            "train_auc": float(roc_auc_score(y_train, probability)),
            "positive_pairs": int(y_train.sum()),
            "negative_pairs": int(len(y_train) - y_train.sum()),
            "weights": {
                name: float(weight)
                for name, weight in zip(MODEL_FEATURES, self.classifier.coef_[0])
            },
            "intercept": float(self.classifier.intercept_[0]),
        }
        return self

    def predict_reuse_probability(self, features: list[float]) -> float:
        x_scaled = self.scaler.transform(np.array([features], dtype=float))
        return float(self.classifier.predict_proba(x_scaled)[0, 1])


def task_view(task: Task) -> TaskView:
    graph = build_task_graph(task)
    return TaskView(
        task=task,
        graph_features=dict(plan_aware_wl_features(graph)),
        semantic_features=dict(_semantic_feature_counter(task)),
        success_cost=_success_path_cost(graph),
    )


def build_task_views(tasks: list[Task]) -> list[TaskView]:
    return [task_view(task) for task in tasks]


def pair_feature_dict(query: TaskView, memory: TaskView) -> dict[str, float]:
    query_meta = query.task["metadata"]
    memory_meta = memory.task["metadata"]
    cost_gap = abs(query.success_cost - memory.success_cost)
    success_cost_similarity = 0.0 if math.isinf(cost_gap) else 1.0 / (1.0 + cost_gap)
    return {
        "wl_kernel": _cosine(query.graph_features, memory.graph_features),
        "semantic_cosine": _cosine(query.semantic_features, memory.semantic_features),
        "same_category": 1.0 if query.task.get("category") == memory.task.get("category") else 0.0,
        "same_object_family": 1.0 if query_meta.get("object_family") == memory_meta.get("object_family") else 0.0,
        "same_target_family": 1.0 if query_meta.get("target_family") == memory_meta.get("target_family") else 0.0,
        "same_relation": 1.0 if query_meta.get("relation") == memory_meta.get("relation") else 0.0,
        "same_risk_type": 1.0 if query_meta.get("risk_type") == memory_meta.get("risk_type") else 0.0,
        "same_strategy": 1.0 if query_meta.get("required_strategy") == memory_meta.get("required_strategy") else 0.0,
        "target_label_cosine": _token_cosine(query_meta.get("target_label"), memory_meta.get("target_label")),
        "object_label_cosine": _token_cosine(query_meta.get("object_label"), memory_meta.get("object_label")),
        "success_cost_similarity": success_cost_similarity,
    }


def pair_feature_vector(query: TaskView, memory: TaskView) -> list[float]:
    features = pair_feature_dict(query, memory)
    return [features[name] for name in MODEL_FEATURES]


def reusable_label(query: TaskView, memory: TaskView) -> int:
    query_meta = query.task["metadata"]
    memory_meta = memory.task["metadata"]
    same_target = query_meta["target_family"] == memory_meta["target_family"]
    same_relation = query_meta["relation"] == memory_meta["relation"]
    same_risk = query_meta["risk_type"] == memory_meta["risk_type"]
    same_strategy = query_meta["required_strategy"] == memory_meta["required_strategy"]
    same_object = query_meta["object_family"] == memory_meta["object_family"]
    return int((same_target and same_relation and same_strategy) or (same_object and same_risk and same_relation))


def sample_training_pairs(
    query_views: list[TaskView],
    memory_views: list[TaskView],
    pair_count: int,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    positives_by_key: dict[tuple[str, str, str], list[TaskView]] = {}
    for memory in memory_views:
        meta = memory.task["metadata"]
        key = (meta["target_family"], meta["relation"], meta["required_strategy"])
        positives_by_key.setdefault(key, []).append(memory)

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for index in range(pair_count):
        query = rng.choice(query_views)
        query_meta = query.task["metadata"]
        desired_positive = index % 2 == 0
        if desired_positive:
            key = (query_meta["target_family"], query_meta["relation"], query_meta["required_strategy"])
            candidates = positives_by_key.get(key)
            memory = rng.choice(candidates) if candidates else rng.choice(memory_views)
        else:
            memory = rng.choice(memory_views)
            attempts = 0
            while reusable_label(query, memory) == 1 and attempts < 50:
                memory = rng.choice(memory_views)
                attempts += 1
        x_rows.append(pair_feature_vector(query, memory))
        y_rows.append(reusable_label(query, memory))
    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=int)


def retrieve_best_memory(
    query: TaskView,
    memory_views: list[TaskView],
    model: MemoryRetrievalModel,
) -> tuple[TaskView, float]:
    best_memory = memory_views[0]
    best_score = -1.0
    for memory in memory_views:
        score = model.predict_reuse_probability(pair_feature_vector(query, memory))
        if score > best_score:
            best_score = score
            best_memory = memory
    return best_memory, best_score


def baseline_success_probability(query: TaskView) -> float:
    return float(query.task["metadata"]["baseline_success"])


def memory_success_probability(query: TaskView, memory: TaskView) -> float:
    query_meta = query.task["metadata"]
    memory_meta = memory.task["metadata"]
    probability = float(query_meta["baseline_success"]) + 0.05
    if query_meta["target_family"] == memory_meta["target_family"]:
        probability += 0.10
    if query_meta["relation"] == memory_meta["relation"]:
        probability += 0.06
    if query_meta["required_strategy"] == memory_meta["required_strategy"]:
        probability += 0.18
    else:
        probability -= 0.10
    if query_meta["risk_type"] == memory_meta["risk_type"]:
        probability += 0.08
    if query_meta["object_family"] == memory_meta["object_family"]:
        probability += 0.04
    return max(0.02, min(float(query_meta["memory_cap"]), probability))


def rollout_success_rate(
    probabilities: list[float],
    rollouts: int,
    seed: int = 23,
    thresholds: list[list[float]] | None = None,
) -> float:
    rng = random.Random(seed)
    if thresholds is None:
        thresholds = [
            [rng.random() for _ in range(rollouts)]
            for _ in probabilities
        ]
    total = 0
    success = 0
    for index, probability in enumerate(probabilities):
        for threshold in thresholds[index]:
            success += int(threshold < probability)
            total += 1
    return success / total if total else 0.0


def evaluate_memory_system(
    eval_views: list[TaskView],
    memory_views: list[TaskView],
    model: MemoryRetrievalModel,
    rollouts: int = 100,
    seed: int = 29,
) -> dict[str, Any]:
    baseline_probs: list[float] = []
    memory_probs: list[float] = []
    oracle_probs: list[float] = []
    top1_reusable = 0

    for query in eval_views:
        retrieved, score = retrieve_best_memory(query, memory_views, model)
        baseline_probs.append(baseline_success_probability(query))
        memory_probs.append(memory_success_probability(query, retrieved))
        top1_reusable += reusable_label(query, retrieved)

        oracle = max(memory_views, key=lambda memory: memory_success_probability(query, memory))
        oracle_probs.append(memory_success_probability(query, oracle))

    rng = random.Random(seed)
    thresholds = [
        [rng.random() for _ in range(rollouts)]
        for _ in eval_views
    ]
    baseline_rate = rollout_success_rate(baseline_probs, rollouts, thresholds=thresholds)
    memory_rate = rollout_success_rate(memory_probs, rollouts, thresholds=thresholds)
    oracle_rate = rollout_success_rate(oracle_probs, rollouts, thresholds=thresholds)

    return {
        "baseline_expected_success": sum(baseline_probs) / len(baseline_probs),
        "graph_memory_expected_success": sum(memory_probs) / len(memory_probs),
        "oracle_memory_expected_success": sum(oracle_probs) / len(oracle_probs),
        "baseline_success_rate": baseline_rate,
        "graph_memory_success_rate": memory_rate,
        "oracle_memory_success_rate": oracle_rate,
        "absolute_improvement": memory_rate - baseline_rate,
        "relative_improvement": (memory_rate - baseline_rate) / baseline_rate if baseline_rate else 0.0,
        "top1_reusable_rate": top1_reusable / len(eval_views),
        "eval_tasks": len(eval_views),
        "rollouts_per_task": rollouts,
        "total_rollouts_per_system": len(eval_views) * rollouts,
    }
