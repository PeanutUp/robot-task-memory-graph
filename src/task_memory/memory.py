from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from .constants import ACTION_NAMES, HANDLING_ACTIONS, OBJECT_CATALOG, TARGET_CATALOG
from .graph import summarize_graph
from .schema import GraphSummary, MemoryMatch, TaskSpec, TaskState


class MemoryBank:
    def __init__(self, summaries: list[GraphSummary], source_split: str = "train") -> None:
        self.summaries = summaries
        self.source_split = source_split

    def retrieve(self, query: GraphSummary, top_k: int = 15, exclude_task_id: str | None = None) -> list[MemoryMatch]:
        matches = [
            graph_similarity(query, candidate)
            for candidate in self.summaries
            if candidate.task_id != exclude_task_id
        ]
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def summary_by_id(self, task_id: str) -> GraphSummary:
        for summary in self.summaries:
            if summary.task_id == task_id:
                return summary
        raise KeyError(task_id)


TASK_REGISTRY: dict[str, TaskSpec] = {}
GRAPH_SUMMARY_CACHE: dict[str, GraphSummary] = {}
RETRIEVAL_CACHE: dict[tuple[int, str, int, str | None], list[MemoryMatch]] = {}


def register_tasks(tasks: list[TaskSpec]) -> None:
    for task in tasks:
        TASK_REGISTRY[task.task_id] = task


def summarize_graph_cached(task: TaskSpec) -> GraphSummary:
    if task.task_id not in GRAPH_SUMMARY_CACHE:
        GRAPH_SUMMARY_CACHE[task.task_id] = summarize_graph(task)
    return GRAPH_SUMMARY_CACHE[task.task_id]


def build_memory_bank(tasks: list[TaskSpec]) -> MemoryBank:
    return MemoryBank([summarize_graph_cached(task) for task in tasks], source_split="train")


def counter_cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    numerator = sum(left[key] * right[key] for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def counter_jaccard(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    keys = set(left) | set(right)
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return float(intersection / union) if union else 0.0


def graph_similarity(query: GraphSummary, candidate: GraphSummary) -> MemoryMatch:
    node_overlap = counter_jaccard(query.node_types, candidate.node_types)
    edge_overlap = counter_jaccard(query.edge_types, candidate.edge_types)
    object_family_match = float(query.object_family == candidate.object_family)
    target_family_match = float(query.target_family == candidate.target_family)
    required_key_match = float(query.required_key == candidate.required_key)
    door_dependency_match = float(
        ("open_door" in query.action_sequence) == ("open_door" in candidate.action_sequence)
    )
    cost_similarity = 1.0 - min(
        1.0,
        abs(query.path_cost - candidate.path_cost) / max(query.path_cost, candidate.path_cost, 1.0),
    )
    wl_similarity = counter_cosine(query.wl_features, candidate.wl_features)
    subgraph_match = counter_jaccard(query.dependency_patterns, candidate.dependency_patterns)
    action_overlap = counter_jaccard(Counter(query.action_sequence), Counter(candidate.action_sequence))
    similarity = float(
        0.10 * node_overlap
        + 0.10 * edge_overlap
        + 0.08 * object_family_match
        + 0.06 * target_family_match
        + 0.07 * required_key_match
        + 0.07 * door_dependency_match
        + 0.10 * cost_similarity
        + 0.18 * wl_similarity
        + 0.20 * subgraph_match
        + 0.04 * action_overlap
    )
    return MemoryMatch(
        task_id=candidate.task_id,
        similarity=similarity,
        node_type_overlap=node_overlap,
        edge_type_overlap=edge_overlap,
        object_family_match=object_family_match,
        target_family_match=target_family_match,
        required_key_match=required_key_match,
        door_dependency_match=door_dependency_match,
        success_path_cost_similarity=cost_similarity,
        wl_graph_kernel_similarity=wl_similarity,
        subgraph_match_score=subgraph_match,
        action_sequence=candidate.action_sequence,
    )


def memory_feature_vector(
    task: TaskSpec,
    state: TaskState,
    memory_bank: MemoryBank,
    top_k: int = 15,
    exclude_task_id: str | None = None,
) -> tuple[list[float], list[MemoryMatch]]:
    del state
    query = summarize_graph_cached(task)
    cache_key = (id(memory_bank), task.task_id, top_k, exclude_task_id)
    if cache_key not in RETRIEVAL_CACHE:
        RETRIEVAL_CACHE[cache_key] = memory_bank.retrieve(query, top_k=top_k, exclude_task_id=exclude_task_id)
    matches = RETRIEVAL_CACHE[cache_key]
    if not matches:
        return [0.0] * memory_feature_size(), []

    weights = np.array([max(match.similarity, 1e-6) for match in matches], dtype=float)
    weights = weights / weights.sum()

    def weighted_mean(attr: str) -> float:
        return float(sum(weight * getattr(match, attr) for weight, match in zip(weights, matches)))

    values = [
        matches[0].similarity,
        float(np.mean([match.similarity for match in matches])),
        weighted_mean("node_type_overlap"),
        weighted_mean("edge_type_overlap"),
        weighted_mean("object_family_match"),
        weighted_mean("target_family_match"),
        weighted_mean("required_key_match"),
        weighted_mean("door_dependency_match"),
        weighted_mean("success_path_cost_similarity"),
        weighted_mean("wl_graph_kernel_similarity"),
        weighted_mean("subgraph_match_score"),
    ]

    values.extend(
        float(sum(weight for weight, match in zip(weights, matches) if action in match.action_sequence))
        for action in ACTION_NAMES
    )
    for action in ACTION_NAMES:
        position_sum = 0.0
        weight_sum = 0.0
        for weight, match in zip(weights, matches):
            if action in match.action_sequence:
                position_sum += weight * (match.action_sequence.index(action) + 1) / len(match.action_sequence)
                weight_sum += weight
        values.append(float(position_sum / weight_sum) if weight_sum else 0.0)
    values.append(float(len(matches[0].action_sequence)) / 12.0)
    return values, matches


def memory_feature_size() -> int:
    return 11 + len(ACTION_NAMES) + len(ACTION_NAMES) + 1


def make_randomized_memory_bank(train_tasks: list[TaskSpec], seed: int = 42, sample_size: int = 25) -> MemoryBank:
    rng = random.Random(seed)
    randomized: list[TaskSpec] = []
    sampled = rng.sample(train_tasks, k=min(sample_size, len(train_tasks)))
    for task in sampled:
        object_name, object_family = rng.choice(OBJECT_CATALOG)
        target_name, target_family = rng.choice(TARGET_CATALOG)
        handling = rng.choice(HANDLING_ACTIONS)
        required_key = rng.random() < 0.5
        randomized.append(
            TaskSpec(
                task_id=f"random_memory_{task.task_id}",
                split="train",
                split_seed=task.split_seed,
                object_name=object_name,
                object_family=object_family,
                target_name=target_name,
                target_family=target_family,
                risk_type="normal" if handling == "none" else "special",
                handling_action=handling,
                required_key=required_key,
                door_state="locked" if required_key else "none",
                obstacle_state="blocked" if rng.random() < 0.5 else "clear",
                object_location=task.object_location,
                target_location=task.target_location,
            )
        )
    register_tasks(randomized)
    return build_memory_bank(randomized)
