from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    split_seed: int
    object_name: str
    object_family: str
    target_name: str
    target_family: str
    risk_type: str
    handling_action: str
    required_key: bool
    door_state: str
    obstacle_state: str
    object_location: str
    target_location: str


@dataclass(frozen=True)
class TaskState:
    location: str
    has_key: bool
    door_open: bool
    obstacle_cleared: bool
    object_checked: bool
    cart_ready: bool
    lid_secured: bool
    object_in_hand: bool
    delivered: bool


@dataclass
class RolloutFrame:
    step: int
    state: TaskState
    predicted_action: str
    action_probs: dict[str, float]
    success: bool
    failed: bool
    message: str
    top_memory_match: str | None = None
    top_memory_similarity: float = 0.0


@dataclass
class GraphSummary:
    task_id: str
    split_seed: int
    object_family: str
    target_family: str
    required_key: bool
    obstacle_blocked: bool
    handling_action: str
    path_cost: float
    action_sequence: tuple[str, ...]
    node_types: Counter[str]
    edge_types: Counter[str]
    dependency_patterns: Counter[tuple[str, str, str]]
    wl_features: Counter[str]
    graph: nx.DiGraph


@dataclass
class MemoryMatch:
    task_id: str
    similarity: float
    node_type_overlap: float
    edge_type_overlap: float
    object_family_match: float
    target_family_match: float
    required_key_match: float
    door_dependency_match: float
    success_path_cost_similarity: float
    wl_graph_kernel_similarity: float
    subgraph_match_score: float
    action_sequence: tuple[str, ...]
