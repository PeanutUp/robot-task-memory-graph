from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .constants import ACTION_TO_ID, LOCATIONS, OBJECT_FAMILIES, TARGET_FAMILIES
from .expert import expert_bfs_plan
from .memory import MemoryBank, memory_feature_vector
from .schema import TaskSpec, TaskState


def base_feature_vector(task: TaskSpec, state: TaskState) -> list[float]:
    values: list[float] = []
    values.extend([1.0 if state.location == location else 0.0 for location in LOCATIONS])
    special_ready = state.object_checked and state.cart_ready and state.lid_secured
    values.extend(
        [
            float(state.has_key),
            float(state.door_open),
            float(state.obstacle_cleared),
            float(special_ready),
            float(state.object_in_hand),
            float(state.delivered),
            float(task.required_key),
            float(task.door_state == "locked"),
            float(task.obstacle_state == "blocked"),
            float(task.risk_type == "special"),
        ]
    )
    values.extend([1.0 if task.object_family == family else 0.0 for family in OBJECT_FAMILIES])
    values.extend([1.0 if task.target_family == family else 0.0 for family in TARGET_FAMILIES])
    return values


def build_supervised_arrays(
    tasks: list[TaskSpec],
    memory_bank: MemoryBank,
    use_memory: bool,
    top_k: int = 15,
    leave_one_out: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    task_memory_cache: dict[tuple[str, TaskState], list[float]] = {}

    for task in tasks:
        for state, action in expert_bfs_plan(task):
            base = base_feature_vector(task, state)
            if use_memory:
                cache_key = (task.task_id, state)
                if cache_key not in task_memory_cache:
                    memory_values, matches = memory_feature_vector(
                        task,
                        state,
                        memory_bank,
                        top_k=top_k,
                        exclude_task_id=task.task_id if leave_one_out else None,
                    )
                    task_memory_cache[cache_key] = memory_values
                    top_match = matches[0].task_id if matches else None
                    top_similarity = matches[0].similarity if matches else 0.0
                else:
                    memory_values = task_memory_cache[cache_key]
                    top_match = None
                    top_similarity = 0.0
                row = base + memory_values
            else:
                row = base
                top_match = None
                top_similarity = 0.0
            rows.append(row)
            labels.append(ACTION_TO_ID[action])
            metadata.append(
                {
                    "task_id": task.task_id,
                    "split_seed": task.split_seed,
                    "state": asdict(state),
                    "label": action,
                    "top_memory_match": top_match,
                    "top_memory_similarity": top_similarity,
                }
            )
    return np.array(rows, dtype=float), np.array(labels, dtype=int), metadata
