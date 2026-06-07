from __future__ import annotations

import random

from .constants import HANDLING_ACTIONS, OBJECT_CATALOG, TARGET_CATALOG
from .schema import TaskSpec


def sample_handling_action(object_family: str, rng: random.Random) -> str:
    weights = {
        "document": [0.72, 0.18, 0.02, 0.08],
        "vessel": [0.25, 0.12, 0.06, 0.57],
        "tool": [0.34, 0.12, 0.48, 0.06],
        "decor": [0.28, 0.58, 0.08, 0.06],
        "toy": [0.68, 0.16, 0.12, 0.04],
        "device": [0.26, 0.52, 0.10, 0.12],
        "parcel": [0.38, 0.10, 0.44, 0.08],
    }[object_family]
    return rng.choices(HANDLING_ACTIONS, weights=weights, k=1)[0]


def key_probability(target_family: str) -> float:
    return {
        "storage": 0.48,
        "disposal": 0.18,
        "serving": 0.22,
        "workspace": 0.38,
    }[target_family]


def obstacle_probability(object_family: str, target_family: str) -> float:
    base = {
        "document": 0.18,
        "vessel": 0.30,
        "tool": 0.35,
        "decor": 0.32,
        "toy": 0.22,
        "device": 0.28,
        "parcel": 0.38,
    }[object_family]
    if target_family == "storage":
        base += 0.08
    if target_family == "disposal":
        base -= 0.06
    return min(0.55, max(0.08, base))


def generate_task(task_id: str, split: str, split_seed: int) -> TaskSpec:
    rng = random.Random(split_seed)
    object_name, object_family = rng.choice(OBJECT_CATALOG)
    target_name, target_family = rng.choice(TARGET_CATALOG)
    handling_action = sample_handling_action(object_family, rng)
    required_key = rng.random() < key_probability(target_family)

    return TaskSpec(
        task_id=task_id,
        split=split,
        split_seed=split_seed,
        object_name=object_name,
        object_family=object_family,
        target_name=target_name,
        target_family=target_family,
        risk_type="normal" if handling_action == "none" else "special",
        handling_action=handling_action,
        required_key=required_key,
        door_state="locked" if required_key else rng.choice(["none", "open"]),
        obstacle_state="blocked" if rng.random() < obstacle_probability(object_family, target_family) else "clear",
        object_location="object_area",
        target_location="target_area",
    )


def generate_dataset(n_train: int = 1200, n_test: int = 350, seed: int = 42) -> tuple[list[TaskSpec], list[TaskSpec]]:
    train_tasks = [
        generate_task(f"train_{index:05d}", "train", 100_000 + seed * 10_000 + index)
        for index in range(n_train)
    ]
    test_tasks = [
        generate_task(f"test_{index:05d}", "test", 900_000 + seed * 10_000 + index)
        for index in range(n_test)
    ]
    return train_tasks, test_tasks
