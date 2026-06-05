from __future__ import annotations

import random
from typing import Any

from .graph_builder import Task


OBJECT_CATALOG = [
    ("book", "reading_item"),
    ("notebook", "reading_item"),
    ("cup", "drinkware"),
    ("mug", "drinkware"),
    ("toy car", "toy"),
    ("block", "toy_part"),
    ("pen", "stationery"),
    ("marker", "stationery"),
    ("remote control", "device"),
    ("game controller", "device"),
    ("paper ball", "waste"),
    ("plastic wrapper", "waste"),
    ("plate", "dishware"),
    ("spoon", "utensil"),
    ("snack bag", "food_package"),
    ("medicine bottle", "bottle"),
    ("key", "small_tool"),
    ("eraser", "stationery"),
    ("towel", "cloth"),
    ("charger", "cable"),
]

TARGET_CATALOG = [
    ("storage box", "box", "into"),
    ("toy basket", "basket", "into"),
    ("trash bin", "trash_bin", "into"),
    ("drawer", "drawer", "into"),
    ("pen holder", "holder", "into"),
    ("utensil holder", "holder", "into"),
    ("cup rack", "rack", "on"),
    ("serving tray", "tray", "on"),
    ("bookshelf", "shelf", "on"),
    ("medicine box", "box", "into"),
    ("cabinet", "cabinet", "into"),
    ("charging dock", "dock", "on"),
]

RISK_PROFILES = {
    "easy": {
        "strategy": "generic_pick_place",
        "baseline_success": 0.88,
        "memory_cap": 0.94,
        "tags": ["normal"],
    },
    "slippery_grasp": {
        "strategy": "retry_grasp",
        "baseline_success": 0.45,
        "memory_cap": 0.84,
        "tags": ["grasp", "retry"],
    },
    "blocked_path": {
        "strategy": "reroute",
        "baseline_success": 0.42,
        "memory_cap": 0.82,
        "tags": ["blocked", "reroute"],
    },
    "object_missing": {
        "strategy": "search_then_pick",
        "baseline_success": 0.35,
        "memory_cap": 0.80,
        "tags": ["missing", "search"],
    },
    "fragile_object": {
        "strategy": "careful_grasp",
        "baseline_success": 0.50,
        "memory_cap": 0.86,
        "tags": ["fragile", "careful"],
    },
    "heavy_object": {
        "strategy": "two_hand_pick",
        "baseline_success": 0.48,
        "memory_cap": 0.85,
        "tags": ["heavy", "two_hand"],
    },
}


def _grid_scene(rng: random.Random) -> dict[str, Any]:
    width, height = 6, 5
    robot = [rng.randrange(0, 2), rng.randrange(0, height)]
    obj = [rng.randrange(1, 4), rng.randrange(0, height)]
    target = [rng.randrange(4, 6), rng.randrange(0, height)]
    obstacle_count = rng.choice([0, 1, 1, 2])
    occupied = {tuple(robot), tuple(obj), tuple(target)}
    obstacles: list[list[int]] = []
    while len(obstacles) < obstacle_count:
        candidate = [rng.randrange(1, 5), rng.randrange(0, height)]
        if tuple(candidate) not in occupied:
            obstacles.append(candidate)
            occupied.add(tuple(candidate))
    return {
        "grid_size": [width, height],
        "robot_start": robot,
        "object_start": obj,
        "target": target,
        "obstacles": obstacles,
    }


def _base_nodes(object_label: str, target_label: str, relation: str) -> list[dict[str, Any]]:
    return [
        {"id": "start", "type": "state", "role": "start", "label": "task start", "template": "start task"},
        {"id": "object", "type": "object", "role": "object", "label": object_label},
        {"id": "target", "type": "location", "role": "target", "label": target_label},
        {"id": "detect", "type": "subtask", "role": "detect", "label": f"detect {object_label}", "template": "detect {object}"},
        {"id": "move_to_object", "type": "action", "role": "move_to_object", "label": f"move to {object_label}", "template": "move to {object}"},
        {"id": "pick", "type": "action", "role": "pick", "label": f"pick {object_label}", "template": "pick {object}"},
        {"id": "in_hand", "type": "state", "role": "in_hand", "label": f"{object_label} in hand", "template": "{object} in hand"},
        {"id": "move_to_target", "type": "action", "role": "move_to_target", "label": f"move to {target_label}", "template": "move to {target}"},
        {"id": "place", "type": "action", "role": "place", "label": f"place {object_label} {relation} {target_label}", "template": "place {object} " + relation + " {target}"},
        {"id": "success", "type": "result", "role": "success", "label": "success", "template": "success"},
    ]


def _core_edges() -> list[dict[str, Any]]:
    return [
        {"source": "object", "target": "detect", "type": "object_to_subtask", "cost": 1},
        {"source": "target", "target": "place", "type": "spatial", "cost": 1},
        {"source": "start", "target": "detect", "type": "temporal", "cost": 1},
        {"source": "detect", "target": "move_to_object", "type": "temporal", "cost": 1},
        {"source": "move_to_object", "target": "pick", "type": "reachable", "cost": 3},
        {"source": "pick", "target": "in_hand", "type": "causal", "cost": 1},
        {"source": "in_hand", "target": "move_to_target", "type": "temporal", "cost": 1},
        {"source": "move_to_target", "target": "place", "type": "reachable", "cost": 3},
        {"source": "place", "target": "success", "type": "causal", "cost": 1},
    ]


def _apply_risk_branch(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], risk_type: str) -> None:
    if risk_type == "slippery_grasp":
        nodes.extend([
            {"id": "grasp_failed", "type": "state", "role": "grasp_failed", "label": "grasp failed", "template": "grasp failed"},
            {"id": "retry_pick", "type": "action", "role": "retry_pick", "label": "retry pick", "template": "retry pick {object}"},
        ])
        edges.append({"source": "pick", "target": "grasp_failed", "type": "causal", "cost": 5})
        edges.append({"source": "grasp_failed", "target": "retry_pick", "type": "temporal", "cost": 5})
        edges.append({"source": "retry_pick", "target": "in_hand", "type": "causal", "cost": 1})
    elif risk_type == "blocked_path":
        edges[:] = [edge for edge in edges if not (edge["source"] == "move_to_target" and edge["target"] == "place")]
        nodes.extend([
            {"id": "path_blocked", "type": "state", "role": "path_blocked", "label": "path blocked", "template": "path blocked"},
            {"id": "reroute", "type": "action", "role": "reroute", "label": "reroute", "template": "reroute around obstacle"},
        ])
        edges.append({"source": "move_to_target", "target": "path_blocked", "type": "causal", "cost": 5})
        edges.append({"source": "path_blocked", "target": "reroute", "type": "temporal", "cost": 5})
        edges.append({"source": "reroute", "target": "place", "type": "reachable", "cost": 3})
    elif risk_type == "object_missing":
        edges[:] = [edge for edge in edges if not (edge["source"] == "start" and edge["target"] == "detect")]
        nodes.extend([
            {"id": "object_missing", "type": "state", "role": "object_missing", "label": "object missing", "template": "{object} missing"},
            {"id": "search", "type": "subtask", "role": "search", "label": "search object", "template": "search for {object}"},
        ])
        edges.append({"source": "start", "target": "object_missing", "type": "causal", "cost": 5})
        edges.append({"source": "object_missing", "target": "search", "type": "temporal", "cost": 3})
        edges.append({"source": "search", "target": "detect", "type": "temporal", "cost": 2})
    elif risk_type == "fragile_object":
        nodes.append({"id": "careful_grasp", "type": "action", "role": "careful_grasp", "label": "careful grasp", "template": "carefully grasp {object}"})
        edges[:] = [edge for edge in edges if not (edge["source"] == "move_to_object" and edge["target"] == "pick")]
        edges[:] = [edge for edge in edges if not (edge["source"] == "pick" and edge["target"] == "in_hand")]
        edges.append({"source": "move_to_object", "target": "careful_grasp", "type": "reachable", "cost": 4})
        edges.append({"source": "careful_grasp", "target": "in_hand", "type": "causal", "cost": 1})
    elif risk_type == "heavy_object":
        nodes.append({"id": "two_hand_pick", "type": "action", "role": "two_hand_pick", "label": "two hand pick", "template": "use two hands to pick {object}"})
        edges[:] = [edge for edge in edges if not (edge["source"] == "move_to_object" and edge["target"] == "pick")]
        edges[:] = [edge for edge in edges if not (edge["source"] == "pick" and edge["target"] == "in_hand")]
        edges.append({"source": "move_to_object", "target": "two_hand_pick", "type": "reachable", "cost": 4})
        edges.append({"source": "two_hand_pick", "target": "in_hand", "type": "causal", "cost": 1})


def generate_synthetic_task(task_id: str, rng: random.Random) -> Task:
    object_label, object_family = rng.choice(OBJECT_CATALOG)
    target_label, target_family, relation = rng.choice(TARGET_CATALOG)
    risk_type = rng.choices(
        list(RISK_PROFILES.keys()),
        weights=[0.34, 0.16, 0.16, 0.12, 0.12, 0.10],
        k=1,
    )[0]
    risk_profile = RISK_PROFILES[risk_type]

    title = f"{object_label} -> {target_label} ({risk_type})"
    nodes = _base_nodes(object_label, target_label, relation)
    edges = _core_edges()
    _apply_risk_branch(nodes, edges, risk_type)

    return {
        "task_id": task_id,
        "title": title,
        "goal": f"put {object_label} {relation} {target_label}",
        "category": "pick_place" if risk_type == "easy" else "adaptive_pick_place",
        "tags": ["synthetic", object_family, target_family, relation, risk_type] + risk_profile["tags"],
        "metadata": {
            "object_label": object_label,
            "target_label": target_label,
            "object_family": object_family,
            "target_family": target_family,
            "relation": relation,
            "risk_type": risk_type,
            "required_strategy": risk_profile["strategy"],
            "baseline_success": risk_profile["baseline_success"],
            "memory_cap": risk_profile["memory_cap"],
        },
        "scene": _grid_scene(rng),
        "nodes": nodes,
        "edges": edges,
    }


def generate_synthetic_tasks(count: int, seed: int = 7, prefix: str = "syn") -> list[Task]:
    rng = random.Random(seed)
    return [generate_synthetic_task(f"{prefix}_{index:05d}", rng) for index in range(count)]

