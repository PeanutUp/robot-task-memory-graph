from __future__ import annotations

from collections import deque
from typing import Any

from .graph_builder import Task, get_task_context


Position = tuple[int, int]


def _as_position(value: list[int] | tuple[int, int], fallback: Position) -> Position:
    if not value:
        return fallback
    return int(value[0]), int(value[1])


def _neighbors(pos: Position, width: int, height: int, obstacles: set[Position]) -> list[Position]:
    x, y = pos
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [
        item
        for item in candidates
        if 0 <= item[0] < width and 0 <= item[1] < height and item not in obstacles
    ]


def _grid_path(start: Position, goal: Position, width: int, height: int, obstacles: set[Position]) -> list[Position]:
    if start == goal:
        return [start]

    queue: deque[Position] = deque([start])
    came_from: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in _neighbors(current, width, height, obstacles):
            if nxt not in came_from:
                came_from[nxt] = current
                queue.append(nxt)

    if goal not in came_from:
        return [start]

    path = [goal]
    while path[-1] != start:
        previous = came_from[path[-1]]
        if previous is None:
            break
        path.append(previous)
    path.reverse()
    return path


def build_execution_frames(plan: dict[str, Any], query_task: Task) -> list[dict[str, Any]]:
    scene = query_task.get("scene", {})
    width, height = scene.get("grid_size", [6, 5])
    robot_pos = _as_position(scene.get("robot_start", [0, 0]), (0, 0))
    object_pos = _as_position(scene.get("object_start", [1, 1]), (1, 1))
    target_pos = _as_position(scene.get("target", [5, 3]), (5, 3))
    obstacles = {
        _as_position(item, (0, 0))
        for item in scene.get("obstacles", [])
    }
    context = get_task_context(query_task)

    frames: list[dict[str, Any]] = []
    carrying = False
    step_counter = 0

    def add_frame(step: dict[str, Any], message: str) -> None:
        nonlocal step_counter
        frames.append(
            {
                "frame_index": step_counter,
                "active_node": step["node_id"],
                "active_role": step["role"],
                "action": step["label"],
                "message": message,
                "robot_pos": robot_pos,
                "object_pos": object_pos,
                "target_pos": target_pos,
                "carrying": carrying,
                "grid_size": (width, height),
                "obstacles": sorted(obstacles),
                "object_label": context["object"],
                "target_label": context["target"],
            }
        )
        step_counter += 1

    for step in plan["steps"]:
        role = step["role"]

        if role == "move_to_object":
            for pos in _grid_path(robot_pos, object_pos, width, height, obstacles)[1:]:
                robot_pos = pos
                add_frame(step, f"Moving to {context['object']}")
        elif role == "pick":
            carrying = True
            object_pos = robot_pos
            add_frame(step, f"Picked up {context['object']}")
        elif role == "move_to_target":
            for pos in _grid_path(robot_pos, target_pos, width, height, obstacles)[1:]:
                robot_pos = pos
                if carrying:
                    object_pos = robot_pos
                add_frame(step, f"Carrying {context['object']} to {context['target']}")
        elif role == "place":
            object_pos = target_pos
            carrying = False
            add_frame(step, f"Placed {context['object']} at {context['target']}")
        elif role in {"grasp_failed", "path_blocked", "object_missing"}:
            add_frame(step, f"Recovery state: {step['label']}")
        elif role in {"retry_pick", "reroute", "search"}:
            add_frame(step, f"Recovery action: {step['label']}")
        else:
            add_frame(step, step["label"])

    return frames

