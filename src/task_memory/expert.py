from __future__ import annotations

from collections import deque

from .constants import ACTION_NAMES
from .environment import initial_state, transition
from .schema import TaskSpec, TaskState


def expert_bfs_plan(task: TaskSpec) -> list[tuple[TaskState, str]]:
    start = initial_state(task)
    queue: deque[TaskState] = deque([start])
    parent: dict[TaskState, tuple[TaskState | None, str | None]] = {start: (None, None)}
    goal_state: TaskState | None = None

    while queue:
        current = queue.popleft()
        if current.delivered:
            goal_state = current
            break
        for action in ACTION_NAMES:
            nxt = transition(task, current, action)
            if nxt is None or nxt in parent:
                continue
            parent[nxt] = (current, action)
            queue.append(nxt)

    if goal_state is None:
        raise RuntimeError(f"No expert plan found for {task.task_id}")

    reversed_steps: list[tuple[TaskState, str]] = []
    cursor = goal_state
    while True:
        previous, action = parent[cursor]
        if previous is None or action is None:
            break
        reversed_steps.append((previous, action))
        cursor = previous
    reversed_steps.reverse()
    return reversed_steps


def planned_action_sequence(task: TaskSpec) -> list[str]:
    actions: list[str] = []
    if task.required_key:
        actions.extend(["move_to_key", "pickup_key", "move_to_door", "open_door"])
    if task.obstacle_state == "blocked":
        actions.append("clear_obstacle")
    actions.append("move_to_object")
    if task.handling_action != "none":
        actions.append(task.handling_action)
    actions.extend(["pick_object", "move_to_target", "place_object"])
    return actions
