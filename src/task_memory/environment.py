from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .constants import ACTION_NAMES
from .schema import TaskSpec, TaskState


def initial_state(task: TaskSpec) -> TaskState:
    needs_inspection = task.handling_action == "inspect_object"
    needs_cart = task.handling_action == "use_cart"
    needs_lid = task.handling_action == "secure_lid"
    return TaskState(
        location="start",
        has_key=False,
        door_open=not task.required_key or task.door_state == "open",
        obstacle_cleared=task.obstacle_state != "blocked",
        object_checked=not needs_inspection,
        cart_ready=not needs_cart,
        lid_secured=not needs_lid,
        object_in_hand=False,
        delivered=False,
    )


def replace_state(state: TaskState, **updates: Any) -> TaskState:
    values = asdict(state)
    values.update(updates)
    return TaskState(**values)


def transition(task: TaskSpec, state: TaskState, action: str) -> TaskState | None:
    if state.delivered:
        return None
    if action == "move_to_key":
        if task.required_key and not state.has_key and state.location != "key_area":
            return replace_state(state, location="key_area")
        return None
    if action == "pickup_key":
        if task.required_key and state.location == "key_area" and not state.has_key:
            return replace_state(state, has_key=True)
        return None
    if action == "move_to_door":
        if task.required_key and state.has_key and not state.door_open and state.location != "door":
            return replace_state(state, location="door")
        return None
    if action == "open_door":
        if task.required_key and state.has_key and state.location == "door" and not state.door_open:
            return replace_state(state, door_open=True)
        return None
    if action == "clear_obstacle":
        if task.obstacle_state == "blocked" and not state.obstacle_cleared:
            return replace_state(state, obstacle_cleared=True)
        return None
    if action == "move_to_object":
        if (
            not state.object_in_hand
            and state.location != "object_area"
            and state.door_open
            and state.obstacle_cleared
        ):
            return replace_state(state, location="object_area")
        return None
    if action == "inspect_object":
        if task.handling_action == "inspect_object" and state.location == "object_area" and not state.object_checked:
            return replace_state(state, object_checked=True)
        return None
    if action == "use_cart":
        if task.handling_action == "use_cart" and state.location == "object_area" and not state.cart_ready:
            return replace_state(state, cart_ready=True)
        return None
    if action == "secure_lid":
        if task.handling_action == "secure_lid" and state.location == "object_area" and not state.lid_secured:
            return replace_state(state, lid_secured=True)
        return None
    if action == "pick_object":
        if (
            state.location == "object_area"
            and not state.object_in_hand
            and state.object_checked
            and state.cart_ready
            and state.lid_secured
        ):
            return replace_state(state, object_in_hand=True)
        return None
    if action == "move_to_target":
        if state.object_in_hand and state.location != "target_area":
            return replace_state(state, location="target_area")
        return None
    if action == "place_object":
        if state.location == "target_area" and state.object_in_hand:
            return replace_state(state, delivered=True, object_in_hand=False)
        return None
    return None


def legal_actions(task: TaskSpec, state: TaskState) -> list[str]:
    return [action for action in ACTION_NAMES if transition(task, state, action) is not None]


def state_phase(task: TaskSpec, state: TaskState) -> str:
    if task.required_key and not state.has_key:
        return "need_key"
    if task.required_key and state.has_key and not state.door_open:
        return "need_door"
    if task.obstacle_state == "blocked" and not state.obstacle_cleared:
        return "need_clear"
    if not state.object_in_hand and state.location != "object_area":
        return "need_object_area"
    if not state.object_checked:
        return "need_inspect"
    if not state.cart_ready:
        return "need_cart"
    if not state.lid_secured:
        return "need_lid"
    if not state.object_in_hand:
        return "need_pick"
    if state.object_in_hand and state.location != "target_area":
        return "need_target_area"
    if state.object_in_hand:
        return "need_place"
    return "terminal"
