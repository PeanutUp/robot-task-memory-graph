from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import networkx as nx

from .constants import ACTION_COST
from .expert import planned_action_sequence
from .schema import GraphSummary, TaskSpec


def build_task_graph(task: TaskSpec) -> nx.DiGraph:
    graph = nx.DiGraph(task_id=task.task_id)

    def add_node(node_id: str, node_type: str, role: str, label: str) -> None:
        graph.add_node(node_id, type=node_type, role=role, label=label)

    add_node(f"object:{task.object_name}", "object", "object", task.object_name)
    add_node(f"target:{task.target_name}", "location", "target", task.target_name)
    add_node("location:start", "location", "start", "start")
    add_node("location:object_area", "location", "object_area", task.object_location)
    add_node("location:target_area", "location", "target_area", task.target_location)
    add_node("state:object_in_hand", "state", "object_in_hand", "object_in_hand")
    add_node("result:success", "result", "success", "success")
    add_node("result:failure", "result", "failure", "failure")

    if task.required_key:
        add_node("object:key", "object", "key", "key")
        add_node("location:door", "location", "door", "door")
        add_node("state:has_key", "state", "has_key", "has_key")
        add_node("state:door_open", "state", "door_open", "door_open")
    if task.obstacle_state == "blocked":
        add_node("state:blocked", "state", "blocked", "blocked")
        add_node("state:obstacle_cleared", "state", "obstacle_cleared", "obstacle_cleared")
    if task.handling_action == "inspect_object":
        add_node("state:object_checked", "state", "object_checked", "object_checked")
    if task.handling_action == "use_cart":
        add_node("state:cart_ready", "state", "cart_ready", "cart_ready")
    if task.handling_action == "secure_lid":
        add_node("state:lid_secured", "state", "lid_secured", "lid_secured")

    actions = planned_action_sequence(task)
    for action in actions:
        add_node(f"action:{action}", "action", action, action)
    for left, right in zip(actions[:-1], actions[1:]):
        graph.add_edge(f"action:{left}", f"action:{right}", type="temporal", cost=ACTION_COST[left])

    graph.add_edge(f"action:{actions[-1]}", "result:success", type="causal", cost=ACTION_COST[actions[-1]])
    graph.add_edge("location:start", "location:object_area", type="reachable", cost=3)
    graph.add_edge("location:object_area", "location:target_area", type="reachable", cost=3)
    graph.add_edge(f"object:{task.object_name}", "location:object_area", type="spatial", cost=1)
    graph.add_edge(f"target:{task.target_name}", "location:target_area", type="spatial", cost=1)
    graph.add_edge(f"object:{task.object_name}", "action:pick_object", type="requires", cost=1)
    graph.add_edge("state:object_in_hand", "action:move_to_target", type="requires", cost=1)

    if task.required_key:
        graph.add_edge("object:key", "action:pickup_key", type="requires", cost=1)
        graph.add_edge("action:pickup_key", "state:has_key", type="causal", cost=1)
        graph.add_edge("state:has_key", "action:open_door", type="requires", cost=1)
        graph.add_edge("action:open_door", "state:door_open", type="causal", cost=1)
        graph.add_edge("state:door_open", "action:move_to_object", type="requires", cost=1)
        graph.add_edge("location:start", "location:door", type="reachable", cost=2)
        graph.add_edge("location:door", "location:object_area", type="reachable", cost=2)
    if task.obstacle_state == "blocked":
        graph.add_edge("state:blocked", "action:clear_obstacle", type="requires", cost=1)
        graph.add_edge("action:clear_obstacle", "state:obstacle_cleared", type="causal", cost=1)
        graph.add_edge("state:obstacle_cleared", "action:move_to_object", type="requires", cost=1)
    if task.handling_action == "inspect_object":
        graph.add_edge("action:inspect_object", "state:object_checked", type="causal", cost=2)
        graph.add_edge("state:object_checked", "action:pick_object", type="requires", cost=1)
    if task.handling_action == "use_cart":
        graph.add_edge("action:use_cart", "state:cart_ready", type="causal", cost=3)
        graph.add_edge("state:cart_ready", "action:pick_object", type="requires", cost=1)
    if task.handling_action == "secure_lid":
        graph.add_edge("action:secure_lid", "state:lid_secured", type="causal", cost=2)
        graph.add_edge("state:lid_secured", "action:pick_object", type="requires", cost=1)

    graph.add_edge("action:pick_object", "state:object_in_hand", type="causal", cost=1)
    graph.add_edge("action:place_object", "result:success", type="causal", cost=1)
    return graph


def summarize_graph(task: TaskSpec) -> GraphSummary:
    graph = build_task_graph(task)
    action_sequence = tuple(planned_action_sequence(task))
    return GraphSummary(
        task_id=task.task_id,
        split_seed=task.split_seed,
        object_family=task.object_family,
        target_family=task.target_family,
        required_key=task.required_key,
        obstacle_blocked=task.obstacle_state == "blocked",
        handling_action=task.handling_action,
        path_cost=float(sum(ACTION_COST[action] for action in action_sequence)),
        action_sequence=action_sequence,
        node_types=Counter(data["type"] for _, data in graph.nodes(data=True)),
        edge_types=Counter(data["type"] for _, _, data in graph.edges(data=True)),
        dependency_patterns=dependency_pattern_counter(graph),
        wl_features=wl_feature_counter(graph),
        graph=graph,
    )


def dependency_pattern_counter(graph: nx.DiGraph) -> Counter[tuple[str, str, str]]:
    patterns: Counter[tuple[str, str, str]] = Counter()
    for source, target, data in graph.edges(data=True):
        source_role = graph.nodes[source].get("role", "")
        target_role = graph.nodes[target].get("role", "")
        patterns[(source_role, data.get("type", ""), target_role)] += 1
    return patterns


def wl_feature_counter(graph: nx.DiGraph, iterations: int = 2) -> Counter[str]:
    labels = {
        node: f'{data.get("type", "")}:{data.get("role", "")}:{data.get("label", "")}'
        for node, data in graph.nodes(data=True)
    }
    features: Counter[str] = Counter(labels.values())
    for _ in range(iterations):
        next_labels: dict[Any, str] = {}
        for node in graph.nodes:
            outgoing = sorted(
                f'out:{graph.edges[node, nbr].get("type", "")}:{labels[nbr]}'
                for nbr in graph.successors(node)
            )
            incoming = sorted(
                f'in:{graph.edges[nbr, node].get("type", "")}:{labels[nbr]}'
                for nbr in graph.predecessors(node)
            )
            label = labels[node] + "|" + "|".join(incoming + outgoing)
            next_labels[node] = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
        labels = next_labels
        features.update(labels.values())
    return features
