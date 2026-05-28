from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism


CORE_ROLES = {
    "object",
    "target",
    "detect",
    "move_to_object",
    "pick",
    "in_hand",
    "move_to_target",
    "place",
    "success",
}

ROLE_ORDER = [
    "object",
    "detect",
    "move_to_object",
    "pick",
    "in_hand",
    "move_to_target",
    "place",
    "success",
]


def _core_graph(graph: nx.DiGraph) -> nx.DiGraph:
    nodes = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("role") in CORE_ROLES
    ]
    return graph.subgraph(nodes).copy()


def _node_match(history_attrs: dict[str, Any], query_attrs: dict[str, Any]) -> bool:
    if history_attrs.get("type") != query_attrs.get("type"):
        return False
    query_role = query_attrs.get("role")
    history_role = history_attrs.get("role")
    if query_role in {"object", "target"}:
        return history_role == query_role
    return history_role == query_role


def _edge_match(history_attrs: dict[str, Any], query_attrs: dict[str, Any]) -> bool:
    return history_attrs.get("type") == query_attrs.get("type")


def find_reusable_subgraph(
    history_graph: nx.DiGraph,
    query_graph: nx.DiGraph,
) -> dict[str, Any]:
    pattern = _core_graph(query_graph)
    matcher = isomorphism.DiGraphMatcher(
        history_graph,
        pattern,
        node_match=_node_match,
        edge_match=_edge_match,
    )

    if not matcher.subgraph_is_isomorphic():
        return {
            "found": False,
            "matched_nodes": set(),
            "matched_edges": set(),
            "structure": "",
            "mapping": {},
        }

    mapping = next(matcher.subgraph_isomorphisms_iter())
    matched_nodes = set(mapping.keys())
    pattern_edges = set(pattern.edges())
    reverse_mapping = {pattern_node: history_node for history_node, pattern_node in mapping.items()}
    matched_edges = {
        (reverse_mapping[src], reverse_mapping[dst])
        for src, dst in pattern_edges
        if src in reverse_mapping and dst in reverse_mapping
    }

    role_by_node = {
        node_id: history_graph.nodes[node_id].get("role")
        for node_id in matched_nodes
    }
    matched_roles = [role for role in ROLE_ORDER if role in set(role_by_node.values())]

    return {
        "found": True,
        "matched_nodes": matched_nodes,
        "matched_edges": matched_edges,
        "structure": " -> ".join(matched_roles),
        "mapping": mapping,
    }

