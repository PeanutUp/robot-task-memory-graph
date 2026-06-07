import networkx as nx

from src.task_memory import (
    ACTION_NAMES,
    TaskState,
    build_memory_bank,
    build_task_graph,
    evaluate_rollouts,
    expert_bfs_plan,
    generate_dataset,
    memory_feature_size,
    train_models,
)


def test_dataset_split_has_no_seed_or_task_leakage() -> None:
    train_tasks, test_tasks = generate_dataset(n_train=60, n_test=25, seed=7)
    train_ids = {task.task_id for task in train_tasks}
    test_ids = {task.task_id for task in test_tasks}
    train_seeds = {task.split_seed for task in train_tasks}
    test_seeds = {task.split_seed for task in test_tasks}

    assert train_ids.isdisjoint(test_ids)
    assert train_seeds.isdisjoint(test_seeds)


def test_bfs_uses_extended_task_state() -> None:
    train_tasks, _ = generate_dataset(n_train=200, n_test=1, seed=9)
    task = next(task for task in train_tasks if task.required_key and task.obstacle_state == "blocked")
    plan = expert_bfs_plan(task)
    states = [state for state, _ in plan]
    actions = [action for _, action in plan]

    assert "pickup_key" in actions
    assert "open_door" in actions
    assert "clear_obstacle" in actions
    assert all(isinstance(state, TaskState) for state in states)
    assert states[0].has_key is False
    assert any(state.has_key for state in states)
    assert any(state.door_open for state in states)


def test_task_graph_has_required_node_and_edge_types() -> None:
    train_tasks, _ = generate_dataset(n_train=80, n_test=1, seed=12)
    task = next(task for task in train_tasks if task.required_key and task.handling_action != "none")
    graph = build_task_graph(task)
    node_types = {data["type"] for _, data in graph.nodes(data=True)}
    edge_types = {data["type"] for _, _, data in graph.edges(data=True)}

    assert isinstance(graph, nx.DiGraph)
    assert {"object", "location", "action", "state", "result"}.issubset(node_types)
    assert {"temporal", "causal", "requires", "spatial", "reachable"}.issubset(edge_types)


def test_memory_bank_is_train_only_and_features_have_expected_size() -> None:
    train_tasks, test_tasks = generate_dataset(n_train=50, n_test=20, seed=15)
    bank = build_memory_bank(train_tasks)
    train_ids = {task.task_id for task in train_tasks}
    test_ids = {task.task_id for task in test_tasks}

    assert {summary.task_id for summary in bank.summaries}.issubset(train_ids)
    assert {summary.task_id for summary in bank.summaries}.isdisjoint(test_ids)
    assert memory_feature_size() == 36


def test_models_train_and_memory_rollout_is_not_worse() -> None:
    train_tasks, test_tasks = generate_dataset(n_train=180, n_test=70, seed=21)
    trained = train_models(train_tasks, test_tasks, seed=21, top_k=10)
    result = evaluate_rollouts(
        trained["baseline_model"],
        trained["memory_model"],
        test_tasks,
        trained["memory_bank"],
        max_steps=16,
        top_k=10,
    )

    assert trained["report"]["baseline_train_samples"] > 1000
    assert trained["report"]["memory_train_samples"] == trained["report"]["baseline_train_samples"]
    assert set(trained["baseline_model"].classes_).issubset(set(range(len(ACTION_NAMES))))
    assert set(trained["memory_model"].classes_).issubset(set(range(len(ACTION_NAMES))))
    assert result["memory_success_rate"] >= result["baseline_success_rate"]
