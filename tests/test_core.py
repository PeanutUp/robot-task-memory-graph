from pathlib import Path

from src.graph_builder import build_task_graph, load_tasks
from src.graph_matcher import find_reusable_subgraph
from src.graph_similarity import rank_tasks
from src.memory_learning import (
    MemoryRetrievalModel,
    build_task_views,
    evaluate_memory_system,
    sample_training_pairs,
)
from src.planner import generate_plan, shortest_success_path
from src.simulator import build_execution_frames
from src.synthetic_tasks import generate_synthetic_tasks


ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "data" / "historical_tasks"
QUERY_DIR = ROOT / "data" / "new_tasks"


def test_loads_all_task_graphs() -> None:
    historical_tasks = load_tasks(HISTORY_DIR)
    query_tasks = load_tasks(QUERY_DIR)

    assert len(historical_tasks) == 12
    assert len(query_tasks) == 4

    for task in historical_tasks + query_tasks:
        graph = build_task_graph(task)
        assert graph.number_of_nodes() >= 10
        assert graph.number_of_edges() >= 9
        assert "start" in graph
        assert "success" in graph


def test_block_query_prefers_book_box_memory() -> None:
    historical_tasks = load_tasks(HISTORY_DIR)
    query_task = load_tasks(QUERY_DIR)[0]

    rankings = rank_tasks(query_task, historical_tasks, top_k=3)

    assert rankings[0]["task"]["task_id"] == "task_001"
    assert rankings[0]["score"] > rankings[1]["score"]
    assert rankings[0]["breakdown"]["algorithm"] == "weakly_supervised_logistic_ranker"
    assert rankings[0]["breakdown"]["wl_kernel"] > 0
    assert rankings[0]["breakdown"]["semantic_cosine"] > 0
    assert rankings[0]["breakdown"]["learned_score"] > 0


def test_subgraph_matching_finds_reusable_structure() -> None:
    historical_tasks = load_tasks(HISTORY_DIR)
    query_task = load_tasks(QUERY_DIR)[0]
    best = rank_tasks(query_task, historical_tasks, top_k=1)[0]

    match = find_reusable_subgraph(best["graph"], build_task_graph(query_task))

    assert match["found"]
    assert "detect" in match["structure"]
    assert "place" in match["structure"]
    assert "success" in match["structure"]


def test_shortest_path_avoids_high_cost_failure_branch() -> None:
    task = load_tasks(HISTORY_DIR)[0]
    graph = build_task_graph(task)

    path, cost = shortest_success_path(graph)

    assert "grasp_failed" not in path
    assert "retry_pick" not in path
    assert cost == 11.0


def test_simulation_places_object_at_target() -> None:
    historical_tasks = load_tasks(HISTORY_DIR)
    query_task = load_tasks(QUERY_DIR)[0]
    best = rank_tasks(query_task, historical_tasks, top_k=1)[0]
    plan = generate_plan(best["graph"], query_task)

    frames = build_execution_frames(plan, query_task)

    assert frames
    assert frames[-1]["active_role"] == "success"
    assert frames[-1]["object_pos"] == frames[-1]["target_pos"]
    assert frames[-1]["carrying"] is False


def test_learned_memory_model_beats_no_memory_on_synthetic_rollouts() -> None:
    memory_tasks = generate_synthetic_tasks(80, seed=101, prefix="memory_test")
    train_queries = generate_synthetic_tasks(40, seed=102, prefix="train_test")
    eval_tasks = generate_synthetic_tasks(25, seed=103, prefix="eval_test")

    memory_views = build_task_views(memory_tasks)
    train_views = build_task_views(train_queries)
    eval_views = build_task_views(eval_tasks)
    x_train, y_train = sample_training_pairs(train_views, memory_views, 240, seed=104)
    model = MemoryRetrievalModel().fit(x_train, y_train)
    result = evaluate_memory_system(eval_views, memory_views, model, rollouts=20, seed=105)

    assert model.training_report["train_auc"] >= 0.9
    assert result["graph_memory_success_rate"] > result["baseline_success_rate"]
