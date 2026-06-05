from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_learning import (
    MemoryRetrievalModel,
    build_task_views,
    evaluate_memory_system,
    sample_training_pairs,
)
from src.synthetic_tasks import generate_synthetic_tasks


def markdown_report(summary: dict) -> str:
    weights = summary["training_report"]["weights"]
    top_weights = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:8]
    weight_lines = "\n".join(f"- `{name}`: {value:.3f}" for name, value in top_weights)

    return f"""# Synthetic Memory Experiment Results

This experiment uses generated robot tabletop tasks to evaluate whether a learned graph-memory retriever improves high-level task completion.

## Setup

- Memory tasks: {summary["memory_tasks"]}
- Training query tasks: {summary["train_query_tasks"]}
- Training task pairs: {summary["train_pairs"]}
- Evaluation tasks: {summary["eval_tasks"]}
- Rollouts per task: {summary["rollouts_per_task"]}
- Total rollouts per system: {summary["total_rollouts_per_system"]}
- Model: LogisticRegression retrieval model over graph/semantic pair features

## Learned Model

- Training accuracy: {summary["training_report"]["train_accuracy"]:.3f}
- Training AUC: {summary["training_report"]["train_auc"]:.3f}
- Positive pairs: {summary["training_report"]["positive_pairs"]}
- Negative pairs: {summary["training_report"]["negative_pairs"]}

Top learned feature weights:

{weight_lines}

## Rollout Success Rate

| System | Success Rate |
|---|---:|
| No memory baseline | {summary["evaluation"]["baseline_success_rate"]:.3f} |
| Learned graph memory | {summary["evaluation"]["graph_memory_success_rate"]:.3f} |
| Oracle memory upper bound | {summary["evaluation"]["oracle_memory_success_rate"]:.3f} |

Absolute improvement: **{summary["evaluation"]["absolute_improvement"]:.3f}**

Relative improvement: **{summary["evaluation"]["relative_improvement"]:.1%}**

Top-1 reusable retrieval rate: **{summary["evaluation"]["top1_reusable_rate"]:.3f}**

Expected success probabilities before stochastic rollout:

| System | Expected Success |
|---|---:|
| No memory baseline | {summary["evaluation"]["baseline_expected_success"]:.3f} |
| Learned graph memory | {summary["evaluation"]["graph_memory_expected_success"]:.3f} |
| Oracle memory upper bound | {summary["evaluation"]["oracle_memory_expected_success"]:.3f} |

## Interpretation

The no-memory baseline uses a generic high-level pick-and-place program. It does not retrieve failure recovery experience, so tasks with missing objects, blocked paths, slippery grasps, fragile objects, or heavy objects have lower success probabilities.

The learned graph-memory system first retrieves a similar historical task graph, then reuses the historical success path or recovery strategy. The learned model is trained from synthetic task pairs and predicts whether a memory graph is reusable for a new task.
"""


def run(args: argparse.Namespace) -> dict:
    memory_tasks = generate_synthetic_tasks(args.memory_tasks, seed=args.seed, prefix="memory")
    train_query_tasks = generate_synthetic_tasks(args.train_queries, seed=args.seed + 1, prefix="train_query")
    eval_tasks = generate_synthetic_tasks(args.eval_tasks, seed=args.seed + 2, prefix="eval")

    memory_views = build_task_views(memory_tasks)
    train_query_views = build_task_views(train_query_tasks)
    eval_views = build_task_views(eval_tasks)

    x_train, y_train = sample_training_pairs(
        train_query_views,
        memory_views,
        args.train_pairs,
        seed=args.seed + 3,
    )
    model = MemoryRetrievalModel().fit(x_train, y_train)
    evaluation = evaluate_memory_system(
        eval_views,
        memory_views,
        model,
        rollouts=args.rollouts,
        seed=args.seed + 4,
    )

    summary = {
        "seed": args.seed,
        "memory_tasks": args.memory_tasks,
        "train_query_tasks": args.train_queries,
        "train_pairs": args.train_pairs,
        "eval_tasks": args.eval_tasks,
        "rollouts_per_task": args.rollouts,
        "total_rollouts_per_system": args.eval_tasks * args.rollouts,
        "training_report": model.training_report,
        "evaluation": evaluation,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(args.report_path).write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-tasks", type=int, default=3000)
    parser.add_argument("--train-queries", type=int, default=1200)
    parser.add_argument("--train-pairs", type=int, default=12000)
    parser.add_argument("--eval-tasks", type=int, default=300)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--report-path", default="docs/experiment_results.md")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
