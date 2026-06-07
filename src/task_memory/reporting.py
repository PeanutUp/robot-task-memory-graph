from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/task-graph-memory-matplotlib")

import matplotlib.pyplot as plt

from .memory import TASK_REGISTRY
from .schema import TaskSpec
from .visualization import draw_task_graph


def write_experiment_outputs(result: dict[str, Any], output_dir: str | Path = "results") -> None:
    output_path = Path(output_dir)
    case_dir = output_path / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)

    metrics = {key: value for key, value in result.items() if key not in {"trained", "cases"}}
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "summary.md").write_text(make_summary_markdown(metrics), encoding="utf-8")

    cases = select_case_studies(result["cases"])
    for name, case in cases.items():
        (case_dir / f"{name}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
        task = TaskSpec(**case["task"])
        fig = draw_task_graph(task)
        fig.savefig(case_dir / f"{name}_query_graph.png", dpi=140)
        plt.close(fig)
        top_match = first_memory_match(case)
        if top_match:
            train_task = TASK_REGISTRY.get(top_match)
            if train_task is not None:
                fig = draw_task_graph(train_task)
                fig.savefig(case_dir / f"{name}_top_memory_graph.png", dpi=140)
                plt.close(fig)


def select_case_studies(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not case["baseline_success"] and case["memory_success"] and "baseline_fail_memory_success" not in selected:
            selected["baseline_fail_memory_success"] = case
        if case["baseline_success"] and case["memory_success"] and "both_success" not in selected:
            selected["both_success"] = case
        if not case["baseline_success"] and not case["memory_success"] and "both_fail" not in selected:
            selected["both_fail"] = case
        if len(selected) == 3:
            break
    return selected


def first_memory_match(case: dict[str, Any]) -> str | None:
    for frame in case.get("memory_frames", []):
        if frame.get("top_memory_match"):
            return frame["top_memory_match"]
    return None


def make_summary_markdown(metrics: dict[str, Any]) -> str:
    cls = metrics["classification"]
    rollout = metrics["rollout"]
    sanity = metrics["sanity_checks"]
    warnings = metrics["warnings"]
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No blocking warnings."
    return f"""# Robot Task Memory Graph Experiment

## Setup

- Model: {cls["model"]}
- Train tasks: {cls["train_tasks"]}
- Test tasks: {cls["test_tasks"]}
- Baseline train samples: {cls["baseline_train_samples"]}
- Memory train samples: {cls["memory_train_samples"]}
- Test samples: {cls["test_samples"]}

## Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline RF | {cls["baseline_accuracy"]:.4f} | {cls["baseline_macro_f1"]:.4f} |
| Graph-memory RF | {cls["memory_accuracy"]:.4f} | {cls["memory_macro_f1"]:.4f} |

## Closed-Loop Rollout

| Model | Success Rate | Avg Steps |
|---|---:|---:|
| Baseline RF | {rollout["baseline_success_rate"]:.4f} | {rollout["baseline_avg_steps"]:.2f} |
| Graph-memory RF | {rollout["memory_success_rate"]:.4f} | {rollout["memory_avg_steps"]:.2f} |

Absolute improvement: **{rollout["absolute_improvement"]:.4f}**

## Leakage Checks

- Train/test task_id intersection: {len(sanity["train_test_task_id_intersection"])}
- Train/test split_seed intersection: {len(sanity["train_test_seed_intersection"])}
- Memory bank source split: {sanity["memory_bank_source_split"]}
- Memory bank only train: {sanity["memory_bank_only_train"]}
- Shuffled memory accuracy: {sanity["shuffled_memory_accuracy"]:.4f}
- Random memory-bank success rate: {sanity["random_memory_bank_success_rate"]:.4f}

## Warnings

{warning_text}
"""
