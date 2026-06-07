from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .constants import ACTION_NAMES
from .data import generate_dataset
from .environment import initial_state, transition
from .features import base_feature_vector, build_supervised_arrays
from .memory import MemoryBank, make_randomized_memory_bank, memory_feature_vector
from .modeling import make_classifier, predict_with_probs, train_models
from .schema import MemoryMatch, RolloutFrame, TaskSpec, TaskState


def rollout_task(
    model: RandomForestClassifier,
    task: TaskSpec,
    memory_bank: MemoryBank | None,
    max_steps: int = 16,
    top_k: int = 15,
) -> tuple[bool, list[RolloutFrame]]:
    state = initial_state(task)
    frames: list[RolloutFrame] = []
    visited: Counter[TaskState] = Counter()

    for step in range(max_steps):
        if state.delivered:
            frames.append(
                RolloutFrame(step, state, "stop", {action: 0.0 for action in ACTION_NAMES}, True, False, "success")
            )
            return True, frames

        base = base_feature_vector(task, state)
        matches: list[MemoryMatch] = []
        if memory_bank is None:
            features = base
        else:
            memory_values, matches = memory_feature_vector(task, state, memory_bank, top_k=top_k)
            features = base + memory_values
        predicted_action, probs = predict_with_probs(model, features)
        next_state = transition(task, state, predicted_action)
        top_match = matches[0].task_id if matches else None
        top_similarity = matches[0].similarity if matches else 0.0

        if next_state is None:
            frames.append(
                RolloutFrame(
                    step,
                    state,
                    predicted_action,
                    probs,
                    False,
                    True,
                    f"invalid_action:{predicted_action}",
                    top_match,
                    top_similarity,
                )
            )
            return False, frames

        frames.append(
            RolloutFrame(step, state, predicted_action, probs, False, False, "ok", top_match, top_similarity)
        )
        state = next_state
        visited[state] += 1
        if visited[state] > 3:
            frames.append(
                RolloutFrame(
                    step + 1,
                    state,
                    predicted_action,
                    probs,
                    False,
                    True,
                    "loop_detected",
                    top_match,
                    top_similarity,
                )
            )
            return False, frames

    frames.append(
        RolloutFrame(max_steps, state, "timeout", {action: 0.0 for action in ACTION_NAMES}, False, True, "timeout")
    )
    return False, frames


def evaluate_rollouts(
    baseline_model: RandomForestClassifier,
    memory_model: RandomForestClassifier,
    test_tasks: list[TaskSpec],
    memory_bank: MemoryBank,
    max_steps: int = 16,
    top_k: int = 15,
) -> dict[str, Any]:
    baseline_success = 0
    memory_success = 0
    baseline_steps: list[int] = []
    memory_steps: list[int] = []
    baseline_failures: Counter[str] = Counter()
    memory_failures: Counter[str] = Counter()
    case_records: list[dict[str, Any]] = []

    for task in test_tasks:
        baseline_ok, baseline_frames = rollout_task(baseline_model, task, None, max_steps=max_steps, top_k=top_k)
        memory_ok, memory_frames = rollout_task(memory_model, task, memory_bank, max_steps=max_steps, top_k=top_k)
        baseline_success += int(baseline_ok)
        memory_success += int(memory_ok)
        baseline_steps.append(len(baseline_frames))
        memory_steps.append(len(memory_frames))
        if not baseline_ok:
            baseline_failures[baseline_frames[-1].message] += 1
        if not memory_ok:
            memory_failures[memory_frames[-1].message] += 1
        case_records.append(
            {
                "task": asdict(task),
                "baseline_success": baseline_ok,
                "memory_success": memory_ok,
                "baseline_frames": frames_to_json(baseline_frames),
                "memory_frames": frames_to_json(memory_frames),
            }
        )

    total = len(test_tasks)
    return {
        "episodes": total,
        "max_steps": max_steps,
        "baseline_success_rate": baseline_success / total,
        "memory_success_rate": memory_success / total,
        "absolute_improvement": (memory_success - baseline_success) / total,
        "baseline_avg_steps": float(np.mean(baseline_steps)),
        "memory_avg_steps": float(np.mean(memory_steps)),
        "baseline_failure_reasons": dict(baseline_failures),
        "memory_failure_reasons": dict(memory_failures),
        "cases": case_records,
    }


def frames_to_json(frames: list[RolloutFrame]) -> list[dict[str, Any]]:
    return [
        {
            "step": frame.step,
            "state": asdict(frame.state),
            "predicted_action": frame.predicted_action,
            "success": frame.success,
            "failed": frame.failed,
            "message": frame.message,
            "top_memory_match": frame.top_memory_match,
            "top_memory_similarity": frame.top_memory_similarity,
        }
        for frame in frames
    ]


def run_sanity_checks(
    trained: dict[str, Any],
    seed: int = 42,
    top_k: int = 15,
    max_steps: int = 16,
) -> dict[str, Any]:
    train_tasks: list[TaskSpec] = trained["train_tasks"]
    test_tasks: list[TaskSpec] = trained["test_tasks"]
    memory_bank: MemoryBank = trained["memory_bank"]

    train_ids = {task.task_id for task in train_tasks}
    test_ids = {task.task_id for task in test_tasks}
    train_seeds = {task.split_seed for task in train_tasks}
    test_seeds = {task.split_seed for task in test_tasks}
    memory_ids = {summary.task_id for summary in memory_bank.summaries}

    rng = np.random.default_rng(seed)
    x_memory_train = trained["x_memory_train"].copy()
    x_memory_test = trained["x_memory_test"].copy()
    base_width = trained["x_base_train"].shape[1]
    shuffled_train_tail = x_memory_train[:, base_width:].copy()
    rng.shuffle(shuffled_train_tail, axis=0)
    x_memory_train[:, base_width:] = shuffled_train_tail
    shuffled_model = make_classifier(seed + 222)
    shuffled_model.fit(x_memory_train, trained["y_train"])
    shuffled_accuracy = float(accuracy_score(trained["y_test"], shuffled_model.predict(x_memory_test)))
    shuffled_f1 = float(f1_score(trained["y_test"], shuffled_model.predict(x_memory_test), average="macro"))

    random_bank = make_randomized_memory_bank(train_tasks, seed=seed + 333)
    x_random_test, y_random_test, _ = build_supervised_arrays(
        test_tasks,
        random_bank,
        use_memory=True,
        top_k=top_k,
    )
    random_accuracy = float(accuracy_score(y_random_test, trained["memory_model"].predict(x_random_test)))
    random_f1 = float(f1_score(y_random_test, trained["memory_model"].predict(x_random_test), average="macro"))
    random_rollout = evaluate_rollouts(
        trained["baseline_model"],
        trained["memory_model"],
        test_tasks[: min(120, len(test_tasks))],
        random_bank,
        max_steps=max_steps,
        top_k=top_k,
    )

    return {
        "train_test_task_id_intersection": sorted(train_ids & test_ids),
        "train_test_seed_intersection": sorted(train_seeds & test_seeds),
        "memory_bank_source_split": memory_bank.source_split,
        "memory_bank_task_count": len(memory_bank.summaries),
        "memory_bank_only_train": memory_ids.issubset(train_ids),
        "shuffled_memory_accuracy": shuffled_accuracy,
        "shuffled_memory_macro_f1": shuffled_f1,
        "random_memory_bank_accuracy": random_accuracy,
        "random_memory_bank_macro_f1": random_f1,
        "random_memory_bank_success_rate": random_rollout["memory_success_rate"],
    }


def run_full_experiment(
    n_train: int = 1200,
    n_test: int = 350,
    seed: int = 42,
    top_k: int = 15,
    max_steps: int = 16,
) -> dict[str, Any]:
    train_tasks, test_tasks = generate_dataset(n_train=n_train, n_test=n_test, seed=seed)
    trained = train_models(train_tasks, test_tasks, seed=seed, top_k=top_k)
    rollout_metrics = evaluate_rollouts(
        trained["baseline_model"],
        trained["memory_model"],
        test_tasks,
        trained["memory_bank"],
        max_steps=max_steps,
        top_k=top_k,
    )
    sanity = run_sanity_checks(trained, seed=seed, top_k=top_k, max_steps=max_steps)
    warnings = build_warnings(trained["report"], rollout_metrics, sanity)
    return {
        "config": {
            "n_train": n_train,
            "n_test": n_test,
            "seed": seed,
            "top_k": top_k,
            "max_steps": max_steps,
        },
        "classification": trained["report"],
        "rollout": {key: value for key, value in rollout_metrics.items() if key != "cases"},
        "sanity_checks": sanity,
        "warnings": warnings,
        "cases": rollout_metrics["cases"],
        "trained": trained,
    }


def build_warnings(classification: dict[str, Any], rollout: dict[str, Any], sanity: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if sanity["train_test_task_id_intersection"]:
        warnings.append("train/test task_id intersection is not empty")
    if sanity["train_test_seed_intersection"]:
        warnings.append("train/test split_seed intersection is not empty")
    if not sanity["memory_bank_only_train"]:
        warnings.append("memory bank contains non-train tasks")
    if rollout["baseline_success_rate"] == 0.0:
        warnings.append("baseline rollout success is exactly 0; inspect case studies")
    if rollout["memory_success_rate"] == 1.0:
        warnings.append("memory rollout success is exactly 1; inspect leakage and task difficulty")
    if sanity["random_memory_bank_success_rate"] > rollout["memory_success_rate"] - 0.02:
        warnings.append("random memory bank did not reduce memory rollout success enough")
    if sanity["shuffled_memory_accuracy"] > classification["memory_accuracy"] - 0.02:
        warnings.append("shuffled memory features did not reduce classification accuracy enough")
    return warnings
