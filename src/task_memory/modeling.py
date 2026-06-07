from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .constants import ACTION_NAMES, ID_TO_ACTION
from .features import build_supervised_arrays
from .memory import build_memory_bank, register_tasks
from .schema import TaskSpec


def make_classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def train_models(
    train_tasks: list[TaskSpec],
    test_tasks: list[TaskSpec],
    seed: int = 42,
    top_k: int = 15,
) -> dict[str, Any]:
    register_tasks(train_tasks + test_tasks)
    memory_bank = build_memory_bank(train_tasks)

    x_base_train, y_train, _ = build_supervised_arrays(train_tasks, memory_bank, use_memory=False)
    x_base_test, y_test, test_meta = build_supervised_arrays(test_tasks, memory_bank, use_memory=False)
    x_memory_train, y_memory_train, _ = build_supervised_arrays(
        train_tasks,
        memory_bank,
        use_memory=True,
        top_k=top_k,
        leave_one_out=True,
    )
    x_memory_test, y_memory_test, memory_test_meta = build_supervised_arrays(
        test_tasks,
        memory_bank,
        use_memory=True,
        top_k=top_k,
    )

    baseline_model = make_classifier(seed)
    memory_model = make_classifier(seed + 1)
    baseline_model.fit(x_base_train, y_train)
    memory_model.fit(x_memory_train, y_memory_train)

    baseline_preds = baseline_model.predict(x_base_test)
    memory_preds = memory_model.predict(x_memory_test)
    report = {
        "model": "RandomForestClassifier",
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "baseline_train_samples": int(len(y_train)),
        "memory_train_samples": int(len(y_memory_train)),
        "test_samples": int(len(y_test)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_preds)),
        "memory_accuracy": float(accuracy_score(y_memory_test, memory_preds)),
        "baseline_macro_f1": float(f1_score(y_test, baseline_preds, average="macro")),
        "memory_macro_f1": float(f1_score(y_memory_test, memory_preds, average="macro")),
    }
    return {
        "baseline_model": baseline_model,
        "memory_model": memory_model,
        "memory_bank": memory_bank,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "x_base_train": x_base_train,
        "x_memory_train": x_memory_train,
        "y_train": y_train,
        "x_base_test": x_base_test,
        "x_memory_test": x_memory_test,
        "y_test": y_test,
        "test_meta": test_meta,
        "memory_test_meta": memory_test_meta,
        "report": report,
    }


def predict_with_probs(
    model: RandomForestClassifier,
    features: list[float],
) -> tuple[str, dict[str, float]]:
    probs = model.predict_proba(np.array([features], dtype=float))[0]
    probability_map = {action: 0.0 for action in ACTION_NAMES}
    for class_id, probability in zip(model.classes_, probs):
        probability_map[ID_TO_ACTION[int(class_id)]] = float(probability)
    action = max(probability_map, key=probability_map.get)
    return action, probability_map
