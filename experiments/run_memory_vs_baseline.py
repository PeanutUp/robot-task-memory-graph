from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task_memory import run_full_experiment, write_experiment_outputs


def strip_large_objects(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"trained", "cases"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=int, default=1200)
    parser.add_argument("--test-tasks", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    result = run_full_experiment(
        n_train=args.train_tasks,
        n_test=args.test_tasks,
        seed=args.seed,
        top_k=args.top_k,
        max_steps=args.max_steps,
    )
    write_experiment_outputs(result, output_dir=args.output_dir)
    print(json.dumps(strip_large_objects(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
