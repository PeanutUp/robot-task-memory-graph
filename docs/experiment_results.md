# Synthetic Memory Experiment Results

This experiment uses generated robot tabletop tasks to evaluate whether a learned graph-memory retriever improves high-level task completion.

## Setup

- Memory tasks: 3000
- Training query tasks: 1200
- Training task pairs: 12000
- Evaluation tasks: 300
- Rollouts per task: 100
- Total rollouts per system: 30000
- Model: LogisticRegression retrieval model over graph/semantic pair features

## Learned Model

- Training accuracy: 1.000
- Training AUC: 1.000
- Positive pairs: 6000
- Negative pairs: 6000

Top learned feature weights:

- `same_target_family`: 4.025
- `same_risk_type`: 2.345
- `same_strategy`: 2.345
- `semantic_cosine`: 2.041
- `target_label_cosine`: 1.877
- `wl_kernel`: 1.393
- `same_category`: 0.887
- `success_cost_similarity`: 0.594

## Rollout Success Rate

| System | Success Rate |
|---|---:|
| No memory baseline | 0.588 |
| Learned graph memory | 0.871 |
| Oracle memory upper bound | 0.871 |

Absolute improvement: **0.283**

Relative improvement: **48.0%**

Top-1 reusable retrieval rate: **1.000**

Expected success probabilities before stochastic rollout:

| System | Expected Success |
|---|---:|
| No memory baseline | 0.589 |
| Learned graph memory | 0.869 |
| Oracle memory upper bound | 0.869 |

## Interpretation

The no-memory baseline uses a generic high-level pick-and-place program. It does not retrieve failure recovery experience, so tasks with missing objects, blocked paths, slippery grasps, fragile objects, or heavy objects have lower success probabilities.

The learned graph-memory system first retrieves a similar historical task graph, then reuses the historical success path or recovery strategy. The learned model is trained from synthetic task pairs and predicts whether a memory graph is reusable for a new task.
