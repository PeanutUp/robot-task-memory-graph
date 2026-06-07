# Robot Task Memory Graph Experiment

## Setup

- Model: RandomForestClassifier
- Train tasks: 1200
- Test tasks: 350
- Baseline train samples: 7423
- Memory train samples: 7423
- Test samples: 2196

## Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline RF | 0.9727 | 0.9210 |
| Graph-memory RF | 0.9995 | 0.9986 |

## Closed-Loop Rollout

| Model | Success Rate | Avg Steps |
|---|---:|---:|
| Baseline RF | 0.8286 | 6.59 |
| Graph-memory RF | 0.9971 | 7.26 |

Absolute improvement: **0.1686**

## Leakage Checks

- Train/test task_id intersection: 0
- Train/test split_seed intersection: 0
- Memory bank source split: train
- Memory bank only train: True
- Shuffled memory accuracy: 0.9718
- Random memory-bank success rate: 0.6500

## Warnings

- No blocking warnings.
