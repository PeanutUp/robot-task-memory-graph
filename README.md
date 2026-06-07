# Robot Task Memory Graph

**机器人任务记忆图：基于图结构的长期任务经验复用**

本项目构建了一个合成的机器人高层任务规划 benchmark，用于研究图结构历史经验在任务规划中的复用价值。系统将历史任务表示为属性有向图，并从训练集任务图中构建记忆库；面对新任务时，模型检索相似历史任务图，提取图结构记忆特征，并将其作为机器学习模型的额外输入。

## 项目概述

每个任务描述一个高层机器人操作流程，包含物体、目标位置、钥匙、门、障碍物和特殊处理需求等要素。模型在每个状态下预测一个高层动作，例如：

```text
move_to_key -> pickup_key -> open_door -> clear_obstacle
move_to_object -> inspect_object / use_cart / secure_lid
pick_object -> move_to_target -> place_object
```

项目比较两种设置下的同类型模型：

| 模型 | 输入特征 | 分类器 |
|---|---|---|
| Baseline RF | 当前任务与当前状态特征 | RandomForestClassifier |
| Graph-Memory RF | Baseline 特征 + 图结构记忆特征 | RandomForestClassifier |

两种模型使用相同的训练集、测试集、随机种子、任务分布和模型超参数。Graph-Memory RF 的额外信息只来自训练集历史任务图构成的 memory bank。

## 任务图建模

每个任务会被转换成一张 `NetworkX.DiGraph` 属性有向图。图中的节点和边都带有类型属性，用于表达任务结构。

节点类型：

```text
object, location, action, state, result
```

边类型：

```text
temporal, causal, requires, spatial, reachable
```

任务图用于刻画：

- 动作之间的先后关系；
- 动作对状态的因果影响；
- 动作执行所需的前置条件；
- 物体与地点之间的空间关系；
- 任务成功路径中的结构模式。

## Expert 标签生成

监督学习标签由确定性的 BFS expert solver 生成。BFS 搜索状态包含任务执行所需的符号状态：

```text
location
has_key
door_open
obstacle_cleared
object_checked
cart_ready
lid_secured
object_in_hand
delivered
```

每个 BFS 状态对应一个下一步 expert action label，用于训练随机森林动作分类器。

## 图结构记忆特征

对于一个新任务，系统会在训练集 memory bank 中检索相似历史任务图，并将检索结果转换为可解释的数值特征，包括：

- `node_type_overlap`
- `edge_type_overlap`
- `object_family_match`
- `target_family_match`
- `required_key_match`
- `door_dependency_match`
- `success_path_cost_similarity`
- `wl_graph_kernel_similarity`
- `subgraph_match_score`
- 历史成功路径动作分布特征

这些特征会拼接到 baseline 特征后，作为 Graph-Memory RF 的输入。

## 实验结果

默认实验设置：

```text
训练任务数：1200
测试任务数：350
模型：RandomForestClassifier
```

最新复现实验结果：

```text
Baseline accuracy:       0.9727
Graph-memory accuracy:   0.9995

Baseline macro F1:       0.9210
Graph-memory macro F1:   0.9986

Baseline rollout success rate:     0.8286
Graph-memory rollout success rate: 0.9971
```

防泄露检查：

```text
train/test task_id intersection: 0
train/test split_seed intersection: 0
memory bank source split: train
memory bank only train: True
shuffled memory accuracy: 0.9718
random memory-bank success rate: 0.6500
warnings: none
```

实验输出文件：

```text
results/metrics.json
results/summary.md
results/case_studies/
```

## 目录结构

```text
src/task_memory/
  constants.py        动作、实体类别和代价常量
  schema.py           任务、状态、图摘要和 rollout 数据结构
  data.py             合成任务生成与 train/test split
  environment.py      符号状态转移与合法动作判断
  expert.py           BFS expert 与标准成功动作序列
  graph.py            属性任务图构建与 WL 图特征
  memory.py           训练集 memory bank、图相似度和记忆特征
  features.py         baseline / memory 特征矩阵构造
  modeling.py         随机森林训练与动作概率预测
  evaluation.py       闭环 rollout、评估指标和 sanity checks
  reporting.py        实验结果与 case studies 输出
  visualization.py    任务图和动作序列可视化

experiments/run_memory_vs_baseline.py
  一键运行完整实验

app.py
  Streamlit 可视化 Demo

tests/test_core.py
  数据划分、BFS、任务图、memory bank 和模型训练测试
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行完整实验：

```bash
python experiments/run_memory_vs_baseline.py
```

运行测试：

```bash
pytest
```

启动可视化 Demo：

```bash
streamlit run app.py
```

## 可复现性

默认实验使用固定随机种子，并在任务 seed 层面进行 train/test 划分。Memory bank 只由训练集任务图构建，测试任务不会加入记忆库。实验脚本会自动执行防泄露检查，并将结果保存到 `results/metrics.json`。
