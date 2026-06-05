# Robot Task Memory Graph

这是一个课程设计 Demo：把机器人桌面收纳任务建成有向图，再用历史任务图辅助新任务规划。

项目重点不是接入真实机器人，而是展示图论知识如何转成代码能力：

- 有向图：表示物体、地点、动作、状态和结果。
- 规划感知 WL 图核：把任务图编码成结构特征向量。
- TF-IDF + Cosine：从任务文本字段中计算语义相似度。
- 弱监督学习排序器：从历史任务对中学习检索权重。
- 子图匹配：找到可以复用的任务结构。
- 最短路径：从历史任务中抽取低代价成功路径。
- 可视化：同步展示机器人执行过程和任务图高亮。

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

如果依赖已经装好，可以直接运行：

```bash
streamlit run app.py
```

测试：

```bash
pytest
```

运行大规模合成实验：

```bash
python3 experiments/run_synthetic_memory_experiment.py \
  --memory-tasks 3000 \
  --train-queries 1200 \
  --train-pairs 12000 \
  --eval-tasks 300 \
  --rollouts 100
```

## Demo 说明

打开网页后，左侧选择新任务，例如：

```text
Put block into storage box
```

系统会：

1. 把新任务构造成查询任务图。
2. 和 12 个历史任务图计算相似度。
3. 用规划感知 Weisfeiler-Lehman 图核提取结构特征。
4. 用 TF-IDF + Cosine 计算任务语义相似度。
5. 用历史任务对训练 logistic ranker，学习各检索特征权重。
6. 找到最相似历史任务，例如 `task_001 Put book into storage box`。
7. 用 VF2 子图匹配找到可复用结构。
8. 用 Dijkstra 生成从 `start` 到 `success` 的低代价路径。
9. 在桌面网格中逐步展示机器人移动、抓取和放置过程。
10. 在任务图中同步高亮当前执行节点和边。

## Learned Memory Experiment

除了 12 个手写展示任务，项目还包含一个自动合成数据实验：

```text
历史记忆任务：3000
训练查询任务：1200
训练任务对：12000
评估任务：300
每个任务 rollout：100
每个系统总 rollout：30000
```

训练的是一个可学习的 `LogisticRegression` 检索模型。模型输入是一对任务图之间的图结构和语义特征，输出该历史任务是否值得被新任务复用。

正式实验结果：

```text
No memory baseline success: 0.588
Learned graph memory success: 0.871
Absolute improvement: 0.283
Relative improvement: 48.0%
Top-1 reusable retrieval: 1.000
```

完整结果见 [docs/experiment_results.md](docs/experiment_results.md)。

## 数据结构

每个任务 JSON 包含：

```text
nodes: object, location, subtask, action, state, result
edges: temporal, causal, spatial, reachable, object_to_subtask
cost: 动作或状态转移代价
scene: 离散桌面网格，用于执行动画
```

核心路径示例：

```text
start -> detect -> move_to_object -> pick -> in_hand -> move_to_target -> place -> success
```

失败恢复任务会额外加入：

```text
grasp_failed -> retry_pick
path_blocked -> reroute
object_missing -> search
```

## 项目结构

```text
data/
  historical_tasks/    12 个历史任务图
  new_tasks/           4 个查询任务图
src/
  graph_builder.py     JSON 转 NetworkX 有向图
  graph_similarity.py  图相似度和 Graph Edit Distance 指标
  synthetic_tasks.py   自动生成合成任务图
  memory_learning.py   可学习记忆检索模型和 rollout 评估
  graph_matcher.py     VF2 子图匹配
  planner.py           Dijkstra 成功路径规划
  simulator.py         离散桌面执行模拟
  visualization.py     任务图和执行过程绘图
experiments/
  run_synthetic_memory_experiment.py
app.py                 Streamlit Demo
tests/                 核心逻辑测试
docs/report.md         三分钟汇报稿
docs/experiment_results.md
```

## 三分钟展示重点

这不是普通路径规划。普通路径规划只在当前环境里找路；本项目先从历史任务图中检索相似经验，再把成功结构迁移到新任务中。

检索权重不是人工固定，而是从历史任务对中弱监督学习得到。每个任务对根据是否具有可迁移的物体族、目标族、放置关系和任务结构生成训练标签，再训练一个 logistic ranker 预测历史任务是否适合被复用。

输入特征包括：

```text
WL graph kernel cosine
TF-IDF semantic cosine
target family match
object family match
relation match
target/object label cosine
edge/node type overlap
success path cost similarity
```

其中 WL 图核重点编码最低代价成功路径，避免失败分支干扰可复用经验检索。Graph Edit Distance 作为解释性指标展示，不参与主排序。

这里的“动作”不是神经网络生成的低层关节控制，而是高层任务动作程序：

```text
detect -> move_to_object -> pick -> move_to_target -> place
```

无记忆系统只能使用固定模板；有记忆系统会先检索相似历史经验，再复用历史图中的成功路径和失败恢复结构。

一句话总结：

```text
机器人把过去完成任务的经验存成图，新任务到来时，通过图相似度和子图匹配复用过去的成功路径。
```
