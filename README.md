# Robot Task Memory Graph

这是一个课程设计 Demo：把机器人桌面收纳任务建成有向图，再用历史任务图辅助新任务规划。

项目重点不是接入真实机器人，而是展示图论知识如何转成代码能力：

- 有向图：表示物体、地点、动作、状态和结果。
- 图相似度：从历史经验库中检索相似任务。
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

## Demo 说明

打开网页后，左侧选择新任务，例如：

```text
Put block into storage box
```

系统会：

1. 把新任务构造成查询任务图。
2. 和 12 个历史任务图计算相似度。
3. 找到最相似历史任务，例如 `task_001 Put book into storage box`。
4. 用 VF2 子图匹配找到可复用结构。
5. 用 Dijkstra 生成从 `start` 到 `success` 的低代价路径。
6. 在桌面网格中逐步展示机器人移动、抓取和放置过程。
7. 在任务图中同步高亮当前执行节点和边。

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
  graph_matcher.py     VF2 子图匹配
  planner.py           Dijkstra 成功路径规划
  simulator.py         离散桌面执行模拟
  visualization.py     任务图和执行过程绘图
app.py                 Streamlit Demo
tests/                 核心逻辑测试
docs/report.md         三分钟汇报稿
```

## 三分钟展示重点

这不是普通路径规划。普通路径规划只在当前环境里找路；本项目先从历史任务图中检索相似经验，再把成功结构迁移到新任务中。

一句话总结：

```text
机器人把过去完成任务的经验存成图，新任务到来时，通过图相似度和子图匹配复用过去的成功路径。
```

