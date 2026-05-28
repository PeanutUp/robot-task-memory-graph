# 三分钟汇报稿

## 1. 选题背景

我的课程项目叫“机器人任务记忆图：基于图结构的长期任务经验复用”。

机器人在桌面收纳、厨房整理、取物放置这类任务中，经常会遇到结构相似的任务。比如“把书放进盒子”和“把积木放进收纳盒”，物体不同，但任务结构很像：检测物体、移动到物体、抓取、移动到目标位置、放置、成功。

所以我没有让机器人每次都从零规划，而是把历史任务经验保存成图结构记忆。

## 2. 图论建模

我把一次任务执行过程建成一个有向图。

节点包括：

```text
object, location, subtask, action, state, result
```

边包括：

```text
temporal, causal, spatial, reachable, object_to_subtask
```

例如一条成功路径是：

```text
start -> detect -> move_to_object -> pick -> in_hand -> move_to_target -> place -> success
```

其中有向边表达先后关系和因果关系，边上的 cost 表示动作代价。普通动作代价低，失败重试、路径受阻这类边代价高。

## 3. 算法实现

代码里主要用了三个图论方法。

第一，图相似度检索。系统会比较新任务图和历史任务图的节点类型、节点角色、边类型、语义标签和任务类别，找出最相似的历史任务。

第二，子图匹配。系统用 VF2 算法判断新任务中的核心结构，是否能在历史任务图中找到。

第三，最短路径。找到相似历史任务后，用 Dijkstra 从 `start` 到 `success` 提取总代价最低的成功路径。这样如果历史图里存在失败重试分支，系统会优先选择低代价路径。

## 4. Demo 展示

Demo 中有 12 个历史任务图，包括取放、归位、清理和失败恢复任务。

当我输入新任务：

```text
Put block into storage box
```

系统检索到最相似历史任务：

```text
task_001 Put book into storage box
```

它们物体不同，但目标位置和任务结构相似，所以可以复用历史任务的成功结构：

```text
detect -> move_to_object -> pick -> in_hand -> move_to_target -> place -> success
```

系统最终生成规划：

```text
detect block
move to block
pick block
move to storage box
place block into storage box
success
```

网页左侧会显示机器人在离散桌面网格中移动、抓取和放置；右侧会同步高亮任务图中当前执行到的节点和边。

## 5. 成果总结

这个项目的核心成果是：把机器人任务经验从普通文本或列表，转成可以检索、匹配和规划的图结构。

它体现了图论知识在代码里的应用：

```text
有向图负责建模
图相似度负责经验检索
子图匹配负责结构复用
最短路径负责任务规划
可视化负责展示执行过程
```

最终效果是：机器人面对新任务时，可以从历史任务图中找到相似经验，并复用成功路径辅助规划。

