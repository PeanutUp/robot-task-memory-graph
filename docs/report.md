# 三分钟汇报稿

## 1. 项目背景

我们的题目是 **机器人任务记忆图：基于图结构的长期任务经验复用**。

这个项目不做真实机器人控制，也不做物理仿真。我们关注的是高层任务规划：机器人以前做过很多整理、取放、开门、清障、放置任务，能不能把这些历史经验建成图结构记忆，并在新任务中作为机器学习模型的辅助特征。

任务例子是：

```text
拿钥匙 -> 开门 -> 清障 -> 移动到物体 -> 抓取 -> 移动到目标 -> 放置
```

有些任务还需要特殊处理，比如易碎物体要先检查，重物要用推车，液体容器要先盖紧。

## 2. 图论建模

每个任务都被建模成一张属性有向图 `G=(V,E)`。

节点包括：

```text
object: book, cup, toolbox
location: start, door, object_area, target_area
action: pickup_key, open_door, pick_object
state: has_key, door_open, object_in_hand
result: success, failure
```

边包括：

```text
temporal: 动作先后关系
causal: 动作导致状态变化
requires: 某动作依赖某状态
spatial: 物体和地点关系
reachable: 地点可达关系
```

这部分对应离散数学里的有向图、属性图、路径、边类型、图相似度和子图结构。

## 3. Expert 标签和机器学习模型

训练标签由 BFS expert solver 生成。BFS 使用包含任务条件的扩展状态：

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

这样 key、door、obstacle 和 object-in-hand 都不会被遗漏。

我们训练两个模型，二者都是同一种模型：

```text
RandomForestClassifier
```

baseline 输入无记忆特征，例如当前状态、是否有 key、门是否打开、物体类别、目标类别等。

memory 版本输入同样的 baseline 特征，额外加入从训练集历史任务图 memory bank 检索得到的图结构特征，包括：

```text
node_type_overlap
edge_type_overlap
object_family_match
target_family_match
required_key_match
door_dependency_match
success_path_cost_similarity
WL graph kernel similarity
subgraph_match_score
historical success-path action profile
```

memory 模块只输出特征，不直接替模型执行动作。

## 4. 实验和防泄露

数据是自动生成的 synthetic benchmark。每个任务有独立的 `task_id` 和 `split_seed`，train/test 在 seed 层面划分，保证同一个任务或同一个 seed 不会同时出现在训练和测试里。

防泄露检查包括：

```text
train/test task_id intersection = 0
train/test split_seed intersection = 0
memory bank only from train split
shuffle memory features sanity check
random memory bank sanity check
```

评估不只看分类准确率，还看闭环任务成功率。测试时模型一步步预测高层 action，如果预测非法动作、超时或没完成任务，就算失败。

输出指标包括：

```text
baseline accuracy
memory accuracy
baseline macro F1
memory macro F1
baseline rollout success rate
memory rollout success rate
平均步数
失败原因统计
```

## 5. 结论

这个项目的核心结论是：

```text
历史任务经验可以被表示成任务图；
新任务可以通过图相似度和子图结构检索相似经验；
这些图结构记忆特征可以接入同一种机器学习模型，
提升高层任务规划的稳定性。
```

项目重点是把历史任务里的物体、动作、状态、依赖和成功路径表示成图，并把这些图论特征转成可学习模型的输入。
