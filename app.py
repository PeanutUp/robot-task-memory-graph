from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/robot-task-memory-matplotlib")

import pandas as pd
import streamlit as st

from src.graph_builder import build_task_graph, load_tasks
from src.graph_matcher import find_reusable_subgraph
from src.graph_similarity import rank_tasks
from src.planner import generate_plan
from src.simulator import build_execution_frames
from src.visualization import draw_execution_frame, draw_task_graph


ROOT = Path(__file__).parent
HISTORY_DIR = ROOT / "data" / "historical_tasks"
QUERY_DIR = ROOT / "data" / "new_tasks"


@st.cache_data
def load_project_data():
    return load_tasks(HISTORY_DIR), load_tasks(QUERY_DIR)


def completed_highlights(plan: dict, active_node: str) -> tuple[set[str], set[tuple[str, str]]]:
    if active_node in plan["path"]:
        active_index = plan["path"].index(active_node)
    else:
        active_index = 0
    nodes = set(plan["path"][: active_index + 1])
    edges = set(plan["edges"][:active_index])
    return nodes, edges


def reset_step_if_needed(query_id: str) -> None:
    if st.session_state.get("query_id") != query_id:
        st.session_state["query_id"] = query_id
        st.session_state["step_index"] = 0


def main() -> None:
    st.set_page_config(
        page_title="Robot Task Memory Graph",
        page_icon="RT",
        layout="wide",
    )

    historical_tasks, query_tasks = load_project_data()

    st.title("机器人任务记忆图：长期任务经验复用")
    st.caption("Directed graph modeling + graph similarity + subgraph matching + shortest path planning")

    with st.sidebar:
        st.header("Demo 控制台")
        query_options = {task["title"]: task for task in query_tasks}
        selected_title = st.selectbox("选择新任务", list(query_options.keys()))
        include_ged = st.checkbox("加入 Graph Edit Distance 指标", value=False)
        st.divider()
        st.write("经验库规模")
        st.metric("历史任务图", len(historical_tasks))
        st.metric("查询任务", len(query_tasks))

    query_task = query_options[selected_title]
    reset_step_if_needed(query_task["task_id"])

    rankings = rank_tasks(
        query_task,
        historical_tasks,
        top_k=5,
        include_ged=include_ged,
    )
    best = rankings[0]
    best_graph = best["graph"]
    query_graph = build_task_graph(query_task)
    match = find_reusable_subgraph(best_graph, query_graph)
    plan = generate_plan(best_graph, query_task)
    frames = build_execution_frames(plan, query_task)

    max_step = max(0, len(frames) - 1)
    st.session_state["step_index"] = min(st.session_state.get("step_index", 0), max_step)

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        st.subheader("检索结果")
        c1, c2, c3 = st.columns(3)
        c1.metric("最相似历史任务", best["task"]["task_id"])
        c2.metric("相似度", f"{best['score']:.3f}")
        c3.metric("规划总代价", f"{plan['cost']:.1f}")
        st.write(f"**匹配任务：** {best['task']['title']}")
        st.write(f"**可复用结构：** {match['structure'] if match['found'] else '未找到完整核心子图'}")

    with top_right:
        st.subheader("推荐规划路径")
        for step in plan["steps"]:
            st.write(f"{step['index'] + 1}. {step['label']}")

    rank_rows = []
    for index, item in enumerate(rankings, start=1):
        row = {
            "rank": index,
            "task_id": item["task"]["task_id"],
            "title": item["task"]["title"],
            "score": item["score"],
            "semantic": item["breakdown"]["semantic"],
            "edge_type": item["breakdown"]["edge_type"],
        }
        if include_ged:
            row["graph_edit"] = item["breakdown"]["graph_edit"]
        rank_rows.append(row)
    st.dataframe(pd.DataFrame(rank_rows), width="stretch", hide_index=True)

    st.divider()

    control_a, control_b, control_c, control_d = st.columns([1, 1, 1, 5])
    with control_a:
        if st.button("上一步", width="stretch"):
            st.session_state["step_index"] = max(0, st.session_state["step_index"] - 1)
    with control_b:
        if st.button("下一步", type="primary", width="stretch"):
            st.session_state["step_index"] = min(max_step, st.session_state["step_index"] + 1)
    with control_c:
        if st.button("重置", width="stretch"):
            st.session_state["step_index"] = 0
    with control_d:
        st.session_state["step_index"] = st.slider(
            "执行步骤",
            min_value=0,
            max_value=max_step,
            value=st.session_state["step_index"],
            format="%d",
        )

    frame = frames[st.session_state["step_index"]]
    completed_nodes, completed_edges = completed_highlights(plan, frame["active_node"])

    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("桌面执行动画")
        st.pyplot(draw_execution_frame(frame), clear_figure=True)
        st.info(f"当前动作：{frame['action']}")

    with right:
        st.subheader("任务图同步高亮")
        st.pyplot(
            draw_task_graph(
                best_graph,
                highlighted_nodes=completed_nodes,
                highlighted_edges=completed_edges,
                matched_nodes=match["matched_nodes"],
                active_node=frame["active_node"],
                title=best["task"]["title"],
            ),
            clear_figure=True,
        )

    with st.expander("图论方法说明"):
        st.markdown(
            """
            - **有向图建模**：节点表示物体、地点、动作、状态和结果，边表示时序、因果、空间和可达关系。
            - **图相似度检索**：综合节点类型、节点角色、边类型、语义标签和任务类别计算相似度。
            - **子图匹配**：用 VF2 在历史任务图中查找可复用的核心结构。
            - **最短路径规划**：用 Dijkstra 从 `start` 到 `success` 找低代价成功路径。
            - **执行可视化**：把规划路径映射到离散桌面网格，同步高亮机器人当前执行步骤。
            """
        )


if __name__ == "__main__":
    main()
