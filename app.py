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
        page_title="Robot Task Memory",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    historical_tasks, query_tasks = load_project_data()

    # --- Header ---
    st.title("🤖 机器人任务记忆图：长期任务经验复用")
    st.markdown("基于 **有向图建模**、**规划感知 WL 图核检索** 与 **最短路径复用** 的机器人任务记忆系统。")

    # --- Sidebar ---
    with st.sidebar:
        st.header("🎮 Demo 控制台")
        st.markdown("选择一个新任务，观察系统如何检索历史任务图并复用成功路径。")
        query_options = {task["title"]: task for task in query_tasks}
        selected_title = st.selectbox("🎯 选择新查询任务", list(query_options.keys()))
        
        st.divider()
        st.markdown("📊 **领域经验库规模**")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("历史经验", len(historical_tasks))
        metric_col2.metric("测试用例", len(query_tasks))

    query_task = query_options[selected_title]
    reset_step_if_needed(query_task["task_id"])

    # Compute matches
    rankings = rank_tasks(
        query_task,
        historical_tasks,
        top_k=5,
        include_ged=True,  # 默认开启 GED (Graph Edit Distance)
    )
    best = rankings[0]
    best_graph = best["graph"]
    query_graph = build_task_graph(query_task)
    match = find_reusable_subgraph(best_graph, query_graph)
    plan = generate_plan(best_graph, query_task)
    frames = build_execution_frames(plan, query_task)

    max_step = max(0, len(frames) - 1)
    st.session_state["step_index"] = min(st.session_state.get("step_index", 0), max_step)
    frame = frames[st.session_state["step_index"]]
    completed_nodes, completed_edges = completed_highlights(plan, frame["active_node"])

    # --- Main Content UI ---
    tab1, tab2 = st.tabs(["🕹️ 执行可视化", "📑 检索与分析面板"])

    with tab1:
        st.subheader(f"当前执行阶段：**{best['task']['title']}**")
        
        # UI controls wrapped nicely
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 4])
        with ctrl_col1:
            if st.button("⏪ 上一步", width="stretch"):
                st.session_state["step_index"] = max(0, st.session_state["step_index"] - 1)
        with ctrl_col2:
            if st.button("⏩ 下一步", type="primary", width="stretch"):
                st.session_state["step_index"] = min(max_step, st.session_state["step_index"] + 1)
        with ctrl_col3:
            if st.button("🔄 重置", width="stretch"):
                st.session_state["step_index"] = 0
        with ctrl_col4:
            st.session_state["step_index"] = st.slider(
                "播放进度",
                min_value=0,
                max_value=max_step,
                value=st.session_state["step_index"],
                format="步骤 %d",
                label_visibility="collapsed"
            )

        # Plotly side by side
        plot_col1, plot_col2 = st.columns([1, 1.25])
        with plot_col1:
            st.markdown(f"#### 📺 场景映射动画  \n👉 **动作**: `{frame['action']}`")
            with st.container(border=True):
                st.pyplot(draw_execution_frame(frame), clear_figure=True)
                
        with plot_col2:
            st.markdown("#### 🧠 知识图谱激活态")
            with st.container(border=True):
                st.pyplot(
                    draw_task_graph(
                        best_graph,
                        highlighted_nodes=completed_nodes,
                        highlighted_edges=completed_edges,
                        matched_nodes=match["matched_nodes"],
                        active_node=frame["active_node"],
                        title="Inference graph",
                    ),
                    clear_figure=True,
                )

    with tab2:
        st.subheader("💡 检索细节分析")
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("📌 最佳匹配任务 ID", best["task"]["task_id"])
        m_c2.metric("🌟 综合检索分数", f"{best['score']:.3f}")
        m_c3.metric("🛣️ 最短路径代价", f"{plan['cost']:.1f}")
        m_c4.metric("⚙️ 子图同构度", f"{len(match['matched_nodes'])} Node(s)")

        st.markdown(f"**核心复用结构：** `{match['structure'] if match['found'] else '尚未找到具有完整特征的核心子结构'}`")
        
        with st.expander("📍 推荐路径细节", expanded=True):
            for step in plan["steps"]:
                st.markdown(f"- 🟢 **Step {step['index'] + 1}**: {step['label']}")

        st.markdown("#### 🏆 历史记忆召回排行")
        rank_rows = []
        for index, item in enumerate(rankings, start=1):
            row = {
                "排名": f"Top {index}",
                "历史任务 ID": item["task"]["task_id"],
                "任务描述": item["task"]["title"],
                "学习检索分数": f"{item['score']:.4f}",
                "手工融合对照": f"{item['breakdown']['fusion_score']:.4f}",
                "WL图核": f"{item['breakdown']['wl_kernel']:.4f}",
                "TF-IDF语义": f"{item['breakdown']['semantic_cosine']:.4f}",
                "边类型重合": f"{item['breakdown']['edge_type']:.4f}",
                "GED": f"{item['breakdown']['graph_edit']:.4f}"
            }
            rank_rows.append(row)
        st.dataframe(pd.DataFrame(rank_rows), width="stretch", hide_index=True)

        with st.expander("🧮 学到的检索权重"):
            training_pairs = best["breakdown"]["training_pairs"]
            st.write(
                f"训练样本来自历史任务两两配对：正样本 {training_pairs['positive']}，"
                f"负样本 {training_pairs['negative']}。"
            )
            weights = best["breakdown"]["learned_weights"]
            weight_rows = [
                {"特征": name, "学习权重": value}
                for name, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)
            ]
            st.dataframe(pd.DataFrame(weight_rows), width="stretch", hide_index=True)

    with st.expander("ℹ️ 系统底层逻辑说明"):
        st.markdown(
            """
            - **有向图建模**：将动作抽象为节点(物体、地点、状态)，边表示时序、空间、因果关系。
            - **规划感知 WL 图核**：重点编码从 start 到 success 的低代价成功路径，并保留完整图结构作为辅助特征。
            - **弱监督学习排序器**：从历史任务两两配对中学习特征权重，预测某个历史任务是否值得复用。
            - **TF-IDF + Cosine 检索**：把任务目标、物体、地点、类别等文本字段转为向量，作为学习排序器的输入特征。
            - **子图匹配 (VF2)**：精准抽取可直接复用的历史动作序列逻辑。
            - **Dijkstra 路径复用**：在历史任务图中提取从 start 到 success 的最低代价成功路径。
            - **降维可视化**：将图谱推演映射回二维桌面的空间执行过程。
            """
        )

if __name__ == "__main__":
    main()
