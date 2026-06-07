from __future__ import annotations

import os
from dataclasses import asdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/task-graph-memory-matplotlib")

import pandas as pd
import streamlit as st

from src.task_memory import (
    ACTION_NAMES,
    draw_rollout_timeline,
    draw_task_graph,
    generate_dataset,
    memory_feature_vector,
    rollout_gif,
    rollout_task,
    train_models,
)


@st.cache_resource(show_spinner="Training task-graph memory models...")
def load_demo():
    train_tasks, test_tasks = generate_dataset(n_train=500, n_test=80, seed=13)
    trained = train_models(train_tasks, test_tasks, seed=13, top_k=12)
    return trained


def frame_table(frames):
    return pd.DataFrame(
        [
            {
                "step": frame.step,
                "state": frame.state.location,
                "action": frame.predicted_action,
                "message": frame.message,
                "top_memory_match": frame.top_memory_match or "",
                "similarity": round(frame.top_memory_similarity, 3),
            }
            for frame in frames
        ]
    )


def probability_table(frame):
    return pd.DataFrame(
        [
            {
                "action": action,
                "probability": frame.action_probs.get(action, 0.0),
                "predicted": action == frame.predicted_action,
            }
            for action in ACTION_NAMES
        ]
    )


def main() -> None:
    st.set_page_config(
        page_title="Robot Task Memory Graph",
        page_icon="TG",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Robot Task Memory Graph")
    st.caption("Synthetic graph-based task planning benchmark: same RandomForest model, with or without task-graph memory features.")

    trained = load_demo()
    train_tasks = trained["train_tasks"]
    test_tasks = trained["test_tasks"]
    memory_bank = trained["memory_bank"]
    report = trained["report"]

    with st.sidebar:
        st.header("Demo")
        mode = st.radio("Rollout mode", ["With graph memory", "No memory"], index=0)
        task_index = st.slider("Test task", 0, len(test_tasks) - 1, 0)
        max_steps = st.slider("Max steps", min_value=8, max_value=24, value=16, step=1)
        st.divider()
        st.metric("Model", report["model"])
        st.metric("Train tasks", f"{report['train_tasks']:,}")
        st.metric("Test tasks", f"{report['test_tasks']:,}")
        st.metric("Baseline acc", f"{report['baseline_accuracy']:.3f}")
        st.metric("Memory acc", f"{report['memory_accuracy']:.3f}")

    task = test_tasks[task_index]
    memory_enabled = mode == "With graph memory"
    model = trained["memory_model"] if memory_enabled else trained["baseline_model"]
    success, frames = rollout_task(
        model,
        task,
        memory_bank if memory_enabled else None,
        max_steps=max_steps,
        top_k=12,
    )

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Mode", mode)
    top_b.metric("Success", "Yes" if success else "No")
    top_c.metric("Frames", len(frames))
    top_d.metric("Final", frames[-1].message if frames else "none")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Task Graph")
        st.pyplot(draw_task_graph(task), clear_figure=True)
    with right:
        st.subheader("Task Spec")
        st.dataframe(pd.DataFrame([asdict(task)]), width="stretch", hide_index=True)
        if memory_enabled:
            _, matches = memory_feature_vector(task, frames[0].state, memory_bank, top_k=8)
            st.subheader("Retrieved Historical Graphs")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "task_id": match.task_id,
                            "similarity": round(match.similarity, 3),
                            "node_overlap": round(match.node_type_overlap, 3),
                            "edge_overlap": round(match.edge_type_overlap, 3),
                            "subgraph": round(match.subgraph_match_score, 3),
                            "path": " -> ".join(match.action_sequence),
                        }
                        for match in matches[:5]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.divider()
    video_col, detail_col = st.columns([1.05, 1])
    with video_col:
        st.subheader("Rollout Timeline")
        st.image(rollout_gif(frames), caption="Predicted high-level action sequence", width="stretch")
        st.pyplot(draw_rollout_timeline(frames), clear_figure=True)
    with detail_col:
        st.subheader("Step-Level Prediction")
        frame_index = st.slider("Inspect step", 0, max(0, len(frames) - 1), 0)
        frame = frames[frame_index]
        st.dataframe(frame_table(frames), width="stretch", hide_index=True)
        probs = probability_table(frame)
        st.dataframe(probs, width="stretch", hide_index=True)
        st.bar_chart(probs.set_index("action")["probability"])

    with st.expander("What is compared?"):
        st.markdown(
            """
            Both systems use the same `RandomForestClassifier` family and the same BFS expert labels.

            - **No memory** receives flat current task/state features only.
            - **With graph memory** receives the same features plus graph-similarity features retrieved from a train-only task-graph memory bank.

            The memory bank stores historical task graphs with object, location, action, state, and result nodes; edges represent temporal, causal, requires, spatial, and reachable relations.
            """
        )


if __name__ == "__main__":
    main()
