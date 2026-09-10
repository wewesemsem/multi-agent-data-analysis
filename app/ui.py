"""Streamlit UI for the multi-agent data intelligence MVP."""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.dataset_agent import DatasetAgent
from app.llm import LLMClient
from app.messages import AgentMessage
from app.orchestrator import Orchestrator
from app.state import SharedWorkspace, ensure_workspace

st.set_page_config(
    page_title="Data Intelligence MAS",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_workspace()

EXAMPLES = [
    ("Create 10k e-commerce orders", "Create a dataset of 10,000 fictional e-commerce orders."),
    ("Revenue by category", "How much revenue does each product category generate?"),
    ("Find anomalies", "Find anomalous transactions."),
    ("Visualize revenue by category", "Show me a visualization of revenue by category."),
    (
        "Full acceptance workflow",
        (
            "Create a synthetic e-commerce dataset with 10,000 orders. "
            "Tell me which product categories generate the most revenue, "
            "identify anomalous transactions, and create visualizations showing "
            "revenue by category and the distribution of transaction amounts."
        ),
    ),
]

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #f7f5f1;
        border: 1px solid #e4e0d8;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .mas-banner {
        background: linear-gradient(135deg, #1a2e28 0%, #2d4a42 55%, #3d6b5e 100%);
        color: #f4f1ea;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .mas-banner h1 {
        color: #f4f1ea !important;
        font-size: 1.6rem;
        margin: 0 0 0.35rem 0;
        font-weight: 650;
    }
    .mas-banner p { margin: 0; opacity: 0.85; font-size: 0.95rem; }
    .agent-chip {
        display: inline-block;
        background: #eef6f3;
        color: #1a2e28;
        border: 1px solid #c5ddd4;
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.8rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-idle { background: #eee; color: #555; }
    .status-running { background: #fff3cd; color: #856404; }
    .status-ok { background: #d4edda; color: #155724; }
    .status-warn { background: #f8d7da; color: #721c24; }
    .msg-user, .msg-assistant {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        white-space: pre-wrap;
    }
    .msg-user { background: #eef6f3; border: 1px solid #c5ddd4; }
    .msg-assistant { background: #f7f5f1; border: 1px solid #e4e0d8; }
</style>
"""

def _fresh_state() -> dict:
    return {
        "workspace": SharedWorkspace(),
        "events": [],
        "messages": [],
        "busy": False,
    }


def init_session() -> None:
    for key, value in _fresh_state().items():
        if key not in st.session_state:
            st.session_state[key] = value


def hard_reset_session() -> None:
    """Delete all session + widget state, then recreate app state.

    Streamlit keeps widget values in session_state; assigning empty lists alone
    is not enough to clear the visible conversation/results reliably.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.update(_fresh_state())
    st.session_state.flash = "Workspace reset — conversation and results cleared."


def status_class(status: str) -> str:
    if status == "completed":
        return "status-ok"
    if status in {"completed_with_warnings", "error"}:
        return "status-warn"
    if status in {"planning", "executing", "validating"}:
        return "status-running"
    return "status-idle"


def render_header(ws: SharedWorkspace) -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mas-banner">
          <h1>Data Intelligence</h1>
          <p>Multi-agent workspace — orchestrator delegates to dataset, analysis,
          anomaly, visualization, and validation agents. Numbers come from tools, not the LLM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", ws.task_status)
    c2.metric("Dataset rows", (ws.dataset or {}).get("row_count") or 0)
    c3.metric("Analyses", len(ws.analysis_results))
    c4.metric("Charts", len(ws.visualizations))


def render_progress(events: list[dict]) -> None:
    if not events:
        st.info("Agent progress will appear here after you send a request.")
        return
    for ev in events:
        et = ev.get("type")
        if et == "plan":
            st.markdown(f"**Plan:** {ev.get('summary')}")
            for step in ev.get("plan") or []:
                st.markdown(
                    f'<span class="agent-chip">{step.get("agent")}</span> '
                    f'`{step.get("action")}` — {step.get("rationale", "")}',
                    unsafe_allow_html=True,
                )
        elif et == "step_start":
            step = ev.get("step") or {}
            st.markdown(f"→ Starting **{step.get('agent')}** / `{step.get('action')}`")
        elif et == "step_end":
            step = ev.get("step") or {}
            icon = "✓" if ev.get("success") else "✗"
            st.markdown(f"{icon} Finished **{step.get('agent')}** / `{step.get('action')}`")
            if ev.get("error"):
                st.error(ev["error"])
        elif et == "validation":
            icon = "✓" if ev.get("ok") else "!"
            st.markdown(f"{icon} Validation `{ev.get('check')}`")
            if ev.get("issues"):
                st.warning(ev["issues"])
        elif et == "retry":
            st.markdown(f"Retrying `{ev.get('agent')}`")
        elif et == "final":
            st.success(f"Done — status `{ev.get('status')}`")


def render_dataset_panel(ws: SharedWorkspace) -> None:
    if not ws.dataset:
        st.info("No dataset in the shared workspace yet. Ask the system to create one, or upload a CSV.")
        return
    ds = ws.dataset
    st.markdown(f"**{ds.get('name')}** (`{ds.get('id')}`)")
    c1, c2, c3 = st.columns(3)
    c1.write(f"Rows: **{ds.get('row_count')}**")
    c2.write(f"Columns: **{ds.get('column_count')}**")
    c3.write(f"Source: **{ds.get('source')}**")
    st.markdown("#### Schema")
    st.json(ds.get("schema") or {})
    profile = ds.get("profile") or {}
    if profile.get("numeric"):
        st.markdown("#### Numeric profile")
        st.json(profile["numeric"])
    if profile.get("categorical_top"):
        st.markdown("#### Top categories")
        st.json(profile["categorical_top"])


def render_analysis_panel(ws: SharedWorkspace) -> None:
    if not ws.analysis_results:
        st.info("No analysis results yet.")
        return
    for i, item in enumerate(ws.analysis_results, 1):
        st.markdown(f"#### Analysis {i}")
        st.markdown(item.get("explanation") or "")
        records = item.get("result", {}).get("records") or []
        if records:
            st.dataframe(records, use_container_width=True)
        with st.expander("Query plan / tool output"):
            st.json(
                {
                    "query_plan": item.get("query_plan"),
                    "grounded": item.get("result", {}).get("grounded"),
                    "sql": item.get("result", {}).get("sql")
                    or item.get("result", {}).get("sql_equivalent"),
                }
            )


def render_anomaly_panel(ws: SharedWorkspace) -> None:
    if not ws.anomalies:
        st.info("No anomaly findings yet.")
        return
    for i, item in enumerate(ws.anomalies, 1):
        st.markdown(f"#### Anomaly run {i}")
        st.markdown(
            f"Method **`{item.get('method')}`** on `{item.get('column')}` — "
            f"**{item.get('n_anomalies')}** anomalies "
            f"({(item.get('anomaly_rate') or 0) * 100:.2f}% of rows)"
        )
        st.markdown(item.get("explanation") or "")
        records = item.get("records") or []
        if records:
            st.dataframe(records, use_container_width=True)
        with st.expander("Method parameters"):
            st.json(item.get("parameters") or {})


def render_charts_panel(ws: SharedWorkspace) -> None:
    if not ws.visualizations:
        st.info("No visualizations yet.")
        return
    for viz in ws.visualizations:
        st.markdown(f"#### {viz.get('title')} (`{viz.get('chart_type')}`)")
        fig_json = viz.get("plotly_json")
        if fig_json:
            payload = json.dumps(fig_json) if isinstance(fig_json, dict) else fig_json
            fig = pio.from_json(payload)
            st.plotly_chart(fig, use_container_width=True)
        elif viz.get("html_path"):
            st.caption(f"Chart artifact: `{viz['html_path']}`")
        st.caption(f"{viz.get('n_points')} data points · grounded={viz.get('grounded')}")


def render_history_panel(ws: SharedWorkspace) -> None:
    if not ws.agent_history:
        st.info("No agent history yet.")
        return
    st.json(ws.agent_history)


def render_conversation(messages: list[dict]) -> None:
    if not messages:
        st.caption("Ask in natural language, or pick an example from the sidebar.")
        return
    for msg in messages:
        role = msg.get("role", "assistant")
        css = "msg-user" if role == "user" else "msg-assistant"
        label = "You" if role == "user" else "Agents"
        # Escape-free markdown body; role label only in plain text prefix
        st.markdown(f"**{label}**")
        st.markdown(f'<div class="{css}">{_html_escape(msg.get("content") or "")}</div>', unsafe_allow_html=True)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def run_request(prompt: str) -> None:
    ws: SharedWorkspace = st.session_state.workspace
    messages = list(st.session_state.messages)
    messages.append({"role": "user", "content": prompt})
    events: list[dict] = []

    def on_progress(event: dict) -> None:
        events.append(event)

    with st.spinner("Orchestrator coordinating agents…"):
        orch = Orchestrator()
        ws = orch.run(prompt, workspace=ws, progress_callback=on_progress)

    messages.append({"role": "assistant", "content": ws.final_response or "Completed."})
    st.session_state.workspace = ws
    st.session_state.events = events
    st.session_state.messages = messages


def main() -> None:
    init_session()

    # Handle reset BEFORE any main-panel render so stale UI cannot paint.
    with st.sidebar:
        st.markdown("### Workspace")
        ws_preview: SharedWorkspace = st.session_state.workspace
        st.caption(f"Session `{ws_preview.session_id[:8]}`")
        st.markdown(
            f'<span class="status-pill {status_class(ws_preview.task_status)}">'
            f"{ws_preview.task_status}</span>",
            unsafe_allow_html=True,
        )
        llm = LLMClient()
        st.caption("LLM: " + ("connected" if llm.available else "offline heuristics"))

        if st.button("Reset workspace", use_container_width=True, type="primary", key="reset_workspace_btn"):
            hard_reset_session()
            st.rerun()

        st.divider()
        st.markdown("### Load CSV")
        uploaded = st.file_uploader("Upload a CSV into the shared workspace", type=["csv"], key="csv_uploader")
        if uploaded is not None and st.button("Load into Dataset Agent", use_container_width=True, key="csv_load_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            agent = DatasetAgent()
            msg = AgentMessage(
                task_id=str(uuid.uuid4()),
                source_agent="orchestrator",
                target_agent="dataset_agent",
                action="load_csv",
                parameters={"path": tmp_path, "name": uploaded.name},
            )
            result = agent.handle(msg, st.session_state.workspace)
            if result.success:
                st.session_state.flash = f"Loaded `{uploaded.name}`"
                st.rerun()
            st.error(result.error or "Failed to load CSV")

        st.divider()
        st.markdown("### Try an example")
        pending_from_example: str | None = None
        for label, prompt in EXAMPLES:
            if st.button(label, use_container_width=True, key=f"example_{label}"):
                pending_from_example = prompt

    # Re-read after possible sidebar mutations
    ws: SharedWorkspace = st.session_state.workspace
    render_header(ws)

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        st.markdown("### Conversation")
        with st.container(height=520):
            render_conversation(st.session_state.messages)

        prompt = st.chat_input("Ask the multi-agent system…")
        if pending_from_example:
            prompt = pending_from_example
        if prompt:
            run_request(prompt)
            st.rerun()

    with right:
        st.markdown("### Results workspace")
        tabs = st.tabs(["Progress", "Dataset", "Analysis", "Anomalies", "Charts", "Agent log"])
        with tabs[0]:
            render_progress(st.session_state.events)
        with tabs[1]:
            render_dataset_panel(ws)
        with tabs[2]:
            render_analysis_panel(ws)
        with tabs[3]:
            render_anomaly_panel(ws)
        with tabs[4]:
            render_charts_panel(ws)
        with tabs[5]:
            render_history_panel(ws)


if __name__ == "__main__":
    main()
