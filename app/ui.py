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

# Numbered progression — run these in order
STEP_EXAMPLES = [
    ("Create 10k e-commerce orders", "Create a dataset of 10,000 fictional e-commerce orders."),
    ("Revenue by category", "How much revenue does each product category generate?"),
    ("Find anomalies", "Find anomalous transactions."),
    ("Visualize revenue by category", "Show me a visualization of revenue by category."),
]

# Alternative: one-shot instead of steps 1–4
FULL_WORKFLOW_EXAMPLE = (
    "Full acceptance workflow",
    (
        "Create a synthetic e-commerce dataset with 10,000 orders. "
        "Tell me which product categories generate the most revenue, "
        "identify anomalous transactions, and create visualizations showing "
        "revenue by category and the distribution of transaction amounts."
    ),
)

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
    .example-arrow {
        text-align: center;
        color: #6b7c76;
        font-size: 0.95rem;
        line-height: 1;
        margin: -0.15rem 0 0.15rem 0;
        opacity: 0.85;
    }
    .example-or {
        text-align: center;
        color: #6b7c76;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        margin: 0.65rem 0 0.55rem 0;
        text-transform: lowercase;
    }
    .tour-popup {
        position: relative;
        background: linear-gradient(145deg, #1a2e28 0%, #2d4a42 100%);
        color: #f4f1ea;
        border-radius: 12px;
        padding: 0.95rem 1.1rem 1rem 1.1rem;
        margin: 0.35rem 0 0.85rem 0;
        box-shadow: 0 10px 28px rgba(26, 46, 40, 0.28);
        border: 1px solid #3d6b5e;
    }
    .tour-popup::after {
        content: "";
        position: absolute;
        bottom: -9px;
        left: var(--tour-pointer, 8%);
        width: 0;
        height: 0;
        border-left: 9px solid transparent;
        border-right: 9px solid transparent;
        border-top: 9px solid #2d4a42;
    }
    .tour-popup .tour-step {
        display: inline-block;
        background: #c5ddd4;
        color: #1a2e28;
        border-radius: 999px;
        padding: 0.1rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .tour-popup h4 {
        color: #f4f1ea !important;
        margin: 0 0 0.35rem 0;
        font-size: 1.02rem;
    }
    .tour-popup p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .tour-popup .tour-agent {
        margin-top: 0.55rem;
        font-size: 0.8rem;
        opacity: 0.8;
    }
</style>
"""

TOUR_STEPS = [
    {
        "tab": "Progress",
        "pointer": "8%",
        "title": "Progress",
        "agent": "Orchestrator / Root Agent",
        "body": (
            "Watch the execution plan and live hand-offs here. The orchestrator "
            "interprets your request, delegates to specialists, and tracks each step — "
            "it does not invent numbers itself."
        ),
    },
    {
        "tab": "Dataset",
        "pointer": "22%",
        "title": "Dataset",
        "agent": "Dataset Agent",
        "body": (
            "Schema, row counts, and profiles for the active dataset. The Dataset Agent "
            "builds a structured spec, then deterministic code generates or loads the data "
            "into the shared workspace."
        ),
    },
    {
        "tab": "Analysis",
        "pointer": "38%",
        "title": "Analysis",
        "agent": "Analysis / Question Agent",
        "body": (
            "Answers grounded in real computation. The agent plans a query, DuckDB/pandas "
            "execute it, then the LLM only explains the returned numbers — never fabricates them."
        ),
    },
    {
        "tab": "Anomalies",
        "pointer": "54%",
        "title": "Anomalies",
        "agent": "Anomaly Detection Agent",
        "body": (
            "Unusual rows from statistical/ML methods (IQR, Z-score, Isolation Forest). "
            "The agent chooses a method; the outlier labels come from the calculation, "
            "then are explained in plain language."
        ),
    },
    {
        "tab": "Charts",
        "pointer": "70%",
        "title": "Charts",
        "agent": "Visualization Agent",
        "body": (
            "Data-driven Plotly charts from a structured viz spec. The LLM picks chart type "
            "and columns; a deterministic renderer draws the figure from actual query/dataset values."
        ),
    },
    {
        "tab": "Agent log",
        "pointer": "88%",
        "title": "Agent log",
        "agent": "Validation / Critic + full history",
        "body": (
            "Structured messages between agents: what each received, did, and produced. "
            "The Validation Agent checks grounding — dataset exists, numbers came from tools, "
            "and charts reference real data — before the final answer is trusted."
        ),
    },
]

def _fresh_state() -> dict:
    return {
        "workspace": SharedWorkspace(),
        "events": [],
        "messages": [],
        "busy": False,
        "tour_active": True,
        "tour_step": 0,
    }


# Only app-owned keys — never delete Streamlit widget keys in the same click handler.
_APP_KEYS = ("workspace", "events", "messages", "busy", "pending_prompt", "flash", "tour_active", "tour_step")


def init_session() -> None:
    for key, value in _fresh_state().items():
        if key not in st.session_state:
            st.session_state[key] = value


def hard_reset_session() -> None:
    """Clear conversation + results without destroying active widget keys mid-click."""
    fresh = _fresh_state()
    for key in _APP_KEYS:
        st.session_state.pop(key, None)
    st.session_state.workspace = fresh["workspace"]
    st.session_state.events = fresh["events"]
    st.session_state.messages = fresh["messages"]
    st.session_state.busy = False
    st.session_state.flash = "Workspace reset — conversation and results cleared."


def apply_reset_if_requested() -> bool:
    """Honor ?reset=1 from the Reset link/button. Returns True if a reset ran."""
    try:
        reset_flag = st.query_params.get("reset")
    except Exception:  # noqa: BLE001
        reset_flag = None
    if reset_flag != "1":
        return False
    hard_reset_session()
    try:
        st.query_params.clear()
    except Exception:  # noqa: BLE001
        pass
    return True


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
    top_l, top_r = st.columns([4, 1])
    with top_l:
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
    with top_r:
        st.write("")
        # link_button navigates to ?reset=1 — more reliable than widget on_click for full clears
        st.link_button(
            "Reset workspace",
            url="?reset=1",
            use_container_width=True,
            type="primary",
            help="Clear conversation, dataset, analyses, anomalies, charts, and agent log",
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
    for i, viz in enumerate(ws.visualizations):
        chart_id = viz.get("id") or f"idx_{i}"
        st.markdown(f"#### {viz.get('title')} (`{viz.get('chart_type')}`)")
        fig_json = viz.get("plotly_json")
        if fig_json:
            payload = json.dumps(fig_json) if isinstance(fig_json, dict) else fig_json
            fig = pio.from_json(payload)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"plotly_{chart_id}_{i}_{st.session_state.get('tour_step', 0)}_{bool(st.session_state.get('tour_active'))}",
            )
        elif viz.get("html_path"):
            st.caption(f"Chart artifact: `{viz['html_path']}`")
        st.caption(f"{viz.get('n_points')} data points · grounded={viz.get('grounded')}")


def render_history_panel(ws: SharedWorkspace) -> None:
    if not ws.agent_history:
        st.info("No agent history yet.")
        return
    st.json(ws.agent_history)


def render_tour_popup(step_index: int) -> None:
    """Numbered popup callout that points up toward the active results tab."""
    step = TOUR_STEPS[step_index]
    n = len(TOUR_STEPS)
    st.markdown(
        f"""
        <div class="tour-popup" style="--tour-pointer: {step["pointer"]};">
          <div class="tour-step">Step {step_index + 1} of {n}</div>
          <h4>{step["title"]}</h4>
          <p>{step["body"]}</p>
          <div class="tour-agent">Agent focus: <strong>{step["agent"]}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        if st.button("Back", use_container_width=True, disabled=step_index <= 0, key="tour_back"):
            st.session_state.tour_step = max(0, step_index - 1)
            st.rerun()
    with c2:
        if st.button(
            "Next" if step_index < n - 1 else "Done",
            use_container_width=True,
            type="primary",
            key="tour_next",
        ):
            if step_index >= n - 1:
                st.session_state.tour_active = False
            else:
                st.session_state.tour_step = step_index + 1
            st.rerun()
    with c3:
        if st.button("Skip tour", use_container_width=True, key="tour_skip"):
            st.session_state.tour_active = False
            st.rerun()
    with c4:
        st.caption(f"Tab → **{step['tab']}**")


def section_help(title: str, agent: str, body: str) -> None:
    """Inline ? popover for each results section."""
    head, tip = st.columns([6, 1])
    with head:
        st.markdown(f"#### {title}")
    with tip:
        with st.popover("?"):
            st.markdown(f"**{agent}**")
            st.write(body)


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
    apply_reset_if_requested()

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
        st.link_button("Reset workspace", url="?reset=1", use_container_width=True, type="primary")

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
        st.caption(
            "Steps don’t have to run in order, but create or load a dataset first "
            "(step 1) so analysis, anomalies, and charts have data to work with — "
            "or use the full workflow instead."
        )
        pending_from_example: str | None = None
        for i, (label, prompt) in enumerate(STEP_EXAMPLES, start=1):
            if st.button(f"{i}. {label}", use_container_width=True, key=f"example_step_{i}"):
                pending_from_example = prompt
            if i < len(STEP_EXAMPLES):
                st.markdown('<div class="example-arrow">↓</div>', unsafe_allow_html=True)

        st.markdown('<div class="example-or">———— or ————</div>', unsafe_allow_html=True)
        full_label, full_prompt = FULL_WORKFLOW_EXAMPLE
        if st.button(full_label, use_container_width=True, key="example_full_workflow"):
            pending_from_example = full_prompt

    # Always bind panels to the latest session_state after callbacks (e.g. reset on_click)
    ws: SharedWorkspace = st.session_state.workspace
    render_header(ws)

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        st.markdown("### Conversation")
        with st.container(height=520):
            render_conversation(list(st.session_state.get("messages") or []))

        prompt = st.chat_input("Ask the multi-agent system…")
        if pending_from_example:
            prompt = pending_from_example
        if prompt:
            run_request(prompt)
            st.rerun()

    with right:
        title_col, tour_col = st.columns([3.2, 1.3])
        with title_col:
            st.markdown("### Results workspace")
        with tour_col:
            if st.button(
                "Agent tour",
                use_container_width=True,
                key="start_agent_tour",
                help="Numbered walkthrough of each results tab and its agent",
            ):
                st.session_state.tour_active = True
                st.session_state.tour_step = 0
                st.rerun()

        tour_active = bool(st.session_state.get("tour_active"))
        tour_step = int(st.session_state.get("tour_step") or 0)
        tour_step = max(0, min(tour_step, len(TOUR_STEPS) - 1))
        st.session_state.tour_step = tour_step

        if tour_active:
            render_tour_popup(tour_step)

        tab_labels = [s["tab"] for s in TOUR_STEPS]
        default_tab = TOUR_STEPS[tour_step]["tab"] if tour_active else "Progress"
        # Remount tabs when the tour step changes so the pointed tab becomes active
        tabs = st.tabs(
            tab_labels,
            default=default_tab,
            key=f"results_tabs_tour_{tour_step}" if tour_active else "results_tabs_main",
        )

        with tabs[0]:
            section_help(
                "Progress",
                TOUR_STEPS[0]["agent"],
                TOUR_STEPS[0]["body"],
            )
            render_progress(list(st.session_state.get("events") or []))
        with tabs[1]:
            section_help(
                "Dataset",
                TOUR_STEPS[1]["agent"],
                TOUR_STEPS[1]["body"],
            )
            render_dataset_panel(ws)
        with tabs[2]:
            section_help(
                "Analysis",
                TOUR_STEPS[2]["agent"],
                TOUR_STEPS[2]["body"],
            )
            render_analysis_panel(ws)
        with tabs[3]:
            section_help(
                "Anomalies",
                TOUR_STEPS[3]["agent"],
                TOUR_STEPS[3]["body"],
            )
            render_anomaly_panel(ws)
        with tabs[4]:
            section_help(
                "Charts",
                TOUR_STEPS[4]["agent"],
                TOUR_STEPS[4]["body"],
            )
            render_charts_panel(ws)
        with tabs[5]:
            section_help(
                "Agent log",
                TOUR_STEPS[5]["agent"],
                TOUR_STEPS[5]["body"],
            )
            render_history_panel(ws)


if __name__ == "__main__":
    main()
