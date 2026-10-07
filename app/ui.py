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
from app.llm import (
    DEFAULT_PROVIDER,
    LLMClient,
    MAX_MODEL_CHOICES,
    PROVIDERS,
    clear_model_cache,
    latest_model_for,
    list_models,
    resolve_provider,
)
from app.context import ConversationContext
from app.messages import AgentMessage
from app.orchestrator import Orchestrator
from app.state import SharedWorkspace, ensure_workspace
st.set_page_config(
    page_title="Data Intelligence MAS",
    page_icon=str(ROOT / ".streamlit" / "favicon.ico"),
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_workspace()

# Numbered progression — run these in order
STEP_EXAMPLES = [
    ("Create 10k e-commerce orders", "Create a dataset of 10,000 fictional e-commerce orders."),
    ("Revenue by category", "How much revenue does each product category generate?"),
    ("Find anomalies", "Find anomalous transactions."),
    ("Forecast revenue", "Forecast revenue for the next 12 months."),
    ("Visualize revenue by category", "Show me a visualization of revenue by category."),
    (
        "Draft a report",
        "Write a detailed report summarizing the analysis, anomalies, and forecast.",
    ),
]

# Alternative: one-shot instead of steps 1–6
FULL_WORKFLOW_EXAMPLE = (
    "Full acceptance workflow",
    (
        "Create a synthetic e-commerce dataset with 10,000 orders. "
        "Tell me which product categories generate the most revenue, "
        "identify anomalous transactions, forecast revenue for the next 12 months, "
        "create visualizations showing revenue by category and the distribution of "
        "transaction amounts, and draft a report summarizing the findings."
    ),
)

# Second stepped demo — same pattern as STEP_EXAMPLES, but on a generic (non-ecommerce) dataset.
TOOLKIT_STEP_EXAMPLES = [
    (
        "Create a generic synthetic dataset",
        (
            "Create a generic synthetic dataset with 5,000 rows "
            "(not e-commerce — use template=generic)."
        ),
        "Load generic tabular data · Dataset Agent",
    ),
    (
        "Explore the data",
        "Explore the data — show how columns relate, missing values, and typical ranges.",
        "Links between columns, blank values, and typical ranges · "
        "EDA · correlation · missingness · quantiles → Analysis · Dataset",
    ),
    (
        "Summarize by category",
        (
            "For each category, show the spread of values, "
            "how many unique ids, and each category's percent share of total value."
        ),
        "Spread, unique counts, and percent of total · "
        "Aggs · std · nunique · pct → Analysis Agent",
    ),
    (
        "Show box, heat, and dual charts",
        (
            "Create a box plot of values by category, a heatmap of how numeric "
            "columns relate, and a dual-axis chart of total value and row count by category."
        ),
        "Three chart styles for the same dataset · "
        "Charts · box · heatmap · dual_axis → Visualization Agent",
    ),
]

# One-shot for the toolkit demo (always creates a generic dataset first).
TOOLKIT_FULL_EXAMPLE = (
    "Run all of the above",
    (
        "Create a generic synthetic dataset with 5,000 rows "
        "(not e-commerce — use template=generic). "
        "Explore the data — show how columns relate, missing values, and typical ranges. "
        "For each category, show the spread of values, how many unique ids, "
        "and each category's percent share of total value. "
        "Create a box plot of values by category, a heatmap of how numeric columns relate, "
        "and a dual-axis chart of total value and row count by category."
    ),
    "Exploration, summaries, and new charts in one go · "
    "EDA + Aggs + Charts → Dataset · Analysis · Visualization",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #1a2420;
        border: 1px solid #2d3f38;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .mas-banner {
        background: linear-gradient(135deg, #0f1a16 0%, #1a2e28 55%, #243f36 100%);
        color: #e6eeea;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        border: 1px solid #2d4a42;
    }
    .mas-banner h1 {
        color: #e6eeea !important;
        font-size: 1.6rem;
        margin: 0 0 0.35rem 0;
        font-weight: 650;
    }
    .mas-banner p { margin: 0; opacity: 0.85; font-size: 0.95rem; }
    .agent-chip {
        display: inline-block;
        background: #243f36;
        color: #c5ddd4;
        border: 1px solid #3d6b5e;
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
    .status-idle { background: #2a3330; color: #b0bbb6; }
    .status-running { background: #3d3420; color: #f0d78c; }
    .status-ok { background: #1e3d2f; color: #8fd4a8; }
    .status-warn { background: #3d2226; color: #f0a0a8; }
    .msg-user, .msg-assistant {
        border-radius: 12px;
        padding: 0.85rem 1.05rem;
        margin: 0.55rem 0 0.85rem 0;
        color: #e6eeea;
    }
    .msg-user {
        background: #1e332c;
        border: 1px solid #3d6b5e;
        white-space: pre-wrap;
        line-height: 1.45;
    }
    .msg-assistant {
        background: #1a2420;
        border: 1px solid #2d3f38;
    }
    .msg-assistant h2, .msg-assistant h3, .msg-assistant h4 {
        color: #e6eeea !important;
        margin: 0.85rem 0 0.4rem 0;
        line-height: 1.3;
    }
    .msg-assistant h2 { font-size: 1.15rem; }
    .msg-assistant h3 { font-size: 1.02rem; }
    .msg-assistant h4 { font-size: 0.95rem; }
    .msg-assistant p { margin: 0.35rem 0; line-height: 1.5; }
    .msg-assistant ul { margin: 0.35rem 0 0.55rem 1.1rem; padding: 0; }
    .msg-assistant li { margin: 0.2rem 0; line-height: 1.45; }
    .msg-assistant table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.45rem 0 0.75rem 0;
        font-size: 0.86rem;
    }
    .msg-assistant th, .msg-assistant td {
        border: 1px solid #2d3f38;
        padding: 0.35rem 0.5rem;
        text-align: left;
    }
    .msg-assistant th {
        background: #243f36;
        color: #c5ddd4;
        font-weight: 600;
    }
    .msg-assistant code {
        background: #243f36;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.84em;
    }
    .msg-role {
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #8aa399;
        margin: 0.35rem 0 0.15rem 0;
        font-weight: 600;
    }
    .msg-greeting-wave {
        display: inline-flex;
        vertical-align: -0.2em;
        margin-right: 0.4rem;
        width: 1.15rem;
        height: 1.15rem;
        color: #8fd4a8;
        transform-origin: 70% 90%;
        animation: wave-hand 1.6s ease-in-out 3;
    }
    .msg-greeting-wave svg {
        width: 100%;
        height: 100%;
        display: block;
    }
    @keyframes wave-hand {
        0%   { transform: rotate(0deg); }
        12%  { transform: rotate(16deg); }
        24%  { transform: rotate(-10deg); }
        36%  { transform: rotate(16deg); }
        48%  { transform: rotate(-6deg); }
        60%  { transform: rotate(10deg); }
        72%, 100% { transform: rotate(0deg); }
    }
    .example-arrow {
        text-align: center;
        color: #8aa399;
        font-size: 0.95rem;
        line-height: 1;
        margin: -0.15rem 0 0.15rem 0;
        opacity: 0.85;
    }
    .example-or {
        text-align: center;
        color: #8aa399;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        margin: 0.65rem 0 0.55rem 0;
        text-transform: lowercase;
    }
    .tour-popup {
        position: relative;
        background: linear-gradient(145deg, #14201c 0%, #1f332c 100%);
        color: #e6eeea;
        border-radius: 12px;
        padding: 0.95rem 1.1rem 1rem 1.1rem;
        margin: 0.35rem 0 0.85rem 0;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
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
        border-top: 9px solid #1f332c;
    }
    .tour-popup .tour-step {
        display: inline-block;
        background: #3d6b5e;
        color: #e6eeea;
        border-radius: 999px;
        padding: 0.1rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .tour-popup h4 {
        color: #e6eeea !important;
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
    .site-footer {
        margin-top: 2.5rem;
        padding: 1.35rem 1rem 1.1rem 1rem;
        border-top: 1px solid #2d3f38;
        text-align: center;
        color: #8aa399;
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .site-footer a {
        color: #6bc4a6;
        text-decoration: underline;
        text-underline-offset: 2px;
    }
    .site-footer a:hover {
        color: #9ad9c4;
    }
    .demo-card {
        background: #141c19;
        border: 1px solid #2d3f38;
        border-radius: 10px;
        padding: 0.75rem 0.85rem 0.35rem 0.85rem;
        margin: 0.55rem 0 0.15rem 0;
    }
    .demo-card .demo-title {
        color: #e6eeea;
        font-weight: 650;
        font-size: 0.95rem;
        margin: 0 0 0.25rem 0;
        line-height: 1.3;
    }
    .demo-card .demo-plain {
        color: #a8bdb4;
        font-size: 0.8rem;
        margin: 0 0 0.45rem 0;
        line-height: 1.35;
    }
    .demo-card .demo-meta {
        color: #7a9a8c;
        font-size: 0.72rem;
        margin: 0 0 0.15rem 0;
        line-height: 1.35;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .demo-card .demo-agent {
        color: #6bc4a6;
        font-size: 0.72rem;
        margin: 0 0 0.35rem 0;
        line-height: 1.35;
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
        "pointer": "48%",
        "title": "Anomalies",
        "agent": "Anomaly Detection Agent",
        "body": (
            "Unusual rows from statistical/ML methods (IQR, Z-score, Isolation Forest). "
            "The agent chooses a method; the outlier labels come from the calculation, "
            "then are explained in plain language."
        ),
    },
    {
        "tab": "Forecasts",
        "pointer": "55%",
        "title": "Forecasts",
        "agent": "Forecasting Agent",
        "body": (
            "Quantitative forecasts from historical time series. The agent inspects the dataset, "
            "compares methods on a holdout window, and tools compute future values — "
            "the LLM only explains the grounded forecast artifact."
        ),
    },
    {
        "tab": "Charts",
        "pointer": "66%",
        "title": "Charts",
        "agent": "Visualization Agent",
        "body": (
            "Data-driven Plotly charts from a structured viz spec. The LLM picks chart type "
            "and columns; a deterministic renderer draws the figure from actual query/dataset values."
        ),
    },
    {
        "tab": "Drafts",
        "pointer": "78%",
        "title": "Drafts",
        "agent": "Drafting Agent",
        "body": (
            "Reports, summaries, memos, and creditworthiness assessments synthesized from "
            "validated specialist outputs. The Drafting Agent does not recalculate metrics — "
            "it organizes evidence, interpretation, recommendations, and limitations."
        ),
    },
    {
        "tab": "Agent log",
        "pointer": "90%",
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
        "conversation_context": ConversationContext(),
        "events": [],
        "messages": [],
        "busy": False,
        "tour_active": True,
        "tour_step": 0,
        "toolkit_demos": {},
    }


# Only app-owned keys — never delete Streamlit widget keys in the same click handler.
_APP_KEYS = (
    "workspace",
    "conversation_context",
    "events",
    "messages",
    "busy",
    "pending_prompt",
    "flash",
    "tour_active",
    "tour_step",
    "toolkit_demos",
)


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
    st.session_state.conversation_context = fresh["conversation_context"]
    st.session_state.events = fresh["events"]
    st.session_state.messages = fresh["messages"]
    st.session_state.busy = False
    st.session_state.toolkit_demos = fresh["toolkit_demos"]
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
              anomaly, forecasting, visualization, and validation agents. Numbers come from tools, not the LLM.</p>
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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Status", ws.task_status)
    c2.metric("Dataset rows", (ws.dataset or {}).get("row_count") or 0)
    c3.metric("Analyses", len(ws.analysis_results))
    c4.metric("Forecasts", len(ws.forecasts))
    c5.metric("Charts", len(ws.visualizations))
    c6.metric("Drafts", len(getattr(ws, "drafts", None) or []))


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
    eda = ds.get("eda") or {}
    if eda:
        st.markdown("#### Exploration (Dataset Agent)")
        miss = eda.get("missingness") or {}
        st.caption(
            f"Blank cells: **{miss.get('total_nulls', 0)}** "
            f"({(miss.get('overall_null_rate') or 0) * 100:.2f}%)"
        )
        pairs = (eda.get("correlation") or {}).get("pairs") or []
        if pairs:
            st.markdown("Top column links")
            st.dataframe(pairs[:8], use_container_width=True)


def render_analysis_panel(ws: SharedWorkspace) -> None:
    if not ws.analysis_results:
        st.info("No analysis results yet.")
        return
    for i, item in enumerate(ws.analysis_results, 1):
        result = item.get("result") or {}
        is_eda = (item.get("query_plan") or {}).get("mode") == "eda" or result.get("operation") == "eda"
        st.markdown(f"#### Analysis {i}" + (" — Exploration" if is_eda else ""))
        st.markdown(item.get("explanation") or "")

        if is_eda:
            pairs = (result.get("correlation") or {}).get("pairs") or result.get("records") or []
            st.markdown("**How columns relate**")
            if pairs:
                st.dataframe(pairs, use_container_width=True)
            miss = result.get("missingness") or {}
            st.markdown("**Missing values**")
            st.caption(
                f"Blank cells: **{miss.get('total_nulls', 0)}** "
                f"({(miss.get('overall_null_rate') or 0) * 100:.2f}% overall)"
            )
            if miss.get("columns"):
                st.dataframe(miss["columns"], use_container_width=True)
            qstats = (result.get("quantiles") or {}).get("stats") or {}
            st.markdown("**Typical ranges**")
            if qstats:
                rows = [{"column": col, **vals} for col, vals in qstats.items()]
                st.dataframe(rows, use_container_width=True)
        else:
            records = result.get("records") or []
            if records:
                st.dataframe(records, use_container_width=True)

        with st.expander("Query plan / tool output"):
            st.json(
                {
                    "query_plan": item.get("query_plan"),
                    "grounded": result.get("grounded"),
                    "sql": result.get("sql") or result.get("sql_equivalent"),
                    "operation": result.get("operation"),
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


def render_forecast_panel(ws: SharedWorkspace) -> None:
    if not ws.forecasts:
        st.info("No forecasts yet.")
        return
    for i, item in enumerate(ws.forecasts, 1):
        st.markdown(f"#### Forecast run {i}")
        if not item.get("suitable"):
            st.warning(item.get("explanation") or item.get("reason") or "Forecast not suitable.")
            reqs = item.get("requirements") or []
            if reqs:
                st.markdown("**What to provide**")
                for req in reqs:
                    st.markdown(f"- {req}")
            continue
        st.markdown(
            f"Target **`{item.get('target_column')}`** · "
            f"time `{item.get('time_column')}` · "
            f"frequency **{item.get('frequency')}** · "
            f"horizon **{item.get('forecast_horizon')}** · "
            f"method **`{item.get('selected_method')}`**"
        )
        st.markdown(item.get("explanation") or "")
        values = item.get("forecast_values") or []
        if values:
            st.dataframe(values, use_container_width=True)
        metrics = item.get("evaluation_metrics") or []
        if metrics:
            with st.expander("Holdout evaluation"):
                st.dataframe(metrics, use_container_width=True)
                st.caption(f"Baseline: {item.get('baseline_metrics')}")
        with st.expander("Assumptions / warnings"):
            st.json(
                {
                    "assumptions": item.get("assumptions"),
                    "warnings": item.get("warnings"),
                    "seasonality": item.get("seasonality"),
                    "trend": item.get("trend"),
                    "validation": item.get("validation"),
                }
            )


def render_charts_panel(ws: SharedWorkspace) -> None:
    if not ws.visualizations:
        st.info("No visualizations yet.")
        return
    # Newest charts first (workspace appends in creation order).
    for i, viz in enumerate(reversed(ws.visualizations)):
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


def render_drafts_panel(ws: SharedWorkspace) -> None:
    drafts = getattr(ws, "drafts", None) or []
    if not drafts:
        st.info("No drafts yet.")
        return
    from app.tools import export_tools

    for i, item in enumerate(drafts, 1):
        draft_id = str(item.get("id") or f"idx_{i}")
        title = item.get("title") or f"Draft {i}"
        st.markdown(f"#### {title} (`{item.get('type')}`)")
        st.caption(
            f"grounded={item.get('grounded')} · "
            f"evidence_sufficient={item.get('evidence_sufficient')} · "
            f"sources={len(item.get('source_artifacts') or [])}"
        )
        st.markdown(item.get("content") or "")
        with st.expander("Sections / assumptions / warnings"):
            st.json(
                {
                    "sections": item.get("sections"),
                    "assumptions": item.get("assumptions"),
                    "warnings": item.get("warnings"),
                    "limitations": item.get("limitations"),
                    "source_artifacts": item.get("source_artifacts"),
                    "metadata": item.get("metadata"),
                }
            )

        # Explicit export actions — do not auto-generate after every draft.
        st.markdown("**Export**")
        c_pdf, c_docx = st.columns(2)
        pdf_err_key = f"draft_export_pdf_err_{draft_id}"
        docx_err_key = f"draft_export_docx_err_{draft_id}"
        pdf_data_key = f"draft_export_pdf_data_{draft_id}"
        docx_data_key = f"draft_export_docx_data_{draft_id}"

        with c_pdf:
            if st.button("Export as PDF", key=f"btn_export_pdf_{draft_id}", use_container_width=True):
                try:
                    st.session_state[pdf_data_key] = export_tools.export_draft_pdf(item, ws)
                    st.session_state[pdf_err_key] = None
                except Exception as exc:  # noqa: BLE001
                    st.session_state[pdf_data_key] = None
                    st.session_state[pdf_err_key] = str(exc)
            if st.session_state.get(pdf_err_key):
                st.error(f"PDF export failed: {st.session_state[pdf_err_key]}")
            elif st.session_state.get(pdf_data_key):
                st.download_button(
                    "Download PDF",
                    data=st.session_state[pdf_data_key],
                    file_name=export_tools.safe_export_filename(str(title), "pdf"),
                    mime="application/pdf",
                    key=f"dl_pdf_{draft_id}",
                    use_container_width=True,
                )

        with c_docx:
            if st.button("Export as DOCX", key=f"btn_export_docx_{draft_id}", use_container_width=True):
                try:
                    st.session_state[docx_data_key] = export_tools.export_draft_docx(item, ws)
                    st.session_state[docx_err_key] = None
                except Exception as exc:  # noqa: BLE001
                    st.session_state[docx_data_key] = None
                    st.session_state[docx_err_key] = str(exc)
            if st.session_state.get(docx_err_key):
                st.error(f"DOCX export failed: {st.session_state[docx_err_key]}")
            elif st.session_state.get(docx_data_key):
                st.download_button(
                    "Download DOCX",
                    data=st.session_state[docx_data_key],
                    file_name=export_tools.safe_export_filename(str(title), "docx"),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_docx_{draft_id}",
                    use_container_width=True,
                )


def render_history_panel(ws: SharedWorkspace) -> None:
    if not ws.agent_history:
        st.info("No agent history yet.")
        return
    st.json(ws.agent_history)


def render_toolkit_panel(demos: dict, ws: SharedWorkspace | None = None) -> None:
    """Show toolkit outputs from agent runs (and optional cached demos)."""
    eda_block = (demos or {}).get("eda")
    aggs_block = (demos or {}).get("aggs")
    charts_block = (demos or {}).get("charts")

    if ws is not None:
        if not eda_block and (ws.dataset or {}).get("eda"):
            eda_block = (ws.dataset or {}).get("eda")
        if not eda_block:
            for item in ws.analysis_results:
                result = item.get("result") or {}
                if result.get("operation") == "eda" or (item.get("query_plan") or {}).get("mode") == "eda":
                    eda_block = {
                        "correlation": result.get("correlation"),
                        "missingness": result.get("missingness"),
                        "quantiles": result.get("quantiles"),
                    }
                    break
        if not aggs_block:
            found: dict = {}
            for item in ws.analysis_results:
                result = item.get("result") or {}
                agg = result.get("agg")
                if agg in {"std", "nunique", "pct"} and agg not in found:
                    found[agg] = result
            if found:
                aggs_block = found
        if not charts_block:
            charts_block = [
                v
                for v in ws.visualizations
                if (v.get("chart_type") or "") in {"box", "heatmap", "dual_axis", "boxplot"}
            ]

    if not eda_block and not aggs_block and not charts_block:
        st.info(
            "Nothing here yet. In the sidebar under **Try another example**, start with "
            "step 1 (generic synthetic dataset), then Explore / Summarize / Charts — "
            "or use **Run all of the above**."
        )
        return

    if eda_block:
        eda = eda_block
        st.markdown("#### How columns relate to each other")
        st.caption(
            "Closer to 1 or −1 means a stronger link between two numeric columns."
        )
        corr = eda.get("correlation") if isinstance(eda.get("correlation"), dict) else eda
        pairs = (corr or {}).get("pairs") or []
        if pairs:
            st.dataframe(pairs, use_container_width=True)
        with st.expander("Full relationship table"):
            st.json((eda.get("correlation") or {}).get("matrix") or eda.get("matrix") or {})

        st.markdown("#### Missing or blank values")
        miss = eda.get("missingness") or {}
        st.caption(
            f"Share of blank cells: **{(miss.get('overall_null_rate') or 0) * 100:.2f}%** · "
            f"blank cells total: **{miss.get('total_nulls', '—')}**"
        )
        if miss.get("columns"):
            st.dataframe(miss.get("columns") or [], use_container_width=True)

        st.markdown("#### Typical ranges (low → mid → high)")
        st.caption("Shows where most values sit — from the smallest, through the middle, to the largest.")
        qstats = (eda.get("quantiles") or {}).get("stats") or {}
        rows = [{"column": col, **vals} for col, vals in qstats.items()]
        if rows:
            st.dataframe(rows, use_container_width=True)

    if aggs_block:
        aggs = aggs_block
        st.markdown("#### Summaries by product category")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**How spread out** order totals are")
            st.dataframe((aggs.get("std") or {}).get("records") or [], use_container_width=True)
        with c2:
            st.markdown("**How many different** customers")
            st.dataframe((aggs.get("nunique") or {}).get("records") or [], use_container_width=True)
        with c3:
            st.markdown("**Share of revenue** (%)")
            st.dataframe((aggs.get("pct") or {}).get("records") or [], use_container_width=True)

    if charts_block:
        st.markdown("#### New chart styles")
        st.caption("These also appear under the **Charts** tab.")
        for i, viz in enumerate(charts_block):
            st.markdown(f"**{viz.get('title')}** (`{viz.get('chart_type')}`)")
            fig_json = viz.get("plotly_json")
            if fig_json:
                payload = json.dumps(fig_json) if isinstance(fig_json, dict) else fig_json
                fig = pio.from_json(payload)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"toolkit_chart_{viz.get('id')}_{i}",
                )


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


def render_footer() -> None:
    st.markdown(
        """
        <footer class="site-footer">
          <div>Email: <a href="mailto:hello@wesamtechnologies.com">hello@wesamtechnologies.com</a></div>
          <div>© Copyright 2026</div>
          <div>Wesam Technologies LLC</div>
          <div><a href="https://wesamtechnologies.com/privacy" target="_blank" rel="noopener noreferrer">Privacy Policy</a></div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_conversation(messages: list[dict]) -> None:
    if not messages:
        st.markdown('<div class="msg-role">Agents</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="msg-assistant">'
            '<p><span class="msg-greeting-wave" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M9 11.5V5.5a1.5 1.5 0 0 1 3 0V11"/>'
            '<path d="M12 10.5V4.5a1.5 1.5 0 0 1 3 0V11"/>'
            '<path d="M15 10.5V6.5a1.5 1.5 0 0 1 3 0v7.5a6 6 0 0 1-6 6h-1.5'
            'a6.5 6.5 0 0 1-6.5-6.5V11a1.5 1.5 0 0 1 3 0v1.5"/>'
            '<path d="M6 12.5V11a1.5 1.5 0 0 1 3 0v2"/>'
            "</svg></span>"
            "Hi — I'm your data analysis team. Ask me anything in natural language, "
            "or pick an example from the sidebar to get started.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    for msg in messages:
        role = msg.get("role", "assistant")
        content = (msg.get("content") or "").strip()
        if role == "user":
            st.markdown('<div class="msg-role">You</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="msg-user">{_html_escape(content)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="msg-role">Agents</div>', unsafe_allow_html=True)
            # Render markdown inside the bubble (headings, lists, tables)
            st.markdown(
                f'<div class="msg-assistant">{_markdown_to_safe_html(content)}</div>',
                unsafe_allow_html=True,
            )


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _markdown_to_safe_html(text: str) -> str:
    """Lightweight markdown → HTML for chat bubbles (no raw HTML from the model)."""
    import html
    import re

    # Escape first so agent text cannot inject tags
    t = html.escape(text or "")

    # Fenced code blocks
    def _code_block(m: re.Match[str]) -> str:
        return f"<pre><code>{m.group(1).strip()}</code></pre>"

    t = re.sub(r"```[\w]*\n([\s\S]*?)```", _code_block, t)

    # Tables: consecutive lines starting with |
    def _convert_tables(block: str) -> str:
        lines = block.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|?\s*:?-{3,}", lines[i + 1]
            ):
                header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                i += 2
                rows: list[list[str]] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                html_rows = "".join(
                    "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
                )
                html_rows += "".join(
                    "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
                )
                out.append(f"<table>{html_rows}</table>")
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    t = _convert_tables(t)

    # Headings (longest first)
    t = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", t, flags=re.MULTILINE)
    t = re.sub(r"^### (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = re.sub(r"^## (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = re.sub(r"^# (.+)$", r"<h2>\1</h2>", t, flags=re.MULTILINE)

    # Bold / italic / inline code
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"^---$", r"<hr>", t, flags=re.MULTILINE)

    # Unordered lists
    def _lists(block: str) -> str:
        lines = block.split("\n")
        out: list[str] = []
        in_list = False
        for line in lines:
            m = re.match(r"^[-*] (.+)$", line)
            if m:
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{m.group(1)}</li>")
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(line)
        if in_list:
            out.append("</ul>")
        return "\n".join(out)

    t = _lists(t)

    # Turn remaining text lines into paragraphs; keep block elements intact
    block_prefixes = ("<h2", "<h3", "<h4", "<ul", "</ul", "<table", "<pre", "<hr", "<li")
    rendered: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        body = "<br>".join(buf)
        rendered.append(f"<p>{body}</p>")
        buf.clear()

    for line in t.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if s.startswith(block_prefixes):
            flush()
            rendered.append(s)
        else:
            buf.append(s)
    flush()
    return "\n".join(rendered)


def build_llm_from_session() -> LLMClient:
    """Create an LLM client from sidebar/session provider + model choices."""
    provider = resolve_provider(st.session_state.get("llm_provider") or DEFAULT_PROVIDER)
    model = st.session_state.get("llm_model") or latest_model_for(provider)
    return LLMClient(provider=provider, model=model)


def run_request(prompt: str) -> None:
    ws: SharedWorkspace = st.session_state.workspace
    ctx: ConversationContext = st.session_state.conversation_context
    messages = list(st.session_state.messages)
    # Recent window excludes the brand-new user turn; orchestrator receives it as user_request.
    prior_messages = list(messages)
    messages.append({"role": "user", "content": prompt})
    events: list[dict] = []

    def on_progress(event: dict) -> None:
        events.append(event)

    with st.spinner("Orchestrator coordinating agents…"):
        orch = Orchestrator(llm=build_llm_from_session())
        ws = orch.run(
            prompt,
            workspace=ws,
            progress_callback=on_progress,
            conversation_context=ctx,
            recent_messages=prior_messages,
        )

    messages.append({"role": "assistant", "content": ws.final_response or "Completed."})
    st.session_state.workspace = ws
    st.session_state.conversation_context = ctx
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
        st.divider()
        st.markdown("### AI model")
        provider_ids = list(PROVIDERS.keys())
        provider_labels = {pid: PROVIDERS[pid]["label"] for pid in provider_ids}
        if "llm_provider" not in st.session_state:
            st.session_state.llm_provider = resolve_provider()
        provider = st.selectbox(
            "Provider",
            options=provider_ids,
            format_func=lambda pid: provider_labels[pid],
            key="llm_provider",
        )
        prev_provider = st.session_state.get("_llm_provider_loaded")
        refresh_col, _ = st.columns([1, 2])
        with refresh_col:
            if st.button("Refresh models", use_container_width=True, key="llm_refresh_models"):
                clear_model_cache(provider)
                st.session_state.pop("llm_model", None)
                st.session_state.pop("_llm_model_options", None)

        # Reload when provider changes or cached options are missing/stale.
        cached_options = st.session_state.get("_llm_model_options")
        need_reload = prev_provider != provider or not cached_options
        if need_reload:
            clear_model_cache(provider)
            with st.spinner("Loading latest models…"):
                # Hard-cap in the UI so the dropdown never exceeds 3 chat models.
                model_options = list_models(provider, force_refresh=True)[:MAX_MODEL_CHOICES]
            st.session_state._llm_model_options = model_options
            st.session_state._llm_provider_loaded = provider
            if "llm_model" in st.session_state and st.session_state.llm_model not in model_options:
                st.session_state.pop("llm_model", None)
        else:
            model_options = list(cached_options)[:MAX_MODEL_CHOICES]

        latest = model_options[0] if model_options else latest_model_for(provider)
        if "llm_model" not in st.session_state or st.session_state.llm_model not in model_options:
            st.session_state.llm_model = latest
        st.selectbox(
            "Model",
            options=model_options,
            key="llm_model",
            help="Latest 3 chat models from the provider API (newest first). Default is the newest.",
        )
        llm = build_llm_from_session()
        st.caption(llm.status_label)
        if model_options:
            st.caption(f"Latest: `{latest}` · {len(model_options)} chat models")
        env_key = PROVIDERS[provider]["env_key"]
        if not llm.available:
            st.error(
                f"Live mode requires `{env_key}` in `.env`. "
                "Without it, requests will fail instead of inventing data."
            )
        st.link_button("Reset workspace", url="?reset=1", use_container_width=True, type="primary")

        st.divider()
        st.markdown("### Load CSV")
        st.caption("Upload real data before analysis. Synthetic datasets are created only when you ask.")
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
            "(step 1) so analysis, anomalies, forecasts, charts, and drafts have "
            "data to work with — or use the full workflow instead."
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

        st.divider()
        st.markdown("### Try another example")
        st.caption(
            "Same idea as above: create or load a dataset first (step 1), then run "
            "Explore, category summaries, and the new chart types — or use the full "
            "toolkit workflow instead."
        )
        for i, (label, prompt, meta) in enumerate(TOOLKIT_STEP_EXAMPLES, start=1):
            if st.button(f"{i}. {label}", use_container_width=True, key=f"toolkit_step_{i}"):
                pending_from_example = prompt
            st.caption(meta)
            if i < len(TOOLKIT_STEP_EXAMPLES):
                st.markdown('<div class="example-arrow">↓</div>', unsafe_allow_html=True)

        st.markdown('<div class="example-or">———— or ————</div>', unsafe_allow_html=True)
        full_label, full_prompt, full_meta = TOOLKIT_FULL_EXAMPLE
        if st.button(full_label, use_container_width=True, key="toolkit_full_workflow"):
            pending_from_example = full_prompt
        st.caption(full_meta)

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

        tab_labels = [s["tab"] for s in TOUR_STEPS] + ["Toolkit"]
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
                "Forecasts",
                TOUR_STEPS[4]["agent"],
                TOUR_STEPS[4]["body"],
            )
            render_forecast_panel(ws)
        with tabs[5]:
            section_help(
                "Charts",
                TOUR_STEPS[5]["agent"],
                TOUR_STEPS[5]["body"],
            )
            render_charts_panel(ws)
        with tabs[6]:
            section_help(
                "Drafts",
                TOUR_STEPS[6]["agent"],
                TOUR_STEPS[6]["body"],
            )
            render_drafts_panel(ws)
        with tabs[7]:
            section_help(
                "Agent log",
                TOUR_STEPS[7]["agent"],
                TOUR_STEPS[7]["body"],
            )
            render_history_panel(ws)
        with tabs[8]:
            section_help(
                "Toolkit",
                "Dataset · Analysis · Forecasting · Visualization · Drafting agents",
                (
                    "Results from the newer tools wired into agents: explore how columns relate, "
                    "find blank values and value ranges, summarize by category "
                    "(spread, unique counts, percent share), forecasts, drafts/reports, "
                    "and box / heat / dual charts."
                ),
            )
            render_toolkit_panel(dict(st.session_state.get("toolkit_demos") or {}), ws)

    render_footer()


if __name__ == "__main__":
    main()
