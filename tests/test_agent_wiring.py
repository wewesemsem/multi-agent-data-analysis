"""Agent wiring tests for Tier-0 tools connected to specialists."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator
from app.state import SharedWorkspace
from app.tools import dataset_tools


def test_explore_routes_to_analysis_and_dataset_agents():
    orch = Orchestrator()
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=400, seed=21)
    prompt = "Explore the data — show how columns relate, missing values, and typical ranges."
    # CI enables MAS_ALLOW_OFFLINE_HEURISTICS → heuristic plan still runs explore agents
    ws = orch.run(prompt, workspace=ws)
    agents = {h["agent"] for h in ws.agent_history if h.get("success")}
    assert "analysis_agent" in agents
    assert "dataset_agent" in agents
    assert (ws.dataset or {}).get("eda")
    assert any(
        (a.get("query_plan") or {}).get("mode") == "eda" or (a.get("result") or {}).get("operation") == "eda"
        for a in ws.analysis_results
    )
    # Final answer must surface the three asked facets
    final = (ws.final_response or "").lower()
    assert "relate" in final or "correlation" in final or "vs" in final
    assert "missing" in final or "blank" in final
    assert "range" in final or "median" in final or "p25" in final or "percentile" in final
    assert ws.task_status in {"completed", "completed_with_warnings"}


def test_multi_agg_std_nunique_pct_via_analysis_agent():
    orch = Orchestrator()
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=500, seed=22)
    ws = orch.run(
        "For each product category, show the spread of order totals, "
        "how many unique customers, and each category's percent share of revenue.",
        workspace=ws,
    )
    aggs = {
        (a.get("result") or {}).get("agg")
        for a in ws.analysis_results
        if (a.get("result") or {}).get("agg")
    }
    assert {"std", "nunique", "pct"}.issubset(aggs)
    assert ws.task_status in {"completed", "completed_with_warnings"}


def test_new_chart_types_via_visualization_agent():
    orch = Orchestrator()
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=400, seed=23)
    # Call visualization agent directly so the test is not flaky on LLM network
    from app.agents.visualization_agent import VisualizationAgent
    from app.messages import AgentMessage

    agent = VisualizationAgent()
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={
                "user_request": (
                    "Create a box plot of order amounts by category, a heatmap of how numeric "
                    "columns relate, and a dual-axis chart of revenue and order count by category."
                ),
                "force_heuristic": True,
            },
        ),
        ws,
    )
    assert result.success, result.error
    types = {c.get("chart_type") for c in result.data.get("charts", [])}
    assert "box" in types
    assert "heatmap" in types
    assert "dual_axis" in types
    assert all(c.get("grounded") for c in result.data.get("charts", []))


def test_heatmap_ignores_bad_data_records_and_dedupes_llm_specs():
    """Regression: LLM heatmap+data_records used to abort the whole viz step;
    duplicate specs used to stack into 9 charts (3×3)."""
    from app.agents.visualization_agent import VisualizationAgent
    from app.messages import AgentMessage

    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=300, seed=24)
    agent = VisualizationAgent()

    # Simulate a bad LLM plan: heatmap with one-metric records, plus duplicates
    bad_specs = [
        {
            "chart_type": "box",
            "title": "Box",
            "x": "product_category",
            "y": "total_amount",
            "use_raw_dataset": True,
        },
        {
            "chart_type": "heatmap",
            "title": "Bad heatmap",
            "data_records": [{"product_category": "A", "value": 10.0}],
            "x": "product_category",
            "y": "value",
        },
        {
            "chart_type": "dual_axis",
            "title": "Dual",
            "x": "product_category",
            "y": "total_amount",
        },
        # duplicates that previously became a 3×3 grid
        {"chart_type": "box", "x": "product_category", "y": "total_amount", "use_raw_dataset": True},
        {"chart_type": "heatmap", "data_records": [{"a": 1}]},
        {"chart_type": "dual_axis", "x": "product_category", "y": "total_amount"},
        {"chart_type": "box", "x": "product_category", "y": "total_amount", "use_raw_dataset": True},
        {"chart_type": "heatmap"},
        {"chart_type": "dual_axis", "x": "product_category", "y": "total_amount"},
    ]
    request = (
        "Create a box plot of order amounts by category, a heatmap of how numeric "
        "columns relate, and a dual-axis chart of revenue and order count by category."
    )
    # Explicit specs are normalized/deduped (heatmap recovers via raw dataset)
    result = agent.handle(
        AgentMessage(
            task_id="t2",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={"user_request": request, "specs": bad_specs},
        ),
        ws,
    )
    assert result.success, result.error
    charts = result.data.get("charts", [])
    assert len(charts) == 3, [c.get("chart_type") for c in charts]
    types = {c.get("chart_type") for c in charts}
    assert types == {"box", "heatmap", "dual_axis"}

    # Non-toolkit request with explicit bad heatmap specs still recovers via normalize
    ws2 = SharedWorkspace()
    ws2.dataset = dataset_tools.create_ecommerce_orders(n_rows=200, seed=25)
    result2 = agent.handle(
        AgentMessage(
            task_id="t3",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={
                "user_request": "show a correlation heatmap",
                "specs": [
                    {
                        "chart_type": "heatmap",
                        "data_records": [{"product_category": "A", "value": 1.0}],
                        "x": "product_category",
                        "y": "value",
                    }
                ],
            },
        ),
        ws2,
    )
    assert result2.success, result2.error
    assert result2.data["charts"][0]["chart_type"] == "heatmap"


def test_explore_without_dataset_is_not_llm_unavailable():
    """Explore with an empty workspace must fail closed clearly — not as 'LLM unavailable'."""
    orch = Orchestrator()
    ws = SharedWorkspace()
    ws = orch.run(
        "Explore the data — show how columns relate, missing values, and typical ranges.",
        workspace=ws,
    )
    assert ws.task_status == "failed"
    final = (ws.final_response or "").lower()
    assert "llm unavailable" not in final
    assert "dataset" in final
    assert ws.dataset is None


def test_explore_falls_back_when_llm_returns_empty_steps():
    """Live LLM can return [] for Explore; capability intents must still run agents."""

    class EmptyPlanLLM:
        def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
            return {"summary": "no steps", "steps": []}

        def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
            return ""

    orch = Orchestrator(llm=EmptyPlanLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=300, seed=26)
    ws = orch.run(
        "Explore the data — show how columns relate, missing values, and typical ranges.",
        workspace=ws,
    )
    agents = {h["agent"] for h in ws.agent_history if h.get("success")}
    assert "analysis_agent" in agents
    assert "dataset_agent" in agents
    assert ws.task_status in {"completed", "completed_with_warnings"}
    assert "llm unavailable" not in (ws.final_response or "").lower()
