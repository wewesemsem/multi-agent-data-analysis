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
    assert orch._prefer_heuristic_plan(prompt) is True
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
    ws = orch.run(
        "Create a box plot of order amounts by category, a heatmap of how numeric columns relate, "
        "and a dual-axis chart of revenue and order count by category.",
        workspace=ws,
    )
    types = {v.get("chart_type") for v in ws.visualizations}
    assert "box" in types
    assert "heatmap" in types
    assert "dual_axis" in types
    assert all(v.get("grounded") for v in ws.visualizations)
    assert ws.task_status in {"completed", "completed_with_warnings"}
