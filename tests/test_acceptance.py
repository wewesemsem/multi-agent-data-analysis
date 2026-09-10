"""Acceptance and unit tests for the multi-agent MVP.

Tools are tested with real computation. Orchestrator acceptance runs offline
(heuristic planning) so CI does not require an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator
from app.state import SharedWorkspace
from app.tools import anomaly_tools, chart_tools, dataset_tools, query_tools


def test_dataset_generation_not_llm_rows():
    meta = dataset_tools.create_ecommerce_orders(n_rows=1000, seed=7)
    assert meta["row_count"] == 1000
    df = dataset_tools.load_dataset(meta)
    assert len(df) == 1000
    assert "total_amount" in df.columns
    assert "product_category" in df.columns


def test_query_aggregation_grounded():
    meta = dataset_tools.create_ecommerce_orders(n_rows=500, seed=1)
    result = query_tools.execute_aggregation(
        meta,
        group_by="product_category",
        metric_column="total_amount",
        agg="sum",
    )
    assert result["grounded"] is True
    assert len(result["records"]) >= 1
    # Cross-check with pandas
    df = dataset_tools.load_dataset(meta)
    expected = df.groupby("product_category")["total_amount"].sum().sort_values(ascending=False)
    got = {r["product_category"]: r["value"] for r in result["records"]}
    for cat, val in expected.items():
        assert abs(got[cat] - float(val)) < 1e-6


def test_anomaly_iqr_grounded():
    meta = dataset_tools.create_ecommerce_orders(n_rows=2000, seed=2)
    result = anomaly_tools.detect_anomalies(meta, column="total_amount", method="iqr")
    assert result["grounded"] is True
    assert result["method"] == "iqr"
    assert result["n_anomalies"] > 0


def test_chart_render_from_aggregation():
    meta = dataset_tools.create_ecommerce_orders(n_rows=300, seed=3)
    chart = chart_tools.render_chart(
        chart_type="bar",
        title="Revenue by Category",
        dataset_meta=meta,
        aggregation={"group_by": "product_category", "metric_column": "total_amount", "agg": "sum"},
    )
    assert chart["grounded"] is True
    assert Path(chart["html_path"]).exists()
    assert chart["n_points"] > 0


def test_acceptance_end_to_end():
    """MVP acceptance test from the product brief."""
    request = (
        "Create a synthetic e-commerce dataset with 10,000 orders. "
        "Tell me which product categories generate the most revenue, "
        "identify anomalous transactions, and create visualizations showing "
        "revenue by category and the distribution of transaction amounts."
    )
    events = []
    orch = Orchestrator()
    ws = orch.run(request, workspace=SharedWorkspace(), progress_callback=events.append)

    assert ws.dataset is not None
    assert ws.dataset["row_count"] == 10_000
    assert len(ws.analysis_results) >= 1
    assert ws.analysis_results[0]["result"]["grounded"] is True
    assert len(ws.anomalies) >= 1
    assert ws.anomalies[0]["grounded"] is True
    assert ws.anomalies[0]["n_anomalies"] > 0
    assert len(ws.visualizations) >= 2
    assert all(v.get("grounded") for v in ws.visualizations)
    assert ws.final_response
    assert ws.task_status in {"completed", "completed_with_warnings"}

    agents_used = {h["agent"] for h in ws.agent_history}
    assert "orchestrator" in agents_used
    assert "dataset_agent" in agents_used
    assert "analysis_agent" in agents_used
    assert "anomaly_agent" in agents_used
    assert "visualization_agent" in agents_used
    assert "validation_agent" in agents_used

    # Analysis numbers must match tool recompute
    records = ws.analysis_results[0]["result"]["records"]
    assert records
    df = dataset_tools.load_dataset(ws.dataset)
    expected = df.groupby("product_category")["total_amount"].sum().to_dict()
    for row in records:
        cat = row.get("product_category")
        if cat in expected:
            assert abs(float(row["value"]) - float(expected[cat])) < 1e-4
