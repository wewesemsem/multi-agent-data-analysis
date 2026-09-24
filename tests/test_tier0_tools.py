"""Unit tests for Tier-0 additive tools (EDA, aggs, chart types).

These do not change orchestrator routing; acceptance path stays untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools import chart_tools, dataset_tools, eda_tools, query_tools


def test_correlation_matrix_grounded():
    meta = dataset_tools.create_ecommerce_orders(n_rows=400, seed=11)
    result = eda_tools.correlation_matrix(
        meta,
        columns=["quantity", "unit_price", "total_amount"],
    )
    assert result["grounded"] is True
    assert result["method"] == "pearson"
    assert set(result["columns"]) == {"quantity", "unit_price", "total_amount"}
    assert abs(result["matrix"]["total_amount"]["total_amount"] - 1.0) < 1e-9
    assert len(result["pairs"]) == 3
    assert all("correlation" in p for p in result["pairs"])


def test_missingness_summary_grounded():
    meta = dataset_tools.create_ecommerce_orders(n_rows=200, seed=12)
    result = eda_tools.missingness_summary(meta)
    assert result["grounded"] is True
    assert result["row_count"] == 200
    assert result["column_count"] >= 1
    assert len(result["columns"]) == result["column_count"]
    assert result["overall_null_rate"] is not None
    # Synthetic generator should be complete
    assert result["total_nulls"] == 0


def test_quantile_stats_grounded():
    meta = dataset_tools.create_ecommerce_orders(n_rows=500, seed=13)
    result = eda_tools.quantile_stats(meta, columns=["total_amount"])
    assert result["grounded"] is True
    stats = result["stats"]["total_amount"]
    assert stats["count"] == 500
    assert stats["p0"] <= stats["p25"] <= stats["p50"] <= stats["p75"] <= stats["p100"]
    df = dataset_tools.load_dataset(meta)
    assert abs(stats["p50"] - float(df["total_amount"].median())) < 1e-6


def test_aggregation_std_nunique_pct():
    meta = dataset_tools.create_ecommerce_orders(n_rows=600, seed=14)
    df = dataset_tools.load_dataset(meta)

    std_res = query_tools.execute_aggregation(
        meta, group_by="product_category", metric_column="total_amount", agg="std"
    )
    assert std_res["grounded"] is True
    assert std_res["agg"] == "std"
    expected_std = df.groupby("product_category")["total_amount"].std()
    got_std = {r["product_category"]: r["value"] for r in std_res["records"]}
    for cat, val in expected_std.items():
        assert abs(got_std[cat] - float(val)) < 1e-6

    nunique_res = query_tools.execute_aggregation(
        meta, group_by="product_category", metric_column="customer_id", agg="nunique"
    )
    expected_n = df.groupby("product_category")["customer_id"].nunique()
    got_n = {r["product_category"]: r["value"] for r in nunique_res["records"]}
    for cat, val in expected_n.items():
        assert got_n[cat] == float(val)

    pct_res = query_tools.execute_aggregation(
        meta, group_by="product_category", metric_column="total_amount", agg="pct"
    )
    total = float(df["total_amount"].sum())
    got_pct = {r["product_category"]: r["value"] for r in pct_res["records"]}
    assert abs(sum(got_pct.values()) - 100.0) < 1e-4
    for cat, share in (df.groupby("product_category")["total_amount"].sum() / total * 100).items():
        assert abs(got_pct[cat] - float(share)) < 1e-4


def test_chart_box_heatmap_dual_axis():
    meta = dataset_tools.create_ecommerce_orders(n_rows=350, seed=15)

    box = chart_tools.render_chart(
        chart_type="box",
        title="Amount by Category",
        dataset_meta=meta,
        x="product_category",
        y="total_amount",
    )
    assert box["grounded"] is True
    assert box["chart_type"] == "box"
    assert Path(box["html_path"]).exists()

    heat = chart_tools.render_chart(
        chart_type="heatmap",
        title="Numeric Correlations",
        dataset_meta=meta,
    )
    assert heat["grounded"] is True
    assert heat["chart_type"] == "heatmap"
    assert heat["n_points"] > 0
    assert Path(heat["html_path"]).exists()

    dual = chart_tools.render_chart(
        chart_type="dual_axis",
        title="Revenue vs Orders",
        data_records=[
            {"category": "A", "revenue": 100.0, "orders": 10},
            {"category": "B", "revenue": 200.0, "orders": 25},
            {"category": "C", "revenue": 150.0, "orders": 18},
        ],
        x="category",
        y="revenue",
        y2="orders",
    )
    assert dual["grounded"] is True
    assert dual["chart_type"] == "dual_axis"
    assert dual["y2"] == "orders"
    assert Path(dual["html_path"]).exists()


def test_choose_chart_type_new_intents_preserve_old():
    assert chart_tools.choose_chart_type("show a box plot of amounts") == "box"
    assert chart_tools.choose_chart_type("correlation heatmap") == "heatmap"
    assert chart_tools.choose_chart_type("dual-axis revenue and count") == "dual_axis"
    # Existing behaviors unchanged
    assert chart_tools.choose_chart_type("distribution of amounts") == "histogram"
    assert chart_tools.choose_chart_type("correlation between price and qty") == "scatter"
    assert chart_tools.choose_chart_type("revenue by category") == "bar"
