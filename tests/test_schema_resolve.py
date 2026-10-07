"""NL → schema column resolution (dataset-agnostic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools.schema_resolve import resolve_metric_columns, rank_columns_for_request


def test_flow_metric_prefers_aggregate_over_unit_price():
    cols = ["order_date", "unit_price", "quantity", "line_total", "customer_id"]
    ranked = resolve_metric_columns("Forecast revenue for the next 12 months.", cols)
    assert ranked
    assert ranked[0] == "line_total"
    assert ranked[0] != "unit_price"


def test_explicit_schema_column_wins():
    cols = ["Year", "company_revenue", "unit_price", "EBITDA"]
    ranked = resolve_metric_columns("Forecast company_revenue for the next 3 years.", cols)
    assert ranked[0] == "company_revenue"


def test_price_request_can_select_unit_price():
    cols = ["date", "unit_price", "total_amount", "quantity"]
    ranked = resolve_metric_columns("Predict unit price over the next 6 months", cols)
    assert ranked
    assert ranked[0] == "unit_price"


def test_arbitrary_schema_synonym_mapping():
    cols = ["ts", "widget_throughput", "scrap_rate", "energy_cost"]
    ranked = resolve_metric_columns("Forecast volume for the next 8 weeks", cols)
    assert ranked
    assert ranked[0] == "widget_throughput"


def test_rank_scores_are_ordered():
    cols = ["a_price", "b_total_amount", "c_id"]
    ranked = rank_columns_for_request("project sales next quarter", cols)
    assert ranked
    assert ranked[0][0] == "b_total_amount"
    # Unit-price columns are outranked / filtered for flow metrics like sales
    assert all(c != "a_price" for c, _ in ranked)
