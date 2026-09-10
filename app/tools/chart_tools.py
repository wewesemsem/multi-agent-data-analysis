"""Deterministic visualization rendering from structured specs + real data."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from app.state import CHARTS_DIR, ensure_workspace
from app.tools.dataset_tools import load_dataset
from app.tools.query_tools import _jsonify


def render_chart(
    *,
    chart_type: str,
    title: str,
    data_records: list[dict[str, Any]] | None = None,
    dataset_meta: dict[str, Any] | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    aggregation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a Plotly chart from either explicit records or dataset + aggregation."""
    ensure_workspace()
    chart_type = (chart_type or "bar").lower()

    if data_records is not None:
        df = pd.DataFrame(data_records)
    elif dataset_meta is not None and aggregation:
        df = _aggregate(dataset_meta, aggregation)
        x = x or aggregation.get("group_by")
        y = y or "value"
    elif dataset_meta is not None:
        df = load_dataset(dataset_meta)
    else:
        raise ValueError("Chart rendering requires data_records or dataset_meta.")

    if df.empty:
        raise ValueError("No data available to plot.")

    # Infer columns if missing
    if x is None:
        x = df.columns[0]
    if y is None and len(df.columns) > 1:
        numeric = df.select_dtypes(include="number").columns.tolist()
        y = numeric[0] if numeric else df.columns[1]

    fig = _build_figure(chart_type, df, x=x, y=y, color=color, title=title)

    chart_id = f"chart_{uuid.uuid4().hex[:10]}"
    html_path = CHARTS_DIR / f"{chart_id}.html"
    json_path = CHARTS_DIR / f"{chart_id}.json"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    fig_json = json.loads(pio.to_json(fig))
    json_path.write_text(json.dumps(fig_json))

    # Compact data preview for grounding / UI
    preview = [{k: _jsonify(v) for k, v in row.items()} for row in df.head(50).to_dict(orient="records")]

    return {
        "id": chart_id,
        "chart_type": chart_type,
        "title": title,
        "x": x,
        "y": y,
        "color": color,
        "html_path": str(html_path),
        "json_path": str(json_path),
        "n_points": int(len(df)),
        "data_preview": preview,
        "grounded": True,
        "source_dataset_id": (dataset_meta or {}).get("id"),
        "plotly_json": fig_json,
    }


def _aggregate(meta: dict[str, Any], aggregation: dict[str, Any]) -> pd.DataFrame:
    from app.tools.query_tools import execute_aggregation

    metric = aggregation.get("metric_column")
    if not metric or metric in {"None", "null"}:
        schema = meta.get("schema") or {}
        cols = list(schema.keys())
        metric = next(
            (
                c
                for c in cols
                if any(k in c.lower() for k in ("total_amount", "amount", "revenue", "value", "price"))
            ),
            None,
        )
        if not metric:
            raise ValueError("Aggregation requires a metric_column present in the dataset.")
        aggregation = {**aggregation, "metric_column": metric}

    group_by = aggregation.get("group_by")
    if group_by in {None, "None", "null"}:
        group_by = None

    result = execute_aggregation(
        meta,
        group_by=group_by,
        metric_column=metric,
        agg=aggregation.get("agg", "sum"),
        order_desc=aggregation.get("order_desc", True),
        limit=aggregation.get("limit", 50),
    )
    return pd.DataFrame(result["records"])


def _build_figure(
    chart_type: str,
    df: pd.DataFrame,
    *,
    x: str,
    y: str | None,
    color: str | None,
    title: str,
) -> go.Figure:
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, color=color, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x if y is None else y or x, color=color, title=title, nbins=40)
    else:
        fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), template="plotly_white")
    return fig


def choose_chart_type(intent: str, columns_info: dict[str, Any] | None = None) -> str:
    intent_l = intent.lower()
    if any(k in intent_l for k in ("distribution", "histogram", "spread")):
        return "histogram"
    if any(k in intent_l for k in ("over time", "trend", "timeseries", "time series")):
        return "line"
    if any(k in intent_l for k in ("scatter", "relationship", "correlation")):
        return "scatter"
    return "bar"
