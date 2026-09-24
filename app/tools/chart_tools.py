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
from app.tools.query_tools import _jsonify, coerce_column_ref


def render_chart(
    *,
    chart_type: str,
    title: str,
    data_records: list[dict[str, Any]] | None = None,
    dataset_meta: dict[str, Any] | None = None,
    x: str | None = None,
    y: str | None = None,
    y2: str | None = None,
    z: str | None = None,
    color: str | None = None,
    aggregation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a Plotly chart from either explicit records or dataset + aggregation."""
    ensure_workspace()
    chart_type = (chart_type or "bar").lower().replace("-", "_")
    if chart_type in {"dualaxis", "dual"}:
        chart_type = "dual_axis"
    if chart_type in {"heat_map", "corr_heatmap", "correlation_heatmap"}:
        chart_type = "heatmap"
    x = coerce_column_ref(x)
    y = coerce_column_ref(y)
    y2 = coerce_column_ref(y2)
    z = coerce_column_ref(z)
    color = coerce_column_ref(color)

    # Heatmaps must use the full numeric frame (or correlation matrix), never a
    # one-metric aggregation result the LLM may have attached.
    if chart_type == "heatmap" and dataset_meta is not None:
        data_records = None
        aggregation = None

    if data_records is not None:
        df = pd.DataFrame(data_records)
    elif chart_type == "heatmap" and dataset_meta is not None and not aggregation and data_records is None:
        # Default heatmap path: numeric correlation matrix (additive; opt-in via chart_type)
        from app.tools.eda_tools import correlation_matrix

        corr = correlation_matrix(dataset_meta)
        rows = []
        for a, row in corr["matrix"].items():
            for b, val in row.items():
                rows.append({"column_a": a, "column_b": b, "correlation": val})
        df = pd.DataFrame(rows)
        x = x or "column_a"
        y = y or "column_b"
        z = z or "correlation"
    elif dataset_meta is not None and aggregation:
        df = _aggregate(dataset_meta, aggregation)
        x = x or coerce_column_ref(aggregation.get("group_by"))
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
    if chart_type == "dual_axis" and y2 is None:
        numeric = [c for c in df.select_dtypes(include="number").columns.tolist() if c != y]
        if not numeric:
            raise ValueError("dual_axis charts require a second numeric column (y2).")
        y2 = numeric[0]
    if chart_type == "heatmap" and z is None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        z = numeric[0] if numeric else None

    fig = _build_figure(chart_type, df, x=x, y=y, y2=y2, z=z, color=color, title=title)

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
        "y2": y2,
        "z": z,
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

    metric = coerce_column_ref(aggregation.get("metric_column"))
    if not metric:
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

    group_by = coerce_column_ref(aggregation.get("group_by"))

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
    y2: str | None = None,
    z: str | None = None,
    color: str | None,
    title: str,
) -> go.Figure:
    if chart_type == "pie":
        names = x
        values = y if y is not None else (
            df.select_dtypes(include="number").columns.tolist() or [df.columns[-1]]
        )[0]
        fig = px.pie(df, names=names, values=values, title=title)
    elif chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, color=color, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x if y is None else y or x, color=color, title=title, nbins=40)
    elif chart_type in {"box", "boxplot"}:
        fig = px.box(df, x=x if x in df.columns else None, y=y or x, color=color, title=title)
    elif chart_type == "heatmap":
        if z and x in df.columns and y in df.columns and z in df.columns:
            pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
            fig = px.imshow(
                pivot,
                title=title,
                labels={"x": x, "y": y, "color": z},
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1 if z == "correlation" else None,
                zmax=1 if z == "correlation" else None,
            )
        else:
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] < 2:
                raise ValueError("heatmap requires a z column or at least two numeric columns.")
            fig = px.imshow(
                numeric.corr(),
                title=title,
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
    elif chart_type == "dual_axis":
        if y is None or y2 is None:
            raise ValueError("dual_axis requires y and y2 columns.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df[x], y=df[y], name=str(y), yaxis="y"))
        fig.add_trace(go.Scatter(x=df[x], y=df[y2], name=str(y2), yaxis="y2", mode="lines+markers"))
        fig.update_layout(
            title=title,
            yaxis=dict(title=str(y)),
            yaxis2=dict(title=str(y2), overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
    else:
        fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), template="plotly_white")
    return fig


def choose_chart_type(intent: str, columns_info: dict[str, Any] | None = None) -> str:
    intent_l = intent.lower()
    if any(k in intent_l for k in ("pie", "donut", "share of", "proportion", "percentage breakdown")):
        return "pie"
    if any(k in intent_l for k in ("box plot", "boxplot", "box chart")):
        return "box"
    if any(k in intent_l for k in ("heatmap", "heat map")):
        return "heatmap"
    if any(k in intent_l for k in ("dual axis", "dual-axis", "two axis", "secondary axis")):
        return "dual_axis"
    if any(k in intent_l for k in ("distribution", "histogram", "spread")):
        return "histogram"
    if any(k in intent_l for k in ("over time", "trend", "timeseries", "time series")):
        return "line"
    # Keep correlation → scatter so existing callers are unchanged
    if any(k in intent_l for k in ("scatter", "relationship", "correlation")):
        return "scatter"
    return "bar"
