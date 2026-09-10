"""Deterministic query / analysis tools. Numerical results always come from execution."""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from app.tools.dataset_tools import load_dataset


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|copy|create|pragma|export|import)\b",
    re.IGNORECASE,
)


def execute_sql(meta: dict[str, Any], sql: str, *, limit: int = 500) -> dict[str, Any]:
    """Run a read-only SQL query against the dataset via DuckDB."""
    if _FORBIDDEN.search(sql):
        raise ValueError("Only read-only SELECT-style queries are allowed.")

    df = load_dataset(meta)
    con = duckdb.connect()
    con.register("data", df)
    # Soft-limit large result sets
    trimmed = sql.strip().rstrip(";")
    if not re.search(r"\blimit\b", trimmed, re.IGNORECASE):
        trimmed = f"SELECT * FROM ({trimmed}) AS _q LIMIT {limit}"
    result = con.execute(trimmed).fetchdf()
    con.close()

    records = result.where(pd.notnull(result), None).to_dict(orient="records")
    # Convert non-JSON-friendly types
    clean_records = []
    for row in records:
        clean_records.append({k: _jsonify(v) for k, v in row.items()})

    summary_stats: dict[str, Any] = {}
    for col in result.select_dtypes(include="number").columns:
        summary_stats[col] = {
            "sum": _jsonify(result[col].sum()),
            "mean": _jsonify(result[col].mean()),
            "min": _jsonify(result[col].min()),
            "max": _jsonify(result[col].max()),
        }

    return {
        "sql": sql,
        "executed_sql": trimmed,
        "row_count": int(len(result)),
        "columns": list(result.columns),
        "records": clean_records,
        "summary_stats": summary_stats,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
    }


def execute_aggregation(
    meta: dict[str, Any],
    *,
    group_by: str | None = None,
    metric_column: str | None,
    agg: str = "sum",
    order_desc: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    df = load_dataset(meta)
    metric_column = _resolve_metric_column(df, metric_column)
    if group_by in {None, "None", "null", ""}:
        group_by = None
    elif group_by not in df.columns:
        # Try case-insensitive / alias repair before failing
        repaired = _resolve_column_name(df, group_by, ("product_category", "category", "state"))
        if repaired is None:
            raise ValueError(f"Group-by column '{group_by}' not in dataset.")
        group_by = repaired

    if metric_column not in df.columns:
        raise ValueError(f"Metric column '{metric_column}' not in dataset.")
    if group_by and group_by not in df.columns:
        raise ValueError(f"Group-by column '{group_by}' not in dataset.")

    agg = agg.lower()
    if agg not in {"sum", "mean", "avg", "count", "min", "max", "median"}:
        raise ValueError(f"Unsupported aggregation: {agg}")
    if agg == "avg":
        agg = "mean"

    if group_by:
        if agg == "count":
            out = df.groupby(group_by, dropna=False).size().reset_index(name="value")
        else:
            out = df.groupby(group_by, dropna=False)[metric_column].agg(agg).reset_index(name="value")
        out = out.sort_values("value", ascending=not order_desc).head(limit)
        records = [
            {group_by: _jsonify(r[group_by]), "value": _jsonify(r["value"])}
            for _, r in out.iterrows()
        ]
        sql_equiv = f"SELECT {group_by}, {agg.upper()}({metric_column}) AS value FROM data GROUP BY {group_by}"
    else:
        if agg == "count":
            value = float(len(df))
        else:
            value = float(getattr(df[metric_column], agg)())
        records = [{"value": _jsonify(value)}]
        sql_equiv = f"SELECT {agg.upper()}({metric_column}) AS value FROM data"

    return {
        "operation": "aggregation",
        "group_by": group_by,
        "metric_column": metric_column,
        "agg": agg,
        "records": records,
        "sql_equivalent": sql_equiv,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
    }


def _resolve_metric_column(df: pd.DataFrame, metric_column: str | None) -> str:
    if metric_column and metric_column not in {"None", "null", "undefined"} and metric_column in df.columns:
        return metric_column
    resolved = _resolve_column_name(
        df,
        metric_column,
        ("total_amount", "revenue", "amount", "unit_price", "value", "price", "sales"),
    )
    if resolved:
        return resolved
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        return str(numeric[0])
    raise ValueError("No usable metric column found in dataset for aggregation.")


def _resolve_column_name(
    df: pd.DataFrame,
    name: str | None,
    hints: tuple[str, ...],
) -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    if name and name not in {"None", "null", "undefined"}:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return str(lower_map[name.lower()])
        for c in cols:
            if name.lower() in str(c).lower():
                return str(c)
    for hint in hints:
        if hint.lower() in lower_map:
            return str(lower_map[hint.lower()])
    for c in cols:
        cl = str(c).lower()
        for hint in hints:
            if hint.lower() in cl:
                return str(c)
    return None


def _jsonify(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return str(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (float, str, bool, int)):
        return v
    try:
        import numpy as np

        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:  # noqa: BLE001
        pass
    return str(v)
