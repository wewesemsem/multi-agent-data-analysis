"""Deterministic query / analysis tools. Numerical results always come from execution."""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from app.tools.dataset_tools import load_dataset


_MISSING_COL_NAMES = frozenset({"", "none", "null", "undefined"})


def coerce_column_ref(value: Any) -> str | None:
    """Normalize LLM column refs (plain strings or dicts like {"name": "col"})."""
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("name", "column", "field", "col", "id", "value"):
            if key in value and value[key] not in (None, ""):
                return coerce_column_ref(value[key])
        if len(value) == 1:
            return coerce_column_ref(next(iter(value.values())))
        return None
    if isinstance(value, (list, tuple)) and value:
        return coerce_column_ref(value[0])
    text = str(value).strip()
    if text.lower() in _MISSING_COL_NAMES:
        return None
    return text


def _is_missing_col(value: Any) -> bool:
    return coerce_column_ref(value) is None


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
    group_by = coerce_column_ref(group_by)
    if group_by is not None and group_by not in df.columns:
        # Try case-insensitive / alias repair before failing
        repaired = _resolve_column_name(
            df, group_by, ("category", "segment", "type", "group", "region", "state")
        )
        if repaired is None:
            raise ValueError(f"Group-by column '{group_by}' not in dataset.")
        group_by = repaired

    if metric_column not in df.columns:
        raise ValueError(f"Metric column '{metric_column}' not in dataset.")
    if group_by and group_by not in df.columns:
        raise ValueError(f"Group-by column '{group_by}' not in dataset.")

    agg = agg.lower()
    if agg not in {"sum", "mean", "avg", "count", "min", "max", "median", "std", "nunique", "pct"}:
        raise ValueError(f"Unsupported aggregation: {agg}")
    if agg == "avg":
        agg = "mean"

    if group_by:
        out = _grouped_values(df, group_by=group_by, metric_column=metric_column, agg=agg)
        out = out.sort_values("value", ascending=not order_desc).head(limit)
        records = [
            {group_by: _jsonify(r[group_by]), "value": _jsonify(r["value"])}
            for _, r in out.iterrows()
        ]
        sql_equiv = _sql_equiv(group_by=group_by, metric_column=metric_column, agg=agg)
    else:
        value = _ungrouped_value(df, metric_column=metric_column, agg=agg)
        records = [{"value": _jsonify(value)}]
        sql_equiv = _sql_equiv(group_by=None, metric_column=metric_column, agg=agg)

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


def _grouped_values(
    df: pd.DataFrame,
    *,
    group_by: str,
    metric_column: str,
    agg: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_by, dropna=False)
    if agg == "count":
        out = grouped.size().reset_index(name="value")
    elif agg == "nunique":
        out = grouped[metric_column].nunique(dropna=True).reset_index(name="value")
    elif agg == "pct":
        totals = grouped[metric_column].sum()
        denom = float(totals.sum())
        shares = (totals / denom * 100.0) if denom else totals * 0.0
        out = shares.reset_index(name="value")
    elif agg == "std":
        out = grouped[metric_column].std().reset_index(name="value")
    else:
        out = grouped[metric_column].agg(agg).reset_index(name="value")
    return out


def _ungrouped_value(df: pd.DataFrame, *, metric_column: str, agg: str) -> float:
    if agg == "count":
        return float(len(df))
    if agg == "nunique":
        return float(df[metric_column].nunique(dropna=True))
    if agg == "pct":
        return 100.0 if len(df) else 0.0
    if agg == "std":
        return float(df[metric_column].std())
    return float(getattr(df[metric_column], agg)())


def _sql_equiv(*, group_by: str | None, metric_column: str, agg: str) -> str:
    if agg == "pct":
        expr = f"100.0 * SUM({metric_column}) / SUM(SUM({metric_column})) OVER ()"
    elif agg == "nunique":
        expr = f"COUNT(DISTINCT {metric_column})"
    elif agg == "std":
        expr = f"STDDEV({metric_column})"
    elif agg == "count":
        expr = "COUNT(*)"
    else:
        expr = f"{agg.upper()}({metric_column})"
    if group_by:
        return f"SELECT {group_by}, {expr} AS value FROM data GROUP BY {group_by}"
    return f"SELECT {expr} AS value FROM data"


def _resolve_metric_column(df: pd.DataFrame, metric_column: str | None) -> str:
    metric_column = coerce_column_ref(metric_column)
    if metric_column and metric_column in df.columns:
        return metric_column
    # Resolve user/LLM aliases against the live schema, then first numeric column.
    if metric_column:
        resolved = _resolve_column_name(
            df,
            metric_column,
            ("revenue", "amount", "sales", "value", "price", "total"),
        )
        if resolved:
            return resolved
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        return str(numeric[0])
    raise ValueError("No usable metric column found in dataset for aggregation.")


def _resolve_column_name(
    df: pd.DataFrame,
    name: Any,
    hints: tuple[str, ...],
) -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    name = coerce_column_ref(name)
    if name:
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
