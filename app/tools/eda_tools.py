"""Deterministic EDA helpers — additive; unused until agents/orchestrator opt in."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.tools.dataset_tools import load_dataset
from app.tools.query_tools import _jsonify


def correlation_matrix(
    meta: dict[str, Any],
    *,
    columns: list[str] | None = None,
    method: str = "pearson",
    max_columns: int = 20,
) -> dict[str, Any]:
    """Pairwise correlation among numeric columns (tool-computed only)."""
    df = load_dataset(meta)
    method = (method or "pearson").lower()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError(f"Unsupported correlation method: {method}")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not in dataset: {missing}")
        non_numeric = [c for c in columns if c not in numeric]
        if non_numeric:
            raise ValueError(f"Correlation requires numeric columns; got: {non_numeric}")
        cols = list(columns)
    else:
        cols = numeric[:max_columns]

    if len(cols) < 2:
        raise ValueError("Need at least two numeric columns for a correlation matrix.")

    corr = df[cols].corr(method=method)
    matrix = {
        str(row): {str(col): _jsonify(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }
    # Flatten upper triangle for easy charting / summaries
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "correlation": _jsonify(corr.loc[a, b]),
                }
            )
    pairs.sort(key=lambda p: abs(p["correlation"] or 0), reverse=True)

    return {
        "operation": "correlation_matrix",
        "method": method,
        "columns": cols,
        "matrix": matrix,
        "pairs": pairs,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
    }


def missingness_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Per-column null counts and rates."""
    df = load_dataset(meta)
    n = len(df)
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        columns.append(
            {
                "column": str(col),
                "null_count": null_count,
                "null_rate": _jsonify(null_count / n if n else 0.0),
                "non_null_count": int(n - null_count),
            }
        )
    columns.sort(key=lambda r: r["null_count"], reverse=True)
    total_cells = n * len(df.columns) if df.columns.size else 0
    total_nulls = int(sum(r["null_count"] for r in columns))

    return {
        "operation": "missingness_summary",
        "row_count": int(n),
        "column_count": int(len(df.columns)),
        "total_nulls": total_nulls,
        "overall_null_rate": _jsonify(total_nulls / total_cells if total_cells else 0.0),
        "columns": columns,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
    }


def quantile_stats(
    meta: dict[str, Any],
    *,
    columns: list[str] | None = None,
    quantiles: list[float] | None = None,
    max_columns: int = 20,
) -> dict[str, Any]:
    """Quantile / percentile summary for numeric columns."""
    df = load_dataset(meta)
    qs = quantiles or [0.0, 0.25, 0.5, 0.75, 1.0]
    for q in qs:
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1]; got {q}")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not in dataset: {missing}")
        non_numeric = [c for c in columns if c not in numeric]
        if non_numeric:
            raise ValueError(f"Quantiles require numeric columns; got: {non_numeric}")
        cols = list(columns)
    else:
        cols = numeric[:max_columns]

    if not cols:
        raise ValueError("No numeric columns available for quantile stats.")

    stats: dict[str, Any] = {}
    for col in cols:
        s = df[col].dropna()
        qvals = s.quantile(qs) if len(s) else pd.Series(dtype=float)
        entry: dict[str, Any] = {
            "count": int(len(s)),
            "mean": _jsonify(float(s.mean()) if len(s) else None),
            "std": _jsonify(float(s.std()) if len(s) else None),
        }
        for q in qs:
            key = f"p{int(round(float(q) * 100))}"
            entry[key] = _jsonify(float(qvals.loc[q]) if len(s) else None)
        stats[col] = entry

    return {
        "operation": "quantile_stats",
        "quantiles": [float(q) for q in qs],
        "columns": cols,
        "stats": stats,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
    }
