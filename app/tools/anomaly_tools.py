"""Deterministic anomaly detection tools (IQR, Z-score, Isolation Forest)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.tools.dataset_tools import load_dataset
from app.tools.query_tools import _jsonify


Method = Literal["iqr", "zscore", "isolation_forest", "auto"]


def detect_anomalies(
    meta: dict[str, Any],
    *,
    column: str | None = None,
    method: Method = "auto",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.02,
    max_records: int = 25,
) -> dict[str, Any]:
    df = load_dataset(meta)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for anomaly detection.")

    if column is None:
        # Prefer amount-like columns
        preferred = [c for c in numeric_cols if any(k in c.lower() for k in ("amount", "price", "total", "value", "revenue"))]
        column = preferred[0] if preferred else numeric_cols[0]
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if column not in numeric_cols:
        raise ValueError(f"Column '{column}' is not numeric.")

    series = df[column].astype(float)
    chosen = method
    if method == "auto":
        # Skewed financial data → IQR; otherwise z-score; IsolationForest if many dims requested later
        skew = float(series.skew()) if len(series) > 2 else 0.0
        chosen = "iqr" if abs(skew) > 1.0 else "zscore"

    if chosen == "iqr":
        mask, scores, details = _iqr(series, iqr_multiplier)
    elif chosen == "zscore":
        mask, scores, details = _zscore(series, z_threshold)
    elif chosen == "isolation_forest":
        mask, scores, details = _isolation_forest(df[numeric_cols], column, contamination)
    else:
        raise ValueError(f"Unknown method: {chosen}")

    anomaly_df = df.loc[mask].copy()
    anomaly_df["_anomaly_score"] = scores[mask]
    anomaly_df = anomaly_df.sort_values("_anomaly_score", ascending=False).head(max_records)

    records = []
    for _, row in anomaly_df.iterrows():
        records.append({k: _jsonify(v) for k, v in row.items()})

    return {
        "method": chosen,
        "column": column,
        "n_anomalies": int(mask.sum()),
        "n_rows_scanned": int(len(df)),
        "anomaly_rate": float(mask.mean()),
        "parameters": details,
        "records": records,
        "grounded": True,
        "source_dataset_id": meta.get("id"),
        "explanation_facts": {
            "method": chosen,
            "column": column,
            "threshold_info": details,
            "top_values": [_jsonify(v) for v in anomaly_df[column].head(5).tolist()] if len(anomaly_df) else [],
        },
    }


def _iqr(series: pd.Series, multiplier: float) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    # Score = distance beyond fence, normalized by IQR
    scores = np.zeros(len(series), dtype=float)
    scores[series > upper] = ((series[series > upper] - upper) / (iqr + 1e-9)).to_numpy()
    scores[series < lower] = ((lower - series[series < lower]) / (iqr + 1e-9)).to_numpy()
    details = {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper, "multiplier": multiplier}
    return mask, scores, details


def _zscore(series: pd.Series, threshold: float) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
    mean = float(series.mean())
    std = float(series.std()) or 1e-9
    z = (series - mean) / std
    mask = z.abs() > threshold
    scores = z.abs().to_numpy()
    details = {"mean": mean, "std": std, "threshold": threshold}
    return mask, scores, details


def _isolation_forest(
    numeric_df: pd.DataFrame,
    focus_column: str,
    contamination: float,
) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
    X = numeric_df.fillna(numeric_df.median(numeric_only=True))
    model = IsolationForest(
        n_estimators=100,
        contamination=min(max(contamination, 0.001), 0.2),
        random_state=42,
    )
    preds = model.fit_predict(X)
    raw_scores = -model.score_samples(X)  # higher = more anomalous
    mask = pd.Series(preds == -1, index=numeric_df.index)
    details = {
        "contamination": contamination,
        "n_features": int(X.shape[1]),
        "focus_column": focus_column,
        "features": list(X.columns),
    }
    return mask, raw_scores, details
