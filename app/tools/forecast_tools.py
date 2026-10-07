"""Deterministic time-series forecasting tools.

LLMs select/configure; numerical forecasts come only from these tools.
No domain-specific column names are hard-coded as required targets.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.tools.dataset_tools import load_dataset
from app.tools.query_tools import _jsonify
from app.tools.schema_resolve import rank_columns_for_request


MethodName = Literal[
    "naive",
    "moving_average",
    "exponential_smoothing",
    "seasonal_naive",
    "trend_regression",
    "causal_regression",
]

MIN_OBSERVATIONS = 8
_TIME_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "period",
    "year",
    "month",
    "week",
    "day",
    "quarter",
    "fiscal",
)
_TARGET_NAME_HINTS = (
    "revenue",
    "sales",
    "amount",
    "total",
    "value",
    "price",
    "ebitda",
    "profit",
    "income",
    "debt",
    "cash",
    "customers",
    "quantity",
    "volume",
    "units",
    "demand",
)


@dataclass
class ForecastResult:
    """Serializable forecast artifact stored in SharedWorkspace.forecasts."""

    id: str
    suitable: bool
    grounded: bool
    target_column: str | None
    time_column: str | None
    frequency: str | None
    forecast_horizon: int | None
    historical_observations: int
    historical_values: list[dict[str, Any]] = field(default_factory=list)
    forecast_values: list[dict[str, Any]] = field(default_factory=list)
    selected_method: str | None = None
    candidate_methods: list[str] = field(default_factory=list)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    trend: str | None = None
    seasonality: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    requirements: list[str] = field(default_factory=list)
    predictors_used: list[str] = field(default_factory=list)
    source_dataset_id: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def forecast_series(
    meta: dict[str, Any],
    *,
    target_column: str | None = None,
    time_column: str | None = None,
    horizon: int | None = None,
    user_request: str | None = None,
    method: str | None = None,
    aggregate: str = "sum",
) -> dict[str, Any]:
    """Inspect dataset, select models, backtest, and forecast.

    Returns a serializable dict (ForecastResult) with grounded=True.
    When unsuitable, suitable=False and no fabricated forecast_values.
    """
    df = load_dataset(meta)
    assessment = assess_forecastability(
        df,
        time_column=time_column,
        target_column=target_column,
        user_request=user_request,
    )
    result = ForecastResult(
        id=f"fc_{uuid.uuid4().hex[:10]}",
        suitable=False,
        grounded=True,
        target_column=assessment.get("target_column"),
        time_column=assessment.get("time_column"),
        frequency=assessment.get("frequency"),
        forecast_horizon=None,
        historical_observations=int(assessment.get("n_observations") or 0),
        source_dataset_id=meta.get("id"),
        warnings=list(assessment.get("warnings") or []),
        requirements=list(assessment.get("requirements") or []),
        reason=assessment.get("reason"),
    )

    if not assessment.get("suitable"):
        result.validation = {"ok": False, "issues": list(assessment.get("issues") or [])}
        return result.to_dict()

    time_col = assessment["time_column"]
    target_col = assessment["target_column"]
    series_df = assessment["series_df"]
    frequency = assessment["frequency"]
    seasonality = assessment.get("seasonality") or {}
    predictors = assessment.get("predictor_columns") or []

    hz, hz_assumption = resolve_horizon(horizon, user_request, frequency, len(series_df))
    result.forecast_horizon = hz
    if hz_assumption:
        result.assumptions.append(hz_assumption)

    y = series_df[target_col].astype(float).to_numpy()
    times = series_df[time_col]
    trend = _detect_trend(y)
    result.trend = trend
    result.seasonality = seasonality
    result.historical_observations = int(len(y))
    result.historical_values = [
        {"time": _jsonify(t), "value": _jsonify(v)} for t, v in zip(times.tolist(), y.tolist())
    ]

    if float(np.nanstd(y)) == 0.0 or np.allclose(y, y[0]):
        result.reason = f"Target '{target_col}' is constant; forecasting is not informative."
        result.warnings.append(result.reason)
        result.validation = {"ok": False, "issues": [result.reason]}
        return result.to_dict()

    exog = None
    if predictors:
        exog = series_df[predictors].astype(float)
        result.predictors_used = list(predictors)

    forced = _normalize_method(method) if method and method != "auto" else None
    candidates = _candidate_methods(
        n=len(y),
        seasonality=seasonality,
        has_exog=exog is not None and len(predictors) > 0,
        forced=forced,
    )
    result.candidate_methods = list(candidates)

    eval_rows, baseline = _evaluate_candidates(y, candidates, seasonality, exog=exog)
    result.evaluation_metrics = eval_rows
    result.baseline_metrics = baseline

    if not eval_rows:
        result.reason = "All candidate forecasting models failed to fit during evaluation."
        result.warnings.append(result.reason)
        result.validation = {"ok": False, "issues": [result.reason]}
        return result.to_dict()

    selected = min(eval_rows, key=lambda r: (r.get("rmse") is None, r.get("rmse", math.inf)))
    selected_method = selected["method"]
    result.selected_method = selected_method
    if selected_method != "naive" and baseline:
        base_rmse = baseline.get("rmse")
        sel_rmse = selected.get("rmse")
        if base_rmse is not None and sel_rmse is not None and sel_rmse > base_rmse * 1.05:
            result.warnings.append(
                "Selected model did not clearly beat the naive baseline on holdout RMSE; "
                "results should be treated cautiously."
            )

    seasonal_period = int(seasonality.get("period") or 0)
    forecast_y, lower, upper = _generate_forecast(
        y,
        method=selected_method,
        horizon=hz,
        seasonal_period=seasonal_period,
        exog=exog,
    )
    future_times = _future_timestamps(times, frequency, hz)
    result.forecast_values = [
        {
            "time": _jsonify(t),
            "value": _jsonify(v),
            "lower": _jsonify(lo),
            "upper": _jsonify(hi),
        }
        for t, v, lo, hi in zip(future_times, forecast_y, lower, upper)
    ]
    result.assumptions.append(
        f"Selected '{selected_method}' after time-series holdout comparison among: "
        f"{', '.join(candidates)}."
    )
    if predictors:
        result.assumptions.append(
            "Causal/regression predictors for future periods use last observed values "
            "(held constant) when extrapolating."
        )
    if aggregate and assessment.get("aggregated"):
        result.assumptions.append(
            f"Duplicate timestamps were aggregated with '{assessment.get('aggregate')}'."
        )
    if assessment.get("resampled") and assessment.get("requested_frequency"):
        result.assumptions.append(
            f"Series was resampled from {assessment.get('native_frequency')} to "
            f"{assessment.get('requested_frequency')} to match the requested forecast grain."
        )

    # Non-negative history → clip intervals/values below zero (e.g. revenue)
    if y.min() >= 0:
        for row in result.forecast_values:
            for key in ("value", "lower", "upper"):
                if row.get(key) is not None:
                    try:
                        row[key] = max(0.0, float(row[key]))
                    except (TypeError, ValueError):
                        pass

    issues = validate_forecast_payload(result.to_dict())
    result.suitable = not issues
    result.validation = {"ok": not issues, "issues": issues}
    if issues:
        result.reason = "Forecast failed deterministic validation: " + "; ".join(issues)
        result.forecast_values = []
        result.suitable = False
    else:
        result.reason = None
        result.suitable = True
    return result.to_dict()


def assess_forecastability(
    df: pd.DataFrame,
    *,
    time_column: str | None = None,
    target_column: str | None = None,
    aggregate: str = "sum",
    user_request: str | None = None,
) -> dict[str, Any]:
    """Determine whether the dataframe can support quantitative forecasting."""
    issues: list[str] = []
    warnings: list[str] = []
    requirements: list[str] = []

    if df is None or df.empty:
        return _unsuitable(
            "Dataset is empty.",
            requirements=["Provide a non-empty tabular dataset with a time column and numeric target."],
        )

    time_col = time_column or detect_time_column(df)
    if not time_col:
        return _unsuitable(
            "No usable time/date column was found.",
            issues=["No time column"],
            requirements=[
                "Include a date/time/period column (e.g. date, month, year, order_date).",
                "Ensure the column is ordered and parseable.",
            ],
        )

    parsed = _parse_time_series(df[time_col])
    if parsed is None:
        return _unsuitable(
            f"Time column '{time_col}' could not be parsed into ordered timestamps/periods.",
            issues=[f"Unparseable time column: {time_col}"],
            requirements=[f"Provide parseable dates/periods in '{time_col}'."],
        )

    work = df.copy()
    work["_ts"] = parsed
    work = work.dropna(subset=["_ts"])
    if len(work) < MIN_OBSERVATIONS:
        return _unsuitable(
            f"Only {len(work)} usable time observations after parsing; need at least {MIN_OBSERVATIONS}.",
            issues=["Insufficient observations"],
            requirements=[f"Provide at least {MIN_OBSERVATIONS} historical observations."],
            time_column=time_col,
        )

    numeric_cols = [
        c
        for c in work.select_dtypes(include=[np.number]).columns.tolist()
        if c != "_ts" and c != time_col
    ]
    # Include object columns that are numeric-like
    for c in work.columns:
        if c in numeric_cols or c in {"_ts", time_col}:
            continue
        if pd.api.types.is_bool_dtype(work[c]):
            continue
        coerced = pd.to_numeric(work[c], errors="coerce")
        if coerced.notna().mean() >= 0.9:
            work[c] = coerced
            numeric_cols.append(c)

    if target_column and target_column not in work.columns:
        return _unsuitable(
            f"Requested target column '{target_column}' not found.",
            issues=[f"Missing target: {target_column}"],
            requirements=[f"Provide numeric column '{target_column}' or choose another target."],
            time_column=time_col,
        )

    if target_column:
        if target_column not in numeric_cols:
            return _unsuitable(
                f"Target '{target_column}' is not numeric.",
                issues=["Non-numeric target"],
                requirements=[f"Make '{target_column}' numeric."],
                time_column=time_col,
            )
        target = target_column
    else:
        candidates = suggest_target_columns(
            work, numeric_cols, time_col, user_request=user_request
        )
        if not candidates:
            return _unsuitable(
                "No numeric target columns available for forecasting.",
                issues=["No numeric target"],
                requirements=["Include at least one numeric metric column to forecast."],
                time_column=time_col,
            )
        target = candidates[0]
        if len(candidates) > 1:
            warnings.append(
                f"No target specified; using '{target}'. Other candidates: {', '.join(candidates[1:5])}."
            )

    # Aggregate duplicate timestamps
    agg_fn = aggregate if aggregate in {"sum", "mean", "median", "last"} else "sum"
    if target.lower() in {"rate", "ratio", "margin", "pct", "percent"} or any(
        k in target.lower() for k in ("rate", "ratio", "margin", "pct")
    ):
        agg_fn = "mean"

    grouped = (
        work.groupby("_ts", as_index=False)
        .agg({target: agg_fn, **{c: "mean" for c in numeric_cols if c != target}})
        .sort_values("_ts")
    )
    aggregated = len(work) != len(grouped)
    if aggregated:
        warnings.append(
            f"Found duplicate timestamps on '{time_col}'; aggregated '{target}' with {agg_fn}."
        )

    if grouped[target].isna().mean() > 0.3:
        return _unsuitable(
            f"Target '{target}' has excessive missing values after aggregation "
            f"({grouped[target].isna().mean()*100:.1f}%).",
            issues=["Excessive missing data"],
            requirements=["Reduce missing values in the target series."],
            time_column=time_col,
            target_column=target,
        )

    grouped = grouped.dropna(subset=[target])
    if len(grouped) < MIN_OBSERVATIONS:
        return _unsuitable(
            f"Only {len(grouped)} observations after cleaning; need at least {MIN_OBSERVATIONS}.",
            issues=["Insufficient observations"],
            requirements=[f"Provide at least {MIN_OBSERVATIONS} historical points for '{target}'."],
            time_column=time_col,
            target_column=target,
        )

    # Duplicate check after aggregation should be clean
    if grouped["_ts"].duplicated().any():
        issues.append("Duplicate timestamps remain after aggregation.")

    native_frequency = infer_frequency(grouped["_ts"])
    if native_frequency == "irregular":
        warnings.append(
            "Time intervals appear irregular; forecasts assume an approximate step based on the median gap."
        )

    requested_frequency = detect_requested_frequency(user_request)
    frequency = native_frequency
    resampled = False
    if requested_frequency and _should_resample(native_frequency, requested_frequency):
        grouped, resample_note = _resample_to_frequency(
            grouped,
            target=target,
            predictors=[c for c in numeric_cols if c != target and c in grouped.columns],
            frequency=requested_frequency,
            aggregate=agg_fn,
        )
        if grouped is None or len(grouped) < MIN_OBSERVATIONS:
            n = 0 if grouped is None else len(grouped)
            return _unsuitable(
                f"Requested {requested_frequency} aggregation leaves only {n} observations; "
                f"need at least {MIN_OBSERVATIONS}.",
                issues=["Insufficient observations after resample"],
                requirements=[
                    f"Provide longer history or request a finer grain than {requested_frequency}."
                ],
                time_column=time_col,
                target_column=target,
            )
        frequency = requested_frequency
        resampled = True
        warnings.append(resample_note)
        aggregated = True

    gaps = _count_gaps(grouped["_ts"], frequency)
    if gaps > 0:
        warnings.append(f"Detected approximately {gaps} gap(s) in the time index.")

    seasonality = detect_seasonality(grouped[target].to_numpy(dtype=float), frequency)
    predictors = [
        c
        for c in numeric_cols
        if c != target and c in grouped.columns and grouped[c].notna().mean() > 0.8
    ][:5]

    series_df = grouped.rename(columns={"_ts": time_col})[[time_col, target] + predictors].copy()

    return {
        "suitable": not issues,
        "time_column": time_col,
        "target_column": target,
        "frequency": frequency,
        "native_frequency": native_frequency,
        "requested_frequency": requested_frequency,
        "resampled": resampled,
        "n_observations": int(len(series_df)),
        "seasonality": seasonality,
        "predictor_columns": predictors,
        "series_df": series_df,
        "warnings": warnings,
        "issues": issues,
        "requirements": requirements,
        "reason": None if not issues else "; ".join(issues),
        "aggregated": aggregated,
        "aggregate": agg_fn,
    }


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Heuristically find a time/date/period column without assuming a schema."""
    # Prefer datetime dtypes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    scored: list[tuple[float, str]] = []
    for col in df.columns:
        name = str(col).lower()
        name_score = 1.0 if any(h in name for h in _TIME_NAME_HINTS) else 0.0
        parsed = _parse_time_series(df[col])
        if parsed is None:
            continue
        parse_rate = float(parsed.notna().mean())
        if parse_rate < 0.8:
            continue
        # Prefer columns that are mostly unique / ordered
        nunique = parsed.nunique(dropna=True)
        unique_ratio = nunique / max(len(parsed.dropna()), 1)
        score = name_score * 2 + parse_rate + min(unique_ratio, 1.0)
        scored.append((score, col))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def suggest_target_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    time_col: str,
    user_request: str | None = None,
) -> list[str]:
    """Rank numeric columns that are reasonable forecast targets.

    When a user request is present, NL→schema scoring dominates so metric
    language maps onto the active schema instead of a fixed column name.
    """
    usable: list[str] = []
    coverage_bonus: dict[str, float] = {}
    for col in numeric_cols:
        if col == time_col or col == "_ts":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < MIN_OBSERVATIONS:
            continue
        if float(series.std(skipna=True) or 0) == 0:
            continue
        usable.append(col)
        coverage_bonus[col] = float(series.notna().mean())

    if not usable:
        return []

    if user_request and user_request.strip():
        nl_ranked = rank_columns_for_request(user_request, usable, min_score=0.0)
        scored: list[tuple[float, str]] = []
        for col, nl_score in nl_ranked:
            name = col.lower()
            hint = 0.5 if any(h in name for h in _TARGET_NAME_HINTS) else 0.0
            scored.append((nl_score * 10.0 + hint + coverage_bonus.get(col, 0.0), col))
        scored.sort(reverse=True)
        # Use NL ranking whenever any column got a positive metric score
        if scored and any(nl > 0 for _, nl in nl_ranked):
            return [c for _, c in scored]

    ranked: list[tuple[float, str]] = []
    for col in usable:
        name = col.lower()
        hint = 2.0 if any(h in name for h in _TARGET_NAME_HINTS) else 0.0
        # Mild granularity prior when no NL signal: aggregates over unit measures
        tokens = set(name.replace("-", "_").split("_"))
        if tokens & {"total", "amount", "revenue", "sales", "sum", "gross", "net"}:
            hint += 1.0
        if "unit" in tokens or ("price" in tokens and "total" not in tokens):
            hint -= 1.0
        ranked.append((hint + coverage_bonus.get(col, 0.0), col))
    ranked.sort(reverse=True)
    return [c for _, c in ranked]


def infer_frequency(times: pd.Series) -> str:
    """Infer a coarse frequency label from timestamp gaps."""
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return "unknown"
    # Integer year-like
    if times.dtype != "datetime64[ns]" and set(pd.Series(times).map(type)) <= {int, np.integer}:
        diffs = pd.Series(times).sort_values().diff().dropna()
        if not diffs.empty and diffs.median() == 1:
            return "yearly"

    deltas = ts.diff().dropna()
    if deltas.empty:
        return "unknown"
    median = deltas.median()
    days = median / pd.Timedelta(days=1)
    if days <= 1.5:
        return "daily"
    if 6 <= days <= 8:
        return "weekly"
    if 27 <= days <= 32:
        return "monthly"
    if 88 <= days <= 95:
        return "quarterly"
    if 360 <= days <= 370:
        return "yearly"
    return "irregular"


_FREQ_RANK = {
    "daily": 1,
    "weekly": 2,
    "monthly": 3,
    "quarterly": 4,
    "yearly": 5,
}


def detect_requested_frequency(text: str | None) -> str | None:
    """Infer the forecasting grain the user asked for (e.g. monthly from '12 months')."""
    if not text:
        return None
    q = text.lower()
    # Explicit grain words first
    if re.search(r"\bmonthly\b|\bby\s+month\b|\bper\s+month\b|\baggregat\w*\s+by\s+month", q):
        return "monthly"
    if re.search(r"\bweekly\b|\bby\s+week\b|\bper\s+week\b", q):
        return "weekly"
    if re.search(r"\bquarterly\b|\bby\s+quarter\b|\bper\s+quarter\b", q):
        return "quarterly"
    if re.search(r"\byearly\b|\bannually\b|\bby\s+year\b|\bper\s+year\b|\bannual\b", q):
        return "yearly"
    if re.search(r"\bdaily\b|\bby\s+day\b|\bper\s+day\b", q):
        return "daily"
    # Horizon unit implies grain: "next 12 months" → monthly
    unit_patterns = [
        (r"\d+\s*(?:month|months)\b", "monthly"),
        (r"\d+\s*(?:week|weeks)\b", "weekly"),
        (r"\d+\s*(?:quarter|quarters)\b", "quarterly"),
        (r"\d+\s*(?:year|years)\b", "yearly"),
        (r"\d+\s*(?:day|days)\b", "daily"),
        (r"(?:next|coming)\s+(?:month)\b", "monthly"),
        (r"(?:next|coming)\s+(?:week)\b", "weekly"),
        (r"(?:next|coming)\s+(?:quarter)\b", "quarterly"),
        (r"(?:next|coming)\s+(?:year)\b", "yearly"),
        (r"(?:next|coming)\s+(?:day)\b", "daily"),
    ]
    for pat, freq in unit_patterns:
        if re.search(pat, q):
            return freq
    return None


def detect_seasonality(y: np.ndarray, frequency: str) -> dict[str, Any]:
    """Lightweight seasonality probe using lag autocorrelation."""
    period_map = {"daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}
    period = period_map.get(frequency, 0)
    if period <= 1 or len(y) < period * 2:
        return {"detected": False, "period": period if period > 1 else None, "strength": 0.0}

    y0 = y - np.mean(y)
    denom = float(np.dot(y0, y0)) + 1e-12
    lag = y0[period:]
    lead = y0[:-period]
    strength = float(np.dot(lag, lead) / denom)
    detected = strength >= 0.3
    return {"detected": detected, "period": period, "strength": strength}


def resolve_horizon(
    horizon: int | None,
    user_request: str | None,
    frequency: str,
    n_obs: int,
) -> tuple[int, str | None]:
    """Resolve forecast horizon from explicit args / NL request / frequency default.

    When the user (or planner LLM) supplies a bare integer like 12 alongside
    text such as "next 12 months", prefer the text parse so the integer is
    interpreted in the requested unit after frequency alignment — not as
    native daily steps.
    """
    parsed = parse_horizon_from_text(user_request or "", frequency)
    if parsed is not None:
        hz = max(1, min(int(parsed), max(n_obs * 2, 1)))
        return hz, None

    if horizon is not None:
        # Bare integer with no unit in the request: treat as steps at `frequency`.
        hz = max(1, min(int(horizon), max(n_obs * 2, 1)))
        return hz, None

    defaults = {
        "daily": 14,
        "weekly": 8,
        "monthly": 6,
        "quarterly": 4,
        "yearly": 3,
        "irregular": max(3, min(6, n_obs // 4 or 3)),
        "unknown": max(3, min(6, n_obs // 4 or 3)),
    }
    hz = defaults.get(frequency, 6)
    hz = max(1, min(hz, max(n_obs, 1)))
    assumption = (
        f"No horizon specified; defaulting to {hz} {frequency} period(s) "
        f"based on detected frequency '{frequency}'."
    )
    return hz, assumption


def parse_horizon_from_text(text: str, frequency: str) -> int | None:
    """Extract horizons like 'next 12 months', '8 quarters', '30 days'."""
    if not text:
        return None
    q = text.lower()

    patterns = [
        (r"(?:next|coming|following)\s+(\d+)\s*(day|days|daily)", "daily"),
        (r"(\d+)\s*(day|days)\b", "daily"),
        (r"(?:next|coming|following)\s+(\d+)\s*(week|weeks|weekly)", "weekly"),
        (r"(\d+)\s*(week|weeks)\b", "weekly"),
        (r"(?:next|coming|following)\s+(\d+)\s*(month|months|monthly)", "monthly"),
        (r"(\d+)\s*(month|months)\b", "monthly"),
        (r"(?:next|coming|following)\s+(\d+)\s*(quarter|quarters|quarterly)", "quarterly"),
        (r"(\d+)\s*(quarter|quarters)\b", "quarterly"),
        (r"(?:next|coming|following)\s+(\d+)\s*(year|years|yearly|annual)", "yearly"),
        (r"(\d+)\s*(year|years)\b", "yearly"),
        (r"(?:next|coming)\s+(?:one|1)\s+(day|week|month|quarter|year)", None),
    ]
    unit_to_freq = {
        "day": "daily",
        "days": "daily",
        "daily": "daily",
        "week": "weekly",
        "weeks": "weekly",
        "weekly": "weekly",
        "month": "monthly",
        "months": "monthly",
        "monthly": "monthly",
        "quarter": "quarterly",
        "quarters": "quarterly",
        "quarterly": "quarterly",
        "year": "yearly",
        "years": "yearly",
        "yearly": "yearly",
        "annual": "yearly",
    }

    for pat, _forced in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        if m.lastindex == 1 and m.group(1) in unit_to_freq:
            n = 1
            unit_freq = unit_to_freq[m.group(1)]
        else:
            n = int(m.group(1))
            unit = m.group(2)
            unit_freq = unit_to_freq.get(unit, frequency)
        return _convert_horizon(n, unit_freq, frequency)

    # Phrases without numbers
    if "next quarter" in q or "coming quarter" in q:
        return _convert_horizon(1, "quarterly", frequency)
    if "next year" in q or "coming year" in q:
        return _convert_horizon(1, "yearly", frequency)
    if "next month" in q:
        return _convert_horizon(1, "monthly", frequency)
    if "next week" in q:
        return _convert_horizon(1, "weekly", frequency)
    return None


def validate_forecast_payload(payload: dict[str, Any]) -> list[str]:
    """Deterministic validation of a forecast artifact."""
    issues: list[str] = []
    if not payload.get("suitable") and not payload.get("forecast_values"):
        # Unsuitable refusal is valid — no further structural checks
        return list((payload.get("validation") or {}).get("issues") or [])

    if not payload.get("time_column"):
        issues.append("Missing time_column.")
    if not payload.get("target_column"):
        issues.append("Missing target_column.")
    if not payload.get("selected_method"):
        issues.append("Model evaluation did not select a method.")
    hz = payload.get("forecast_horizon")
    values = payload.get("forecast_values") or []
    if not hz or int(hz) < 1:
        issues.append("Invalid forecast horizon.")
    elif len(values) != int(hz):
        issues.append(f"Forecast length {len(values)} does not match horizon {hz}.")

    hist = payload.get("historical_values") or []
    times = []
    for row in hist:
        times.append(row.get("time"))
    if len(times) != len(set(map(str, times))):
        issues.append("Historical timestamps contain duplicates.")

    for i, row in enumerate(values):
        v = row.get("value")
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            issues.append(f"Forecast value[{i}] is NaN/infinite.")
        lo, hi = row.get("lower"), row.get("upper")
        if lo is not None and hi is not None:
            try:
                if float(lo) > float(hi):
                    issues.append(f"Prediction interval[{i}] has lower > upper.")
            except (TypeError, ValueError):
                issues.append(f"Prediction interval[{i}] is not numeric.")

    if hist and values:
        # Chronology: first forecast time should be after last historical time when comparable
        try:
            last_hist = pd.to_datetime(hist[-1]["time"], errors="coerce")
            first_fc = pd.to_datetime(values[0]["time"], errors="coerce")
            if pd.notna(last_hist) and pd.notna(first_fc) and first_fc <= last_hist:
                issues.append("Forecast dates do not continue after historical data.")
        except Exception:  # noqa: BLE001
            pass

    metrics = payload.get("evaluation_metrics") or []
    if payload.get("suitable") and not metrics:
        issues.append("Model evaluation metrics missing.")
    if payload.get("suitable") and not payload.get("baseline_metrics"):
        issues.append("Baseline comparison missing.")
    return issues


# --- internals -----------------------------------------------------------------


def _unsuitable(
    reason: str,
    *,
    issues: list[str] | None = None,
    requirements: list[str] | None = None,
    time_column: str | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    return {
        "suitable": False,
        "reason": reason,
        "issues": issues or [reason],
        "requirements": requirements or [],
        "warnings": [],
        "time_column": time_column,
        "target_column": target_column,
        "frequency": None,
        "n_observations": 0,
        "seasonality": {},
        "predictor_columns": [],
        "series_df": None,
        "aggregated": False,
        "aggregate": None,
    }


def _parse_time_series(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    # Pure integer years / periods (avoid treating metric floats as timestamps)
    if pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_float_dtype(series) and series.dropna().mod(1).eq(0).all()
    ):
        vals = series.dropna()
        if not vals.empty and vals.min() >= 1900 and vals.max() <= 2100:
            return pd.to_datetime(series.astype("Int64").astype(str), format="%Y", errors="coerce")
        return None

    if pd.api.types.is_numeric_dtype(series):
        return None

    # Month period strings like 2021-01
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return None
    if sample.str.match(r"^\d{4}-\d{2}$").mean() > 0.8:
        return pd.to_datetime(series.astype(str), format="%Y-%m", errors="coerce")

    # Only attempt free-form parsing when values look date-like
    dateish = sample.str.contains(
        r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        regex=True,
        na=False,
    ).mean()
    if dateish < 0.8:
        return None

    parsed = pd.to_datetime(series, errors="coerce")
    if float(parsed.notna().mean()) >= 0.8:
        return parsed
    return None


def _count_gaps(times: pd.Series, frequency: str) -> int:
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(ts) < 3 or frequency in {"unknown", "irregular"}:
        return 0
    freq_alias = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS",
        "quarterly": "QS",
        "yearly": "YS",
    }.get(frequency)
    if not freq_alias:
        return 0
    expected = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=freq_alias)
    # Approximate gap count
    return max(0, int(len(expected) - len(ts)))


def _detect_trend(y: np.ndarray) -> str:
    if len(y) < 3:
        return "unknown"
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    scale = float(np.std(y)) + 1e-9
    if abs(slope) < 0.01 * scale:
        return "flat"
    return "increasing" if slope > 0 else "decreasing"


def _normalize_method(method: str | None) -> MethodName | None:
    if not method:
        return None
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ses": "exponential_smoothing",
        "ets": "exponential_smoothing",
        "exp_smoothing": "exponential_smoothing",
        "ma": "moving_average",
        "sma": "moving_average",
        "seasonal": "seasonal_naive",
        "regression": "trend_regression",
        "linear_regression": "trend_regression",
        "causal": "causal_regression",
    }
    key = aliases.get(key, key)
    valid = {
        "naive",
        "moving_average",
        "exponential_smoothing",
        "seasonal_naive",
        "trend_regression",
        "causal_regression",
    }
    return key if key in valid else None  # type: ignore[return-value]


def _candidate_methods(
    *,
    n: int,
    seasonality: dict[str, Any],
    has_exog: bool,
    forced: MethodName | None,
) -> list[MethodName]:
    """Build the auto-comparison set. Causal regression is forced-method only."""
    del has_exog  # reserved: auto-causal with held-constant siblings is misleading
    if forced:
        return [forced]
    methods: list[MethodName] = ["naive", "moving_average", "exponential_smoothing", "trend_regression"]
    if seasonality.get("detected") and seasonality.get("period") and n >= int(seasonality["period"]) * 2:
        methods.append("seasonal_naive")
    return methods


def _evaluate_candidates(
    y: np.ndarray,
    candidates: list[MethodName],
    seasonality: dict[str, Any],
    *,
    exog: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(y)
    holdout = max(2, min(max(3, n // 5), n // 3))
    if n - holdout < 4:
        holdout = max(1, n // 4)
    train_y = y[:-holdout]
    test_y = y[-holdout:]
    train_exog = exog.iloc[:-holdout] if exog is not None else None
    test_exog = exog.iloc[-holdout:] if exog is not None else None
    period = int(seasonality.get("period") or 0)

    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}
    for method in candidates:
        try:
            pred, _, _ = _generate_forecast(
                train_y,
                method=method,
                horizon=len(test_y),
                seasonal_period=period,
                exog=train_exog,
                future_exog=test_exog,
            )
            metrics = _error_metrics(test_y, pred)
            row = {"method": method, **metrics, "holdout_size": int(len(test_y))}
            rows.append(row)
            if method == "naive":
                baseline = row
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": method,
                    "mae": None,
                    "rmse": None,
                    "mape": None,
                    "error": str(exc),
                    "holdout_size": int(len(test_y)),
                }
            )
    rows = [r for r in rows if r.get("rmse") is not None]
    if not baseline:
        # Ensure baseline exists even if naive failed oddly
        try:
            pred, _, _ = _generate_forecast(
                train_y, method="naive", horizon=len(test_y), seasonal_period=period
            )
            baseline = {"method": "naive", **_error_metrics(test_y, pred), "holdout_size": int(len(test_y))}
        except Exception:  # noqa: BLE001
            baseline = {}
    return rows, baseline


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # MAPE only where |actual| is meaningfully above zero
    mask = np.abs(actual) > 1e-8
    if mask.any():
        mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100.0)
    else:
        mape = None
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _generate_forecast(
    y: np.ndarray,
    *,
    method: str,
    horizon: int,
    seasonal_period: int = 0,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    residual_scale = float(np.std(np.diff(y))) if len(y) > 1 else float(np.std(y) or 1.0)
    residual_scale = residual_scale if residual_scale > 0 else float(np.std(y) or 1.0)

    if method == "naive":
        fc = np.full(horizon, y[-1], dtype=float)
    elif method == "moving_average":
        window = max(2, min(6, len(y) // 3 or 2))
        fc = np.full(horizon, float(np.mean(y[-window:])), dtype=float)
    elif method == "exponential_smoothing":
        fc = _ses_forecast(y, horizon)
    elif method == "seasonal_naive":
        period = max(2, seasonal_period or 2)
        fc = np.array([y[-period + (i % period)] for i in range(horizon)], dtype=float)
    elif method == "trend_regression":
        fc = _regression_forecast(y, horizon, exog=None, future_exog=None)
    elif method == "causal_regression":
        fc = _regression_forecast(y, horizon, exog=exog, future_exog=future_exog)
    else:
        raise ValueError(f"Unknown forecasting method: {method}")

    # Simple residual-based intervals (not statistical PI; labeled as such)
    z = 1.96
    steps = np.sqrt(np.arange(1, horizon + 1))
    lower = fc - z * residual_scale * steps
    upper = fc + z * residual_scale * steps
    return fc, lower, upper


def _ses_forecast(y: np.ndarray, horizon: int, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        # Grid-search alpha on one-step in-sample SSE
        best_a, best_sse = 0.3, math.inf
        for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            level = y[0]
            sse = 0.0
            for t in range(1, len(y)):
                pred = level
                sse += (y[t] - pred) ** 2
                level = a * y[t] + (1 - a) * level
            if sse < best_sse:
                best_sse, best_a = sse, a
        alpha = best_a
    level = y[0]
    for t in range(1, len(y)):
        level = alpha * y[t] + (1 - alpha) * level
    return np.full(horizon, float(level), dtype=float)


def _regression_forecast(
    y: np.ndarray,
    horizon: int,
    *,
    exog: pd.DataFrame | None,
    future_exog: pd.DataFrame | None,
) -> np.ndarray:
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    if exog is not None and len(exog) == n:
        X = np.hstack([t, exog.to_numpy(dtype=float)])
    else:
        X = t
    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(n, n + horizon).reshape(-1, 1)
    if exog is not None and len(exog) == n:
        if future_exog is not None and len(future_exog) == horizon:
            future_x = future_exog.to_numpy(dtype=float)
        else:
            last = exog.to_numpy(dtype=float)[-1]
            future_x = np.tile(last, (horizon, 1))
        Xf = np.hstack([future_t, future_x])
    else:
        Xf = future_t
    return model.predict(Xf).astype(float)


def _future_timestamps(times: pd.Series, frequency: str, horizon: int) -> list[Any]:
    ts = pd.to_datetime(times, errors="coerce")
    if ts.notna().all():
        last = ts.iloc[-1]
        freq_alias = {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
            "quarterly": "QS",
            "yearly": "YS",
            "irregular": None,
            "unknown": None,
        }.get(frequency)
        if freq_alias:
            # Start after last timestamp
            rng = pd.date_range(last, periods=horizon + 1, freq=freq_alias)[1:]
            return list(rng)
        # Fallback: median delta
        deltas = ts.sort_values().diff().dropna()
        step = deltas.median() if not deltas.empty else pd.Timedelta(days=1)
        return [last + step * (i + 1) for i in range(horizon)]

    # Non-datetime (should be rare after parsing)
    last = times.iloc[-1]
    try:
        base = float(last)
        return [base + i + 1 for i in range(horizon)]
    except (TypeError, ValueError):
        return [f"t+{i+1}" for i in range(horizon)]


def _convert_horizon(n: int, from_freq: str, to_freq: str) -> int:
    """Convert a horizon expressed in one frequency into dataset frequency steps."""
    if from_freq == to_freq or to_freq in {"unknown", "irregular"}:
        return n
    # Approximate conversion into target steps
    days = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365,
    }
    if from_freq not in days or to_freq not in days:
        return n
    total_days = n * days[from_freq]
    steps = max(1, int(round(total_days / days[to_freq])))
    return steps


def _should_resample(native: str, requested: str) -> bool:
    """True when the user asked for a coarser grain than the native series."""
    if requested not in _FREQ_RANK:
        return False
    if native in {"unknown", "irregular"}:
        return True
    return _FREQ_RANK.get(requested, 0) > _FREQ_RANK.get(native, 0)


def _resample_to_frequency(
    grouped: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    frequency: str,
    aggregate: str,
) -> tuple[pd.DataFrame | None, str]:
    """Roll a finer time series up to weekly/monthly/quarterly/yearly periods."""
    rule = {
        "weekly": "W-MON",
        "monthly": "MS",
        "quarterly": "QS",
        "yearly": "YS",
    }.get(frequency)
    if not rule:
        return grouped, f"Could not resample to '{frequency}'; keeping native grain."

    work = grouped.copy()
    work["_ts"] = pd.to_datetime(work["_ts"], errors="coerce")
    work = work.dropna(subset=["_ts"]).sort_values("_ts")
    if work.empty:
        return None, "Resample produced an empty series."

    work = work.set_index("_ts")
    agg_map: dict[str, str] = {target: aggregate if aggregate in {"sum", "mean", "median", "last"} else "sum"}
    for col in predictors:
        if col in work.columns and col != target:
            agg_map[col] = "mean"
    # Keep only columns we know how to aggregate
    keep = [c for c in agg_map if c in work.columns]
    if target not in keep:
        return None, f"Target '{target}' missing during resample."
    resampled = work[keep].resample(rule).agg({c: agg_map[c] for c in keep})
    resampled = resampled.dropna(subset=[target]).reset_index()
    note = (
        f"Resampled series to {frequency} periods ({rule}) using '{agg_map[target]}' "
        f"for '{target}' to match the requested forecast grain."
    )
    return resampled, note
