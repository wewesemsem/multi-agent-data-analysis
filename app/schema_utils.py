"""Schema-aware column selection — no domain-specific (e.g. ecommerce) defaults."""

from __future__ import annotations

from typing import Any


def _dtype_str(schema: dict[str, Any], col: str) -> str:
    return str(schema.get(col, "")).lower()


def is_numeric_dtype(dtype: str) -> bool:
    d = dtype.lower()
    return any(k in d for k in ("int", "float", "double", "decimal", "number")) and "bool" not in d


def is_datetime_dtype(dtype: str) -> bool:
    d = dtype.lower()
    return any(k in d for k in ("date", "time", "datetime", "timestamp"))


def is_categorical_dtype(dtype: str) -> bool:
    d = dtype.lower()
    if is_numeric_dtype(d) or is_datetime_dtype(d):
        return False
    return any(k in d for k in ("object", "string", "str", "category", "bool")) or not d


def numeric_columns(schema: dict[str, Any]) -> list[str]:
    return [c for c in schema if is_numeric_dtype(_dtype_str(schema, c))]


def datetime_columns(schema: dict[str, Any]) -> list[str]:
    return [c for c in schema if is_datetime_dtype(_dtype_str(schema, c))]


def categorical_columns(schema: dict[str, Any]) -> list[str]:
    return [c for c in schema if is_categorical_dtype(_dtype_str(schema, c))]


def default_metric_column(schema: dict[str, Any], cols: list[str] | None = None) -> str | None:
    """First numeric column in schema order (live-safe default)."""
    cols = cols if cols is not None else list(schema.keys())
    for c in cols:
        if c in schema and is_numeric_dtype(_dtype_str(schema, c)):
            return c
    nums = numeric_columns(schema)
    return nums[0] if nums else None


def default_group_column(schema: dict[str, Any], cols: list[str] | None = None) -> str | None:
    """First categorical / string column suitable for grouping."""
    cols = cols if cols is not None else list(schema.keys())
    for c in cols:
        if c in schema and is_categorical_dtype(_dtype_str(schema, c)):
            # Prefer non-id-looking labels when several exist
            cl = c.lower()
            if cl.endswith("_id") or cl == "id":
                continue
            return c
    cats = [c for c in categorical_columns(schema) if not c.lower().endswith("_id") and c.lower() != "id"]
    if cats:
        return cats[0]
    cats = categorical_columns(schema)
    return cats[0] if cats else None


def default_id_column(schema: dict[str, Any], cols: list[str] | None = None) -> str | None:
    """Prefer an identifier-like column for distinct counts; else first categorical."""
    cols = cols if cols is not None else list(schema.keys())
    for c in cols:
        cl = c.lower()
        if cl == "id" or cl.endswith("_id"):
            return c
    return default_group_column(schema, cols)


def default_time_column(schema: dict[str, Any], cols: list[str] | None = None) -> str | None:
    cols = cols if cols is not None else list(schema.keys())
    for c in cols:
        if c in schema and is_datetime_dtype(_dtype_str(schema, c)):
            return c
    times = datetime_columns(schema)
    if times:
        return times[0]
    # Name-based fallback when dtype is object but column is clearly temporal
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ("date", "time", "timestamp", "datetime")):
            return c
    return None


def match_column(cols: list[str], *candidates: str) -> str | None:
    """Match a column by exact or substring name (user/LLM aliases), not domain defaults."""
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if not cand:
            continue
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand and cand.lower() in cl:
                return c
    return None
