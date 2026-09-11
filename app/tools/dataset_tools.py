"""Deterministic dataset tools — LLM never hallucinates rows."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from app.state import DATASETS_DIR, ensure_workspace, utc_now


CATEGORY_DEFAULTS = [
    "Electronics",
    "Clothing",
    "Home & Kitchen",
    "Sports",
    "Beauty",
    "Books",
    "Toys",
    "Grocery",
]

US_STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "dataset"


def create_ecommerce_orders(
    *,
    n_rows: int = 10_000,
    seed: int = 42,
    name: str = "ecommerce_orders",
    extra_columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a synthetic e-commerce orders dataset with deterministic RNG."""
    ensure_workspace()
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    n_rows = max(1, min(int(n_rows), 100_000))  # MVP cap
    categories = CATEGORY_DEFAULTS
    cat_idx = rng.integers(0, len(categories), size=n_rows)
    category = np.array(categories)[cat_idx]

    # Price ranges by category — intentional skew so anomalies are detectable
    base_prices = {
        "Electronics": (40, 1200),
        "Clothing": (15, 180),
        "Home & Kitchen": (20, 400),
        "Sports": (20, 350),
        "Beauty": (8, 120),
        "Books": (5, 60),
        "Toys": (10, 150),
        "Grocery": (5, 80),
    }
    amounts = np.empty(n_rows, dtype=float)
    for i, cat in enumerate(categories):
        mask = category == cat
        lo, hi = base_prices[cat]
        amounts[mask] = rng.uniform(lo, hi, size=mask.sum())

    # Inject ~1.5% extreme outliers for anomaly detection demos
    n_outliers = max(1, int(n_rows * 0.015))
    outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    amounts[outlier_idx] = amounts[outlier_idx] * rng.uniform(8, 25, size=n_outliers)

    quantities = rng.integers(1, 6, size=n_rows)
    start = np.datetime64("2024-01-01")
    days = rng.integers(0, 540, size=n_rows)
    order_dates = start + days.astype("timedelta64[D]")

    customer_ids = [f"CUST-{rng.integers(1000, 9999)}" for _ in range(n_rows)]
    states = rng.choice(US_STATES, size=n_rows, p=_state_weights(len(US_STATES), rng))

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{i+1:06d}" for i in range(n_rows)],
            "customer_id": customer_ids,
            "order_date": pd.to_datetime(order_dates),
            "product_category": category,
            "product_name": [fake.catch_phrase() for _ in range(n_rows)],
            "quantity": quantities,
            "unit_price": np.round(amounts, 2),
            "total_amount": np.round(amounts * quantities, 2),
            "state": states,
            "payment_method": rng.choice(
                ["credit_card", "debit_card", "paypal", "apple_pay"], size=n_rows
            ),
            "is_returned": rng.choice([False, True], size=n_rows, p=[0.92, 0.08]),
        }
    )

    if extra_columns:
        normalized = _normalize_columns_spec(extra_columns) or {}
        for col, spec in normalized.items():
            if col in df.columns:
                continue
            if not isinstance(spec, dict):
                spec = {"type": str(spec) if spec else "string"}
            dtype = (spec or {}).get("type", "string")
            if dtype in ("int", "integer"):
                df[col] = rng.integers(0, 100, size=n_rows)
            elif dtype in ("float", "number"):
                df[col] = np.round(rng.uniform(0, 100, size=n_rows), 2)
            else:
                df[col] = [fake.word() for _ in range(n_rows)]

    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    path = DATASETS_DIR / f"{dataset_id}.parquet"
    df.to_parquet(path, index=False)

    schema = {c: str(df[c].dtype) for c in df.columns}
    profile = profile_dataframe(df)

    meta = {
        "id": dataset_id,
        "name": name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "schema": schema,
        "location": str(path),
        "format": "parquet",
        "created_at": utc_now(),
        "seed": seed,
        "profile": profile,
        "source": "synthetic_generator",
    }
    meta_path = DATASETS_DIR / f"{dataset_id}.meta.json"
    meta_path.write_text(json.dumps(meta, default=str, indent=2))
    return meta


def _state_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    # Mild geographic skew (CA/TX/NY heavier)
    w = np.ones(n)
    w[0] = 3.0  # CA
    w[1] = 2.5  # TX
    w[3] = 2.2  # NY
    w = w / w.sum()
    return w


def load_csv(path: str | Path, *, name: str | None = None) -> dict[str, Any]:
    ensure_workspace()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    out = DATASETS_DIR / f"{dataset_id}.parquet"
    df.to_parquet(out, index=False)
    schema = {c: str(df[c].dtype) for c in df.columns}
    meta = {
        "id": dataset_id,
        "name": name or path.stem,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "schema": schema,
        "location": str(out),
        "format": "parquet",
        "created_at": utc_now(),
        "source": "csv_upload",
        "original_path": str(path),
        "profile": profile_dataframe(df),
    }
    (DATASETS_DIR / f"{dataset_id}.meta.json").write_text(json.dumps(meta, default=str, indent=2))
    return meta


def load_dataset(meta: dict[str, Any]) -> pd.DataFrame:
    path = Path(meta["location"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file missing: {path}")
    return pd.read_parquet(path)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    stats: dict[str, Any] = {"numeric": {}, "categorical_top": {}, "null_counts": {}}
    for col in numeric:
        s = df[col]
        stats["numeric"][col] = {
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
        }
    for col in categorical[:12]:
        vc = df[col].astype(str).value_counts().head(8)
        stats["categorical_top"][col] = {str(k): int(v) for k, v in vc.items()}
    stats["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    stats["numeric_columns"] = numeric
    stats["categorical_columns"] = categorical
    stats["datetime_columns"] = datetime_cols
    return stats


def _normalize_columns_spec(columns: Any) -> dict[str, Any] | None:
    """Accept dict or list column specs from the LLM; always return dict[name] -> spec."""
    if not columns:
        return None
    if isinstance(columns, dict):
        return columns
    if isinstance(columns, list):
        out: dict[str, Any] = {}
        for item in columns:
            if isinstance(item, str):
                out[item] = {"type": "string"}
            elif isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("field")
                if not name:
                    continue
                col_spec = {k: v for k, v in item.items() if k not in {"name", "column", "field"}}
                if "type" not in col_spec:
                    col_spec["type"] = "string"
                out[str(name)] = col_spec
        return out or None
    return None


def generate_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Create a dataset from an LLM-produced structured specification.

    Spec shape:
      {
        "name": "...",
        "n_rows": 10000,
        "template": "ecommerce_orders" | "generic",
        "seed": 42,
        "columns": {"col": {"type": "float"}}  # or list[{name, type}] from LLM
      }
    """
    template = (spec.get("template") or "ecommerce_orders").lower()
    n_rows = int(spec.get("n_rows") or spec.get("rows") or 1000)
    seed = int(spec.get("seed") or 42)
    name = spec.get("name") or _slug(template)
    columns = _normalize_columns_spec(spec.get("columns"))

    if "e-commerce" in template or "ecommerce" in template or "order" in template:
        # Ecommerce generator already has a full schema; only pass true extras.
        builtin = {
            "order_id",
            "customer_id",
            "order_date",
            "product_category",
            "product_name",
            "quantity",
            "unit_price",
            "total_amount",
            "state",
            "payment_method",
            "is_returned",
        }
        extras = {k: v for k, v in (columns or {}).items() if k not in builtin} or None
        return create_ecommerce_orders(
            n_rows=n_rows,
            seed=seed,
            name=name,
            extra_columns=extras,
        )

    # Generic fallback: simple tabular data
    ensure_workspace()
    rng = np.random.default_rng(seed)
    columns = columns or {
        "id": {"type": "int"},
        "value": {"type": "float"},
        "category": {"type": "string"},
    }
    data: dict[str, Any] = {}
    for col, col_spec in columns.items():
        if not isinstance(col_spec, dict):
            col_spec = {"type": str(col_spec) if col_spec else "string"}
        dtype = (col_spec or {}).get("type", "string")
        if dtype in ("int", "integer"):
            data[col] = rng.integers(0, 10_000, size=n_rows)
        elif dtype in ("float", "number"):
            data[col] = np.round(rng.normal(100, 25, size=n_rows), 2)
        elif dtype in ("bool", "boolean"):
            data[col] = rng.choice([True, False], size=n_rows)
        else:
            cats = (col_spec or {}).get("categories") or ["A", "B", "C", "D"]
            data[col] = rng.choice(cats, size=n_rows)
    df = pd.DataFrame(data)
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    path = DATASETS_DIR / f"{dataset_id}.parquet"
    df.to_parquet(path, index=False)
    meta = {
        "id": dataset_id,
        "name": name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "schema": {c: str(df[c].dtype) for c in df.columns},
        "location": str(path),
        "format": "parquet",
        "created_at": utc_now(),
        "seed": seed,
        "profile": profile_dataframe(df),
        "source": "synthetic_generator",
        "spec": spec,
    }
    (DATASETS_DIR / f"{dataset_id}.meta.json").write_text(json.dumps(meta, default=str, indent=2))
    return meta
