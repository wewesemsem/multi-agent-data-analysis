"""Schema-aware NL → column resolution.

Maps user metric language onto whatever columns exist in the active schema.
No dataset-specific column names are preferred; scoring uses token overlap,
synonym families, and measure granularity (aggregate vs unit-level).
"""

from __future__ import annotations

import re
from typing import Iterable

# Synonym families: tokens that refer to the same *kind* of measure.
# Used only for scoring — never as a hard-coded preferred column list.
_METRIC_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "revenue",
            "sales",
            "income",
            "turnover",
            "proceeds",
            "receipts",
            "amount",
            "total",
            "value",
            "sum",
            "gross",
            "net",
            "takings",
        }
    ),
    frozenset({"price", "cost", "fee", "tariff", "rate"}),
    frozenset({"quantity", "qty", "units", "volume", "count", "throughput"}),
    frozenset({"profit", "margin", "earnings", "ebitda", "ebit"}),
    frozenset({"debt", "liability", "liabilities", "leverage"}),
    frozenset({"cash", "liquidity", "balance"}),
    frozenset({"customers", "users", "accounts", "subscribers"}),
)

_UNIT_LEVEL_MARKERS = (
    "unit_",
    "per_",
    "_per_",
    "unitprice",
    "unit_price",
    "avg_price",
    "average_price",
)
_AGGREGATE_MARKERS = ("total", "sum", "gross", "net", "aggregate", "overall")

_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "for",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "next",
        "over",
        "with",
        "from",
        "into",
        "by",
        "is",
        "are",
        "be",
        "me",
        "my",
        "our",
        "please",
        "show",
        "tell",
        "give",
        "create",
        "make",
        "run",
        "do",
        "how",
        "what",
        "which",
        "when",
        "where",
        "forecast",
        "predict",
        "prediction",
        "project",
        "projection",
        "estimate",
        "analyze",
        "analyse",
        "analysis",
        "chart",
        "plot",
        "graph",
        "visualize",
        "visualise",
        "month",
        "months",
        "year",
        "years",
        "day",
        "days",
        "week",
        "weeks",
        "quarter",
        "quarters",
        "period",
        "periods",
        "future",
        "look",
        "like",
        "will",
        "would",
        "should",
        "could",
        "can",
        "data",
        "dataset",
        "column",
        "metric",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t and t not in _STOP]


def column_tokens(column: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (column or "").lower()) if t}


def extract_metric_mentions(text: str, columns: Iterable[str]) -> list[str]:
    """Pull metric-like mentions from NL, preferring explicit schema hits."""
    cols = list(columns)
    q = text or ""
    q_lower = q.lower()
    found: list[str] = []

    for col in cols:
        if re.search(rf"\b{re.escape(col.lower())}\b", q_lower):
            found.append(col.lower())

    for pat in (
        r"(?:forecast|predict|project|estimate|analyze|analyse)\s+([a-zA-Z_][\w]*)",
        r"(?:of|for)\s+([a-zA-Z_][\w]*)\s+(?:for the next|over the next|next)",
        r"\b([a-zA-Z_][\w]*)\s+(?:forecast|prediction|projection)\b",
    ):
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            tok = m.group(1).lower()
            if tok not in _STOP and tok not in found:
                found.append(tok)

    # Fallback: non-stop content tokens (still schema-scored later)
    if not found:
        for tok in tokenize(q):
            if tok.isalpha() and len(tok) > 2 and tok not in found:
                found.append(tok)
    return found


def _family_overlap(user_token: str, col_token_set: set[str], column_lower: str) -> bool:
    for fam in _METRIC_FAMILIES:
        user_hit = user_token in fam or any(
            len(t) > 2 and (user_token.startswith(t) or t.startswith(user_token)) for t in fam
        )
        if not user_hit:
            continue
        if col_token_set & fam:
            return True
        if any(t in column_lower for t in fam if len(t) > 2):
            return True
    return False


def _is_unit_level(column: str) -> bool:
    cl = column.lower().replace(" ", "_")
    if any(m in cl for m in _UNIT_LEVEL_MARKERS):
        return True
    tokens = column_tokens(column)
    # "price" alone (or with unit) is usually a unit measure, not a flow total
    if "price" in tokens and "total" not in tokens:
        return True
    return False


def _is_unit_level_request(token: str) -> bool:
    t = (token or "").lower()
    return t in {"price", "cost", "fee", "rate", "tariff", "unit", "unit_price"} or t.startswith(
        "unit"
    )


def _is_aggregate_level(column: str) -> bool:
    cl = column.lower()
    tokens = column_tokens(column)
    return any(m in cl for m in _AGGREGATE_MARKERS) or bool(tokens & {"total", "sum", "amount"})


def score_column_for_metrics(column: str, metric_tokens: Iterable[str]) -> float:
    """Higher = better match between NL metric tokens and a schema column."""
    metrics = [m.lower() for m in metric_tokens if m]
    if not metrics:
        return 0.0

    col_l = column.lower()
    col_toks = column_tokens(column)
    score = 0.0

    for mt in metrics:
        if mt == col_l:
            score += 8.0
        elif mt in col_toks:
            score += 5.0
        elif mt in col_l:
            score += 3.5
        if _family_overlap(mt, col_toks, col_l):
            score += 3.0

    # Prefer aggregate/flow columns when the user did not ask for a unit measure
    asked_unit = any(_is_unit_level_request(m) for m in metrics)
    if _is_aggregate_level(column) and not asked_unit:
        score += 1.25
    if _is_unit_level(column) and not asked_unit:
        score -= 2.5
    if asked_unit and _is_unit_level(column):
        score += 1.5

    return score


def rank_columns_for_request(
    request: str,
    columns: Iterable[str],
    *,
    min_score: float = 1.0,
) -> list[tuple[str, float]]:
    """Rank schema columns by relevance to the user request."""
    cols = [c for c in columns if c]
    mentions = extract_metric_mentions(request, cols)
    if not mentions:
        mentions = tokenize(request)
    scored = [(c, score_column_for_metrics(c, mentions)) for c in cols]
    scored = [(c, s) for c, s in scored if s >= min_score]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def resolve_metric_columns(
    request: str,
    columns: Iterable[str],
    *,
    limit: int = 5,
) -> list[str]:
    """Best-matching schema columns for a natural-language metric request."""
    ranked = rank_columns_for_request(request, columns)
    return [c for c, _ in ranked[:limit]]
