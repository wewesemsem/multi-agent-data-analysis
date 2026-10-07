"""Deterministic drafting helpers — evidence selection, artifact shape, grounding checks.

LLMs may narrate drafts; numerical claims must come from SharedWorkspace artifacts.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


DRAFT_TYPES = frozenset(
    {
        "summary",
        "report",
        "financial_report",
        "forecast_report",
        "risk_report",
        "creditworthiness_assessment",
        "memo",
        "recommendation",
        "analysis_explanation",
        "decision_brief",
        "anomaly_summary",
        "forecast_explanation",
        "scenario_summary",
    }
)


@dataclass
class DraftArtifact:
    """Generic structured document produced by the Drafting Agent."""

    id: str
    type: str
    title: str
    content: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    source_artifacts: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    grounded: bool = True
    evidence_sufficient: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "content": self.content,
            "sections": self.sections,
            "tables": self.tables,
            "charts": self.charts,
            "source_artifacts": self.source_artifacts,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "metadata": self.metadata,
            "grounded": self.grounded,
            "evidence_sufficient": self.evidence_sufficient,
        }


def infer_draft_type(user_request: str) -> str:
    """Infer artifact type from natural language (not a hard schema)."""
    q = (user_request or "").lower()
    if any(
        k in q
        for k in (
            "creditworthiness",
            "credit worthiness",
            "credit risk",
            "credit assessment",
            "credit grade",
            "credit rating",
        )
    ):
        return "creditworthiness_assessment"
    if "financial health" in q:
        return "creditworthiness_assessment"
    if any(k in q for k in ("investment memo", "financial memo", "decision memo", "memo for")):
        return "memo"
    if "decision brief" in q or "decision-brief" in q:
        return "decision_brief"
    if any(k in q for k in ("recommend", "action plan", "next step", "next steps")):
        return "recommendation"
    if any(k in q for k in ("anomaly summary", "summarize the anomal", "summarize anomal")):
        return "anomaly_summary"
    if any(
        k in q
        for k in (
            "explain the forecast",
            "forecast explanation",
            "explain this forecast",
            "explain the projection",
        )
    ):
        return "forecast_explanation"
    if "forecast report" in q:
        return "forecast_report"
    if "scenario summary" in q or "scenario" in q and "summar" in q:
        return "scenario_summary"
    if any(k in q for k in ("risk report", "risk outlook", "risk summary")):
        return "risk_report"
    if any(
        k in q
        for k in (
            "financial report",
            "financial analysis report",
            "financial performance",
            "business performance",
            "management report",
        )
    ):
        return "financial_report"
    if any(k in q for k in ("explain the chart", "explain the analysis", "explain these findings")):
        return "analysis_explanation"
    if "executive summary" in q:
        return "summary"
    if any(k in q for k in ("detailed report", "write a report", "analytical report", "research summary")):
        return "report"
    if "summar" in q:
        return "summary"
    if "memo" in q:
        return "memo"
    if "report" in q:
        return "report"
    # Bare "assessment" is not enough to force a creditworthiness artifact.
    return "summary"


def request_needs_draft(user_request: str) -> bool:
    q = (user_request or "").lower()
    markers = (
        "summar",
        "executive summary",
        "write a report",
        "detailed report",
        "analytical report",
        "financial report",
        "financial analysis report",
        "business performance",
        "management report",
        "research summary",
        "risk report",
        "risk outlook",
        "forecast report",
        "explain the forecast",
        "forecast explanation",
        "explain the result",
        "explain this forecast",
        "scenario summary",
        "creditworthiness",
        "credit worthiness",
        "credit risk",
        "credit assessment",
        "credit grade",
        "credit rating",
        "financial health",
        "anomaly summary",
        "recommend",
        "recommendation",
        "decision memo",
        "investment memo",
        "financial memo",
        "decision brief",
        "action plan",
        "next step",
        "memo for",
        "turn these findings",
        "draft a",
        "write me a",
        "give me a summary",
        "give me an assessment",
        "assessment for",
        "explain the chart",
        "explain the analysis",
        "explain these findings",
    )
    return any(m in q for m in markers)


def draft_needs_supporting_analysis(user_request: str) -> bool:
    """True when a draft request typically needs computed evidence if none exists yet."""
    q = (user_request or "").lower()
    if any(
        k in q
        for k in (
            "creditworthiness",
            "credit worthiness",
            "credit risk",
            "credit assessment",
            "credit grade",
            "financial health",
            "financial report",
            "financial analysis",
            "financial performance",
            "business performance",
            "management report",
            "investment memo",
            "risk report",
            "detailed report",
            "analytical report",
        )
    ):
        return True
    # Pure summarize / explain / turn-findings should not force new analysis.
    return False


def build_evidence_pack(
    workspace: Any,
    *,
    user_request: str,
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select targeted evidence from the workspace for drafting (not a full dump)."""
    draft_type = infer_draft_type(user_request)
    ds = workspace.dataset
    dataset_meta: dict[str, Any] | None = None
    if ds:
        dataset_meta = {
            "id": ds.get("id"),
            "name": ds.get("name"),
            "row_count": ds.get("row_count"),
            "schema": list((ds.get("schema") or {}).keys()),
            "source": ds.get("source"),
        }

    analyses = [_slim_analysis(a, i) for i, a in enumerate(workspace.analysis_results or [])]
    forecasts = [_slim_forecast(f, i) for i, f in enumerate(workspace.forecasts or [])]
    anomalies = [_slim_anomaly(a, i) for i, a in enumerate(workspace.anomalies or [])]
    visualizations = [_slim_viz(v, i) for i, v in enumerate(workspace.visualizations or [])]
    validations = [
        {
            "check": r.get("check"),
            "ok": r.get("ok"),
            "issues": list(r.get("issues") or [])[:8],
        }
        for r in (workspace.validation_reports or [])[-5:]
    ]

    # Prefer recent / relevant slices for large workspaces
    if draft_type in {"forecast_explanation", "forecast_report", "scenario_summary"}:
        analyses = analyses[-2:]
        forecasts = forecasts[-3:]
        anomalies = anomalies[-1:]
        visualizations = [v for v in visualizations if v.get("chart_type") == "forecast"] or visualizations[-2:]
    elif draft_type == "anomaly_summary":
        analyses = analyses[-1:]
        forecasts = []
        anomalies = anomalies[-3:]
        visualizations = visualizations[-1:]
    elif draft_type in {"summary", "analysis_explanation"}:
        analyses = analyses[-3:]
        forecasts = forecasts[-2:]
        anomalies = anomalies[-2:]
        visualizations = visualizations[-3:]
    else:
        analyses = analyses[-4:]
        forecasts = forecasts[-3:]
        anomalies = anomalies[-3:]
        visualizations = visualizations[-4:]

    assumptions: list[str] = []
    warnings: list[str] = []
    findings: list[str] = []
    recommendations: list[str] = []

    ctx = conversation_context or {}
    for item in ctx.get("findings") or []:
        if isinstance(item, str) and item.strip():
            findings.append(item.strip())
    for item in ctx.get("assumptions") or []:
        if isinstance(item, str) and item.strip():
            assumptions.append(item.strip())
    for item in ctx.get("decisions") or []:
        if isinstance(item, str) and item.strip():
            recommendations.append(f"User decision: {item.strip()}")

    for fc in forecasts:
        for a in fc.get("assumptions") or []:
            assumptions.append(str(a))
        for w in fc.get("warnings") or []:
            warnings.append(str(w))

    for analysis in analyses:
        expl = (analysis.get("explanation") or "").strip()
        if expl:
            findings.append(expl[:240])

    for anomaly in anomalies:
        findings.append(
            f"Anomalies on `{anomaly.get('column')}` via {anomaly.get('method')}: "
            f"{anomaly.get('n_anomalies')} flagged"
        )

    source_artifacts: list[dict[str, Any]] = []
    for a in analyses:
        source_artifacts.append({"kind": "analysis", "index": a["index"]})
    for f in forecasts:
        source_artifacts.append({"kind": "forecast", "index": f["index"], "id": f.get("id")})
    for a in anomalies:
        source_artifacts.append({"kind": "anomaly", "index": a["index"]})
    for v in visualizations:
        source_artifacts.append({"kind": "visualization", "index": v["index"], "id": v.get("id")})

    sufficient, missing = assess_evidence_sufficiency(draft_type, analyses, forecasts, anomalies, dataset_meta)

    return {
        "user_request": user_request,
        "draft_type": draft_type,
        "conversation_context": {
            "current_goal": ctx.get("current_goal"),
            "active_entities": ctx.get("active_entities") or {},
            "conversation_summary": ctx.get("conversation_summary"),
            "decisions": list(ctx.get("decisions") or [])[:8],
            "unresolved_questions": list(ctx.get("unresolved_questions") or [])[:8],
        },
        "dataset": dataset_meta,
        "analysis_results": analyses,
        "forecasts": forecasts,
        "anomalies": anomalies,
        "visualizations": visualizations,
        "validation_reports": validations,
        "findings": findings[:12],
        "assumptions": _unique(assumptions)[:12],
        "warnings": _unique(warnings)[:12],
        "recommendations": recommendations[:8],
        "source_artifacts": source_artifacts,
        "evidence_sufficient": sufficient,
        "missing_evidence": missing,
        "evidence_numbers": sorted(
            collect_evidence_numbers_from_pack_parts(
                analyses, forecasts, anomalies, dataset_meta, visualizations
            )
        ),
    }


def assess_evidence_sufficiency(
    draft_type: str,
    analyses: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    dataset_meta: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    has_any = bool(analyses or forecasts or anomalies)
    if draft_type in {"forecast_explanation", "forecast_report", "scenario_summary"}:
        suitable = [f for f in forecasts if f.get("suitable")]
        if not suitable:
            missing.append("No validated suitable forecast results are available.")
        return (bool(suitable), missing)
    if draft_type == "anomaly_summary":
        if not anomalies:
            missing.append("No validated anomaly results are available.")
        return (bool(anomalies), missing)
    if draft_type == "creditworthiness_assessment":
        if not analyses and not forecasts:
            missing.append(
                "No validated analysis or forecast evidence is available for a defensible "
                "creditworthiness / credit risk assessment."
            )
        if not dataset_meta:
            missing.append("No dataset is loaded in the shared workspace.")
        # Partial evidence can still support a limited assessment
        sufficient = bool(analyses or forecasts) and dataset_meta is not None
        return (sufficient, missing)
    if draft_type in {"financial_report", "report", "memo", "decision_brief", "risk_report"}:
        if not has_any:
            missing.append("No validated analysis, forecast, or anomaly results are available.")
        return (has_any, missing)
    if not has_any:
        missing.append("No validated specialist outputs are available to synthesize.")
    return (has_any, missing)


def collect_evidence_numbers_from_pack_parts(
    analyses: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    dataset_meta: dict[str, Any] | None,
    visualizations: list[dict[str, Any]] | None = None,
) -> set[str]:
    blob_parts: list[str] = []
    if dataset_meta:
        blob_parts.append(str(dataset_meta))
    for item in analyses + forecasts + anomalies + list(visualizations or []):
        blob_parts.append(str(item))
    return extract_number_tokens("\n".join(blob_parts))


def extract_number_tokens(text: str) -> set[str]:
    """Normalize numeric tokens for grounding comparisons."""
    tokens: set[str] = set()
    for raw in re.findall(r"[-+]?\d[\d.,]*%?", text or ""):
        for val in _parse_numeric_token(raw):
            tokens.update(_norm_variants(val))
    return tokens


def _parse_numeric_token(raw: str) -> set[float]:
    """Return plausible numeric interpretations of a raw token.

    Handles US (1,234.56) and European (1.234,56 / 251,68) grouping/decimal marks.
    Ambiguous forms like 290,308 yield both thousands and decimal-comma readings so
    grounding can match either evidence representation.
    """
    s = (raw or "").rstrip("%").strip()
    if not s or s in {"+", "-"}:
        return set()

    values: set[float] = set()

    def _add(candidate: str) -> None:
        try:
            values.add(float(candidate))
        except ValueError:
            return

    # Plain / US: strip grouping commas, period as decimal
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s.replace(",", "")):
        _add(s.replace(",", ""))

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # European: 1.234,56
            _add(s.replace(".", "").replace(",", "."))
        else:
            # US: 1,234.56 (already covered by strip-commas, keep explicit)
            _add(s.replace(",", ""))
    elif "," in s and "." not in s:
        # Ambiguous: thousands (1,234 / 290,308) and/or decimal comma (251,68 / 290,308)
        _add(s.replace(",", ""))
        _add(s.replace(",", "."))
    elif "." in s and "," not in s:
        # Ambiguous European thousands vs US decimal: 1.234
        _add(s)
        if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+", s):
            _add(s.replace(".", ""))

    return values


def _norm_num(val: float) -> str:
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e15:
        return str(int(round(val)))
    return f"{val:.6g}"


def _norm_variants(val: float) -> set[str]:
    norms = {_norm_num(val)}
    if abs(val - round(val)) < 1e-9:
        norms.add(str(int(round(val))))
    return norms


def find_ungrounded_numbers(content: str, allowed: set[str]) -> list[str]:
    """Return numeric tokens in content that are not present in allowed evidence numbers."""
    if not content:
        return []
    bad: list[str] = []
    for raw in re.findall(r"[-+]?\d[\d.,]*%?", content):
        values = _parse_numeric_token(raw)
        if not values:
            continue
        norms: set[str] = set()
        for val in values:
            norms.update(_norm_variants(val))
        if norms & allowed:
            continue
        # Allow small integers used as list counts / section numbers (1-20)
        if any(abs(v) <= 20 and abs(v - round(v)) < 1e-9 for v in values):
            continue
        bad.append(raw)
    return bad


def validate_draft_payload(draft: dict[str, Any], workspace: Any) -> list[str]:
    """Deterministic grounding checks for a draft artifact."""
    issues: list[str] = []
    if not draft.get("grounded"):
        issues.append("Draft is not marked grounded.")
    if not draft.get("type"):
        issues.append("Draft missing type.")
    elif draft.get("type") not in DRAFT_TYPES:
        issues.append(f"Draft type '{draft.get('type')}' is not recognized.")
    if not draft.get("title"):
        issues.append("Draft missing title.")
    if not (draft.get("content") or "").strip():
        issues.append("Draft missing content.")
    if "sections" not in draft:
        issues.append("Draft missing sections list.")
    if "source_artifacts" not in draft:
        issues.append("Draft missing source_artifacts.")
    if "assumptions" not in draft or "warnings" not in draft:
        issues.append("Draft must preserve assumptions and warnings lists.")
    if "limitations" not in draft:
        issues.append("Draft missing limitations.")
    if "evidence_sufficient" not in draft:
        issues.append("Draft missing evidence_sufficient flag.")

    # Source references must point at existing workspace artifacts
    for ref in draft.get("source_artifacts") or []:
        kind = ref.get("kind")
        idx = ref.get("index")
        if kind == "analysis":
            if not isinstance(idx, int) or idx < 0 or idx >= len(workspace.analysis_results or []):
                issues.append(f"Draft cites missing analysis index {idx}.")
        elif kind == "forecast":
            if not isinstance(idx, int) or idx < 0 or idx >= len(workspace.forecasts or []):
                issues.append(f"Draft cites missing forecast index {idx}.")
        elif kind == "anomaly":
            if not isinstance(idx, int) or idx < 0 or idx >= len(workspace.anomalies or []):
                issues.append(f"Draft cites missing anomaly index {idx}.")
        elif kind == "visualization":
            if not isinstance(idx, int) or idx < 0 or idx >= len(workspace.visualizations or []):
                issues.append(f"Draft cites missing visualization index {idx}.")

    pack_nums = collect_evidence_numbers_from_pack_parts(
        [_slim_analysis(a, i) for i, a in enumerate(workspace.analysis_results or [])],
        [_slim_forecast(f, i) for i, f in enumerate(workspace.forecasts or [])],
        [_slim_anomaly(a, i) for i, a in enumerate(workspace.anomalies or [])],
        {
            "id": (workspace.dataset or {}).get("id"),
            "row_count": (workspace.dataset or {}).get("row_count"),
        }
        if workspace.dataset
        else None,
        [_slim_viz(v, i) for i, v in enumerate(workspace.visualizations or [])],
    )
    # Also allow numbers explicitly listed on the draft metadata evidence pack
    meta = draft.get("metadata") or {}
    for n in meta.get("evidence_numbers") or []:
        pack_nums.add(str(n))

    ungrounded = find_ungrounded_numbers(draft.get("content") or "", pack_nums)
    # If draft claims sufficiency but invents numbers, fail
    if ungrounded and draft.get("evidence_sufficient"):
        issues.append(
            "Draft contains numerical claims not present in workspace evidence: "
            + ", ".join(ungrounded[:8])
        )

    # Forecast assumptions/warnings should be preserved when forecasts were cited
    cited_forecasts = [r for r in (draft.get("source_artifacts") or []) if r.get("kind") == "forecast"]
    if cited_forecasts and (workspace.forecasts or []):
        expected_assumptions: list[str] = []
        expected_warnings: list[str] = []
        for ref in cited_forecasts:
            idx = ref.get("index")
            if isinstance(idx, int) and 0 <= idx < len(workspace.forecasts):
                fc = workspace.forecasts[idx]
                expected_assumptions.extend(str(a) for a in (fc.get("assumptions") or []))
                expected_warnings.extend(str(w) for w in (fc.get("warnings") or []))
        draft_assumptions = {str(a) for a in (draft.get("assumptions") or [])}
        draft_warnings = {str(w) for w in (draft.get("warnings") or [])}
        for a in expected_assumptions:
            if a and a not in draft_assumptions:
                issues.append("Draft dropped a forecast assumption from source evidence.")
                break
        for w in expected_warnings:
            if w and w not in draft_warnings:
                issues.append("Draft dropped a forecast warning from source evidence.")
                break

    # Insufficient-evidence credit assessments must acknowledge limitations
    if draft.get("type") == "creditworthiness_assessment" and draft.get("evidence_sufficient") is False:
        lim = " ".join(draft.get("limitations") or []).lower()
        content = (draft.get("content") or "").lower()
        if "insufficient" not in lim and "limited" not in lim and "insufficient" not in content and "limited" not in content:
            issues.append("Insufficient-evidence creditworthiness draft must acknowledge limitations.")

    return issues


def heuristic_draft(pack: dict[str, Any]) -> DraftArtifact:
    """Offline / fallback draft built only from the evidence pack."""
    draft_type = pack.get("draft_type") or "summary"
    title = _title_for(draft_type, pack)
    sections: list[dict[str, Any]] = []
    assumptions = list(pack.get("assumptions") or [])
    warnings = list(pack.get("warnings") or [])
    limitations = list(pack.get("missing_evidence") or [])
    sufficient = bool(pack.get("evidence_sufficient"))

    entities = (pack.get("conversation_context") or {}).get("active_entities") or {}
    company = entities.get("company")

    if not sufficient:
        sections.append(
            {
                "heading": "Limitations",
                "kind": "limitation",
                "body": _join_bullets(limitations)
                or "Available validated evidence is insufficient for the requested artifact.",
            }
        )
        if draft_type == "creditworthiness_assessment":
            sections.insert(
                0,
                {
                    "heading": "Overall Assessment",
                    "kind": "assessment",
                    "body": (
                        "A defensible creditworthiness / credit risk assessment cannot be completed "
                        "from the currently available validated evidence. "
                        "No formal credit score was produced."
                    ),
                }
            )
            if pack.get("dataset"):
                sections.append(
                    {
                        "heading": "Available Evidence",
                        "kind": "evidence",
                        "body": (
                            f"Dataset `{pack['dataset'].get('name')}` is loaded "
                            f"({pack['dataset'].get('row_count')} rows), but validated analysis/"
                            f"forecast outputs required for assessment are missing."
                        ),
                    }
                )
        else:
            sections.insert(
                0,
                {
                    "heading": "Summary",
                    "kind": "interpretation",
                    "body": (
                        "I can only synthesize outputs already produced by the specialist agents. "
                        "Required evidence for this request is missing or incomplete."
                    ),
                }
            )
    else:
        sections.extend(_sections_for_type(draft_type, pack, company))

    if assumptions:
        sections.append(
            {
                "heading": "Assumptions",
                "kind": "limitation",
                "body": _join_bullets(assumptions),
            }
        )
    if warnings:
        sections.append(
            {
                "heading": "Warnings",
                "kind": "limitation",
                "body": _join_bullets(warnings),
            }
        )
    if sufficient and limitations:
        sections.append(
            {
                "heading": "Limitations",
                "kind": "limitation",
                "body": _join_bullets(limitations),
            }
        )

    content = sections_to_markdown(title, sections, draft_type)
    return DraftArtifact(
        id=f"draft_{uuid.uuid4().hex[:10]}",
        type=draft_type if draft_type in DRAFT_TYPES else "summary",
        title=title,
        content=content,
        sections=sections,
        tables=[],
        charts=[v.get("id") for v in pack.get("visualizations") or [] if v.get("id")],
        source_artifacts=list(pack.get("source_artifacts") or []),
        assumptions=assumptions,
        warnings=warnings,
        limitations=limitations
        or (
            []
            if sufficient
            else ["Assessment/report limited by missing validated specialist outputs."]
        ),
        metadata={
            "user_request": pack.get("user_request"),
            "evidence_numbers": list(pack.get("evidence_numbers") or []),
            "generator": "heuristic",
        },
        grounded=True,
        evidence_sufficient=sufficient,
    )


def _sections_for_type(
    draft_type: str,
    pack: dict[str, Any],
    company: str | None,
) -> list[dict[str, Any]]:
    subject = company or "the active entity"
    analyses = pack.get("analysis_results") or []
    forecasts = pack.get("forecasts") or []
    anomalies = pack.get("anomalies") or []
    visualizations = pack.get("visualizations") or []
    findings = pack.get("findings") or []

    evidence_body = _evidence_block(analyses, forecasts, anomalies, visualizations)
    interpretation = (
        f"Based solely on validated workspace outputs for {subject}, "
        f"the specialist agents produced {len(analyses)} analysis result(s), "
        f"{len(forecasts)} forecast(s), {len(anomalies)} anomaly run(s), and "
        f"{len(visualizations)} chart(s)."
    )
    if findings:
        interpretation += " Key grounded findings: " + "; ".join(findings[:4])

    if draft_type == "creditworthiness_assessment":
        return [
            {
                "heading": "Overall Assessment",
                "kind": "assessment",
                "body": (
                    f"Creditworthiness assessment for {subject} based on available validated evidence. "
                    "This is a financial risk / credit risk outlook synthesized from existing analysis "
                    "and forecasts — not a formal credit score or established scoring methodology."
                ),
            },
            {"heading": "Financial Evidence", "kind": "evidence", "body": evidence_body},
            {
                "heading": "Forecast Outlook",
                "kind": "forecast",
                "body": _forecast_block(forecasts) or "No forecast evidence available.",
            },
            {
                "heading": "Anomalies / Concerns",
                "kind": "evidence",
                "body": _anomaly_block(anomalies) or "No anomaly evidence available.",
            },
            {
                "heading": "Assessment Rationale",
                "kind": "interpretation",
                "body": interpretation,
            },
            {
                "heading": "Key Strengths",
                "kind": "interpretation",
                "body": _strengths_from_evidence(analyses, forecasts),
            },
            {
                "heading": "Key Risks",
                "kind": "interpretation",
                "body": _risks_from_evidence(forecasts, anomalies),
            },
        ]

    if draft_type in {"forecast_explanation", "forecast_report", "scenario_summary"}:
        return [
            {"heading": "Forecast Summary", "kind": "forecast", "body": _forecast_block(forecasts)},
            {
                "heading": "Historical Context",
                "kind": "evidence",
                "body": _analysis_block(analyses) or "See forecast historical observations in evidence.",
            },
            {"heading": "Interpretation", "kind": "interpretation", "body": interpretation},
        ]

    if draft_type == "anomaly_summary":
        return [
            {"heading": "Anomaly Summary", "kind": "evidence", "body": _anomaly_block(anomalies)},
            {"heading": "Interpretation", "kind": "interpretation", "body": interpretation},
        ]

    if draft_type in {"memo", "decision_brief"}:
        return [
            {"heading": "Purpose", "kind": "interpretation", "body": f"Decision-oriented memo for {subject}."},
            {"heading": "Key Findings", "kind": "evidence", "body": evidence_body},
            {
                "heading": "Recommendations",
                "kind": "recommendation",
                "body": _recommendations_block(pack),
            },
        ]

    if draft_type == "recommendation":
        return [
            {"heading": "Recommendations", "kind": "recommendation", "body": _recommendations_block(pack)},
            {"heading": "Supporting Evidence", "kind": "evidence", "body": evidence_body},
        ]

    if draft_type in {"financial_report", "report", "risk_report"}:
        return [
            {"heading": "Executive Summary", "kind": "interpretation", "body": interpretation},
            {"heading": "Key Findings", "kind": "evidence", "body": evidence_body},
            {"heading": "Historical Performance", "kind": "evidence", "body": _analysis_block(analyses)},
            {"heading": "Forecast", "kind": "forecast", "body": _forecast_block(forecasts) or "No forecast available."},
            {
                "heading": "Risks / Anomalies",
                "kind": "evidence",
                "body": _anomaly_block(anomalies) or "No anomaly runs available.",
            },
            {
                "heading": "Recommendations",
                "kind": "recommendation",
                "body": _recommendations_block(pack),
            },
        ]

    # summary / analysis_explanation / default
    return [
        {"heading": "Summary", "kind": "interpretation", "body": interpretation},
        {"heading": "Evidence", "kind": "evidence", "body": evidence_body},
        {
            "heading": "Charts referenced",
            "kind": "evidence",
            "body": _viz_block(visualizations) or "No charts available.",
        },
    ]


def _title_for(draft_type: str, pack: dict[str, Any]) -> str:
    entities = (pack.get("conversation_context") or {}).get("active_entities") or {}
    company = entities.get("company")
    suffix = f" — {company}" if company else ""
    mapping = {
        "summary": f"Summary{suffix}",
        "report": f"Analytical Report{suffix}",
        "financial_report": f"Financial Analysis Report{suffix}",
        "forecast_report": f"Forecast Report{suffix}",
        "forecast_explanation": f"Forecast Explanation{suffix}",
        "risk_report": f"Risk Report{suffix}",
        "creditworthiness_assessment": f"Creditworthiness Assessment{suffix}",
        "memo": f"Management Memo{suffix}",
        "recommendation": f"Recommendations{suffix}",
        "analysis_explanation": f"Analysis Explanation{suffix}",
        "decision_brief": f"Decision Brief{suffix}",
        "anomaly_summary": f"Anomaly Summary{suffix}",
        "scenario_summary": f"Scenario Summary{suffix}",
    }
    return mapping.get(draft_type, f"Draft{suffix}")


def sections_to_markdown(title: str, sections: list[dict[str, Any]], draft_type: str) -> str:
    parts = [f"# {title}", "", f"_Artifact type: `{draft_type}`_", ""]
    for sec in sections:
        heading = sec.get("heading") or "Section"
        kind = sec.get("kind") or "interpretation"
        body = (sec.get("body") or "").strip()
        parts.append(f"## {heading}")
        parts.append(f"_({kind})_")
        parts.append(body)
        parts.append("")
    parts.append(
        "---\n"
        "_This draft synthesizes validated specialist outputs. "
        "It does not recalculate metrics or invent numerical results._"
    )
    return "\n".join(parts)


def _evidence_block(
    analyses: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    visualizations: list[dict[str, Any]],
) -> str:
    chunks = [
        _analysis_block(analyses),
        _forecast_block(forecasts),
        _anomaly_block(anomalies),
        _viz_block(visualizations),
    ]
    return "\n\n".join(c for c in chunks if c) or "No structured evidence available."


def _analysis_block(analyses: list[dict[str, Any]]) -> str:
    if not analyses:
        return ""
    lines = ["**Computed analysis facts:**"]
    for a in analyses:
        lines.append(
            f"- Analysis[{a.get('index')}] mode=`{a.get('mode')}` · "
            f"{a.get('n_records', 0)} record(s)"
        )
        if a.get("explanation"):
            lines.append(f"  - {a['explanation'][:220]}")
        for row in (a.get("sample_records") or [])[:3]:
            lines.append(f"  - record: `{row}`")
    return "\n".join(lines)


def _forecast_block(forecasts: list[dict[str, Any]]) -> str:
    if not forecasts:
        return ""
    lines = ["**Forecast facts:**"]
    for f in forecasts:
        if not f.get("suitable"):
            lines.append(
                f"- Forecast[{f.get('index')}] unsuitable: {f.get('reason') or 'n/a'}"
            )
            continue
        lines.append(
            f"- Forecast[{f.get('index')}] target=`{f.get('target_column')}` "
            f"horizon={f.get('forecast_horizon')} freq=`{f.get('frequency')}` "
            f"method=`{f.get('selected_method')}` trend={f.get('trend')}"
        )
        for row in (f.get("sample_forecast_values") or [])[:5]:
            lines.append(
                f"  - {row.get('time')}: value={row.get('value')} "
                f"(lower={row.get('lower')}, upper={row.get('upper')})"
            )
        metrics = f.get("evaluation_metrics") or []
        if metrics:
            top = metrics[0] if isinstance(metrics, list) else metrics
            if isinstance(top, dict):
                lines.append(
                    f"  - holdout: method={top.get('method')} "
                    f"MAE={top.get('mae')} RMSE={top.get('rmse')}"
                )
    return "\n".join(lines)


def _anomaly_block(anomalies: list[dict[str, Any]]) -> str:
    if not anomalies:
        return ""
    lines = ["**Anomaly facts:**"]
    for a in anomalies:
        lines.append(
            f"- Anomaly[{a.get('index')}] method=`{a.get('method')}` "
            f"column=`{a.get('column')}` count={a.get('n_anomalies')} "
            f"rate={a.get('anomaly_rate')}"
        )
    return "\n".join(lines)


def _viz_block(visualizations: list[dict[str, Any]]) -> str:
    if not visualizations:
        return ""
    lines = ["**Visualization metadata:**"]
    for v in visualizations:
        lines.append(
            f"- Chart[{v.get('index')}] `{v.get('chart_type')}` — {v.get('title')} "
            f"(x={v.get('x')}, y={v.get('y')}, n_points={v.get('n_points')})"
        )
    return "\n".join(lines)


def _recommendations_block(pack: dict[str, Any]) -> str:
    recs = list(pack.get("recommendations") or [])
    if pack.get("warnings"):
        recs.append("Review forecast/analysis warnings before acting on projections.")
    if pack.get("anomalies"):
        recs.append("Investigate flagged anomalies before relying on aggregate metrics.")
    if not recs:
        recs.append(
            "Continue monitoring the validated metrics and refresh forecasts as new data arrives."
        )
    return _join_bullets(recs)


def _strengths_from_evidence(analyses: list[dict[str, Any]], forecasts: list[dict[str, Any]]) -> str:
    bits: list[str] = []
    if analyses:
        bits.append("Validated analysis results are available to support historical performance review.")
    suitable = [f for f in forecasts if f.get("suitable")]
    if suitable:
        trend = suitable[-1].get("trend")
        bits.append(f"A grounded forecast exists (latest trend signal: {trend}).")
    return _join_bullets(bits) or "No clear strength signals in the current evidence pack."


def _risks_from_evidence(forecasts: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> str:
    bits: list[str] = []
    for f in forecasts:
        for w in f.get("warnings") or []:
            bits.append(str(w))
        if f.get("suitable") is False:
            bits.append(f.get("reason") or "Forecast unsuitable for this dataset.")
    if anomalies:
        bits.append("Anomaly detection flagged unusual observations that may affect risk outlook.")
    return _join_bullets(bits) or "No explicit risk flags in the current evidence pack."


def _join_bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items if x)


def _slim_analysis(item: dict[str, Any], index: int) -> dict[str, Any]:
    result = item.get("result") or {}
    records = result.get("records") or []
    return {
        "index": index,
        "question": item.get("question"),
        "mode": (item.get("query_plan") or {}).get("mode") or result.get("operation"),
        "explanation": (item.get("explanation") or "")[:400],
        "n_records": len(records),
        "sample_records": records[:5],
        "grounded": bool(result.get("grounded") or item.get("grounded")),
    }


def _slim_forecast(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": item.get("id"),
        "suitable": item.get("suitable"),
        "grounded": item.get("grounded"),
        "target_column": item.get("target_column"),
        "time_column": item.get("time_column"),
        "frequency": item.get("frequency"),
        "forecast_horizon": item.get("forecast_horizon"),
        "selected_method": item.get("selected_method"),
        "candidate_methods": item.get("candidate_methods"),
        "evaluation_metrics": item.get("evaluation_metrics"),
        "baseline_metrics": item.get("baseline_metrics"),
        "trend": item.get("trend"),
        "seasonality": item.get("seasonality"),
        "assumptions": list(item.get("assumptions") or []),
        "warnings": list(item.get("warnings") or []),
        "reason": item.get("reason"),
        "historical_observations": item.get("historical_observations"),
        "sample_forecast_values": (item.get("forecast_values") or [])[:8],
        "sample_historical_values": (item.get("historical_values") or [])[-5:],
        "explanation": (item.get("explanation") or "")[:400],
    }


def _slim_anomaly(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "method": item.get("method"),
        "column": item.get("column"),
        "n_anomalies": item.get("n_anomalies"),
        "anomaly_rate": item.get("anomaly_rate"),
        "grounded": item.get("grounded"),
        "explanation": (item.get("explanation") or "")[:300],
        "sample_records": (item.get("records") or [])[:3],
    }


def _slim_viz(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": item.get("id"),
        "chart_type": item.get("chart_type"),
        "title": item.get("title"),
        "x": item.get("x"),
        "y": item.get("y"),
        "y2": item.get("y2"),
        "n_points": item.get("n_points"),
        "grounded": item.get("grounded"),
        "data_preview": (item.get("data_preview") or [])[:3],
    }


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
