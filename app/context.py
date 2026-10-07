"""Short-term conversational memory for the multi-agent session.

ConversationContext captures what the conversation *means* (goals, entities,
decisions). SharedWorkspace remains the source of truth for datasets and
computed artifacts. The UI `messages` list remains the full transcript.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from app.llm import LLMClient
from app.state import SharedWorkspace

# Last N chat turns for LLM prompts (user + assistant ≈ 2 messages per turn).
RECENT_MESSAGE_WINDOW = 6
_MAX_LIST_ITEMS = 8
_MAX_SUMMARY_CHARS = 600


@dataclass
class ActiveEntities:
    """Lightweight pointers to what the user is currently talking about."""

    company: str | None = None
    dataset: str | None = None
    analysis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "dataset": self.dataset,
            "analysis": self.analysis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ActiveEntities:
        if not data:
            return cls()
        return cls(
            company=_as_optional_str(data.get("company")),
            dataset=_as_optional_str(data.get("dataset")),
            analysis=_as_optional_str(data.get("analysis")),
        )


@dataclass
class ConversationContext:
    """Compressed, structured short-term memory for the current Streamlit session."""

    conversation_summary: str = ""
    current_goal: str = ""
    active_entities: ActiveEntities = field(default_factory=ActiveEntities)
    findings: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)

    def clear(self) -> None:
        """Reset all fields in place (same object identity for session_state)."""
        self.conversation_summary = ""
        self.current_goal = ""
        self.active_entities = ActiveEntities()
        self.findings = []
        self.decisions = []
        self.assumptions = []
        self.unresolved_questions = []

    def is_empty(self) -> bool:
        return not any(
            [
                self.conversation_summary,
                self.current_goal,
                self.active_entities.company,
                self.active_entities.dataset,
                self.active_entities.analysis,
                self.findings,
                self.decisions,
                self.assumptions,
                self.unresolved_questions,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_summary": self.conversation_summary,
            "current_goal": self.current_goal,
            "active_entities": self.active_entities.to_dict(),
            "findings": list(self.findings),
            "decisions": list(self.decisions),
            "assumptions": list(self.assumptions),
            "unresolved_questions": list(self.unresolved_questions),
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConversationContext:
        if not data:
            return cls()
        return cls(
            conversation_summary=str(data.get("conversation_summary") or ""),
            current_goal=str(data.get("current_goal") or ""),
            active_entities=ActiveEntities.from_dict(data.get("active_entities")),
            findings=_as_str_list(data.get("findings")),
            decisions=_as_str_list(data.get("decisions")),
            assumptions=_as_str_list(data.get("assumptions")),
            unresolved_questions=_as_str_list(data.get("unresolved_questions")),
        )

    def apply(self, other: ConversationContext) -> None:
        """Copy fields from another context onto this instance."""
        self.conversation_summary = other.conversation_summary
        self.current_goal = other.current_goal
        self.active_entities = ActiveEntities.from_dict(other.active_entities.to_dict())
        self.findings = list(other.findings)
        self.decisions = list(other.decisions)
        self.assumptions = list(other.assumptions)
        self.unresolved_questions = list(other.unresolved_questions)

    def to_prompt_block(self) -> str:
        """Compact block for orchestrator / agent LLM prompts."""
        if self.is_empty():
            return "Conversation context: (empty — new or reset session)"
        entities = self.active_entities
        lines = [
            "Conversation context:",
            f"- Summary: {self.conversation_summary or '(none)'}",
            f"- Current goal: {self.current_goal or '(none)'}",
            f"- Active company/entity: {entities.company or '(none)'}",
            f"- Active dataset: {entities.dataset or '(none)'}",
            f"- Active analysis/task: {entities.analysis or '(none)'}",
            f"- Findings: {_fmt_list(self.findings)}",
            f"- Decisions: {_fmt_list(self.decisions)}",
            f"- Assumptions: {_fmt_list(self.assumptions)}",
            f"- Unresolved questions: {_fmt_list(self.unresolved_questions)}",
        ]
        return "\n".join(lines)


def recent_messages_window(
    messages: list[dict[str, Any]] | None,
    *,
    window: int = RECENT_MESSAGE_WINDOW,
) -> list[dict[str, str]]:
    """Return a capped recent slice of the chat transcript for LLM prompts."""
    if not messages or window <= 0:
        return []
    slim: list[dict[str, str]] = []
    for msg in messages[-window:]:
        role = str(msg.get("role") or "").strip() or "user"
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        # Keep prompts small — truncate long assistant reports.
        if len(content) > 400:
            content = content[:397] + "..."
        slim.append({"role": role, "content": content})
    return slim


def format_recent_messages(messages: list[dict[str, Any]] | None) -> str:
    recent = recent_messages_window(messages)
    if not recent:
        return "(no recent messages)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def workspace_context_summary(workspace: SharedWorkspace) -> str:
    """High-level workspace pointers for planning — not a dump of artifacts."""
    ds = workspace.dataset
    if not ds:
        dataset_line = "none"
    else:
        dataset_line = (
            f"{ds.get('name') or ds.get('id') or 'loaded'} "
            f"({ds.get('row_count', '?')} rows)"
        )
    analysis_n = len(workspace.analysis_results)
    anomaly_n = len(workspace.anomalies)
    forecast_n = len(workspace.forecasts)
    viz_n = len(workspace.visualizations)
    draft_n = len(getattr(workspace, "drafts", None) or [])
    last_analysis = ""
    if workspace.analysis_results:
        last = workspace.analysis_results[-1]
        expl = (last.get("explanation") or "").strip()
        if expl:
            last_analysis = expl[:160]
        else:
            mode = (last.get("query_plan") or {}).get("mode") or "analysis"
            last_analysis = f"mode={mode}"
    last_forecast = ""
    if workspace.forecasts:
        fc = workspace.forecasts[-1]
        if fc.get("suitable"):
            last_forecast = (
                f"target={fc.get('target_column')} method={fc.get('selected_method')} "
                f"horizon={fc.get('forecast_horizon')}"
            )
        else:
            last_forecast = f"unsuitable: {(fc.get('reason') or '')[:120]}"
    return (
        "Workspace state:\n"
        f"- Dataset: {dataset_line}\n"
        f"- Analysis results: {analysis_n}"
        + (f" (latest: {last_analysis})" if last_analysis else "")
        + f"\n- Anomalies: {anomaly_n}\n"
        f"- Forecasts: {forecast_n}"
        + (f" (latest: {last_forecast})" if last_forecast else "")
        + f"\n- Visualizations: {viz_n}\n"
        f"- Drafts: {draft_n}\n"
        f"- Task status: {workspace.task_status}"
    )


def update_conversation_context(
    context: ConversationContext,
    *,
    user_request: str,
    workspace: SharedWorkspace,
    recent_messages: list[dict[str, Any]] | None = None,
    llm: LLMClient | None = None,
) -> ConversationContext:
    """Update short-term memory after a meaningful orchestration run.

    Numerical facts must come from SharedWorkspace artifacts / user statements.
    The updater must not invent financial figures.
    """
    prior = context.to_dict()
    heuristic = _heuristic_update(context, user_request=user_request, workspace=workspace)

    client = llm
    if client is None:
        return _finalize(context, heuristic)

    system = (
        "You maintain short-term conversational memory for a financial/data "
        "intelligence multi-agent app. Return a single JSON object with keys: "
        "conversation_summary, current_goal, active_entities "
        "(object with company, dataset, analysis), findings, decisions, "
        "assumptions, unresolved_questions.\n"
        "Rules:\n"
        "- Compress continuity; do not copy the full transcript.\n"
        "- findings must only restate qualitative conclusions already present in "
        "workspace artifacts or the user request — NEVER invent numbers or "
        "financial facts not present in the workspace summary.\n"
        "- decisions capture user choices (e.g. conservative forecast scenario).\n"
        "- unresolved_questions are gaps the system or user still needs.\n"
        "- Prefer updating/merging prior context over discarding it.\n"
        "- Keep each list to at most 8 short bullet strings.\n"
        "- Keep conversation_summary under 2 sentences."
    )
    user = (
        f"Prior context JSON:\n{prior}\n\n"
        f"Latest user request:\n{user_request}\n\n"
        f"{workspace_context_summary(workspace)}\n\n"
        f"Recent conversation:\n{format_recent_messages(recent_messages)}\n\n"
        f"Grounded artifact hints (do not invent beyond these):\n"
        f"{_artifact_hints(workspace)}"
    )
    out = client.chat_json(system, user)
    if out.get("_offline") or out.get("_fallback") or not isinstance(out, dict):
        return _finalize(context, heuristic)

    merged = _merge_llm_update(heuristic, out, workspace=workspace, user_request=user_request)
    return _finalize(context, merged)


def _finalize(context: ConversationContext, updated: ConversationContext) -> ConversationContext:
    context.apply(_clamp_context(updated))
    return context


def _heuristic_update(
    prior: ConversationContext,
    *,
    user_request: str,
    workspace: SharedWorkspace,
) -> ConversationContext:
    """Offline / fallback updater grounded in request text + workspace metadata."""
    ctx = ConversationContext.from_dict(prior.to_dict())
    request = (user_request or "").strip()
    if request:
        ctx.current_goal = request if len(request) <= 200 else request[:197] + "..."

    company = _extract_company(request) or ctx.active_entities.company
    dataset_name = None
    if workspace.dataset:
        dataset_name = workspace.dataset.get("name") or workspace.dataset.get("id")
    analysis_label = ctx.active_entities.analysis
    if workspace.forecasts:
        fc = workspace.forecasts[-1]
        if fc.get("suitable"):
            analysis_label = f"forecast:{fc.get('target_column')}"
        else:
            analysis_label = analysis_label or "forecast"
    elif workspace.analysis_results:
        last = workspace.analysis_results[-1]
        mode = (last.get("query_plan") or {}).get("mode")
        analysis_label = mode or analysis_label or "financial/data analysis"
    elif _mentions_analysis(request):
        analysis_label = analysis_label or "financial/data analysis"

    ctx.active_entities = ActiveEntities(
        company=company,
        dataset=str(dataset_name) if dataset_name else ctx.active_entities.dataset,
        analysis=analysis_label,
    )

    # Decisions from explicit user phrasing
    decision = _extract_decision(request)
    if decision and decision not in ctx.decisions:
        ctx.decisions = _append_unique(ctx.decisions, decision)

    # Findings from workspace explanations only (no invented numbers)
    for hint in _artifact_hints(workspace).split("\n"):
        hint = hint.strip("- ").strip()
        if hint and hint not in ctx.findings:
            ctx.findings = _append_unique(ctx.findings, hint)

    # Unresolved questions — light heuristic
    for q in _extract_unresolved(request, workspace):
        ctx.unresolved_questions = _append_unique(ctx.unresolved_questions, q)

    parts = []
    if company:
        parts.append(f"Working with {company}")
    if dataset_name:
        parts.append(f"dataset `{dataset_name}`")
    if ctx.current_goal:
        parts.append(f"goal: {ctx.current_goal}")
    ctx.conversation_summary = "; ".join(parts) if parts else (prior.conversation_summary or request[:200])
    return ctx


def _merge_llm_update(
    base: ConversationContext,
    raw: dict[str, Any],
    *,
    workspace: SharedWorkspace,
    user_request: str,
) -> ConversationContext:
    """Blend LLM structured output with heuristic grounding safeguards."""
    entities_raw = raw.get("active_entities") if isinstance(raw.get("active_entities"), dict) else {}
    entities = ActiveEntities.from_dict(entities_raw)

    # Prefer workspace dataset name as authoritative for active dataset pointer
    if workspace.dataset:
        entities.dataset = str(
            workspace.dataset.get("name") or workspace.dataset.get("id") or entities.dataset or ""
        ) or entities.dataset
    if not entities.company:
        entities.company = base.active_entities.company
    if not entities.analysis:
        entities.analysis = base.active_entities.analysis

    findings = _as_str_list(raw.get("findings")) or list(base.findings)
    # Drop findings that look like invented numeric claims not present in artifacts
    artifact_blob = _artifact_hints(workspace).lower() + "\n" + (user_request or "").lower()
    safe_findings: list[str] = []
    for item in findings:
        if _looks_like_numeric_claim(item) and not _claim_grounded(item, artifact_blob):
            continue
        safe_findings.append(item)
    if not safe_findings:
        safe_findings = list(base.findings)

    return ConversationContext(
        conversation_summary=str(raw.get("conversation_summary") or base.conversation_summary or ""),
        current_goal=str(raw.get("current_goal") or base.current_goal or user_request),
        active_entities=entities,
        findings=safe_findings,
        decisions=_as_str_list(raw.get("decisions")) or list(base.decisions),
        assumptions=_as_str_list(raw.get("assumptions")) or list(base.assumptions),
        unresolved_questions=_as_str_list(raw.get("unresolved_questions"))
        or list(base.unresolved_questions),
    )


def _clamp_context(ctx: ConversationContext) -> ConversationContext:
    summary = ctx.conversation_summary or ""
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[: _MAX_SUMMARY_CHARS - 3] + "..."
    return ConversationContext(
        conversation_summary=summary,
        current_goal=(ctx.current_goal or "")[:300],
        active_entities=ctx.active_entities,
        findings=list(ctx.findings)[:_MAX_LIST_ITEMS],
        decisions=list(ctx.decisions)[:_MAX_LIST_ITEMS],
        assumptions=list(ctx.assumptions)[:_MAX_LIST_ITEMS],
        unresolved_questions=list(ctx.unresolved_questions)[:_MAX_LIST_ITEMS],
    )


def _artifact_hints(workspace: SharedWorkspace) -> str:
    hints: list[str] = []
    for analysis in workspace.analysis_results[-3:]:
        expl = (analysis.get("explanation") or "").strip()
        if expl:
            hints.append(f"- analysis: {expl[:220]}")
        else:
            mode = (analysis.get("query_plan") or {}).get("mode")
            n = len((analysis.get("result") or {}).get("records") or [])
            if mode or n:
                hints.append(f"- analysis: completed ({mode or 'query'}, {n} records)")
    for anomaly in workspace.anomalies[-2:]:
        hints.append(
            f"- anomalies: method={anomaly.get('method')} column={anomaly.get('column')} "
            f"count={anomaly.get('n_anomalies')}"
        )
    for forecast in workspace.forecasts[-2:]:
        if forecast.get("suitable"):
            hints.append(
                f"- forecast: target={forecast.get('target_column')} "
                f"method={forecast.get('selected_method')} "
                f"horizon={forecast.get('forecast_horizon')} "
                f"freq={forecast.get('frequency')}"
            )
        else:
            hints.append(f"- forecast: unsuitable ({(forecast.get('reason') or '')[:160]})")
    for viz in workspace.visualizations[-3:]:
        hints.append(f"- chart: {viz.get('title')} ({viz.get('chart_type')})")
    for draft in (getattr(workspace, "drafts", None) or [])[-2:]:
        hints.append(
            f"- draft: type={draft.get('type')} title={draft.get('title')} "
            f"sufficient={draft.get('evidence_sufficient')}"
        )
    if workspace.dataset:
        hints.append(
            f"- dataset: {workspace.dataset.get('name')} "
            f"rows={workspace.dataset.get('row_count')}"
        )
    return "\n".join(hints) if hints else "- (no workspace artifacts yet)"


def _extract_company(text: str) -> str | None:
    if not text:
        return None
    # Prefer full names ("Company A") before "company <token>" splits.
    patterns = [
        r"(?i)\banalyze\s+([A-Za-z][\w&.-]{0,40}(?:\s+[A-Za-z][\w&.-]{0,40}){0,2})(?:'s)?\b",
        r"(?i)\b([A-Za-z][\w&.-]{0,40}(?:\s+[A-Za-z][\w&.-]{0,40}){0,2})(?:'s)?\s+financial\b",
        r"(?i)\b(?:firm|issuer|borrower)\s+([A-Za-z][\w&.-]{0,40}(?:\s+[A-Za-z][\w&.-]{0,40}){0,3})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip().rstrip(".,")
            name = re.sub(r"'s$", "", name, flags=re.IGNORECASE).strip()
            if name.lower() not in {"the", "a", "an", "this", "that", "our", "their"}:
                return name
    return None


def _extract_decision(text: str) -> str | None:
    q = text.lower()
    if "conservative" in q and ("scenario" in q or "forecast" in q or "case" in q):
        return "Use the conservative forecast scenario"
    if "base case" in q or "base-case" in q:
        return "Use the base-case scenario"
    if "aggressive" in q and ("scenario" in q or "forecast" in q):
        return "Use the aggressive forecast scenario"
    if any(
        k in q
        for k in (
            "creditworthiness",
            "credit grade",
            "credit rating",
            "credit assessment",
            "credit risk",
        )
    ):
        return "Produce a creditworthiness assessment"
    return None


def _extract_unresolved(request: str, workspace: SharedWorkspace) -> list[str]:
    out: list[str] = []
    q = request.lower()
    if any(k in q for k in ("maturity", "debt schedule", "refinanc")):
        out.append("Need debt maturity / schedule detail")
    if "credit" in q and not workspace.analysis_results and not workspace.forecasts:
        out.append("Need supporting financial analysis before a creditworthiness assessment")
    return out


def _mentions_analysis(text: str) -> bool:
    q = text.lower()
    return any(
        k in q
        for k in (
            "analy",
            "profit",
            "forecast",
            "credit",
            "performance",
            "revenue",
            "ebitda",
            "debt",
        )
    )


def _looks_like_numeric_claim(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _claim_grounded(claim: str, artifact_blob: str) -> bool:
    """Require numeric tokens from the claim to appear in grounded text."""
    nums = re.findall(r"\d+(?:\.\d+)?", claim)
    if not nums:
        return True
    return all(n in artifact_blob for n in nums)


def _append_unique(items: list[str], value: str) -> list[str]:
    if value in items:
        return items
    return (items + [value])[-_MAX_LIST_ITEMS:]


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return []


def _fmt_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "; ".join(items)
