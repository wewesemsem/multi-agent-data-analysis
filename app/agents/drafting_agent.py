"""Drafting Agent — synthesizes validated specialist outputs into useful artifacts.

Does not recalculate metrics, invent forecasts/anomalies, or implement credit scoring.
LLMs organize and explain; SharedWorkspace tools remain the source of numerical truth.
"""

from __future__ import annotations

import json
from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import draft_tools


class DraftingAgent:
    name = "drafting_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if action not in {
                "draft",
                "create_draft",
                "write_draft",
                "synthesize_draft",
                "compose",
            }:
                return AgentResult(
                    task_id=message.task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    action=action,
                    success=False,
                    error=f"Unknown drafting action: {action}",
                    grounded=False,
                )

            params = message.parameters or {}
            user_request = (
                params.get("user_request")
                or params.get("question")
                or params.get("description")
                or ""
            )
            ctx_dict = {}
            if message.context:
                ctx_dict = message.context.get("conversation_context") or {}

            pack = draft_tools.build_evidence_pack(
                workspace,
                user_request=user_request,
                conversation_context=ctx_dict,
            )
            if params.get("draft_type") in draft_tools.DRAFT_TYPES:
                pack["draft_type"] = params["draft_type"]
            else:
                pack["draft_type"] = self._resolve_draft_type(user_request, pack.get("draft_type"))

            artifact = self._compose(pack, params)
            # Grounding gate: never keep fabricated numeric claims
            issues = draft_tools.validate_draft_payload(artifact.to_dict(), workspace)
            numeric_issues = [i for i in issues if "numerical claims" in i]
            if numeric_issues:
                artifact = draft_tools.heuristic_draft(pack)
                artifact.metadata["generator"] = "heuristic_fallback_ungrounded_llm"
                artifact.warnings = list(artifact.warnings) + [
                    "LLM draft replaced because it contained ungrounded numerical claims."
                ]

            payload = artifact.to_dict()
            # Replace prior drafts so validation retries don't keep a failing Draft[0]
            workspace.drafts = [payload]
            workspace.record(
                agent=self.name,
                action="draft",
                received={
                    "draft_type": payload.get("type"),
                    "user_request": user_request,
                    "source_count": len(payload.get("source_artifacts") or []),
                },
                produced={
                    "id": payload.get("id"),
                    "type": payload.get("type"),
                    "title": payload.get("title"),
                    "evidence_sufficient": payload.get("evidence_sufficient"),
                    "n_sections": len(payload.get("sections") or []),
                },
                success=True,
                limitations=list(payload.get("limitations") or []),
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="draft",
                success=True,
                data=payload,
                grounded=True,
                limitations=list(payload.get("limitations") or []),
            )
        except Exception as exc:  # noqa: BLE001
            workspace.record(
                agent=self.name,
                action=action,
                received=message.model_dump(),
                success=False,
                error=str(exc),
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action=action,
                success=False,
                error=str(exc),
                grounded=False,
            )

    def _resolve_draft_type(self, user_request: str, fallback: str | None) -> str:
        """Ask the LLM for artifact type; keyword infer is only a fallback."""
        fallback_type = fallback if fallback in draft_tools.DRAFT_TYPES else "summary"
        system = (
            "Classify the user's drafting request. Return JSON "
            '{"draft_type":"<one of: '
            + ", ".join(sorted(draft_tools.DRAFT_TYPES))
            + '>"}. '
            "Use creditworthiness_assessment only for explicit credit / credit risk / "
            "financial health assessment requests. Prefer summary for generic asks."
        )
        out = self.llm.chat_json(system, f"User request: {user_request}")
        if out.get("_offline") or out.get("_fallback"):
            return fallback_type
        chosen = out.get("draft_type")
        if isinstance(chosen, str) and chosen in draft_tools.DRAFT_TYPES:
            return chosen
        return fallback_type

    def _compose(self, pack: dict[str, Any], params: dict[str, Any]) -> draft_tools.DraftArtifact:
        """LLM-assisted composition with deterministic heuristic fallback."""
        fallback = draft_tools.heuristic_draft(pack)
        audience = params.get("audience")
        tone = params.get("tone")
        detail = params.get("detail") or params.get("level_of_detail")

        system = (
            "You are the Drafting Agent in a multi-agent data intelligence system. "
            "Specialize agents already computed analysis, forecasts, anomalies, and charts. "
            "Your job is ONLY to synthesize, interpret, organize, and communicate that evidence.\n"
            "Rules:\n"
            "- Do NOT recalculate metrics or invent numbers, forecasts, or anomalies.\n"
            "- Distinguish evidence vs interpretation vs recommendations vs limitations.\n"
            "- Never present interpretation/recommendations as observed facts.\n"
            "- For creditworthiness / credit risk requests: assess only from available evidence; "
            "do NOT invent a formal credit score or pretend an established scoring model exists. "
            "Prefer language like creditworthiness assessment, credit risk assessment, "
            "financial risk assessment, or risk outlook.\n"
            "- If evidence is insufficient, say so clearly.\n"
            "- Preserve all assumptions and warnings from the evidence pack.\n"
            "- Use only numbers that appear in the evidence pack.\n"
            "Return JSON with keys: title (string), sections (array of objects with "
            "heading, kind [evidence|forecast|interpretation|recommendation|assessment|limitation], "
            "body), assumptions (array), warnings (array), limitations (array)."
        )
        user = {
            "request": pack.get("user_request"),
            "draft_type": pack.get("draft_type"),
            "audience": audience,
            "tone": tone,
            "detail": detail,
            "evidence_sufficient": pack.get("evidence_sufficient"),
            "missing_evidence": pack.get("missing_evidence"),
            "evidence_pack": {
                k: pack.get(k)
                for k in (
                    "conversation_context",
                    "dataset",
                    "analysis_results",
                    "forecasts",
                    "anomalies",
                    "visualizations",
                    "validation_reports",
                    "findings",
                    "assumptions",
                    "warnings",
                    "recommendations",
                    "evidence_numbers",
                )
            },
        }
        out = self.llm.chat_json(system, json.dumps(user, default=str))
        if out.get("_offline") or out.get("_fallback") or not isinstance(out.get("sections"), list):
            return fallback

        sections = []
        for raw in out.get("sections") or []:
            if not isinstance(raw, dict):
                continue
            heading = str(raw.get("heading") or "Section").strip()
            body = str(raw.get("body") or "").strip()
            kind = str(raw.get("kind") or "interpretation").strip()
            if not body:
                continue
            sections.append({"heading": heading, "kind": kind, "body": body})
        if not sections:
            return fallback

        assumptions = _as_str_list(out.get("assumptions")) or list(pack.get("assumptions") or [])
        warnings = _as_str_list(out.get("warnings")) or list(pack.get("warnings") or [])
        # Always preserve pack assumptions/warnings
        for a in pack.get("assumptions") or []:
            if a not in assumptions:
                assumptions.append(a)
        for w in pack.get("warnings") or []:
            if w not in warnings:
                warnings.append(w)
        limitations = _as_str_list(out.get("limitations")) or list(pack.get("missing_evidence") or [])
        for m in pack.get("missing_evidence") or []:
            if m not in limitations:
                limitations.append(m)

        title = str(out.get("title") or fallback.title)
        content = draft_tools.sections_to_markdown(title, sections, pack.get("draft_type") or "summary")
        # Reject ungrounded numbers before accepting LLM prose
        allowed = set(pack.get("evidence_numbers") or [])
        if draft_tools.find_ungrounded_numbers(content, allowed):
            return fallback

        return draft_tools.DraftArtifact(
            id=fallback.id,
            type=pack.get("draft_type") if pack.get("draft_type") in draft_tools.DRAFT_TYPES else "summary",
            title=title,
            content=content,
            sections=sections,
            tables=[],
            charts=list(fallback.charts),
            source_artifacts=list(pack.get("source_artifacts") or []),
            assumptions=assumptions,
            warnings=warnings,
            limitations=limitations,
            metadata={
                "user_request": pack.get("user_request"),
                "evidence_numbers": list(pack.get("evidence_numbers") or []),
                "generator": "llm",
                "audience": audience,
                "tone": tone,
            },
            grounded=True,
            evidence_sufficient=bool(pack.get("evidence_sufficient")),
        )


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []
