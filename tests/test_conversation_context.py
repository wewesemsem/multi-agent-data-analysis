"""Focused tests for Phase 1 short-term ConversationContext memory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.context import (
    ActiveEntities,
    ConversationContext,
    RECENT_MESSAGE_WINDOW,
    recent_messages_window,
    update_conversation_context,
)
from app.orchestrator import Orchestrator
from app.state import SharedWorkspace
from app.tools import dataset_tools
from app.ui import _APP_KEYS, _fresh_state


class _FakeLLM:
    """Deterministic offline stand-in; marks itself offline for chat_json."""

    available = False

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
        return {"_offline": True}

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        return ""


def test_conversation_context_roundtrip_and_clear():
    ctx = ConversationContext(
        conversation_summary="Analyzing Company A",
        current_goal="Analyze Company A's financial performance",
        active_entities=ActiveEntities(
            company="Company A",
            dataset="company_a_financials",
            analysis="financial performance",
        ),
        findings=["Revenue concentrated in two segments"],
        decisions=[],
        assumptions=["FY figures are audited"],
        unresolved_questions=["Need debt maturity information"],
    )
    restored = ConversationContext.from_dict(ctx.to_dict())
    assert restored.current_goal == ctx.current_goal
    assert restored.active_entities.company == "Company A"
    assert restored.findings == ctx.findings
    assert not restored.is_empty()

    restored.clear()
    assert restored.is_empty()
    assert restored.active_entities.company is None


def test_recent_messages_window_capped():
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    recent = recent_messages_window(messages)
    assert len(recent) == RECENT_MESSAGE_WINDOW
    assert recent[0]["content"] == "m14"
    assert recent[-1]["content"] == "m19"


def test_context_updater_heuristic_grounds_entities_and_decisions():
    ctx = ConversationContext()
    ws = SharedWorkspace()
    ws.dataset = {
        "id": "ds1",
        "name": "company_a_financials",
        "row_count": 100,
        "schema": {"revenue": "float"},
    }
    ws.analysis_results = [
        {
            "explanation": "Category Electronics leads on revenue.",
            "query_plan": {"mode": "aggregation"},
            "result": {"records": [{"product_category": "Electronics", "value": 10.0}], "grounded": True},
        }
    ]

    update_conversation_context(
        ctx,
        user_request="Analyze Company A's financial performance, then use the conservative forecast scenario.",
        workspace=ws,
        recent_messages=[],
        llm=_FakeLLM(),  # type: ignore[arg-type]
    )

    assert ctx.active_entities.company == "Company A"
    assert ctx.active_entities.dataset == "company_a_financials"
    assert "conservative" in " ".join(ctx.decisions).lower()
    assert any("Electronics" in f or "revenue" in f.lower() for f in ctx.findings)
    assert "Company A" in ctx.conversation_summary or "Company A" in ctx.current_goal


def test_context_updater_does_not_invent_ungrounded_numeric_findings():
    ctx = ConversationContext()
    ws = SharedWorkspace()
    ws.dataset = {"id": "ds1", "name": "orders", "row_count": 10, "schema": {}}

    class InventingLLM:
        def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
            return {
                "conversation_summary": "Follow-up on orders",
                "current_goal": "Check profitability",
                "active_entities": {"company": None, "dataset": "orders", "analysis": "profitability"},
                "findings": ["Debt-to-EBITDA increased to 3.4x"],  # not in workspace
                "decisions": [],
                "assumptions": [],
                "unresolved_questions": [],
            }

        def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
            return ""

    update_conversation_context(
        ctx,
        user_request="What about profitability?",
        workspace=ws,
        recent_messages=[],
        llm=InventingLLM(),  # type: ignore[arg-type]
    )
    assert not any("3.4" in f for f in ctx.findings)


def test_orchestrator_planning_receives_prior_context_for_followups():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=300, seed=31)
    ctx = ConversationContext(
        conversation_summary="Analyzing Company A with ecommerce dataset",
        current_goal="Analyze Company A's financial performance",
        active_entities=ActiveEntities(
            company="Company A",
            dataset=ws.dataset["name"],
            analysis="financial performance",
        ),
    )
    plan = orch._create_plan(
        "t1",
        "What about profitability?",
        ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Here is the performance summary."},
        ],
    )
    agents = {s.agent for s in plan.steps}
    assert "analysis_agent" in agents

    # Planner record path via full run should keep workspace artifacts as SoT
    before_rows = ws.dataset["row_count"]
    ws2 = orch.run(
        "What about profitability?",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Here is the performance summary."},
        ],
    )
    assert ws2.dataset["row_count"] == before_rows
    assert ctx.active_entities.company == "Company A"
    assert ctx.active_entities.dataset == ws2.dataset["name"]
    assert not ctx.is_empty()


def test_context_persists_across_turns_in_same_session_object():
    ctx = ConversationContext()
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=200, seed=32)
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]

    orch.run(
        "Analyze Company A's financial performance.",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[],
    )
    mid = ctx.snapshot()
    assert mid["active_entities"]["company"] == "Company A"

    orch.run(
        "Now forecast it using the conservative scenario.",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Analysis complete."},
        ],
    )
    assert ctx.active_entities.company == "Company A"
    assert any("conservative" in d.lower() for d in ctx.decisions)
    # Same object continued — not a fresh empty context
    assert ctx.conversation_summary


def test_reset_clears_conversation_context_and_does_not_leak():
    """Reset contract: fresh session state replaces workspace + conversation context together."""
    assert "conversation_context" in _APP_KEYS
    dirty = ConversationContext(
        current_goal="Analyze Company A",
        active_entities=ActiveEntities(company="Company A", dataset="company_a_financials"),
        findings=["old finding"],
    )
    assert not dirty.is_empty()

    fresh = _fresh_state()
    assert isinstance(fresh["conversation_context"], ConversationContext)
    assert fresh["conversation_context"].is_empty()
    assert fresh["messages"] == []
    assert fresh["workspace"].dataset is None
    # Distinct object — prior Company A context cannot leak into a reset session
    assert fresh["conversation_context"] is not dirty
    assert fresh["conversation_context"].active_entities.company is None
    assert fresh["conversation_context"].findings == []

    # In-place clear matches Reset workspace semantics for the same session key
    dirty.clear()
    assert dirty.is_empty()


def test_followup_credit_grade_keeps_prior_company():
    ctx = ConversationContext(
        conversation_summary="Company A financial analysis underway",
        current_goal="Analyze Company A's financial performance",
        active_entities=ActiveEntities(company="Company A", dataset="orders", analysis="financial"),
        decisions=["Use the conservative forecast scenario"],
    )
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=150, seed=33)
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    orch.run(
        "Give me a creditworthiness grade.",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Now forecast it using the conservative scenario."},
            {"role": "assistant", "content": "Forecast noted."},
        ],
    )
    assert ctx.active_entities.company == "Company A"
    assert any("credit" in d.lower() for d in ctx.decisions)
    # Workspace remains authoritative for computed artifacts
    assert ws.dataset is not None
    assert "analysis_results" in ws.to_dict()
