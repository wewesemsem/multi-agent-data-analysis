"""Tests for the general-purpose Drafting Agent and orchestrator integration."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.drafting_agent import DraftingAgent
from app.agents.validation_agent import ValidationAgent
from app.context import ActiveEntities, ConversationContext
from app.messages import AgentMessage
from app.orchestrator import Orchestrator
from app.state import DATASETS_DIR, SharedWorkspace, ensure_workspace, utc_now
from app.tools import dataset_tools, draft_tools, forecast_tools


class _FakeLLM:
    available = False

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
        return {"_offline": True}

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        return ""


class _InventingLLM:
    """LLM that fabricates numbers not present in evidence — agent must reject."""

    available = True

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
        return {
            "title": "Fabricated Assessment",
            "sections": [
                {
                    "heading": "Overall Assessment",
                    "kind": "assessment",
                    "body": "Revenue will hit 999999999 next quarter with certainty.",
                }
            ],
            "assumptions": [],
            "warnings": [],
            "limitations": [],
        }

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        return "Revenue will hit 999999999 next quarter."


def _financials_meta(*, n: int = 20, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    years = list(range(2006, 2006 + n))
    revenue = 50 + np.arange(n) * 3 + rng.normal(0, 1.5, size=n)
    ebitda = revenue * 0.25 + rng.normal(0, 0.5, size=n)
    debt = 40 - np.arange(n) * 0.4 + rng.normal(0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "Year": years,
            "company_revenue": revenue,
            "EBITDA": ebitda,
            "debt": debt,
        }
    )
    ensure_workspace()
    ds_id = f"ds_{uuid.uuid4().hex[:10]}"
    loc = DATASETS_DIR / f"{ds_id}.parquet"
    df.to_parquet(loc, index=False)
    meta = {
        "id": ds_id,
        "name": "company_financials",
        "location": str(loc),
        "row_count": int(len(df)),
        "schema": {c: str(df[c].dtype) for c in df.columns},
        "source": "csv_upload",
        "created_at": utc_now(),
    }
    (DATASETS_DIR / f"{ds_id}.meta.json").write_text(json.dumps(meta))
    return meta


def _analysis_payload(question: str = "Revenue by year") -> dict:
    return {
        "question": question,
        "query_plan": {"mode": "aggregation"},
        "result": {
            "records": [
                {"Year": 2024, "company_revenue": 104.5},
                {"Year": 2023, "company_revenue": 98.2},
            ],
            "grounded": True,
        },
        "explanation": "company_revenue was 104.5 in 2024 and 98.2 in 2023.",
        "grounded": True,
    }


def _anomaly_payload() -> dict:
    return {
        "method": "iqr",
        "column": "company_revenue",
        "n_anomalies": 2,
        "anomaly_rate": 0.1,
        "records": [{"Year": 2010, "company_revenue": 999.0, "_anomaly_score": 3.2}],
        "parameters": {"k": 1.5},
        "grounded": True,
        "explanation": "2 anomalies detected on company_revenue via IQR.",
    }


def _seed_workspace_with_evidence() -> SharedWorkspace:
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    fc = forecast_tools.forecast_series(
        ws.dataset,
        target_column="company_revenue",
        horizon=3,
    )
    ws.forecasts = [fc]
    ws.anomalies = [_anomaly_payload()]
    ws.visualizations = [
        {
            "id": "chart_demo",
            "chart_type": "line",
            "title": "Revenue trend",
            "x": "Year",
            "y": "company_revenue",
            "html_path": str(DATASETS_DIR / "noop.html"),
            "n_points": 20,
            "grounded": True,
            "data_preview": [{"Year": 2024, "company_revenue": 104.5}],
        }
    ]
    # Satisfy visualization HTML existence for validation when needed
    Path(ws.visualizations[0]["html_path"]).write_text("<html></html>")
    return ws


def _run_draft(ws: SharedWorkspace, request: str, *, llm=None):
    agent = DraftingAgent(llm=llm or _FakeLLM())  # type: ignore[arg-type]
    return agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="drafting_agent",
            action="draft",
            parameters={"user_request": request},
            context={
                "conversation_context": {
                    "current_goal": request,
                    "active_entities": {"company": "Company A"},
                    "findings": [],
                    "decisions": [],
                    "assumptions": [],
                }
            },
        ),
        ws,
    )


def test_summary_from_analysis_results():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    result = _run_draft(ws, "Summarize the analysis.")
    assert result.success
    assert ws.drafts
    draft = ws.drafts[0]
    assert draft["type"] == "summary"
    assert draft["grounded"] is True
    assert "104.5" in draft["content"] or "98.2" in draft["content"] or draft["evidence_sufficient"]
    assert any(s.get("kind") == "evidence" for s in draft["sections"]) or "Evidence" in draft["content"]


def test_report_from_analysis_and_forecast():
    ws = _seed_workspace_with_evidence()
    result = _run_draft(
        ws,
        "Write a detailed report on the company's financial performance.",
    )
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] in {"financial_report", "report"}
    kinds = {s.get("kind") for s in draft["sections"]}
    assert "evidence" in kinds or "forecast" in kinds
    assert draft["source_artifacts"]
    assert any(r.get("kind") == "forecast" for r in draft["source_artifacts"])
    assert any(r.get("kind") == "analysis" for r in draft["source_artifacts"])


def test_forecast_explanation():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.forecasts = [
        forecast_tools.forecast_series(ws.dataset, target_column="company_revenue", horizon=3)
    ]
    result = _run_draft(ws, "Explain the forecast.")
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] == "forecast_explanation"
    assert draft["evidence_sufficient"] is True
    assert ws.forecasts[0]["selected_method"] in draft["content"] or "Forecast" in draft["content"]


def test_anomaly_summary():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.anomalies = [_anomaly_payload()]
    result = _run_draft(ws, "Give me an anomaly summary.")
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] == "anomaly_summary"
    assert "2" in draft["content"] or "anomal" in draft["content"].lower()


def test_creditworthiness_assessment_with_evidence():
    ws = _seed_workspace_with_evidence()
    result = _run_draft(ws, "Give me a creditworthiness assessment for this company.")
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] == "creditworthiness_assessment"
    assert draft["evidence_sufficient"] is True
    text = draft["content"].lower()
    assert "credit" in text or "risk" in text
    assert "score" not in text or "not a formal credit score" in text or "no formal credit score" in text
    assert any(s.get("kind") == "assessment" for s in draft["sections"])


def test_creditworthiness_insufficient_evidence():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    # Dataset only — no analysis/forecasts/anomalies
    result = _run_draft(ws, "Give me a creditworthiness assessment for Company A.")
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] == "creditworthiness_assessment"
    assert draft["evidence_sufficient"] is False
    blob = (draft["content"] + " " + " ".join(draft.get("limitations") or [])).lower()
    assert "insufficient" in blob or "limited" in blob or "cannot be completed" in blob
    assert "999999" not in draft["content"]


def test_user_created_dataset_summary():
    meta = dataset_tools.create_ecommerce_orders(n_rows=300, seed=21)
    ws = SharedWorkspace()
    ws.dataset = meta
    ws.analysis_results = [
        {
            "question": "Which categories generate revenue?",
            "query_plan": {"mode": "aggregation"},
            "result": {
                "records": [{"product_category": "Electronics", "revenue": 12000}],
                "grounded": True,
            },
            "explanation": "Electronics leads with revenue 12000.",
            "grounded": True,
        }
    ]
    result = _run_draft(ws, "Summarize the analysis.")
    assert result.success
    assert ws.drafts[0]["grounded"] is True
    assert "12000" in ws.drafts[0]["content"] or ws.drafts[0]["evidence_sufficient"]


def test_uploaded_dataset_arbitrary_schema(tmp_path):
    df = pd.DataFrame(
        {
            "obs_date": pd.date_range("2022-01-01", periods=18, freq="MS"),
            "widget_throughput": np.linspace(10, 40, 18),
            "region_code": ["N", "S"] * 9,
        }
    )
    csv_path = tmp_path / "widgets.csv"
    df.to_csv(csv_path, index=False)
    meta = dataset_tools.load_csv(str(csv_path), name="widgets")
    ws = SharedWorkspace()
    ws.dataset = meta
    ws.analysis_results = [
        {
            "question": "Average widget_throughput",
            "query_plan": {"mode": "aggregation"},
            "result": {
                "records": [{"metric": "widget_throughput", "value": 25.0}],
                "grounded": True,
            },
            "explanation": "Average widget_throughput is 25.0.",
            "grounded": True,
        }
    ]
    ws.forecasts = [
        forecast_tools.forecast_series(meta, target_column="widget_throughput", horizon=3)
    ]
    result = _run_draft(ws, "Create an investment memo based on the analysis and forecast.")
    assert result.success
    draft = ws.drafts[0]
    assert draft["type"] == "memo"
    assert draft["evidence_sufficient"] is True


def test_followup_uses_conversation_context():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = _seed_workspace_with_evidence()
    ctx = ConversationContext(
        conversation_summary="Analyzing Company A",
        current_goal="Analyze Company A",
        active_entities=ActiveEntities(
            company="Company A",
            dataset=ws.dataset["name"],
            analysis="financial performance",
        ),
        findings=["company_revenue was 104.5 in 2024"],
    )
    plan = orch._create_plan(
        "t1",
        "Now summarize everything.",
        ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A."},
            {"role": "assistant", "content": "Analysis complete."},
            {"role": "user", "content": "Forecast its revenue."},
            {"role": "assistant", "content": "Forecast complete."},
        ],
    )
    agents = [s.agent for s in plan.steps]
    assert "drafting_agent" in agents
    assert "analysis_agent" not in agents
    assert "forecasting_agent" not in agents

    ws2 = orch.run(
        "Now summarize everything.",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A."},
            {"role": "assistant", "content": "Analysis complete."},
        ],
    )
    assert ws2.drafts
    assert ctx.active_entities.company == "Company A"


def test_drafting_only_request():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    plan = orch._create_plan("t", "Summarize the analysis.", ws)
    agents = {s.agent for s in plan.steps}
    assert "drafting_agent" in agents
    assert "analysis_agent" not in agents
    assert "forecasting_agent" not in agents
    ws = orch.run("Summarize the analysis.", workspace=ws)
    assert ws.drafts
    assert "drafting_agent" in {h["agent"] for h in ws.agent_history if h.get("success")}


def test_request_requiring_multiple_existing_agents():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    plan = orch._create_plan(
        "t",
        "Analyze company_revenue, forecast company_revenue for the next 3 years, "
        "then give me a creditworthiness assessment.",
        ws,
    )
    agents = {s.agent for s in plan.steps}
    assert "analysis_agent" in agents
    assert "forecasting_agent" in agents
    assert "drafting_agent" in agents
    # No invented credit specialist
    assert "credit_assessment_agent" not in agents
    assert all("credit" not in a or a == "drafting_agent" for a in agents)

    ws = orch.run(
        "Analyze company_revenue, forecast company_revenue for the next 3 years, "
        "then give me a creditworthiness assessment.",
        workspace=ws,
    )
    assert ws.analysis_results
    assert ws.forecasts
    assert ws.drafts
    assert ws.drafts[-1]["type"] == "creditworthiness_assessment"


def test_prevention_of_fabricated_numerical_claims():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    result = _run_draft(ws, "Summarize the analysis.", llm=_InventingLLM())  # type: ignore[arg-type]
    assert result.success
    draft = ws.drafts[0]
    assert "999999999" not in draft["content"]
    assert draft["grounded"] is True

    # Validator also flags a handcrafted fabricated draft
    bad = draft_tools.heuristic_draft(
        draft_tools.build_evidence_pack(ws, user_request="Summarize the analysis.")
    ).to_dict()
    bad["content"] = bad["content"] + "\nSecret revenue is 888888888."
    bad["evidence_sufficient"] = True
    issues = draft_tools.validate_draft_payload(bad, ws)
    assert any("numerical claims" in i for i in issues)


def test_european_decimal_formatting_is_grounded():
    """Locale-style decimals in prose must match US-formatted workspace evidence."""
    allowed = draft_tools.extract_number_tokens("revenue 251.68 forecast 290.308 n_points 30")
    content = "Revenue is 251,68 and forecast 290,308 with 30 points. Again 290,308."
    assert draft_tools.find_ungrounded_numbers(content, allowed) == []
    # Still reject invented magnitudes
    assert draft_tools.find_ungrounded_numbers(content + " Secret 888888888.", allowed)


def test_visualization_n_points_count_as_evidence():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    ws.visualizations = [
        {
            "id": "viz_1",
            "chart_type": "line",
            "title": "Revenue",
            "x": "Year",
            "y": "company_revenue",
            "n_points": 30,
            "grounded": True,
            "html_path": "/tmp/does-not-need-to-exist-for-number-check.html",
            "data_preview": [],
        }
    ]
    pack = draft_tools.build_evidence_pack(ws, user_request="Summarize the analysis.")
    assert "30" in pack["evidence_numbers"]
    draft = draft_tools.heuristic_draft(pack).to_dict()
    draft["content"] = draft["content"] + "\nChart uses 30 points."
    draft["evidence_sufficient"] = True
    issues = draft_tools.validate_draft_payload(draft, ws)
    assert not any("numerical claims" in i for i in issues)


def test_drafting_agent_replaces_prior_drafts():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws.analysis_results = [_analysis_payload()]
    _run_draft(ws, "Summarize the analysis.")
    _run_draft(ws, "Summarize the analysis again.")
    assert len(ws.drafts) == 1


def test_preservation_of_assumptions_and_warnings():
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    fc = forecast_tools.forecast_series(ws.dataset, target_column="company_revenue", horizon=3)
    fc["assumptions"] = ["Holdout window used for method selection"]
    fc["warnings"] = ["Short history may reduce forecast reliability"]
    ws.forecasts = [fc]
    result = _run_draft(ws, "Explain the forecast.")
    assert result.success
    draft = ws.drafts[0]
    assert "Holdout window used for method selection" in (draft.get("assumptions") or [])
    assert "Short history may reduce forecast reliability" in (draft.get("warnings") or [])
    assert "Holdout window used for method selection" in draft["content"]
    assert "Short history may reduce forecast reliability" in draft["content"]


def test_existing_agents_continue_functioning():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=400, seed=77)
    ws = orch.run("Detect anomalous transactions in the dataset.", workspace=ws)
    agents = {h["agent"] for h in ws.agent_history if h.get("success")}
    assert "anomaly_agent" in agents
    assert "drafting_agent" not in agents
    assert ws.anomalies

    ws2 = SharedWorkspace()
    ws2.dataset = _financials_meta()
    ws2 = orch.run("Forecast company_revenue for the next 3 years.", workspace=ws2)
    assert ws2.forecasts
    assert "forecasting_agent" in {h["agent"] for h in ws2.agent_history if h.get("success")}


def test_validation_agent_draft_checks():
    ws = _seed_workspace_with_evidence()
    _run_draft(ws, "Summarize the analysis.")
    agent = ValidationAgent()
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="validation_agent",
            action="validate_drafts",
            parameters={"required": True},
        ),
        ws,
    )
    assert result.success


def test_infer_draft_types():
    assert draft_tools.infer_draft_type("Summarize the analysis.") == "summary"
    assert draft_tools.infer_draft_type("Write a detailed report on financial performance.") == "financial_report"
    assert draft_tools.infer_draft_type("Explain the forecast.") == "forecast_explanation"
    assert draft_tools.infer_draft_type("Give me a creditworthiness assessment.") == "creditworthiness_assessment"
    assert draft_tools.infer_draft_type("Turn these findings into a memo for management.") == "memo"
