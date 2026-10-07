"""Focused tests for Forecasting Agent tools, validation, and orchestrator wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.forecasting_agent import ForecastingAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.visualization_agent import VisualizationAgent
from app.context import ConversationContext, ActiveEntities, update_conversation_context
from app.messages import AgentMessage
from app.orchestrator import Orchestrator
from app.state import SharedWorkspace
from app.tools import dataset_tools, forecast_tools


class _FakeLLM:
    available = False

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
        return {"_offline": True}

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        return ""


def _monthly_revenue_csv(path: Path, *, n: int = 36, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="MS")
    base = 100_000 + np.arange(n) * 1200
    noise = rng.normal(0, 2500, size=n)
    season = 8000 * np.sin(np.arange(n) * 2 * np.pi / 12)
    df = pd.DataFrame({"date": dates, "revenue": base + noise + season})
    df.to_csv(path, index=False)
    return path


def _financials_meta(*, n: int = 20, seed: int = 11) -> dict:
    """Yearly financial-style dataset with arbitrary column names."""
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
            "marketing_spend": 5 + np.arange(n) * 0.2 + rng.normal(0, 0.3, size=n),
        }
    )
    from app.state import DATASETS_DIR, ensure_workspace, utc_now
    import json
    import uuid

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


def test_forecast_uploaded_csv_different_column_names(tmp_path):
    csv_path = _monthly_revenue_csv(tmp_path / "rev.csv")
    meta = dataset_tools.load_csv(str(csv_path), name="uploaded_revenue")
    out = forecast_tools.forecast_series(meta, target_column="revenue", horizon=6)
    assert out["grounded"] is True
    assert out["suitable"] is True
    assert out["time_column"] == "date"
    assert out["target_column"] == "revenue"
    assert out["forecast_horizon"] == 6
    assert len(out["forecast_values"]) == 6
    assert out["selected_method"] in out["candidate_methods"]
    assert out["baseline_metrics"]
    assert out["evaluation_metrics"]


def test_forecast_generated_ecommerce_dataset():
    meta = dataset_tools.create_ecommerce_orders(n_rows=800, seed=41)
    out = forecast_tools.forecast_series(
        meta,
        user_request="Forecast total_amount for the next 14 days",
        target_column="total_amount",
        horizon=14,
    )
    assert out["grounded"] is True
    assert out["suitable"] is True
    assert out["time_column"]
    assert len(out["forecast_values"]) == 14


def test_forecast_arbitrary_financial_schema():
    meta = _financials_meta()
    out = forecast_tools.forecast_series(
        meta,
        user_request="Forecast company_revenue for the next three years.",
        target_column="company_revenue",
        horizon=3,
    )
    assert out["suitable"] is True
    assert out["frequency"] in {"yearly", "irregular"}
    assert out["forecast_horizon"] == 3
    assert len(out["forecast_values"]) == 3


def test_naive_moving_average_exponential_smoothing_methods():
    meta = _financials_meta(n=24)
    for method in ("naive", "moving_average", "exponential_smoothing", "trend_regression"):
        out = forecast_tools.forecast_series(
            meta,
            target_column="company_revenue",
            horizon=4,
            method=method,
        )
        assert out["suitable"] is True, method
        assert out["selected_method"] == method
        assert len(out["forecast_values"]) == 4
        assert all(np.isfinite(float(r["value"])) for r in out["forecast_values"])


def test_seasonal_candidate_on_monthly_series(tmp_path):
    csv_path = _monthly_revenue_csv(tmp_path / "seasonal.csv", n=48)
    meta = dataset_tools.load_csv(str(csv_path), name="seasonal_rev")
    out = forecast_tools.forecast_series(meta, target_column="revenue", horizon=12)
    assert out["suitable"] is True
    assert out["frequency"] == "monthly"
    # Seasonality may or may not clear the detection threshold; method set should be sane
    assert "naive" in out["candidate_methods"]
    assert "exponential_smoothing" in out["candidate_methods"]


def test_causal_regression_when_predictors_exist():
    meta = _financials_meta(n=24)
    out = forecast_tools.forecast_series(
        meta,
        target_column="company_revenue",
        horizon=3,
        method="causal_regression",
    )
    assert out["suitable"] is True
    assert out["selected_method"] == "causal_regression"
    assert out["predictors_used"]


def test_model_comparison_includes_baseline():
    meta = _financials_meta(n=24)
    out = forecast_tools.forecast_series(meta, target_column="EBITDA", horizon=4)
    assert out["suitable"] is True
    assert out["baseline_metrics"].get("method") == "naive"
    methods = {m["method"] for m in out["evaluation_metrics"]}
    assert "naive" in methods
    assert out["selected_method"] in methods


def test_horizon_parsing_from_natural_language():
    assert forecast_tools.parse_horizon_from_text("Forecast revenue for the next 12 months", "monthly") == 12
    assert forecast_tools.parse_horizon_from_text("Project the next 8 quarters", "quarterly") == 8
    assert forecast_tools.parse_horizon_from_text("Predict the next 30 days", "daily") == 30
    hz, assumption = forecast_tools.resolve_horizon(None, None, "monthly", 36)
    assert hz == 6
    assert assumption and "defaulting" in assumption.lower()


def test_detect_requested_frequency_from_text():
    assert forecast_tools.detect_requested_frequency("Forecast monthly revenue") == "monthly"
    assert forecast_tools.detect_requested_frequency("next 12 months") == "monthly"
    assert forecast_tools.detect_requested_frequency("next 3 years") == "yearly"
    assert forecast_tools.detect_requested_frequency("next 30 days") == "daily"


def test_monthly_revenue_request_resamples_daily_orders():
    """Regression: 'next 12 months' on daily orders must forecast 12 months, not 12 days."""
    meta = dataset_tools.create_ecommerce_orders(n_rows=600, seed=12)
    request = "Forecast monthly revenue for the next 12 months"
    # Simulate planner LLM passing a bare horizon=12 (days) — tools must still align grain.
    out = forecast_tools.forecast_series(
        meta,
        user_request=request,
        target_column="total_amount",
        horizon=12,
    )
    assert out["suitable"] is True
    assert out["frequency"] == "monthly"
    assert out["forecast_horizon"] == 12
    assert len(out["forecast_values"]) == 12
    assert out["historical_observations"] < 40  # months, not ~365 days
    assert all(float(r["lower"]) >= 0 for r in out["forecast_values"])


def test_resolve_horizon_prefers_text_units_over_bare_integer():
    # Bare 12 would be wrong on daily data; text says months and monthly grain → 12.
    hz, _ = forecast_tools.resolve_horizon(
        12, "Forecast revenue for the next 12 months", "monthly", 18
    )
    assert hz == 12
    hz_daily, _ = forecast_tools.resolve_horizon(
        12, "Forecast revenue for the next 12 months", "daily", 365
    )
    assert hz_daily == 360


def test_validation_rejects_bad_inputs(tmp_path):
    # No time column
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": range(9)})
    p = tmp_path / "notime.csv"
    df.to_csv(p, index=False)
    meta = dataset_tools.load_csv(str(p), name="notime")
    out = forecast_tools.forecast_series(meta)
    assert out["suitable"] is False
    assert not out["forecast_values"]

    # Duplicate timestamps / insufficient after cleanup covered by short series
    short = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=4, freq="D"), "value": [1, 2, 3, 4]}
    )
    p2 = tmp_path / "short.csv"
    short.to_csv(p2, index=False)
    meta2 = dataset_tools.load_csv(str(p2), name="short")
    out2 = forecast_tools.forecast_series(meta2, target_column="value")
    assert out2["suitable"] is False

    # Non-numeric target
    bad = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "label": list("abcdefghijkl"),
        }
    )
    p3 = tmp_path / "nonnum.csv"
    bad.to_csv(p3, index=False)
    meta3 = dataset_tools.load_csv(str(p3), name="nonnum")
    out3 = forecast_tools.forecast_series(meta3, target_column="label")
    assert out3["suitable"] is False


def test_duplicate_timestamps_aggregated(tmp_path):
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"] * 3 + ["2024-03-01"] * 3)
    # Need enough unique periods
    more = pd.date_range("2024-04-01", periods=10, freq="MS")
    all_dates = list(dates) + list(more)
    values = list(range(len(all_dates)))
    df = pd.DataFrame({"month": all_dates, "sales": values})
    p = tmp_path / "dups.csv"
    df.to_csv(p, index=False)
    meta = dataset_tools.load_csv(str(p), name="dups")
    out = forecast_tools.forecast_series(meta, target_column="sales", horizon=3)
    assert out["suitable"] is True
    assert any("duplicate" in w.lower() for w in out.get("warnings") or [])


def test_forecasting_agent_writes_workspace():
    meta = _financials_meta()
    ws = SharedWorkspace()
    ws.dataset = meta
    agent = ForecastingAgent(llm=_FakeLLM())  # type: ignore[arg-type]
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="forecasting_agent",
            action="forecast",
            parameters={"user_request": "Forecast company_revenue for the next 3 years."},
        ),
        ws,
    )
    assert result.success
    assert result.grounded
    assert ws.forecasts
    assert ws.forecasts[0]["target_column"] == "company_revenue"


def test_orchestrator_routes_forecast_request():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ws = orch.run("Forecast company_revenue for the next 3 years.", workspace=ws)
    agents = {h["agent"] for h in ws.agent_history if h.get("success")}
    assert "forecasting_agent" in agents
    assert "visualization_agent" in agents
    assert "drafting_agent" not in agents
    assert ws.forecasts
    assert ws.forecasts[0].get("suitable") is True
    assert any(v.get("chart_type") == "forecast" for v in ws.visualizations)
    assert "forecasts" in ws.to_dict()
    assert ws.task_status in {"completed", "completed_with_warnings"}


def test_plain_forecast_plan_excludes_draft_and_analysis():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=400, seed=3)
    plan = orch._create_plan("t", "Forecast revenue for the next 12 months.", ws)
    agents = {s.agent for s in plan.steps}
    assert "forecasting_agent" in agents
    assert "drafting_agent" not in agents
    assert "analysis_agent" not in agents


def test_no_specialist_intent_skips_pipeline():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=200, seed=4)
    ws = orch.run("hello", workspace=ws)
    assert ws.task_status == "completed"
    assert not ws.analysis_results
    assert not ws.forecasts
    assert not ws.drafts
    assert "analysis_agent" not in {h["agent"] for h in ws.agent_history}


def test_nl_revenue_forecast_selects_aggregate_amount_column():
    """Metric language should resolve onto schema columns (not a fixed name)."""
    meta = dataset_tools.create_ecommerce_orders(n_rows=600, seed=12)
    agent = ForecastingAgent(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = meta
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="forecasting_agent",
            action="forecast",
            parameters={"user_request": "Forecast revenue for the next 12 months."},
        ),
        ws,
    )
    assert result.success
    assert ws.forecasts
    target = ws.forecasts[0].get("target_column")
    assert target in meta["schema"]
    assert "price" not in str(target).lower() or "total" in str(target).lower()
    # For the ecommerce generator schema, order totals are the aggregate measure
    assert target == "total_amount"


def test_validation_agent_forecast_checks():
    meta = _financials_meta()
    ws = SharedWorkspace()
    ws.dataset = meta
    fc = forecast_tools.forecast_series(meta, target_column="debt", horizon=3)
    ws.forecasts = [fc]
    agent = ValidationAgent()
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="validation_agent",
            action="validate_forecasts",
            parameters={"required": True},
        ),
        ws,
    )
    assert result.success


def test_visualization_from_forecast_artifact():
    meta = _financials_meta()
    ws = SharedWorkspace()
    ws.dataset = meta
    ws.forecasts = [
        forecast_tools.forecast_series(meta, target_column="company_revenue", horizon=3)
    ]
    agent = VisualizationAgent(llm=_FakeLLM())  # type: ignore[arg-type]
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={
                "user_request": "Chart the forecast",
                "force_heuristic": True,
            },
        ),
        ws,
    )
    assert result.success
    assert any(c.get("chart_type") == "forecast" for c in ws.visualizations)


def test_forecast_request_uses_forecast_chart_not_plain_line():
    """Regression: LLM line charts must not replace grounded forecast artifacts."""
    meta = dataset_tools.create_ecommerce_orders(n_rows=600, seed=12)
    ws = SharedWorkspace()
    ws.dataset = meta
    ws.forecasts = [
        forecast_tools.forecast_series(
            meta,
            user_request="Forecast monthly revenue for the next 12 months",
            target_column="total_amount",
            horizon=12,
        )
    ]
    assert ws.forecasts[0].get("suitable")

    class _LineLLM(_FakeLLM):
        available = True

        def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict:
            return {
                "specs": [
                    {
                        "chart_type": "line",
                        "title": "Revenue Forecast for the Next 12 Months",
                        "x": "order_date",
                        "y": "total_amount",
                    }
                ]
            }

    agent = VisualizationAgent(llm=_LineLLM())  # type: ignore[arg-type]
    result = agent.handle(
        AgentMessage(
            task_id="t",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={
                "user_request": (
                    "Forecast monthly revenue for the next 12 months using "
                    "fictional_ecommerce_orders, then chart historical revenue and the forecast."
                ),
            },
        ),
        ws,
    )
    assert result.success
    assert any(c.get("chart_type") == "forecast" for c in ws.visualizations)
    assert not any(c.get("chart_type") == "line" for c in ws.visualizations)


def test_followup_revenue_chart_keeps_forecast_visualization():
    """Regression: a new viz must not wipe prior forecast charts (only reset does)."""
    meta = dataset_tools.create_ecommerce_orders(n_rows=400, seed=19)
    ws = SharedWorkspace()
    ws.dataset = meta
    # Seed a forecast chart as if a prior forecast+viz turn completed
    fc = forecast_tools.forecast_series(
        meta, target_column="total_amount", time_column="order_date", horizon=3
    )
    assert fc.get("suitable")
    ws.forecasts = [fc]
    agent = VisualizationAgent(llm=_FakeLLM())  # type: ignore[arg-type]
    forecast_viz = agent.handle(
        AgentMessage(
            task_id="t1",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={"user_request": "Chart the forecast", "force_heuristic": True},
        ),
        ws,
    )
    assert forecast_viz.success
    assert any(c.get("chart_type") == "forecast" for c in ws.visualizations)
    n_before = len(ws.visualizations)

    revenue_viz = agent.handle(
        AgentMessage(
            task_id="t2",
            source_agent="orchestrator",
            target_agent="visualization_agent",
            action="create_visualization",
            parameters={
                "user_request": "Visualize revenue by product category",
                "force_heuristic": True,
            },
        ),
        ws,
    )
    assert revenue_viz.success
    assert len(ws.visualizations) > n_before
    assert any(c.get("chart_type") == "forecast" for c in ws.visualizations)
    assert any(
        c.get("chart_type") in {"bar", "pie"} for c in ws.visualizations
    )

    # Orchestrator follow-up must also preserve charts (no pre-run wipe)
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    forecast_ids = {c["id"] for c in ws.visualizations if c.get("chart_type") == "forecast"}
    ws = orch.run(
        "Create a bar chart of revenue by product category",
        workspace=ws,
    )
    assert forecast_ids.issubset({c["id"] for c in ws.visualizations})


def test_followup_forecast_uses_conversation_context():
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = _financials_meta()
    ctx = ConversationContext(
        conversation_summary="Analyzing Company A financials",
        current_goal="Analyze Company A's financial performance",
        active_entities=ActiveEntities(
            company="Company A",
            dataset=ws.dataset["name"],
            analysis="financial performance",
        ),
    )
    plan = orch._create_plan(
        "t1",
        "Now forecast company_revenue for the next 3 years.",
        ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Here is the performance summary."},
        ],
    )
    agents = {s.agent for s in plan.steps}
    assert "forecasting_agent" in agents
    assert "analysis_agent" not in agents

    ws2 = orch.run(
        "Now forecast company_revenue for the next 3 years.",
        workspace=ws,
        conversation_context=ctx,
        recent_messages=[
            {"role": "user", "content": "Analyze Company A's financial performance."},
            {"role": "assistant", "content": "Here is the performance summary."},
        ],
    )
    assert ctx.active_entities.company == "Company A"
    assert ws2.forecasts
    update_conversation_context(
        ctx,
        user_request="Now forecast company_revenue for the next 3 years.",
        workspace=ws2,
        recent_messages=[],
        llm=_FakeLLM(),  # type: ignore[arg-type]
    )
    assert any("forecast" in (f or "").lower() or "company_revenue" in (f or "") for f in ctx.findings) or (
        ctx.active_entities.analysis and "forecast" in ctx.active_entities.analysis
    )


def test_existing_anomaly_flow_still_works():
    """Regression: adding forecasting must not break anomaly routing."""
    orch = Orchestrator(llm=_FakeLLM())  # type: ignore[arg-type]
    ws = SharedWorkspace()
    ws.dataset = dataset_tools.create_ecommerce_orders(n_rows=400, seed=55)
    ws = orch.run("Detect anomalous transactions in the dataset.", workspace=ws)
    agents = {h["agent"] for h in ws.agent_history if h.get("success")}
    assert "anomaly_agent" in agents
    assert "forecasting_agent" not in agents
    assert ws.anomalies
