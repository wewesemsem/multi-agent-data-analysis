"""Forecasting Agent — LLM/heuristics configure; deterministic tools compute forecasts."""

from __future__ import annotations

import re
from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import forecast_tools
from app.tools.schema_resolve import resolve_metric_columns, rank_columns_for_request


class ForecastingAgent:
    name = "forecasting_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if not workspace.dataset:
                raise ValueError("No dataset loaded in shared workspace.")
            if action not in {"forecast", "run_forecast", "create_forecast"}:
                return AgentResult(
                    task_id=message.task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    action=action,
                    success=False,
                    error=f"Unknown forecasting action: {action}",
                    grounded=False,
                )

            params = self._resolve_params(message.parameters, workspace, message.context)
            targets = params.get("target_columns") or []
            if params.get("target_column") and params["target_column"] not in targets:
                targets = [params["target_column"], *targets]
            if not targets:
                targets = [None]  # let the tool inspect/select

            payloads: list[dict[str, Any]] = []
            for target in targets[:5]:
                result = forecast_tools.forecast_series(
                    workspace.dataset,
                    target_column=target,
                    time_column=params.get("time_column"),
                    horizon=params.get("horizon"),
                    user_request=params.get("user_request"),
                    method=params.get("method"),
                    aggregate=params.get("aggregate") or "sum",
                )
                explanation = self._explain(result, params)
                payload = {**result, "explanation": explanation}
                workspace.forecasts.append(payload)
                payloads.append(payload)

            primary = payloads[0]
            workspace.record(
                agent=self.name,
                action="forecast",
                received={k: v for k, v in params.items() if k != "user_request"},
                produced={
                    "n_forecasts": len(payloads),
                    "suitable": primary.get("suitable"),
                    "selected_method": primary.get("selected_method"),
                    "forecast_horizon": primary.get("forecast_horizon"),
                    "target_columns": [p.get("target_column") for p in payloads],
                },
                success=True,
                limitations=list(primary.get("warnings") or []),
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="forecast",
                success=True,
                data={"forecasts": payloads, **primary},
                grounded=True,
                limitations=list(primary.get("warnings") or []),
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

    def _resolve_params(
        self,
        parameters: dict[str, Any],
        workspace: SharedWorkspace,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = (
            parameters.get("user_request")
            or parameters.get("question")
            or parameters.get("description")
            or ""
        )
        schema = (workspace.dataset or {}).get("schema", {})
        cols = list(schema.keys())
        params: dict[str, Any] = {
            "user_request": request,
            "target_column": parameters.get("target_column") or parameters.get("column"),
            "target_columns": list(parameters.get("target_columns") or []),
            "time_column": parameters.get("time_column"),
            "horizon": parameters.get("horizon") or parameters.get("forecast_horizon"),
            "method": parameters.get("method") or "auto",
            "aggregate": parameters.get("aggregate") or "sum",
        }

        # Schema-aware NL resolution (no dataset-specific column hardcoding)
        if params["target_column"] and params["target_column"] not in schema:
            params["target_column"] = None
        params["target_columns"] = [c for c in params["target_columns"] if c in schema]

        ranked = rank_columns_for_request(request, cols)
        extracted = [c for c, _ in ranked]
        if not params["target_column"] and extracted:
            params["target_column"] = extracted[0]
        for col in extracted:
            if col not in params["target_columns"]:
                params["target_columns"].append(col)

        # Method: only honor a model name the USER typed. Planner/LLM often injects
        # causal_regression and produces flat nonsense on transaction data.
        params["method"] = _method_from_user_request(request) or "auto"

        # Horizon with units ("12 months") must be owned by forecast_tools after
        # grain resampling — drop bare planner integers like horizon=12.
        if _request_has_horizon_units(request):
            params["horizon"] = None

        if params["target_column"] and (
            params.get("horizon") is not None or _request_has_horizon_units(request)
        ):
            return params

        ranked_txt = ", ".join(f"{c} ({s:.1f})" for c, s in ranked[:8]) or "(none)"
        system = (
            "Choose forecasting parameters for a generic tabular time-series dataset. "
            "Return JSON with keys: target_column (string|null), "
            "target_columns (array of strings), time_column (string|null), "
            "horizon (always null — tools parse units like '12 months'), "
            "method (always 'auto'). "
            "Use only column names present in the schema. "
            "Ranked schema matches for the user's metric language are provided — "
            "prefer the highest-ranked column unless the user clearly named another. "
            "If unsure, leave fields null so tools can inspect the dataset. "
            "Do not invent numeric forecasts."
        )
        ctx_prompt = ""
        if context:
            ctx_prompt = str(context.get("conversation_prompt") or "")
        out = self.llm.chat_json(
            system,
            f"Request: {request}\nSchema: {schema}\n"
            f"Ranked metric column matches: {ranked_txt}\n"
            f"Provided params: {parameters}\n{ctx_prompt}",
        )
        if out.get("_offline") or out.get("_fallback"):
            return params

        llm_target = out.get("target_column")
        if not params["target_column"] and isinstance(llm_target, str) and llm_target in schema:
            # Accept LLM pick only if it is competitive with schema scoring
            best = ranked[0][1] if ranked else 0.0
            llm_score = next((s for c, s in ranked if c == llm_target), 0.0)
            if not ranked or llm_score >= best - 1.0:
                params["target_column"] = llm_target
            elif ranked:
                params["target_column"] = ranked[0][0]
        for col in out.get("target_columns") or []:
            if isinstance(col, str) and col in schema and col not in params["target_columns"]:
                params["target_columns"].append(col)
        if not params["time_column"]:
            tc = out.get("time_column")
            if isinstance(tc, str) and tc in schema:
                params["time_column"] = tc
        # Never take horizon/method from the param-selection LLM — tools own grain.
        if not params["target_column"] and not params["target_columns"]:
            params["target_columns"] = resolve_metric_columns(request, cols)
            if params["target_columns"]:
                params["target_column"] = params["target_columns"][0]
        return params

    def _explain(self, result: dict[str, Any], params: dict[str, Any]) -> str:
        if not result.get("suitable"):
            reason = result.get("reason") or "Forecasting is not appropriate for this dataset."
            reqs = result.get("requirements") or []
            req_txt = (" Requirements: " + "; ".join(reqs)) if reqs else ""
            return f"{reason}{req_txt} No forecast values were fabricated."

        facts = {
            "target": result.get("target_column"),
            "time": result.get("time_column"),
            "frequency": result.get("frequency"),
            "horizon": result.get("forecast_horizon"),
            "method": result.get("selected_method"),
            "n_hist": result.get("historical_observations"),
            "trend": result.get("trend"),
            "seasonality": result.get("seasonality"),
            "metrics": result.get("evaluation_metrics"),
            "baseline": result.get("baseline_metrics"),
            "assumptions": result.get("assumptions"),
            "warnings": result.get("warnings"),
            "sample_forecast": (result.get("forecast_values") or [])[:3],
        }
        system = (
            "Explain the forecast in 3-5 sentences using ONLY provided facts. "
            "Mention target, horizon, selected method, and that selection used holdout error metrics. "
            "Do not invent numbers beyond the facts. Note assumptions/warnings briefly."
        )
        text = self.llm.chat_text(system, str(facts))
        if text and not text.startswith("(LLM unavailable"):
            return text

        fc = result.get("forecast_values") or []
        head = ", ".join(
            f"{row.get('time')}→{_fmt(row.get('value'))}" for row in fc[:3]
        )
        return (
            f"Forecasted `{result.get('target_column')}` for the next "
            f"{result.get('forecast_horizon')} {result.get('frequency')} period(s) "
            f"using **{result.get('selected_method')}** "
            f"(selected via holdout MAE/RMSE against candidates "
            f"{result.get('candidate_methods')}). "
            f"Trend appears {result.get('trend')}. "
            f"Next values: {head}. "
            "Values come from the forecasting tool, not LLM judgment."
        )


def _fmt(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1000:
        return f"{f:,.2f}"
    return f"{f:.4g}"


def _request_has_horizon_units(text: str) -> bool:
    q = (text or "").lower()
    return bool(
        re.search(
            r"\d+\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)\b"
            r"|(?:next|coming)\s+(?:day|week|month|quarter|year)\b"
            r"|\bmonthly\b|\bweekly\b|\bquarterly\b|\byearly\b|\bdaily\b",
            q,
        )
    )


def _method_from_user_request(text: str) -> str | None:
    """Return a forced method only when the user explicitly named one."""
    q = (text or "").lower()
    named = {
        "naive": "naive",
        "moving average": "moving_average",
        "moving_average": "moving_average",
        "exponential smoothing": "exponential_smoothing",
        "exponential_smoothing": "exponential_smoothing",
        "seasonal naive": "seasonal_naive",
        "seasonal_naive": "seasonal_naive",
        "trend regression": "trend_regression",
        "linear regression": "trend_regression",
        "causal regression": "causal_regression",
        "causal_regression": "causal_regression",
    }
    for needle, method in named.items():
        if needle in q:
            return method
    return None
