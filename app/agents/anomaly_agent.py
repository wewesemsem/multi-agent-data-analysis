"""Anomaly Detection Agent — method selection via LLM/heuristics; decisions via stats/ML tools."""

from __future__ import annotations

from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import anomaly_tools


class AnomalyAgent:
    name = "anomaly_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if not workspace.dataset:
                raise ValueError("No dataset loaded in shared workspace.")
            if action not in {"detect_anomalies", "find_outliers"}:
                return AgentResult(
                    task_id=message.task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    action=action,
                    success=False,
                    error=f"Unknown anomaly action: {action}",
                    grounded=False,
                )

            params = self._resolve_params(message.parameters, workspace)
            result = anomaly_tools.detect_anomalies(
                workspace.dataset,
                column=params.get("column"),
                method=params.get("method") or "auto",
                z_threshold=float(params.get("z_threshold") or 3.0),
                iqr_multiplier=float(params.get("iqr_multiplier") or 1.5),
                contamination=float(params.get("contamination") or 0.02),
                max_records=int(params.get("max_records") or 25),
            )
            explanation = self._explain(result)
            payload = {**result, "explanation": explanation}
            workspace.anomalies.append(payload)
            workspace.record(
                agent=self.name,
                action="detect_anomalies",
                received=params,
                produced={"n_anomalies": result["n_anomalies"], "method": result["method"]},
                success=True,
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="detect_anomalies",
                success=True,
                data=payload,
                grounded=True,
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

    def _resolve_params(self, parameters: dict[str, Any], workspace: SharedWorkspace) -> dict[str, Any]:
        if parameters.get("column") and parameters.get("method"):
            return parameters
        schema = workspace.dataset.get("schema", {})
        system = (
            "Choose anomaly detection parameters. Return JSON: "
            '{"column":"...","method":"iqr|zscore|isolation_forest|auto"}. '
            "Prefer amount/total columns for transactions. Use iqr for skewed spend data."
        )
        out = self.llm.chat_json(system, f"Request params: {parameters}\nSchema: {schema}")
        if out.get("_offline") or out.get("_fallback") or "column" not in out:
            cols = list(schema.keys())
            column = next(
                (c for c in cols if any(k in c.lower() for k in ("total_amount", "amount", "price", "value"))),
                None,
            )
            return {
                "column": parameters.get("column") or column,
                "method": parameters.get("method") or "auto",
            }
        return {
            "column": parameters.get("column") or out.get("column"),
            "method": parameters.get("method") or out.get("method") or "auto",
            **{k: parameters[k] for k in parameters if k not in {"column", "method"}},
        }

    def _explain(self, result: dict[str, Any]) -> str:
        facts = result.get("explanation_facts") or {}
        system = (
            "Explain anomaly findings in 2-4 sentences using ONLY provided facts. "
            "Mention method, column, count, and why values are unusual. Do not invent records."
        )
        user = (
            f"n_anomalies={result.get('n_anomalies')} method={result.get('method')} "
            f"column={result.get('column')} params={result.get('parameters')} "
            f"facts={facts} sample={result.get('records', [])[:3]}"
        )
        text = self.llm.chat_text(system, user)
        if text and not text.startswith("(LLM unavailable"):
            return text
        params = result.get("parameters") or {}
        return (
            f"Detected {result.get('n_anomalies')} anomalous rows on '{result.get('column')}' "
            f"using {result.get('method')} "
            f"(parameters: {params}). "
            f"Scores and labels come from the statistical/ML tool, not from LLM judgment."
        )
