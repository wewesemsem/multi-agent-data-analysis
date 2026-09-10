"""Validation / Critic Agent — lightweight grounding checks before final response."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace


class ValidationAgent:
    name = "validation_agent"

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if action == "validate_dataset":
                report = self._validate_dataset(workspace, message.parameters)
            elif action == "validate_analysis":
                report = self._validate_analysis(workspace, message.parameters)
            elif action == "validate_anomalies":
                report = self._validate_anomalies(workspace, message.parameters)
            elif action == "validate_visualizations":
                report = self._validate_visualizations(workspace, message.parameters)
            elif action in {"validate_all", "critique"}:
                report = self._validate_all(workspace, message.parameters)
            else:
                return AgentResult(
                    task_id=message.task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    action=action,
                    success=False,
                    error=f"Unknown validation action: {action}",
                    grounded=False,
                )

            workspace.validation_reports.append(report)
            workspace.record(
                agent=self.name,
                action=action,
                received=message.parameters,
                produced={"ok": report["ok"], "issues": report.get("issues", [])},
                success=report["ok"],
                error=None if report["ok"] else "; ".join(report.get("issues", [])),
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action=action,
                success=report["ok"],
                data=report,
                error=None if report["ok"] else "; ".join(report.get("issues", [])),
                grounded=True,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action=action,
                success=False,
                error=str(exc),
                grounded=False,
            )

    def _validate_dataset(self, workspace: SharedWorkspace, params: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        ds = workspace.dataset
        if not ds:
            issues.append("Dataset missing from shared workspace.")
        else:
            if not ds.get("id"):
                issues.append("Dataset id missing.")
            loc = ds.get("location")
            if not loc or not Path(loc).exists():
                issues.append(f"Dataset file missing at {loc}.")
            if not ds.get("schema"):
                issues.append("Dataset schema missing.")
            required = params.get("required_columns") or []
            schema_cols = set((ds.get("schema") or {}).keys())
            for col in required:
                if col not in schema_cols:
                    issues.append(f"Required column missing: {col}")
            if not ds.get("row_count"):
                issues.append("Dataset row_count is zero or missing.")
            if ds.get("source") not in {"synthetic_generator", "csv_upload"}:
                issues.append("Dataset source is unrecognized.")
        return {"check": "dataset", "ok": not issues, "issues": issues}

    def _validate_analysis(self, workspace: SharedWorkspace, params: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        if not workspace.analysis_results:
            if params.get("required", True):
                issues.append("No analysis results present.")
        else:
            for i, item in enumerate(workspace.analysis_results):
                result = item.get("result") or {}
                if not result.get("grounded"):
                    issues.append(f"Analysis[{i}] is not marked grounded.")
                if "records" not in result:
                    issues.append(f"Analysis[{i}] missing computed records.")
                # Ensure explanation doesn't claim numbers without result
                if item.get("explanation") and not result.get("records") and not result.get("summary_stats"):
                    issues.append(f"Analysis[{i}] explanation without computed payload.")
        return {"check": "analysis", "ok": not issues, "issues": issues}

    def _validate_anomalies(self, workspace: SharedWorkspace, params: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        if not workspace.anomalies:
            if params.get("required", True):
                issues.append("No anomaly results present.")
        else:
            for i, item in enumerate(workspace.anomalies):
                if not item.get("grounded"):
                    issues.append(f"Anomaly[{i}] not grounded in a statistical method.")
                if not item.get("method"):
                    issues.append(f"Anomaly[{i}] missing method.")
                if "n_anomalies" not in item:
                    issues.append(f"Anomaly[{i}] missing n_anomalies.")
                if item.get("method") not in {"iqr", "zscore", "isolation_forest"}:
                    issues.append(f"Anomaly[{i}] method not in allowed set.")
        return {"check": "anomalies", "ok": not issues, "issues": issues}

    def _validate_visualizations(self, workspace: SharedWorkspace, params: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        if not workspace.visualizations:
            if params.get("required", True):
                issues.append("No visualizations present.")
        else:
            for i, viz in enumerate(workspace.visualizations):
                if not viz.get("grounded"):
                    issues.append(f"Visualization[{i}] not grounded.")
                html = viz.get("html_path")
                if not html or not Path(html).exists():
                    issues.append(f"Visualization[{i}] HTML artifact missing.")
                if not viz.get("n_points"):
                    issues.append(f"Visualization[{i}] has zero data points.")
                if viz.get("chart_type") not in {"bar", "line", "scatter", "histogram"}:
                    issues.append(f"Visualization[{i}] has unsupported chart_type.")
        return {"check": "visualizations", "ok": not issues, "issues": issues}

    def _validate_all(self, workspace: SharedWorkspace, params: dict[str, Any]) -> dict[str, Any]:
        expected = set(params.get("expected_outputs") or [])
        checks = []
        checks.append(self._validate_dataset(workspace, params))
        if "analysis" in expected or not expected:
            # If plan included analysis steps, require analysis when expected
            require_analysis = "analysis" in expected if expected else bool(workspace.analysis_results) or params.get(
                "require_analysis", False
            )
            checks.append(self._validate_analysis(workspace, {**params, "required": require_analysis}))
        if "anomalies" in expected or params.get("require_anomalies"):
            checks.append(self._validate_anomalies(workspace, {**params, "required": True}))
        elif workspace.anomalies:
            checks.append(self._validate_anomalies(workspace, {**params, "required": True}))
        if "visualizations" in expected or params.get("require_visualizations"):
            checks.append(self._validate_visualizations(workspace, {**params, "required": True}))
        elif workspace.visualizations:
            checks.append(self._validate_visualizations(workspace, {**params, "required": True}))

        # Consistency: final claims should map to tool outputs in history
        history_ok = any(h.get("success") and h.get("agent") != "orchestrator" for h in workspace.agent_history)
        if not history_ok and (workspace.analysis_results or workspace.anomalies or workspace.visualizations):
            checks.append(
                {
                    "check": "history",
                    "ok": False,
                    "issues": ["Outputs exist without successful specialized-agent history entries."],
                }
            )

        issues = [i for c in checks for i in c.get("issues", [])]
        return {
            "check": "all",
            "ok": not issues,
            "issues": issues,
            "subchecks": checks,
            "retry_targets": self._retry_targets(issues),
        }

    @staticmethod
    def _retry_targets(issues: list[str]) -> list[str]:
        targets = []
        joined = " ".join(issues).lower()
        if "dataset" in joined:
            targets.append("dataset_agent")
        if "analysis" in joined:
            targets.append("analysis_agent")
        if "anomaly" in joined:
            targets.append("anomaly_agent")
        if "visual" in joined:
            targets.append("visualization_agent")
        return targets
