"""Orchestrator / Root Agent — plans, delegates, maintains shared state, synthesizes response.

Follows Google Cloud MAS / agentic data-science guidance:
root coordinator delegates to specialized agents; tools do computation; critic validates.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from app.agents.analysis_agent import AnalysisAgent
from app.agents.anomaly_agent import AnomalyAgent
from app.agents.dataset_agent import DatasetAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.visualization_agent import VisualizationAgent
from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult, ExecutionPlan, PlanStep
from app.state import SharedWorkspace, ensure_workspace


class Orchestrator:
    name = "orchestrator"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.dataset_agent = DatasetAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.anomaly_agent = AnomalyAgent(self.llm)
        self.visualization_agent = VisualizationAgent(self.llm)
        self.validation_agent = ValidationAgent()
        self._handlers: dict[str, Callable[[AgentMessage, SharedWorkspace], AgentResult]] = {
            "dataset_agent": self.dataset_agent.handle,
            "analysis_agent": self.analysis_agent.handle,
            "anomaly_agent": self.anomaly_agent.handle,
            "visualization_agent": self.visualization_agent.handle,
            "validation_agent": self.validation_agent.handle,
        }

    def run(
        self,
        user_request: str,
        workspace: SharedWorkspace | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> SharedWorkspace:
        ensure_workspace()
        workspace = workspace or SharedWorkspace()
        workspace.task_status = "planning"
        task_id = str(uuid.uuid4())

        def emit(event: dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(event)

        plan = self._create_plan(task_id, user_request, workspace)
        workspace.plan = [s.model_dump() for s in plan.steps]
        workspace.record(
            agent=self.name,
            action="create_plan",
            received={"user_request": user_request},
            produced={"steps": [s.agent for s in plan.steps], "summary": plan.summary},
            success=True,
        )
        emit({"type": "plan", "plan": workspace.plan, "summary": plan.summary})

        workspace.task_status = "executing"
        max_retries = 1
        pending = list(plan.steps)
        expected_outputs = self._expected_outputs(plan)

        while pending:
            step = pending.pop(0)
            emit({"type": "step_start", "step": step.model_dump()})
            # Visualization/analysis agents should own their tool specs.
            # LLM planner params often invent invalid column names (e.g. metric_column=null).
            step_params = dict(step.parameters or {})
            if step.agent in {"visualization_agent", "analysis_agent", "anomaly_agent"}:
                step_params.pop("specs", None)
                step_params.pop("metric_column", None)
                step_params.pop("aggregation", None)
            message = AgentMessage(
                task_id=task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                target_agent=step.agent,
                action=step.action,
                dataset_id=(workspace.dataset or {}).get("id"),
                parameters={**step_params, "user_request": user_request},
            )
            handler = self._handlers[step.agent]
            result = handler(message, workspace)
            # One automatic heuristic retry for visualization failures
            if (
                not result.success
                and step.agent == "visualization_agent"
                and "metric" in (result.error or "").lower()
            ):
                emit({"type": "retry", "agent": "visualization_agent", "reason": result.error})
                retry_msg = AgentMessage(
                    task_id=task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    target_agent="visualization_agent",
                    action="create_visualization",
                    dataset_id=(workspace.dataset or {}).get("id"),
                    parameters={
                        "user_request": user_request,
                        "description": user_request,
                        "force_heuristic": True,
                    },
                )
                result = handler(retry_msg, workspace)
            emit(
                {
                    "type": "step_end",
                    "step": step.model_dump(),
                    "success": result.success,
                    "error": result.error,
                    "data_keys": list(result.data.keys()),
                }
            )

            # After dataset creation, validate immediately
            if step.agent == "dataset_agent" and result.success:
                vmsg = AgentMessage(
                    task_id=task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    target_agent="validation_agent",
                    action="validate_dataset",
                    dataset_id=(workspace.dataset or {}).get("id"),
                    parameters={},
                )
                vres = self.validation_agent.handle(vmsg, workspace)
                emit({"type": "validation", "check": "dataset", "ok": vres.success, "issues": (vres.data or {}).get("issues")})
                if not vres.success and max_retries > 0:
                    max_retries -= 1
                    pending.insert(0, step)  # retry dataset
                    continue

            if not result.success and step.agent != "validation_agent":
                workspace.task_status = "error"
                workspace.final_response = (
                    f"Step `{step.agent}:{step.action}` failed: {result.error}. "
                    "No fabricated results were returned."
                )
                emit({"type": "error", "message": workspace.final_response})
                return workspace

        # Final critic pass
        workspace.task_status = "validating"
        final_validation = self.validation_agent.handle(
            AgentMessage(
                task_id=task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                target_agent="validation_agent",
                action="validate_all",
                parameters={
                    "expected_outputs": list(expected_outputs),
                    "require_analysis": "analysis" in expected_outputs,
                    "require_anomalies": "anomalies" in expected_outputs,
                    "require_visualizations": "visualizations" in expected_outputs,
                },
            ),
            workspace,
        )
        emit(
            {
                "type": "validation",
                "check": "all",
                "ok": final_validation.success,
                "issues": (final_validation.data or {}).get("issues"),
            }
        )

        # Simple correction loop: re-run failed specialist once
        if not final_validation.success:
            targets = (final_validation.data or {}).get("retry_targets") or []
            for target in targets:
                retry_step = next((s for s in plan.steps if s.agent == target), None)
                if not retry_step:
                    continue
                emit({"type": "retry", "agent": target})
                message = AgentMessage(
                    task_id=task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    target_agent=retry_step.agent,
                    action=retry_step.action,
                    dataset_id=(workspace.dataset or {}).get("id"),
                    parameters={**retry_step.parameters, "user_request": user_request},
                )
                self._handlers[target](message, workspace)

            final_validation = self.validation_agent.handle(
                AgentMessage(
                    task_id=task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    target_agent="validation_agent",
                    action="validate_all",
                    parameters={
                        "expected_outputs": list(expected_outputs),
                        "require_analysis": "analysis" in expected_outputs,
                        "require_anomalies": "anomalies" in expected_outputs,
                        "require_visualizations": "visualizations" in expected_outputs,
                    },
                ),
                workspace,
            )

        workspace.final_response = self._synthesize(user_request, workspace, final_validation)
        workspace.task_status = "completed" if final_validation.success else "completed_with_warnings"
        workspace.record(
            agent=self.name,
            action="synthesize",
            received={"user_request": user_request},
            produced={"task_status": workspace.task_status},
            success=True,
        )
        emit({"type": "final", "status": workspace.task_status, "response": workspace.final_response})
        return workspace

    def _create_plan(self, task_id: str, user_request: str, workspace: SharedWorkspace) -> ExecutionPlan:
        system = (
            "You are a root orchestrator for a multi-agent data intelligence MVP. "
            "Return JSON with keys summary and steps. Each step: "
            "step_id, agent (dataset_agent|analysis_agent|anomaly_agent|visualization_agent|validation_agent), "
            "action, parameters, rationale. "
            "Do NOT include heavy computation in the orchestrator. "
            "Only include agents needed for the request. "
            "Typical actions: create_dataset, answer_question, detect_anomalies, create_visualization, validate_all."
        )
        has_dataset = workspace.dataset is not None
        user = f"User request: {user_request}\nDataset already loaded: {has_dataset}"
        out = self.llm.chat_json(system, user)
        if out.get("_offline") or out.get("_fallback") or not isinstance(out.get("steps"), list) or not out["steps"]:
            return self._heuristic_plan(task_id, user_request, has_dataset)

        steps = []
        for i, raw in enumerate(out["steps"]):
            try:
                steps.append(
                    PlanStep(
                        step_id=str(raw.get("step_id") or f"s{i+1}"),
                        agent=raw["agent"],
                        action=raw["action"],
                        parameters=raw.get("parameters") or {},
                        depends_on=raw.get("depends_on") or [],
                        rationale=raw.get("rationale") or "",
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        if not steps:
            return self._heuristic_plan(task_id, user_request, has_dataset)
        return ExecutionPlan(
            task_id=task_id,
            user_request=user_request,
            steps=steps,
            summary=out.get("summary") or "LLM-generated execution plan",
        )

    def _heuristic_plan(self, task_id: str, user_request: str, has_dataset: bool) -> ExecutionPlan:
        q = user_request.lower()
        steps: list[PlanStep] = []
        n = 1

        needs_dataset = (not has_dataset) and any(
            k in q for k in ("create", "generate", "synthetic", "dataset", "load", "csv")
        )
        # Also create if no dataset and request implies analysis
        if not has_dataset and any(k in q for k in ("anomaly", "visual", "revenue", "category", "orders")):
            needs_dataset = True

        if needs_dataset:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="dataset_agent",
                    action="create_dataset",
                    parameters={},
                    rationale="Create/load dataset into shared workspace",
                )
            )
            n += 1

        needs_analysis = any(
            k in q
            for k in (
                "which",
                "how much",
                "how many",
                "average",
                "revenue",
                "tell me",
                "what is",
                "category",
                "question",
            )
        )
        needs_anomaly = any(k in q for k in ("anomal", "outlier", "unusual"))
        needs_viz = any(k in q for k in ("visual", "chart", "plot", "graph", "distribution", "show me"))

        if needs_analysis:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="analysis_agent",
                    action="answer_question",
                    parameters={"question": user_request},
                    rationale="Answer analytical question from actual data",
                )
            )
            n += 1
        if needs_anomaly:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="anomaly_agent",
                    action="detect_anomalies",
                    parameters={"method": "auto"},
                    rationale="Detect anomalies with statistical/ML tools",
                )
            )
            n += 1
        if needs_viz:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="visualization_agent",
                    action="create_visualization",
                    parameters={"description": user_request},
                    rationale="Render data-driven charts from computed data",
                )
            )
            n += 1

        steps.append(
            PlanStep(
                step_id=f"s{n}",
                agent="validation_agent",
                action="validate_all",
                parameters={},
                rationale="Critic checks grounding and consistency",
            )
        )

        return ExecutionPlan(
            task_id=task_id,
            user_request=user_request,
            steps=steps,
            summary="Heuristic orchestrator plan (offline or LLM fallback)",
        )

    @staticmethod
    def _expected_outputs(plan: ExecutionPlan) -> set[str]:
        out: set[str] = set()
        for s in plan.steps:
            if s.agent == "analysis_agent":
                out.add("analysis")
            elif s.agent == "anomaly_agent":
                out.add("anomalies")
            elif s.agent == "visualization_agent":
                out.add("visualizations")
            elif s.agent == "dataset_agent":
                out.add("dataset")
        return out

    def _synthesize(
        self,
        user_request: str,
        workspace: SharedWorkspace,
        validation: AgentResult,
    ) -> str:
        parts: list[str] = []
        parts.append("## Results")
        parts.append(f"**Request:** {user_request}")

        if workspace.dataset:
            ds = workspace.dataset
            parts.append(
                f"\n### Dataset\n"
                f"- ID: `{ds.get('id')}`\n"
                f"- Name: {ds.get('name')}\n"
                f"- Rows: **{ds.get('row_count')}** (generated/loaded by Dataset Agent tools)\n"
                f"- Columns: {', '.join((ds.get('schema') or {}).keys())}"
            )

        for i, analysis in enumerate(workspace.analysis_results, 1):
            result = analysis.get("result") or {}
            records = result.get("records") or []
            parts.append(f"\n### Analysis {i}")
            parts.append(analysis.get("explanation") or "Analysis completed.")
            if records:
                preview = records[:8]
                parts.append(f"Top computed rows: `{preview}`")

        for i, anomaly in enumerate(workspace.anomalies, 1):
            parts.append(f"\n### Anomalies {i}")
            parts.append(
                f"Method: **{anomaly.get('method')}** on `{anomaly.get('column')}` — "
                f"**{anomaly.get('n_anomalies')}** anomalies "
                f"({(anomaly.get('anomaly_rate') or 0)*100:.2f}% of rows)."
            )
            parts.append(anomaly.get("explanation") or "")
            sample = anomaly.get("records") or []
            if sample:
                keys = [k for k in sample[0].keys() if k in {"order_id", "total_amount", "_anomaly_score", "product_category", "state"}]
                slim = [{k: r.get(k) for k in keys} for r in sample[:5]]
                parts.append(f"Sample anomalies: `{slim}`")

        if workspace.visualizations:
            parts.append("\n### Visualizations")
            for viz in workspace.visualizations:
                parts.append(
                    f"- **{viz.get('title')}** ({viz.get('chart_type')}, {viz.get('n_points')} points) "
                    f"— chart id `{viz.get('id')}`"
                )

        if validation.success:
            parts.append("\n### Validation\nCritic Agent: all grounding checks passed.")
        else:
            issues = (validation.data or {}).get("issues") or [validation.error]
            parts.append(f"\n### Validation warnings\n{issues}")

        parts.append(
            "\n_All numerical claims above are grounded in deterministic tool execution "
            "(dataset generation, SQL/aggregation, statistical anomaly detection, Plotly rendering)._"
        )

        # Optional LLM polish — still constrained to computed facts
        system = (
            "Rewrite the following grounded report into a concise final answer for the user. "
            "Do NOT add any numbers or claims not present in the report."
        )
        polished = self.llm.chat_text(system, "\n".join(parts))
        if polished and not polished.startswith("(LLM unavailable") and len(polished) > 40:
            return polished
        return "\n".join(parts)
