"""Orchestrator / Root Agent — plans, delegates, maintains shared state, synthesizes response.

Follows Google Cloud MAS / agentic data-science guidance:
root coordinator delegates to specialized agents; tools do computation; critic validates.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from app.agents.analysis_agent import AnalysisAgent
from app.agents.anomaly_agent import AnomalyAgent
from app.agents.dataset_agent import DatasetAgent
from app.agents.drafting_agent import DraftingAgent
from app.agents.forecasting_agent import ForecastingAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.visualization_agent import VisualizationAgent
from app.context import (
    ConversationContext,
    format_recent_messages,
    recent_messages_window,
    update_conversation_context,
    workspace_context_summary,
)
from app.llm import LLMClient, LLMUnavailableError, offline_heuristics_allowed
from app.messages import AgentMessage, AgentResult, ExecutionPlan, PlanStep
from app.state import SharedWorkspace, ensure_workspace
from app.tools import draft_tools


@dataclass(frozen=True)
class RequestIntents:
    """Capability flags inferred from the user request + workspace (not LLM)."""

    dataset: bool = False
    explore: bool = False
    analysis: bool = False
    anomaly: bool = False
    forecast: bool = False
    visualization: bool = False
    draft: bool = False

    @property
    def any_specialist(self) -> bool:
        return any(
            (
                self.dataset,
                self.explore,
                self.analysis,
                self.anomaly,
                self.forecast,
                self.visualization,
                self.draft,
            )
        )

    @property
    def needs_loaded_dataset(self) -> bool:
        """True when the request needs data but did not ask to create/synthesize any."""
        return (not self.dataset) and any(
            (self.explore, self.analysis, self.anomaly, self.forecast, self.visualization)
        )

    def allowed_agents(self) -> set[str]:
        agents = {"validation_agent"}
        if self.dataset or self.explore:
            agents.add("dataset_agent")
        if self.explore or self.analysis:
            agents.add("analysis_agent")
        if self.anomaly:
            agents.add("anomaly_agent")
        if self.forecast:
            agents.add("forecasting_agent")
        if self.visualization:
            agents.add("visualization_agent")
        if self.draft:
            agents.add("drafting_agent")
        return agents


class Orchestrator:
    name = "orchestrator"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.dataset_agent = DatasetAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.anomaly_agent = AnomalyAgent(self.llm)
        self.forecasting_agent = ForecastingAgent(self.llm)
        self.visualization_agent = VisualizationAgent(self.llm)
        self.drafting_agent = DraftingAgent(self.llm)
        self.validation_agent = ValidationAgent()
        self._handlers: dict[str, Callable[[AgentMessage, SharedWorkspace], AgentResult]] = {
            "dataset_agent": self.dataset_agent.handle,
            "analysis_agent": self.analysis_agent.handle,
            "anomaly_agent": self.anomaly_agent.handle,
            "forecasting_agent": self.forecasting_agent.handle,
            "visualization_agent": self.visualization_agent.handle,
            "drafting_agent": self.drafting_agent.handle,
            "validation_agent": self.validation_agent.handle,
        }

    def run(
        self,
        user_request: str,
        workspace: SharedWorkspace | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        conversation_context: ConversationContext | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> SharedWorkspace:
        ensure_workspace()
        workspace = workspace or SharedWorkspace()
        ctx = conversation_context if conversation_context is not None else ConversationContext()
        recent = recent_messages_window(recent_messages)
        workspace.task_status = "planning"
        task_id = str(uuid.uuid4())

        def emit(event: dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(event)

        intents = self._infer_intents(
            user_request,
            has_dataset=workspace.dataset is not None,
            conversation_context=ctx,
            workspace=workspace,
        )
        if not intents.any_specialist:
            workspace.task_status = "completed"
            workspace.plan = []
            workspace.final_response = self._no_task_response(user_request, workspace)
            workspace.record(
                agent=self.name,
                action="no_data_task",
                received={"user_request": user_request},
                produced={"response": workspace.final_response},
                success=True,
            )
            emit({"type": "final", "status": workspace.task_status, "response": workspace.final_response})
            if conversation_context is not None:
                update_conversation_context(
                    ctx,
                    user_request=user_request,
                    workspace=workspace,
                    recent_messages=recent,
                    llm=self.llm,
                )
            return workspace

        # Fail closed before planning — do not mislabel "no dataset" as LLM unavailable.
        # (OpenAI often returns an empty step list here; that used to surface as LLMUnavailableError.)
        if workspace.dataset is None and intents.needs_loaded_dataset:
            workspace.task_status = "failed"
            workspace.plan = []
            workspace.final_response = (
                "**No dataset loaded.** Upload a CSV, or run **Create 10k e-commerce orders** "
                "first — then try Explore / analysis again.\n\n"
                "The system will not invent data for exploration prompts."
            )
            workspace.errors.append("No dataset loaded in shared workspace")
            workspace.record(
                agent=self.name,
                action="missing_dataset",
                received={"user_request": user_request},
                produced={"response": workspace.final_response},
                success=False,
                error="No dataset loaded in shared workspace",
            )
            emit({"type": "final", "status": workspace.task_status, "response": workspace.final_response})
            return workspace

        try:
            plan = self._create_plan(
                task_id,
                user_request,
                workspace,
                conversation_context=ctx,
                recent_messages=recent,
                intents=intents,
            )
        except LLMUnavailableError as exc:
            workspace.task_status = "failed"
            err = str(exc)
            if "no usable plan" in err.lower() or "invalid plan" in err.lower():
                workspace.final_response = (
                    f"**Planning failed:** {err}\n\n"
                    "The model responded but did not return runnable agent steps. "
                    "Try again, switch model/provider in the sidebar, or rephrase the request. "
                    "If you have not loaded data yet, upload a CSV or run **Create 10k e-commerce orders** first."
                )
            else:
                workspace.final_response = (
                    f"**LLM unavailable:** {err}\n\n"
                    "Configure a provider API key in `.env` to run live. "
                    "Upload a CSV (or explicitly ask to create synthetic data) before analysis."
                )
            workspace.errors.append(str(exc))
            workspace.record(
                agent=self.name,
                action="create_plan",
                received={"user_request": user_request},
                success=False,
                error=str(exc),
            )
            emit({"type": "final", "status": workspace.task_status, "response": workspace.final_response})
            return workspace
        if not plan.steps:
            workspace.task_status = "failed"
            workspace.plan = []
            workspace.final_response = (
                "**No dataset loaded.** Upload a CSV, or run **Create 10k e-commerce orders** "
                "first — then try Explore / analysis again.\n\n"
                "The system will not invent data for exploration prompts."
                if workspace.dataset is None and intents.needs_loaded_dataset
                else (
                    f"**Planning failed:** {plan.summary or 'LLM returned no usable plan steps.'}\n\n"
                    "Try rephrasing the request, or switch provider/model in the sidebar."
                )
            )
            workspace.errors.append(plan.summary or "Empty execution plan")
            workspace.record(
                agent=self.name,
                action="create_plan",
                received={"user_request": user_request},
                success=False,
                error=plan.summary or "Empty execution plan",
            )
            emit({"type": "final", "status": workspace.task_status, "response": workspace.final_response})
            return workspace
        workspace.plan = [s.model_dump() for s in plan.steps]
        expected_outputs = self._expected_outputs(plan)
        # Replace outputs this run will regenerate so repeated clicks don't stack
        # duplicates. Charts are cumulative until workspace reset — clearing them
        # here would drop prior forecast charts when the user asks for a new viz.
        # Keep results the plan does not touch so stepped sidebar examples can
        # build on each other.
        if "anomalies" in expected_outputs:
            workspace.anomalies = []
        if "analysis" in expected_outputs:
            workspace.analysis_results = []
        if "forecasts" in expected_outputs:
            workspace.forecasts = []
            # Drop stale forecast charts and misleading "Revenue Forecast (line)"
            # artifacts so this run's grounded forecast chart is what the UI shows.
            workspace.visualizations = [
                v
                for v in (workspace.visualizations or [])
                if v.get("chart_type") != "forecast"
                and "forecast" not in str(v.get("title") or "").lower()
            ]
        if "drafts" in expected_outputs:
            workspace.drafts = []
        workspace.validation_reports = []
        workspace.errors = []
        workspace.final_response = None

        workspace.record(
            agent=self.name,
            action="create_plan",
            received={
                "user_request": user_request,
                "conversation_context": ctx.to_dict(),
                "recent_messages": recent,
            },
            produced={"steps": [s.agent for s in plan.steps], "summary": plan.summary},
            success=True,
        )
        emit({"type": "plan", "plan": workspace.plan, "summary": plan.summary})

        workspace.task_status = "executing"
        max_retries = 1
        pending = list(plan.steps)
        short_term = self._agent_context_payload(ctx, recent)

        while pending:
            step = pending.pop(0)
            emit({"type": "step_start", "step": step.model_dump()})
            # Visualization/analysis agents should own their tool specs.
            # LLM planner params often invent invalid column names (e.g. metric_column=null).
            step_params = dict(step.parameters or {})
            if step.agent in {
                "visualization_agent",
                "analysis_agent",
                "anomaly_agent",
                "forecasting_agent",
            }:
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
                context=(
                    short_term
                    if step.agent
                    in {
                        "analysis_agent",
                        "anomaly_agent",
                        "visualization_agent",
                        "forecasting_agent",
                        "drafting_agent",
                    }
                    else {}
                ),
            )
            handler = self._handlers[step.agent]
            result = handler(message, workspace)
            # One automatic heuristic retry for visualization failures
            if (
                not result.success
                and step.agent == "visualization_agent"
                and any(
                    k in (result.error or "").lower()
                    for k in ("metric", "heatmap", "z column", "numeric column", "dual_axis", "spec")
                )
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
                if conversation_context is not None:
                    update_conversation_context(
                        ctx,
                        user_request=user_request,
                        workspace=workspace,
                        recent_messages=recent,
                        llm=self.llm,
                    )
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
                    "require_forecasts": "forecasts" in expected_outputs,
                    "require_visualizations": "visualizations" in expected_outputs,
                    "require_drafts": "drafts" in expected_outputs,
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
                if target == "drafting_agent":
                    # Drop failing drafts so the retry replaces rather than stacks them
                    workspace.drafts = []
                message = AgentMessage(
                    task_id=task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    target_agent=retry_step.agent,
                    action=retry_step.action,
                    dataset_id=(workspace.dataset or {}).get("id"),
                    parameters={**retry_step.parameters, "user_request": user_request},
                    context=(
                        short_term
                        if retry_step.agent
                        in {
                            "analysis_agent",
                            "anomaly_agent",
                            "visualization_agent",
                            "forecasting_agent",
                            "drafting_agent",
                        }
                        else {}
                    ),
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
                        "require_forecasts": "forecasts" in expected_outputs,
                        "require_visualizations": "visualizations" in expected_outputs,
                        "require_drafts": "drafts" in expected_outputs,
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

        # Dedicated short-term memory update (not inline field juggling in the plan loop)
        if conversation_context is not None:
            update_conversation_context(
                ctx,
                user_request=user_request,
                workspace=workspace,
                recent_messages=recent,
                llm=self.llm,
            )
            workspace.record(
                agent=self.name,
                action="update_conversation_context",
                received={"user_request": user_request},
                produced={"conversation_context": ctx.to_dict()},
                success=True,
            )
            emit({"type": "context", "conversation_context": ctx.to_dict()})
        return workspace

    def _create_plan(
        self,
        task_id: str,
        user_request: str,
        workspace: SharedWorkspace,
        *,
        conversation_context: ConversationContext | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        intents: RequestIntents | None = None,
    ) -> ExecutionPlan:
        has_dataset = workspace.dataset is not None
        ctx = conversation_context or ConversationContext()
        intents = intents or self._infer_intents(
            user_request,
            has_dataset=has_dataset,
            conversation_context=ctx,
            workspace=workspace,
        )
        heuristic = self._heuristic_plan(
            task_id,
            user_request,
            has_dataset,
            conversation_context=ctx,
            workspace=workspace,
            intents=intents,
        )

        # When the LLM is available, trust its plan (reconcile only drops unauthorized agents).
        # Heuristic-only planning is reserved for explicit offline mode (CI / MAS_ALLOW_OFFLINE_HEURISTICS).
        system = (
            "You are a root orchestrator for a multi-agent data intelligence system. "
            "Return JSON with keys summary and steps. Each step: "
            "step_id, agent (dataset_agent|analysis_agent|anomaly_agent|forecasting_agent|"
            "visualization_agent|drafting_agent|validation_agent), "
            "action, parameters, rationale. "
            "Do NOT include heavy computation in the orchestrator. "
            "Only include agents needed for the request. "
            "If the user message has no data task (greeting, thanks, small talk), "
            "return steps as an empty list. "
            "Use conversation context and recent messages to resolve follow-ups "
            "(e.g. 'what about profitability?', 'now forecast it', 'summarize everything', "
            "'give me a creditworthiness assessment') "
            "without requiring the user to repeat the company, dataset, or prior goal. "
            "Typical actions: create_dataset, explore_dataset, answer_question, explore, correlate, "
            "detect_anomalies, forecast, create_visualization, draft, validate_all. "
            "ONLY include dataset_agent create_dataset / generate_dataset when the user explicitly "
            "asks to create, generate, or synthesize a dataset. "
            "If no dataset is loaded and the user did not ask to create one, do NOT invent data — "
            "return steps that fail closed (empty list or a single validation note) so the UI can "
            "ask them to upload a CSV. "
            "For forecast / predict / project future values: use forecasting_agent action=forecast "
            "(then visualization_agent if a chart is useful). "
            "Do NOT use drafting_agent for a plain forecast/predict request — drafting is only for "
            "summaries, reports, memos, recommendations, explanations, or assessments the user asked for. "
            "For summaries, reports, memos, recommendations, forecast explanations, anomaly summaries, "
            "or creditworthiness / credit risk / financial health assessments: use drafting_agent "
            "action=draft AFTER any needed specialist agents. Drafting synthesizes existing evidence — "
            "it must NOT recalculate metrics. Do NOT invent a credit assessment specialist agent. "
            "If the user only asks to summarize/report/assess and workspace already has relevant "
            "analysis/forecast/anomaly outputs, drafting_agent alone (plus validation) is enough. "
            "If they ask to analyze/forecast AND write a report/assessment, run those specialists first, "
            "then drafting_agent. "
            "For explore / profile / missing values / quantiles / how columns relate: "
            "MUST include analysis_agent action=explore AND dataset_agent action=explore_dataset "
            "when a dataset is already loaded. "
            "For category summaries asking for spread + unique counts + percent share: "
            "use analysis_agent action=answer_question. "
            "For box plot / heatmap / dual-axis requests: use visualization_agent create_visualization. "
            "For detect_anomalies, parameters.method MUST be one of: iqr, zscore, isolation_forest, auto "
            "(use auto when unsure; never invent method names like 'statistical'). "
            "Chart types may include bar, line, scatter, histogram, pie, box, heatmap, dual_axis. "
            "Use ONLY column names from the workspace schema — never assume ecommerce column names."
        )
        user = (
            f"User request: {user_request}\n"
            f"Dataset already loaded: {has_dataset}\n"
            f"Detected capability intents: {intents}\n"
            f"{workspace_context_summary(workspace)}\n"
            f"{ctx.to_prompt_block()}\n"
            f"Recent conversation:\n{format_recent_messages(recent_messages)}"
        )
        out = self.llm.chat_json(system, user)

        def _offline_heuristic(reason: str) -> ExecutionPlan:
            """Keyword plan only when offline heuristics are explicitly allowed (CI)."""
            if offline_heuristics_allowed():
                heuristic.summary = f"Heuristic orchestrator plan ({reason})"
                return heuristic
            raise LLMUnavailableError(reason)

        # API failures always stop — never execute a keyword plan after a broken LLM call.
        if out.get("_fallback"):
            err = out.get("_llm_error") or "unknown error"
            raise LLMUnavailableError(f"LLM call failed: {err}")
        # No API key: CI may fall back to heuristics; live mode fails loudly.
        if out.get("_offline"):
            return _offline_heuristic("no API key")
        if not isinstance(out.get("steps"), list):
            raise LLMUnavailableError("LLM returned an invalid plan (missing steps list)")
        def _intent_fallback(reason: str) -> ExecutionPlan:
            """Use capability heuristic when the LLM answered but produced no runnable steps.

            Distinct from API/key failure: those still raise. Empty/malformed plans for a
            known intent (e.g. Explore) should not surface as 'LLM unavailable'.
            """
            if not has_dataset and intents.needs_loaded_dataset:
                return ExecutionPlan(
                    task_id=task_id,
                    user_request=user_request,
                    steps=[],
                    summary=out.get("summary") or "No dataset loaded — cannot plan analysis",
                )
            if heuristic.steps:
                heuristic.summary = (
                    f"{out.get('summary') or heuristic.summary or 'Execution plan'} "
                    f"(intent fallback: {reason})"
                )
                return heuristic
            if not intents.any_specialist:
                return ExecutionPlan(
                    task_id=task_id,
                    user_request=user_request,
                    steps=[],
                    summary=out.get("summary") or "No data task in request",
                )
            raise LLMUnavailableError("LLM returned no usable plan steps for this request")

        # Empty steps are valid when no specialist intent was detected, or when the
        # model fail-closes because no dataset is loaded (see system prompt).
        if not out["steps"]:
            if not intents.any_specialist:
                return ExecutionPlan(
                    task_id=task_id,
                    user_request=user_request,
                    steps=[],
                    summary=out.get("summary") or "No data task in request",
                )
            return _intent_fallback("empty steps")

        steps = []
        for i, raw in enumerate(out["steps"]):
            if not isinstance(raw, dict):
                continue
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
            return _intent_fallback("unparseable steps")

        # One visualization step is enough — each call already builds all requested charts.
        # Multiple viz steps caused 3× duplication (e.g. 3 steps × box/heatmap/dual = 9).
        deduped: list[PlanStep] = []
        saw_viz = False
        for step in steps:
            if step.agent == "visualization_agent":
                if saw_viz:
                    continue
                saw_viz = True
            deduped.append(step)
        steps = deduped

        if not any(s.agent == "validation_agent" for s in steps):
            steps.append(
                PlanStep(
                    step_id=f"s{len(steps) + 1}",
                    agent="validation_agent",
                    action="validate_all",
                    parameters={},
                    rationale="Critic checks grounding and consistency",
                )
            )

        llm_plan = ExecutionPlan(
            task_id=task_id,
            user_request=user_request,
            steps=steps,
            summary=out.get("summary") or "LLM-generated execution plan",
        )
        reconciled = self._dedupe_viz_steps(self._reconcile_plan(llm_plan, intents))
        if not reconciled.steps:
            return _intent_fallback("reconcile dropped all steps")
        return reconciled

    @staticmethod
    def _dedupe_viz_steps(plan: ExecutionPlan) -> ExecutionPlan:
        """Keep a single visualization_agent step (also after augment)."""
        deduped: list[PlanStep] = []
        saw_viz = False
        for step in plan.steps:
            if step.agent == "visualization_agent":
                if saw_viz:
                    continue
                saw_viz = True
            deduped.append(step)
        for i, step in enumerate(deduped, 1):
            step.step_id = f"s{i}"
        return ExecutionPlan(
            task_id=plan.task_id,
            user_request=plan.user_request,
            steps=deduped,
            summary=plan.summary,
        )

    def _reconcile_plan(self, plan: ExecutionPlan, intents: RequestIntents) -> ExecutionPlan:
        """Drop LLM-invented specialists that capability intents did not authorize."""
        allowed = intents.allowed_agents()
        kept = [s for s in plan.steps if s.agent in allowed]
        # Always keep a final validation step when any specialist remains
        if kept and not any(s.agent == "validation_agent" for s in kept):
            kept.append(
                PlanStep(
                    step_id="s_val",
                    agent="validation_agent",
                    action="validate_all",
                    parameters={},
                    rationale="Critic checks grounding and consistency",
                )
            )
        for i, step in enumerate(kept, 1):
            step.step_id = f"s{i}"
        return ExecutionPlan(
            task_id=plan.task_id,
            user_request=plan.user_request,
            steps=kept,
            summary=(plan.summary or "Execution plan").rstrip() + " (intents reconciled)",
        )

    def _infer_intents(
        self,
        user_request: str,
        *,
        has_dataset: bool,
        conversation_context: ConversationContext | None = None,
        workspace: SharedWorkspace | None = None,
    ) -> RequestIntents:
        """Infer which specialist capabilities the request actually asks for."""
        q = (user_request or "").lower()
        ctx = conversation_context or ConversationContext()
        has_prior_context = not ctx.is_empty()
        ws = workspace
        has_analysis = bool(ws and ws.analysis_results)
        has_forecasts = bool(ws and ws.forecasts)
        has_anomalies = bool(ws and ws.anomalies)
        has_evidence = has_analysis or has_forecasts or has_anomalies

        needs_draft = draft_tools.request_needs_draft(user_request)
        # Only synthesize/load when the user explicitly asks — never auto-create for analysis.
        needs_dataset = (not has_dataset) and any(
            k in q
            for k in (
                "create a synthetic",
                "create synthetic",
                "generate a synthetic",
                "generate synthetic",
                "create a dataset",
                "generate a dataset",
                "create an ecommerce",
                "create e-commerce",
                "synthetic ecommerce",
                "synthetic e-commerce",
                "synthetic dataset",
                "generate ecommerce",
                "generate e-commerce",
            )
        )

        needs_explore = any(
            k in q
            for k in (
                "explore the data",
                "explore data",
                "eda",
                "profile",
                "correlat",
                "how columns relate",
                "relationship between",
                "missing value",
                "blank value",
                "blank cells",
                "quantile",
                "percentile",
                "typical ranges",
                "links, gaps",
                "columns relate",
            )
        )
        needs_agg_summary = any(
            k in q
            for k in (
                "percent",
                "percentage",
                "share of",
                "unique",
                "distinct",
                "standard deviation",
                "spread",
                "variability",
            )
        )
        needs_forecast = any(
            k in q
            for k in (
                "forecast",
                "predict",
                "prediction",
                "projection",
                "project the next",
                "project next",
                "estimate future",
                "future revenue",
                "future sales",
                "over the next",
                "what will",
                "look like over",
            )
        )
        # Pure explain-the-forecast should not re-forecast when one already exists.
        explain_forecast = any(
            k in q
            for k in (
                "explain the forecast",
                "forecast explanation",
                "explain this forecast",
                "explain the projection",
            )
        )
        if explain_forecast and has_forecasts and "and forecast" not in q:
            needs_forecast = False
            needs_draft = True

        followup = has_prior_context and self._looks_like_followup(q)
        asks_new_analysis = any(
            k in q
            for k in (
                "analyze",
                "analyse",
                "which",
                "how much",
                "how many",
                "detect",
                "outlier",
            )
        )
        draft_only_followup = (
            followup
            and needs_draft
            and has_evidence
            and not needs_forecast
            and not needs_explore
            and not needs_agg_summary
            and not asks_new_analysis
        )
        # Pure forecast / drafting follow-ups should not also force a generic analysis step.
        analysis_followup = followup and not needs_forecast and not draft_only_followup
        explicit_analysis = any(
            k in q
            for k in (
                "which",
                "how much",
                "how many",
                "average",
                "tell me",
                "what is",
                "what about",
                "profit",
                "category",
                "question",
                "analyze",
                "analyse",
            )
        )
        needs_analysis = needs_agg_summary or analysis_followup or (
            (not needs_explore)
            and (not needs_forecast)
            and (not draft_only_followup)
            and explicit_analysis
        )
        if (
            needs_draft
            and has_evidence
            and not asks_new_analysis
            and not needs_agg_summary
            and not needs_explore
        ):
            needs_analysis = False
        if (
            needs_draft
            and not has_evidence
            and draft_tools.draft_needs_supporting_analysis(user_request)
            and not needs_forecast
            and not needs_explore
        ):
            needs_analysis = True
        if needs_draft and asks_new_analysis:
            needs_analysis = True

        needs_anomaly = any(k in q for k in ("anomal", "outlier", "unusual"))
        needs_viz = any(
            k in q
            for k in (
                "visual",
                "chart",
                "plot",
                "graph",
                "distribution",
                "show me",
                "box plot",
                "boxplot",
                "heatmap",
                "heat map",
                "dual axis",
                "dual-axis",
            )
        )
        if needs_forecast:
            needs_viz = True
        if needs_draft and not needs_forecast and not needs_viz:
            needs_viz = False

        return RequestIntents(
            dataset=needs_dataset,
            explore=needs_explore,
            analysis=needs_analysis,
            anomaly=needs_anomaly,
            forecast=needs_forecast,
            visualization=needs_viz,
            draft=needs_draft,
        )

    def _heuristic_plan(
        self,
        task_id: str,
        user_request: str,
        has_dataset: bool,
        *,
        conversation_context: ConversationContext | None = None,
        workspace: SharedWorkspace | None = None,
        intents: RequestIntents | None = None,
    ) -> ExecutionPlan:
        steps: list[PlanStep] = []
        n = 1
        intents = intents or self._infer_intents(
            user_request,
            has_dataset=has_dataset,
            conversation_context=conversation_context,
            workspace=workspace,
        )

        if intents.dataset:
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

        if intents.explore:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="analysis_agent",
                    action="explore",
                    parameters={"question": user_request},
                    rationale="Run EDA tools (correlation, missingness, quantiles)",
                )
            )
            n += 1
            # Also refresh dataset-side EDA profile for the Dataset panel
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="dataset_agent",
                    action="explore_dataset",
                    parameters={},
                    rationale="Attach EDA profile to the shared dataset",
                )
            )
            n += 1
        if intents.analysis:
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
        if intents.anomaly:
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
        if intents.forecast:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="forecasting_agent",
                    action="forecast",
                    parameters={"question": user_request},
                    rationale="Generate a quantitative forecast from the active time series",
                )
            )
            n += 1
        if intents.visualization:
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
        if intents.draft:
            steps.append(
                PlanStep(
                    step_id=f"s{n}",
                    agent="drafting_agent",
                    action="draft",
                    parameters={"question": user_request},
                    rationale="Synthesize validated specialist outputs into the requested artifact",
                )
            )
            n += 1

        if steps:
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

    def _no_task_response(self, user_request: str, workspace: SharedWorkspace) -> str:
        """Reply when the message has no detectable data-specialist intent."""
        ds = workspace.dataset or {}
        if ds:
            cols = list((ds.get("schema") or {}).keys())
            col_preview = ", ".join(f"`{c}`" for c in cols[:8])
            more = f" (+{len(cols) - 8} more)" if len(cols) > 8 else ""
            return (
                f"Hi — I'm ready when you are. There's an active dataset "
                f"**{ds.get('name')}** ({ds.get('row_count')} rows"
                f"{'; columns: ' + col_preview + more if cols else ''}). "
                "Ask a question, request a forecast, chart, anomaly check, or draft."
            )
        return (
            "Hi — ask me to create or load a dataset, analyze it, forecast a metric, "
            "detect anomalies, chart results, or draft a report."
        )

    @staticmethod
    def _expected_outputs(plan: ExecutionPlan) -> set[str]:
        out: set[str] = set()
        for s in plan.steps:
            if s.agent == "analysis_agent":
                out.add("analysis")
            elif s.agent == "anomaly_agent":
                out.add("anomalies")
            elif s.agent == "forecasting_agent":
                out.add("forecasts")
            elif s.agent == "visualization_agent":
                out.add("visualizations")
            elif s.agent == "drafting_agent":
                out.add("drafts")
            elif s.agent == "dataset_agent":
                out.add("dataset")
        return out

    @staticmethod
    def _looks_like_followup(q: str) -> bool:
        """Detect short continuations that rely on prior ConversationContext."""
        markers = (
            "what about",
            "how about",
            "now ",
            "also ",
            "and the ",
            "same ",
            "forecast",
            "creditworth",
            "credit grade",
            "credit assessment",
            "grade it",
            "grade for",
            "scenario",
            "profitability",
            "what if",
            "summar",
            "write a report",
            "explain the",
            "turn these",
            "everything",
        )
        if any(m in q for m in markers):
            return True
        # Pronoun-heavy short follow-ups: "now forecast it."
        words = q.strip().split()
        return len(words) <= 8 and any(p in words for p in ("it", "that", "this", "them"))

    @staticmethod
    def _agent_context_payload(
        ctx: ConversationContext,
        recent: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "conversation_context": ctx.to_dict(),
            "conversation_prompt": ctx.to_prompt_block(),
            "recent_messages": recent,
        }

    def _synthesize(
        self,
        user_request: str,
        workspace: SharedWorkspace,
        validation: AgentResult,
    ) -> str:
        # Prefer drafts only when the user actually asked for a drafted artifact.
        if workspace.drafts and draft_tools.request_needs_draft(user_request):
            draft = workspace.drafts[-1]
            draft_body = (draft.get("content") or "").strip()
            if draft_body:
                parts = [draft_body]
                if validation.success:
                    parts.append("\n### Validation\nAll grounding checks passed.")
                else:
                    issues = (validation.data or {}).get("issues") or [validation.error]
                    parts.append("\n### Validation warnings")
                    for issue in issues if isinstance(issues, list) else [issues]:
                        parts.append(f"- {issue}")
                parts.append(
                    "\n---\n"
                    "_Draft synthesized from validated specialist outputs "
                    "(numbers were not invented by the drafting agent)._"
                )
                grounded_report = "\n".join(parts)
                # Keep structured draft content as-is (already sectioned).
                return grounded_report

        parts: list[str] = []
        parts.append("## Results")
        parts.append(f"**Request:** {user_request}")

        if workspace.dataset:
            ds = workspace.dataset
            cols = list((ds.get("schema") or {}).keys())
            parts.append("\n### Dataset")
            parts.append(f"- **Name:** {ds.get('name')}")
            parts.append(f"- **Rows:** {ds.get('row_count')}")
            parts.append(f"- **Columns:** {', '.join(f'`{c}`' for c in cols)}")

        for i, analysis in enumerate(workspace.analysis_results, 1):
            result = analysis.get("result") or {}
            records = result.get("records") or []
            mode = (analysis.get("query_plan") or {}).get("mode")
            parts.append(f"\n### Analysis {i}")
            explanation = (analysis.get("explanation") or "").strip()
            if explanation:
                parts.append(explanation)

            if mode == "eda" or result.get("operation") == "eda":
                pairs = (result.get("correlation") or {}).get("pairs") or records
                miss = result.get("missingness") or {}
                qstats = (result.get("quantiles") or {}).get("stats") or {}

                parts.append("\n#### How columns relate")
                if pairs:
                    parts.append("| Column A | Column B | Correlation |")
                    parts.append("| --- | --- | ---: |")
                    for p in pairs[:8]:
                        corr = p.get("correlation")
                        try:
                            corr_s = f"{float(corr):.3f}"
                        except (TypeError, ValueError):
                            corr_s = str(corr)
                        parts.append(
                            f"| `{p.get('column_a')}` | `{p.get('column_b')}` | **{corr_s}** |"
                        )

                parts.append("\n#### Missing values")
                parts.append(
                    f"- Blank cells: **{miss.get('total_nulls', 0)}** "
                    f"({(miss.get('overall_null_rate') or 0) * 100:.2f}% of all cells)"
                )

                if qstats:
                    parts.append("\n#### Typical ranges")
                    parts.append("| Column | p25 | Median | p75 |")
                    parts.append("| --- | ---: | ---: | ---: |")
                    for col, stats in list(qstats.items())[:8]:
                        parts.append(
                            f"| `{col}` | {_fmt_num(stats.get('p25'))} | "
                            f"{_fmt_num(stats.get('p50'))} | {_fmt_num(stats.get('p75'))} |"
                        )

            elif mode == "multi_aggregation" or (
                records and any(isinstance(r, dict) and "agg" in r for r in records)
            ):
                by_agg: dict[str, list[dict[str, Any]]] = {}
                for row in records:
                    if not isinstance(row, dict):
                        continue
                    by_agg.setdefault(str(row.get("agg") or "value"), []).append(row)
                labels = {
                    "std": "Spread (std)",
                    "nunique": "Unique count",
                    "pct": "Percent share",
                }
                for agg, rows in by_agg.items():
                    parts.append(f"\n#### {labels.get(agg, agg)}")
                    keys = [k for k in rows[0].keys() if k != "agg"]
                    if not keys:
                        continue
                    parts.append("| " + " | ".join(keys) + " |")
                    parts.append("| " + " | ".join("---" for _ in keys) + " |")
                    for row in rows[:12]:
                        cells = []
                        for k in keys:
                            v = row.get(k)
                            cells.append(_fmt_num(v) if isinstance(v, (int, float)) or _is_number(v) else f"`{v}`" if k != "value" else _fmt_num(v))
                        parts.append("| " + " | ".join(cells) + " |")

            elif records:
                parts.append("\n#### Top results")
                keys = list(records[0].keys())
                parts.append("| " + " | ".join(keys) + " |")
                parts.append("| " + " | ".join("---" for _ in keys) + " |")
                for row in records[:10]:
                    cells = [_fmt_cell(row.get(k)) for k in keys]
                    parts.append("| " + " | ".join(cells) + " |")

        for i, anomaly in enumerate(workspace.anomalies, 1):
            parts.append(f"\n### Anomalies {i}")
            parts.append(
                f"- Method: **{anomaly.get('method')}** on `{anomaly.get('column')}`\n"
                f"- Count: **{anomaly.get('n_anomalies')}** "
                f"({(anomaly.get('anomaly_rate') or 0)*100:.2f}% of rows)"
            )
            explanation = (anomaly.get("explanation") or "").strip()
            if explanation:
                parts.append(explanation)
            sample = anomaly.get("records") or []
            if sample:
                keys = [
                    k
                    for k in sample[0].keys()
                    if k in {"order_id", "total_amount", "_anomaly_score", "product_category", "state"}
                ]
                if keys:
                    parts.append("\n#### Sample anomalies")
                    parts.append("| " + " | ".join(keys) + " |")
                    parts.append("| " + " | ".join("---" for _ in keys) + " |")
                    for row in sample[:5]:
                        parts.append("| " + " | ".join(_fmt_cell(row.get(k)) for k in keys) + " |")

        for i, forecast in enumerate(workspace.forecasts, 1):
            parts.append(f"\n### Forecast {i}")
            if not forecast.get("suitable"):
                parts.append(forecast.get("explanation") or forecast.get("reason") or "Forecast not suitable.")
                reqs = forecast.get("requirements") or []
                for req in reqs:
                    parts.append(f"- {req}")
                continue
            parts.append(
                f"- Target: `{forecast.get('target_column')}` over `{forecast.get('time_column')}`\n"
                f"- Frequency: **{forecast.get('frequency')}** · Horizon: **{forecast.get('forecast_horizon')}**\n"
                f"- Method: **{forecast.get('selected_method')}** "
                f"(candidates: {', '.join(forecast.get('candidate_methods') or [])})\n"
                f"- Trend: {forecast.get('trend')} · "
                f"Historical points: {forecast.get('historical_observations')}"
            )
            explanation = (forecast.get("explanation") or "").strip()
            if explanation:
                parts.append(explanation)
            metrics = forecast.get("evaluation_metrics") or []
            if metrics:
                parts.append("\n#### Holdout metrics")
                parts.append("| Method | MAE | RMSE | MAPE |")
                parts.append("| --- | ---: | ---: | ---: |")
                for row in metrics:
                    mape = row.get("mape")
                    mape_s = "—" if mape is None else f"{mape:.2f}%"
                    parts.append(
                        f"| `{row.get('method')}` | {_fmt_num(row.get('mae'))} | "
                        f"{_fmt_num(row.get('rmse'))} | {mape_s} |"
                    )
            sample = forecast.get("forecast_values") or []
            if sample:
                parts.append("\n#### Forecast values")
                parts.append("| Time | Value | Lower | Upper |")
                parts.append("| --- | ---: | ---: | ---: |")
                for row in sample[:12]:
                    parts.append(
                        f"| {_fmt_cell(row.get('time'))} | {_fmt_num(row.get('value'))} | "
                        f"{_fmt_num(row.get('lower'))} | {_fmt_num(row.get('upper'))} |"
                    )
            for warning in forecast.get("warnings") or []:
                parts.append(f"- Warning: {warning}")
            for assumption in forecast.get("assumptions") or []:
                parts.append(f"- Assumption: {assumption}")

        if workspace.visualizations:
            parts.append("\n### Visualizations")
            for viz in workspace.visualizations:
                parts.append(
                    f"- **{viz.get('title')}** — `{viz.get('chart_type')}` "
                    f"({viz.get('n_points')} points)"
                )
            parts.append("_Open the **Charts** tab to view them._")

        if validation.success:
            parts.append("\n### Validation")
            parts.append("All grounding checks passed.")
        else:
            issues = (validation.data or {}).get("issues") or [validation.error]
            parts.append("\n### Validation warnings")
            for issue in issues if isinstance(issues, list) else [issues]:
                parts.append(f"- {issue}")

        parts.append(
            "\n---\n"
            "_Numbers above come from deterministic tools "
            "(not invented by the language model)._"
        )

        grounded_report = "\n".join(parts)
        has_structured = any(
            (a.get("query_plan") or {}).get("mode") in {"eda", "multi_aggregation"}
            or (a.get("result") or {}).get("operation") == "eda"
            for a in workspace.analysis_results
        )
        if has_structured:
            return grounded_report

        system = (
            "Rewrite the following grounded report into a clear final answer for chat. "
            "Keep markdown headings and bullet lists. Keep every number exactly as given. "
            "Do not add claims that are not in the report."
        )
        polished = self.llm.chat_text(system, grounded_report)
        if polished and not polished.startswith("(LLM unavailable") and len(polished) > 40:
            return polished
        return grounded_report


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1000:
        return f"{f:,.2f}"
    if abs(f) >= 1:
        return f"{f:.2f}"
    return f"{f:.4f}"


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return not isinstance(v, bool)
    except (TypeError, ValueError):
        return False


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if _is_number(v):
        return _fmt_num(v)
    return str(v)
