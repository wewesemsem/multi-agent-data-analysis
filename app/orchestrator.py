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
        has_dataset = workspace.dataset is not None
        heuristic = self._heuristic_plan(task_id, user_request, has_dataset)

        # When the LLM is available, always ask it to plan — then augment so
        # structured toolkit intents (explore / multi-agg / new charts) cannot be dropped.
        # Heuristic-only planning is reserved for true offline / LLM failure.
        system = (
            "You are a root orchestrator for a multi-agent data intelligence MVP. "
            "Return JSON with keys summary and steps. Each step: "
            "step_id, agent (dataset_agent|analysis_agent|anomaly_agent|visualization_agent|validation_agent), "
            "action, parameters, rationale. "
            "Do NOT include heavy computation in the orchestrator. "
            "Only include agents needed for the request. "
            "Typical actions: create_dataset, explore_dataset, answer_question, explore, correlate, "
            "detect_anomalies, create_visualization, validate_all. "
            "For explore / profile / missing values / quantiles / how columns relate: "
            "MUST include analysis_agent action=explore AND dataset_agent action=explore_dataset "
            "(create the dataset first if none exists). "
            "For category summaries asking for spread + unique counts + percent share: "
            "use analysis_agent action=answer_question. "
            "For box plot / heatmap / dual-axis requests: use visualization_agent create_visualization. "
            "For detect_anomalies, parameters.method MUST be one of: iqr, zscore, isolation_forest, auto "
            "(use auto when unsure; never invent method names like 'statistical'). "
            "Chart types may include bar, line, scatter, histogram, pie, box, heatmap, dual_axis."
        )
        user = f"User request: {user_request}\nDataset already loaded: {has_dataset}"
        out = self.llm.chat_json(system, user)
        if out.get("_offline") or out.get("_fallback") or not isinstance(out.get("steps"), list) or not out["steps"]:
            if out.get("_offline"):
                reason = "no API key"
            elif out.get("_fallback"):
                reason = f"LLM call failed: {out.get('_llm_error', 'unknown error')}"
            else:
                reason = "LLM returned an invalid plan"
            heuristic.summary = f"Heuristic orchestrator plan ({reason})"
            return heuristic

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
            heuristic.summary = "Heuristic orchestrator plan (LLM returned no usable steps)"
            return heuristic

        llm_plan = ExecutionPlan(
            task_id=task_id,
            user_request=user_request,
            steps=steps,
            summary=out.get("summary") or "LLM-generated execution plan",
        )
        return self._augment_plan(llm_plan, heuristic)

    def _augment_plan(self, llm_plan: ExecutionPlan, heuristic: ExecutionPlan) -> ExecutionPlan:
        """Ensure capability steps from the heuristic plan are not dropped by the LLM."""
        core = [s for s in llm_plan.steps if s.agent != "validation_agent"]
        validators = [s for s in llm_plan.steps if s.agent == "validation_agent"]
        if not validators:
            validators = [s for s in heuristic.steps if s.agent == "validation_agent"]

        def _has(actions: set[str], *, agent: str | None = None) -> bool:
            return any(
                s.action in actions and (agent is None or s.agent == agent) for s in core
            )

        for hs in heuristic.steps:
            if hs.agent == "validation_agent":
                continue
            if hs.action in {"create_dataset", "generate_dataset", "load_csv"}:
                if any(s.agent == "dataset_agent" for s in core):
                    continue
                core.insert(0, hs.model_copy(deep=True))
                continue
            if hs.action in {"explore", "correlate", "eda"}:
                if _has({"explore", "correlate", "eda"}, agent="analysis_agent"):
                    continue
                core.append(hs.model_copy(deep=True))
                continue
            if hs.action in {"explore_dataset", "profile_dataset"}:
                if _has({"explore_dataset", "profile_dataset"}, agent="dataset_agent"):
                    continue
                core.append(hs.model_copy(deep=True))
                continue
            if hs.action == "answer_question":
                if any(s.agent == "analysis_agent" for s in core):
                    continue
                core.append(hs.model_copy(deep=True))
                continue
            if hs.action in {"create_visualization", "visualize", "create_chart"}:
                if any(s.agent == "visualization_agent" for s in core):
                    continue
                core.append(hs.model_copy(deep=True))
                continue
            if hs.action == "detect_anomalies":
                if any(s.agent == "anomaly_agent" for s in core):
                    continue
                core.append(hs.model_copy(deep=True))
                continue

        merged = core + validators
        for i, step in enumerate(merged, 1):
            step.step_id = f"s{i}"
        return ExecutionPlan(
            task_id=llm_plan.task_id,
            user_request=llm_plan.user_request,
            steps=merged,
            summary=(llm_plan.summary or "LLM-generated execution plan").rstrip()
            + " (capability steps ensured)",
        )

    def _heuristic_plan(self, task_id: str, user_request: str, has_dataset: bool) -> ExecutionPlan:
        q = user_request.lower()
        steps: list[PlanStep] = []
        n = 1

        needs_dataset = (not has_dataset) and any(
            k in q for k in ("create", "generate", "synthetic", "dataset", "load", "csv")
        )
        # Also create if no dataset and request implies analysis / explore / charts
        if not has_dataset and any(
            k in q
            for k in (
                "anomaly",
                "visual",
                "revenue",
                "category",
                "orders",
                "explore",
                "correlat",
                "missing",
                "profile",
                "box",
                "heatmap",
                "heat map",
                "dual",
            )
        ):
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
        needs_analysis = needs_agg_summary or (
            (not needs_explore)
            and any(
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
        )
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

        if needs_explore:
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
