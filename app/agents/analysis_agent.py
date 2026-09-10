"""Analysis / Question Agent — LLM plans the query; DuckDB/pandas execute it."""

from __future__ import annotations

from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import query_tools


class AnalysisAgent:
    name = "analysis_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if not workspace.dataset:
                raise ValueError("No dataset loaded in shared workspace.")

            if action in {"answer_question", "analyze", "run_query"}:
                return self._analyze(message, workspace)
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action=action,
                success=False,
                error=f"Unknown analysis action: {action}",
                grounded=False,
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

    def _analyze(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        question = message.parameters.get("question") or message.parameters.get("user_request") or ""
        schema = workspace.dataset.get("schema", {})
        plan = message.parameters.get("query_plan") or self._plan_query(question, schema)

        if plan.get("mode") == "sql" and plan.get("sql"):
            result = query_tools.execute_sql(workspace.dataset, plan["sql"])
        else:
            result = query_tools.execute_aggregation(
                workspace.dataset,
                group_by=plan.get("group_by"),
                metric_column=plan.get("metric_column") or self._default_metric(schema),
                agg=plan.get("agg") or "sum",
                order_desc=plan.get("order_desc", True),
                limit=int(plan.get("limit") or 50),
            )

        explanation = self._explain(question, result, schema)
        payload = {
            "question": question,
            "query_plan": plan,
            "result": result,
            "explanation": explanation,
            "grounded": True,
        }
        workspace.analysis_results.append(payload)
        workspace.record(
            agent=self.name,
            action="answer_question",
            received={"question": question, "plan": plan},
            produced={"row_count": result.get("row_count", len(result.get("records", [])))},
            success=True,
        )
        return AgentResult(
            task_id=message.task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="answer_question",
            success=True,
            data=payload,
            grounded=True,
        )

    def _plan_query(self, question: str, schema: dict[str, Any]) -> dict[str, Any]:
        cols = list(schema.keys())
        system = (
            "You plan read-only analytics against a DuckDB table named data. "
            "Return JSON with either:\n"
            '  {"mode":"aggregation","group_by":"...|null","metric_column":"...","agg":"sum|mean|count|min|max","order_desc":true,"limit":20}\n'
            "or\n"
            '  {"mode":"sql","sql":"SELECT ..."}\n'
            "Use ONLY columns from the provided schema. Never invent numeric answers."
        )
        user = f"Question: {question}\nSchema columns: {cols}\nDtypes: {schema}"
        out = self.llm.chat_json(system, user)
        if out.get("_offline") or out.get("_fallback") or (
            out.get("mode") not in {"sql", "aggregation"} and "sql" not in out and "metric_column" not in out
        ):
            return self._heuristic_plan(question, cols)
        if out.get("sql") and not out.get("mode"):
            out["mode"] = "sql"
        if out.get("metric_column") and not out.get("mode"):
            out["mode"] = "aggregation"
        return out

    def _heuristic_plan(self, question: str, cols: list[str]) -> dict[str, Any]:
        q = question.lower()
        metric = self._pick(cols, ["total_amount", "revenue", "amount", "unit_price", "value"]) or cols[-1]
        if "categor" in q and ("revenue" in q or "most" in q or "generate" in q):
            group = self._pick(cols, ["product_category", "category"])
            return {
                "mode": "aggregation",
                "group_by": group,
                "metric_column": metric,
                "agg": "sum",
                "order_desc": True,
                "limit": 20,
            }
        if "state" in q and ("customer" in q or "most" in q or "how many" in q):
            group = self._pick(cols, ["state"])
            return {"mode": "aggregation", "group_by": group, "metric_column": metric, "agg": "count", "order_desc": True}
        if "average" in q or "avg" in q or "mean" in q:
            return {"mode": "aggregation", "group_by": None, "metric_column": metric, "agg": "mean"}
        if "over time" in q or "trend" in q:
            date_col = self._pick(cols, ["order_date", "date", "timestamp"])
            if date_col:
                return {
                    "mode": "sql",
                    "sql": (
                        f"SELECT date_trunc('month', {date_col}) AS period, "
                        f"SUM({metric}) AS value FROM data GROUP BY 1 ORDER BY 1"
                    ),
                }
        # default: top categories by metric if available
        group = self._pick(cols, ["product_category", "category", "state"])
        return {
            "mode": "aggregation",
            "group_by": group,
            "metric_column": metric,
            "agg": "sum",
            "order_desc": True,
            "limit": 20,
        }

    def _default_metric(self, schema: dict[str, Any]) -> str:
        return self._pick(list(schema.keys()), ["total_amount", "amount", "value", "unit_price"]) or list(schema.keys())[0]

    @staticmethod
    def _pick(cols: list[str], candidates: list[str]) -> str | None:
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        for c in cols:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    def _explain(self, question: str, result: dict[str, Any], schema: dict[str, Any]) -> str:
        records = result.get("records") or []
        system = (
            "Explain the analytical result in 2-4 concise sentences. "
            "Use ONLY the provided computed numbers. Do not invent values."
        )
        user = f"Question: {question}\nComputed result: {records[:15]}\nSummary stats: {result.get('summary_stats')}"
        text = self.llm.chat_text(system, user)
        if text and not text.startswith("(LLM unavailable"):
            return text
        # Offline grounded explanation
        if not records:
            return "The query executed successfully but returned no rows."
        if len(records) == 1 and "value" in records[0]:
            return f"Computed result for the question: {records[0]['value']} (from actual dataset aggregation)."
        top = records[0]
        keys = list(top.keys())
        return (
            f"Computed {len(records)} grouped result(s) from the dataset. "
            f"Top row: {top}. All values come from executed aggregation/SQL."
        )
