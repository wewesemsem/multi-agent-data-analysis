"""Analysis / Question Agent — LLM plans the query; DuckDB/pandas execute it."""

from __future__ import annotations

from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.schema_utils import (
    default_group_column,
    default_id_column,
    default_metric_column,
    default_time_column,
    match_column,
)
from app.state import SharedWorkspace
from app.tools import eda_tools, query_tools


class AnalysisAgent:
    name = "analysis_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if not workspace.dataset:
                raise ValueError("No dataset loaded in shared workspace.")

            if action in {"correlate", "explore", "eda"}:
                return self._eda(message, workspace, action=action)
            if action in {"answer_question", "analyze", "run_query"}:
                question = self._resolve_question(message)
                # Pure EDA questions (and not multi-summary) → eda tools
                if self._wants_eda(question) and not self._wants_multi_summary(question):
                    return self._eda(message, workspace, action="explore")
                if self._wants_multi_summary(question):
                    return self._multi_agg(message, workspace)
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

    @staticmethod
    def _wants_eda(question: str) -> bool:
        q = question.lower()
        return any(
            k in q
            for k in (
                "correlat",
                "how columns relate",
                "columns relate",
                "relationship between",
                "missing value",
                "blank value",
                "typical ranges",
                "quantile",
                "percentile",
                "explore the data",
                "eda",
            )
        )

    @staticmethod
    def _wants_multi_summary(question: str) -> bool:
        """True when the user asks for several summary stats at once (spread + unique + %)."""
        q = question.lower()
        hits = 0
        if any(k in q for k in ("spread", "std", "standard deviation", "variability")):
            hits += 1
        if any(k in q for k in ("unique", "distinct", "how many different")):
            hits += 1
        if any(k in q for k in ("percent", "percentage", "share of", "pct", "%")):
            hits += 1
        return hits >= 2

    def _multi_agg(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        question = message.parameters.get("question") or message.parameters.get("user_request") or ""
        schema = workspace.dataset.get("schema", {})
        cols = list(schema.keys())
        metric = (
            match_column(cols, "revenue", "amount", "sales", "value", "price", "total")
            or default_metric_column(schema, cols)
            or cols[-1]
        )
        group = (
            match_column(cols, "category", "segment", "type", "group", "region", "state")
            or default_group_column(schema, cols)
        )
        id_col = default_id_column(schema, cols) or metric

        plans = [
            ("std", metric, "spread of values (std)"),
            ("nunique", id_col, "unique count"),
            ("pct", metric, "percent share"),
        ]
        combined_records: list[dict[str, Any]] = []
        last_result: dict[str, Any] = {}
        for agg, col, label in plans:
            result = query_tools.execute_aggregation(
                workspace.dataset,
                group_by=group,
                metric_column=col,
                agg=agg,
                order_desc=True,
                limit=50,
            )
            last_result = result
            payload = {
                "question": f"{question} [{label}]",
                "query_plan": {
                    "mode": "aggregation",
                    "group_by": group,
                    "metric_column": col,
                    "agg": agg,
                },
                "result": result,
                "explanation": self._explain(f"{question} ({label})", result, schema),
                "grounded": True,
            }
            workspace.analysis_results.append(payload)
            for row in result.get("records") or []:
                combined_records.append({"agg": agg, **row})

        workspace.record(
            agent=self.name,
            action="answer_question",
            received={"question": question, "multi_agg": [p[0] for p in plans]},
            produced={"row_count": len(combined_records)},
            success=True,
        )
        return AgentResult(
            task_id=message.task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="answer_question",
            success=True,
            data={
                "question": question,
                "query_plan": {"mode": "multi_aggregation", "aggs": [p[0] for p in plans]},
                "result": {**last_result, "records": combined_records, "grounded": True},
                "explanation": (
                    f"Computed std, nunique, and pct summaries by {group} from the dataset. "
                    "All values come from executed aggregations."
                ),
                "grounded": True,
            },
            grounded=True,
        )

    def _eda(self, message: AgentMessage, workspace: SharedWorkspace, *, action: str) -> AgentResult:
        question = (
            message.parameters.get("question")
            or message.parameters.get("user_request")
            or "Explore the dataset"
        )
        meta = workspace.dataset
        assert meta is not None
        correlation = eda_tools.correlation_matrix(meta)
        missing = eda_tools.missingness_summary(meta)
        quantiles = eda_tools.quantile_stats(meta)

        # Present top correlation pairs as grounded "records" for the Analysis panel
        records = correlation.get("pairs") or []
        result = {
            "operation": "eda",
            "records": records,
            "correlation": correlation,
            "missingness": missing,
            "quantiles": quantiles,
            "grounded": True,
            "source_dataset_id": meta.get("id"),
        }
        explanation = self._explain_eda(question, result)
        payload = {
            "question": question,
            "query_plan": {"mode": "eda", "tools": ["correlation_matrix", "missingness_summary", "quantile_stats"]},
            "result": result,
            "explanation": explanation,
            "grounded": True,
        }
        workspace.analysis_results.append(payload)
        if workspace.dataset is not None:
            workspace.dataset["eda"] = {
                "correlation": correlation,
                "missingness": missing,
                "quantiles": quantiles,
                "grounded": True,
            }
        workspace.record(
            agent=self.name,
            action=action,
            received={"question": question},
            produced={"corr_pairs": len(records), "null_total": missing.get("total_nulls")},
            success=True,
        )
        return AgentResult(
            task_id=message.task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action=action,
            success=True,
            data=payload,
            grounded=True,
        )

    def _explain_eda(self, question: str, result: dict[str, Any]) -> str:
        pairs = (result.get("correlation") or {}).get("pairs") or []
        miss = result.get("missingness") or {}
        qstats = (result.get("quantiles") or {}).get("stats") or {}
        # Always build a grounded offline summary first (numbers guaranteed)
        lines = ["Exploration results from EDA tools:"]
        if pairs:
            lines.append("How columns relate (strongest links):")
            for p in pairs[:5]:
                lines.append(
                    f"- {p.get('column_a')} vs {p.get('column_b')}: {p.get('correlation')}"
                )
        lines.append(
            f"Missing values: {miss.get('total_nulls', 0)} blank cells "
            f"({(miss.get('overall_null_rate') or 0) * 100:.2f}% overall)."
        )
        if qstats:
            lines.append("Typical ranges (25th / median / 75th percentile):")
            for col, stats in list(qstats.items())[:5]:
                lines.append(
                    f"- {col}: {stats.get('p25')} / {stats.get('p50')} / {stats.get('p75')}"
                )
        grounded = "\n".join(lines)

        system = (
            "Rewrite the following grounded EDA findings in 3-6 concise sentences. "
            "Keep every number exactly as given. Do not invent values."
        )
        text = self.llm.chat_text(system, grounded)
        if text and not text.startswith("(LLM unavailable") and len(text) > 40:
            return text
        return grounded


    @staticmethod
    def _resolve_question(message: AgentMessage) -> str:
        """Prefer explicit question; enrich with short-term conversation context when present."""
        question = (
            message.parameters.get("question")
            or message.parameters.get("user_request")
            or ""
        )
        prompt = (message.context or {}).get("conversation_prompt") or ""
        if not prompt or "empty" in prompt.lower():
            return question
        return (
            f"{question}\n\n"
            f"(Interpret follow-ups using this short-term context; "
            f"do not invent numbers.)\n{prompt}"
        )

    def _analyze(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        question = self._resolve_question(message)
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
            '  {"mode":"aggregation","group_by":"...|null","metric_column":"...",'
            '"agg":"sum|mean|count|min|max|median|std|nunique|pct","order_desc":true,"limit":20}\n'
            "or\n"
            '  {"mode":"sql","sql":"SELECT ..."}\n'
            "Use ONLY columns from the provided schema. Never invent numeric answers. "
            "Use agg=pct for percent/share questions, std for spread/variability, "
            "nunique for distinct/unique counts."
        )
        user = f"Question: {question}\nSchema columns: {cols}\nDtypes: {schema}"
        out = self.llm.chat_json(system, user)
        if out.get("_offline") or out.get("_fallback") or (
            out.get("mode") not in {"sql", "aggregation"} and "sql" not in out and "metric_column" not in out
        ):
            return self._heuristic_plan(question, cols, schema)
        if out.get("sql") and not out.get("mode"):
            out["mode"] = "sql"
        if out.get("metric_column") and not out.get("mode"):
            out["mode"] = "aggregation"
        return out

    def _heuristic_plan(self, question: str, cols: list[str], schema: dict[str, Any]) -> dict[str, Any]:
        q = question.lower()
        metric = (
            match_column(cols, "revenue", "amount", "sales", "value", "price", "total")
            or default_metric_column(schema, cols)
            or (cols[-1] if cols else None)
        )
        group = (
            match_column(cols, "category", "segment", "type", "group", "region", "state")
            or default_group_column(schema, cols)
        )

        if any(k in q for k in ("percent", "percentage", "share of", "pct", "% of", "share")):
            return {
                "mode": "aggregation",
                "group_by": group,
                "metric_column": metric,
                "agg": "pct",
                "order_desc": True,
                "limit": 20,
            }
        if any(k in q for k in ("unique", "distinct", "how many different", "nunique")):
            id_col = default_id_column(schema, cols) or metric
            return {
                "mode": "aggregation",
                "group_by": group,
                "metric_column": id_col,
                "agg": "nunique",
                "order_desc": True,
                "limit": 20,
            }
        if any(k in q for k in ("standard deviation", "std ", " std", "spread", "variability", "volatility")):
            return {
                "mode": "aggregation",
                "group_by": group,
                "metric_column": metric,
                "agg": "std",
                "order_desc": True,
                "limit": 20,
            }
        if ("categor" in q or "group" in q or "by " in q) and (
            "revenue" in q or "most" in q or "generate" in q or "sum" in q or "total" in q
        ):
            return {
                "mode": "aggregation",
                "group_by": group,
                "metric_column": metric,
                "agg": "sum",
                "order_desc": True,
                "limit": 20,
            }
        if "average" in q or "avg" in q or "mean" in q:
            return {"mode": "aggregation", "group_by": None, "metric_column": metric, "agg": "mean"}
        if "over time" in q or "trend" in q:
            date_col = default_time_column(schema, cols)
            if date_col and metric:
                return {
                    "mode": "sql",
                    "sql": (
                        f"SELECT date_trunc('month', {date_col}) AS period, "
                        f"SUM({metric}) AS value FROM data GROUP BY 1 ORDER BY 1"
                    ),
                }
        return {
            "mode": "aggregation",
            "group_by": group,
            "metric_column": metric,
            "agg": "sum",
            "order_desc": True,
            "limit": 20,
        }

    def _default_metric(self, schema: dict[str, Any]) -> str:
        cols = list(schema.keys())
        return default_metric_column(schema, cols) or cols[0]

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
        agg = result.get("agg")
        if agg == "pct":
            return (
                f"Computed percent share across {len(records)} group(s) from the dataset. "
                f"Top row: {records[0]}. All values come from executed aggregation."
            )
        if len(records) == 1 and "value" in records[0]:
            return f"Computed result for the question: {records[0]['value']} (from actual dataset aggregation)."
        top = records[0]
        return (
            f"Computed {len(records)} grouped result(s) from the dataset. "
            f"Top row: {top}. All values come from executed aggregation/SQL."
        )
