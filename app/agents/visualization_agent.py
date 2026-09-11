"""Visualization Agent — LLM/heuristics produce a chart spec; Plotly renders from real data."""

from __future__ import annotations

from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import chart_tools


_AMOUNT_HINTS = ("total_amount", "revenue", "amount", "unit_price", "value", "price", "sales")
_CATEGORY_HINTS = ("product_category", "category", "categories", "segment", "type")
_DATE_HINTS = ("order_date", "date", "timestamp", "created_at", "time")


class VisualizationAgent:
    name = "visualization_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if action not in {"create_visualization", "visualize", "create_chart"}:
                return AgentResult(
                    task_id=message.task_id,
                    source_agent=self.name,  # type: ignore[arg-type]
                    action=action,
                    success=False,
                    error=f"Unknown visualization action: {action}",
                    grounded=False,
                )
            if not workspace.dataset and not message.parameters.get("data_records"):
                raise ValueError("No dataset or data_records available for visualization.")

            specs = message.parameters.get("specs")
            if message.parameters.get("force_heuristic"):
                specs = self._heuristic_specs(
                    message.parameters.get("user_request")
                    or message.parameters.get("description")
                    or "",
                    (workspace.dataset or {}).get("schema", {}),
                    workspace,
                )
            elif not specs:
                specs = self._plan_specs(message.parameters, workspace)

            schema = (workspace.dataset or {}).get("schema", {})
            specs = [self._normalize_spec(s, schema, workspace) for s in specs]
            specs = [s for s in specs if s]
            if not specs:
                specs = self._heuristic_specs(
                    message.parameters.get("user_request")
                    or message.parameters.get("description")
                    or "",
                    schema,
                    workspace,
                )
                specs = [self._normalize_spec(s, schema, workspace) for s in specs]
                specs = [s for s in specs if s]
            if not specs:
                raise ValueError(
                    "Could not build a valid visualization spec from the request and dataset schema."
                )

            created = []
            for spec in specs:
                chart = self._render_one(spec, workspace)
                workspace.visualizations.append(chart)
                created.append(
                    {
                        "id": chart["id"],
                        "title": chart["title"],
                        "chart_type": chart["chart_type"],
                        "html_path": chart["html_path"],
                        "n_points": chart["n_points"],
                        "grounded": True,
                        "plotly_json": chart.get("plotly_json"),
                        "data_preview": chart.get("data_preview"),
                    }
                )

            workspace.record(
                agent=self.name,
                action="create_visualization",
                received={"n_specs": len(specs)},
                produced={"chart_ids": [c["id"] for c in created]},
                success=True,
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="create_visualization",
                success=True,
                data={"charts": created},
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

    def _plan_specs(self, parameters: dict[str, Any], workspace: SharedWorkspace) -> list[dict[str, Any]]:
        request = parameters.get("user_request") or parameters.get("description") or ""
        schema = (workspace.dataset or {}).get("schema", {})
        cols = list(schema.keys())

        # Prefer charting grounded analysis results when they already answer the request.
        analysis_specs = self._specs_from_analysis(request, workspace)
        if analysis_specs:
            return analysis_specs + self._extra_distribution_specs(request, schema)

        system = (
            "Return JSON {\"specs\":[...]} where each spec has: "
            "chart_type (bar|line|scatter|histogram|pie), title, "
            "aggregation optional {group_by, metric_column, agg}, "
            "x, y, color optional. "
            f"Use ONLY these exact column names: {cols}. "
            "Never invent column names like 'revenue' — use total_amount when that exists. "
            "metric_column and group_by must be non-null when aggregation is used. "
            "If the user asks for a pie chart, set chart_type to pie. "
            "If the user asks for both category revenue and amount distribution, return TWO specs."
        )
        out = self.llm.chat_json(system, f"Request: {request}\nSchema: {schema}")
        if isinstance(out.get("specs"), list) and out["specs"]:
            specs = out["specs"]
            # Honor explicit pie requests even if the model returned bar
            if self._wants_pie(request):
                for s in specs:
                    if isinstance(s, dict) and (s.get("chart_type") or "").lower() in {"bar", "", "none"}:
                        s["chart_type"] = "pie"
            # Ensure distribution chart is present when requested even if LLM omitted it
            if self._wants_distribution(request) and not any(
                (s.get("chart_type") or "").lower() == "histogram" for s in specs if isinstance(s, dict)
            ):
                specs = list(specs) + self._extra_distribution_specs(request, schema)
            return specs
        return self._heuristic_specs(request, schema, workspace)

    def _specs_from_analysis(self, request: str, workspace: SharedWorkspace) -> list[dict[str, Any]]:
        if not workspace.analysis_results:
            return []
        req = request.lower()
        # Only reuse analysis output for visualization-oriented requests
        if not any(k in req for k in ("visual", "chart", "plot", "graph", "show", "pie")):
            return []

        item = workspace.analysis_results[-1]
        records = (item.get("result") or {}).get("records") or []
        if not records:
            return []
        keys = list(records[0].keys())
        if len(keys) < 2:
            return []
        x = keys[0]
        y = "value" if "value" in keys else keys[1]
        title = "Revenue by Product Category" if "categor" in req else "Analysis Result"
        chart_type = chart_tools.choose_chart_type(request)
        return [
            {
                "chart_type": chart_type,
                "title": title,
                "data_records": records,
                "x": x,
                "y": y,
            }
        ]

    @staticmethod
    def _wants_distribution(request: str) -> bool:
        req = request.lower()
        return "distribution" in req or "histogram" in req or "transaction amount" in req

    @staticmethod
    def _wants_pie(request: str) -> bool:
        req = request.lower()
        return any(k in req for k in ("pie", "donut", "share of", "proportion"))

    def _extra_distribution_specs(self, request: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._wants_distribution(request):
            return []
        amount = self._resolve_column(list(schema.keys()), _AMOUNT_HINTS)
        if not amount:
            return []
        return [
            {
                "chart_type": "histogram",
                "title": "Distribution of Transaction Amounts",
                "x": amount,
                "y": None,
                "use_raw_dataset": True,
            }
        ]

    def _heuristic_specs(
        self, request: str, schema: dict[str, Any], workspace: SharedWorkspace
    ) -> list[dict[str, Any]]:
        cols = list(schema.keys())
        req = request.lower()
        specs: list[dict[str, Any]] = []

        cat = self._resolve_column(cols, _CATEGORY_HINTS)
        amount = self._resolve_column(cols, _AMOUNT_HINTS)
        date = self._resolve_column(cols, _DATE_HINTS)
        chart_type = chart_tools.choose_chart_type(request)

        want_revenue_by_cat = (
            ("categor" in req and any(k in req for k in ("revenue", "visual", "chart", "show", "pie")))
            or "revenue by category" in req
            or self._wants_pie(request)
        )
        want_dist = "distribution" in req or "histogram" in req or "transaction amount" in req

        if want_revenue_by_cat and cat and amount:
            specs.append(
                {
                    "chart_type": chart_type if chart_type in {"bar", "pie"} else "bar",
                    "title": "Revenue by Product Category",
                    "aggregation": {"group_by": cat, "metric_column": amount, "agg": "sum"},
                    "x": cat,
                    "y": "value",
                }
            )
        if want_dist and amount and not self._wants_pie(request):
            specs.append(
                {
                    "chart_type": "histogram",
                    "title": "Distribution of Transaction Amounts",
                    "x": amount,
                    "y": None,
                    "use_raw_dataset": True,
                }
            )
        if not specs:
            analysis_specs = self._specs_from_analysis(request or "visualize", workspace)
            if analysis_specs:
                return analysis_specs + self._extra_distribution_specs(request, schema)
            if cat and amount:
                specs.append(
                    {
                        "chart_type": chart_type if chart_type in {"bar", "pie"} else "bar",
                        "title": f"{amount} by {cat}",
                        "aggregation": {"group_by": cat, "metric_column": amount, "agg": "sum"},
                        "x": cat,
                        "y": "value",
                    }
                )
            elif date and amount:
                specs.append(
                    {
                        "chart_type": "line",
                        "title": f"{amount} over time",
                        "aggregation": {"group_by": date, "metric_column": amount, "agg": "sum", "limit": 100},
                        "x": date,
                        "y": "value",
                    }
                )
            elif amount:
                specs.append(
                    {
                        "chart_type": "histogram",
                        "title": f"Distribution of {amount}",
                        "x": amount,
                        "use_raw_dataset": True,
                    }
                )
        return specs

    def _normalize_spec(
        self,
        spec: dict[str, Any],
        schema: dict[str, Any],
        workspace: SharedWorkspace,
    ) -> dict[str, Any] | None:
        if not isinstance(spec, dict):
            return None
        cols = list(schema.keys())
        out = dict(spec)

        # Already has concrete records — keep if plottable
        if out.get("data_records"):
            records = out["data_records"]
            if isinstance(records, list) and records and isinstance(records[0], dict):
                keys = list(records[0].keys())
                out["x"] = out.get("x") if out.get("x") in keys else keys[0]
                if len(keys) > 1:
                    out["y"] = out.get("y") if out.get("y") in keys else (
                        "value" if "value" in keys else keys[1]
                    )
                out.setdefault("chart_type", "bar")
                out.setdefault("title", "Chart")
                return out
            return None

        agg = out.get("aggregation")
        if isinstance(agg, dict) or out.get("group_by") or out.get("metric_column"):
            agg = dict(agg or {})
            if out.get("group_by") and not agg.get("group_by"):
                agg["group_by"] = out["group_by"]
            if out.get("metric_column") and not agg.get("metric_column"):
                agg["metric_column"] = out["metric_column"]

            group_by = self._map_column(agg.get("group_by"), cols, _CATEGORY_HINTS) or self._resolve_column(
                cols, _CATEGORY_HINTS
            )
            metric = self._map_column(agg.get("metric_column"), cols, _AMOUNT_HINTS) or self._resolve_column(
                cols, _AMOUNT_HINTS
            )
            if not metric:
                return None
            agg["group_by"] = group_by
            agg["metric_column"] = metric
            agg["agg"] = agg.get("agg") or "sum"
            out["aggregation"] = agg
            out["x"] = out.get("x") if out.get("x") in cols else group_by
            out["y"] = "value"
            out.setdefault("chart_type", "bar")
            out.setdefault("title", f"{metric} by {group_by}" if group_by else metric)
            return out

        # Raw / histogram path
        chart_type = (out.get("chart_type") or "bar").lower()
        x = self._map_column(out.get("x"), cols, _AMOUNT_HINTS if chart_type == "histogram" else _CATEGORY_HINTS)
        y = self._map_column(out.get("y"), cols, _AMOUNT_HINTS)
        if chart_type == "histogram":
            x = x or self._resolve_column(cols, _AMOUNT_HINTS)
            if not x:
                return None
            out["x"] = x
            out["use_raw_dataset"] = True
            out.setdefault("title", f"Distribution of {x}")
            return out

        # If LLM asked for a categorical comparison without aggregation, synthesize one
        if x and y:
            out["x"] = x
            out["y"] = y
            out.setdefault("title", "Chart")
            out["use_raw_dataset"] = True
            return out

        # Last resort: build aggregation from schema
        cat = self._resolve_column(cols, _CATEGORY_HINTS)
        amount = self._resolve_column(cols, _AMOUNT_HINTS)
        if cat and amount:
            return {
                "chart_type": "bar",
                "title": out.get("title") or "Revenue by Product Category",
                "aggregation": {"group_by": cat, "metric_column": amount, "agg": "sum"},
                "x": cat,
                "y": "value",
            }
        return None

    @staticmethod
    def _resolve_column(cols: list[str], hints: tuple[str, ...]) -> str | None:
        lower_map = {c.lower(): c for c in cols}
        for hint in hints:
            if hint.lower() in lower_map:
                return lower_map[hint.lower()]
        for c in cols:
            cl = c.lower()
            for hint in hints:
                if hint.lower() in cl:
                    return c
        return None

    @classmethod
    def _map_column(
        cls,
        name: str | None,
        cols: list[str],
        hints: tuple[str, ...],
    ) -> str | None:
        if not name or name in {"None", "null", "undefined"}:
            return None
        lower_map = {c.lower(): c for c in cols}
        if name in cols:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
        # Map aliases like "revenue" → total_amount
        alias_hints = (name.lower(),) + hints
        return cls._resolve_column(cols, alias_hints)

    def _render_one(self, spec: dict[str, Any], workspace: SharedWorkspace) -> dict[str, Any]:
        if spec.get("data_records") is not None:
            return chart_tools.render_chart(
                chart_type=spec.get("chart_type") or "bar",
                title=spec.get("title") or "Chart",
                data_records=spec["data_records"],
                x=spec.get("x"),
                y=spec.get("y"),
                color=spec.get("color"),
            )
        if spec.get("use_raw_dataset"):
            return chart_tools.render_chart(
                chart_type=spec.get("chart_type") or "histogram",
                title=spec.get("title") or "Chart",
                dataset_meta=workspace.dataset,
                x=spec.get("x"),
                y=spec.get("y"),
                color=spec.get("color"),
            )
        aggregation = spec.get("aggregation")
        if aggregation and not aggregation.get("metric_column"):
            raise ValueError("Visualization aggregation is missing metric_column after normalization.")
        return chart_tools.render_chart(
            chart_type=spec.get("chart_type") or "bar",
            title=spec.get("title") or "Chart",
            dataset_meta=workspace.dataset,
            x=spec.get("x"),
            y=spec.get("y"),
            color=spec.get("color"),
            aggregation=aggregation,
        )
