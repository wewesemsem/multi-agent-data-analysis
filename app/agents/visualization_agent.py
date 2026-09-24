"""Visualization Agent — LLM/heuristics produce a chart spec; Plotly renders from real data."""

from __future__ import annotations

from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import chart_tools
from app.tools.query_tools import coerce_column_ref


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

            # Replace charts for this run (avoids stacking when the planner emits
            # multiple visualization steps or the user re-runs a demo).
            workspace.visualizations = []

            request = (
                message.parameters.get("user_request")
                or message.parameters.get("description")
                or ""
            )
            schema = (workspace.dataset or {}).get("schema", {})

            specs = message.parameters.get("specs")
            # Explicit box+heatmap+dual toolkit requests are deterministic — skip flaky
            # LLM specs that often attach one-metric data_records to heatmaps.
            if message.parameters.get("force_heuristic") or self._is_toolkit_multi_chart(request):
                specs = self._heuristic_specs(request, schema, workspace)
            elif not specs:
                specs = self._plan_specs(message.parameters, workspace)

            specs = [self._normalize_spec(s, schema, workspace) for s in specs]
            specs = [s for s in specs if s]
            specs = self._dedupe_specs(specs)
            if not specs:
                specs = self._heuristic_specs(request, schema, workspace)
                specs = [self._normalize_spec(s, schema, workspace) for s in specs]
                specs = [s for s in specs if s]
                specs = self._dedupe_specs(specs)
            if not specs:
                raise ValueError(
                    "Could not build a valid visualization spec from the request and dataset schema."
                )

            created = []
            errors: list[str] = []
            for spec in specs:
                try:
                    chart = self._render_one(spec, workspace)
                except Exception as chart_exc:  # noqa: BLE001
                    # One bad LLM heatmap must not abort box / dual_axis siblings.
                    errors.append(f"{spec.get('chart_type')}: {chart_exc}")
                    continue
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

            # Fill any toolkit chart types that failed or were omitted
            if self._is_toolkit_multi_chart(request) and workspace.dataset:
                have = {c.get("chart_type") for c in created}
                for hs in self._heuristic_specs(request, schema, workspace):
                    ct = self._normalize_chart_type_name(hs.get("chart_type"))
                    if ct in have:
                        continue
                    try:
                        ns = self._normalize_spec(hs, schema, workspace)
                        if not ns:
                            continue
                        chart = self._render_one(ns, workspace)
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
                        have.add(chart["chart_type"])
                    except Exception as fill_exc:  # noqa: BLE001
                        errors.append(f"fill {ct}: {fill_exc}")

            if not created:
                raise ValueError(
                    "; ".join(errors)
                    if errors
                    else "Could not render any visualizations from the request."
                )

            workspace.record(
                agent=self.name,
                action="create_visualization",
                received={"n_specs": len(specs), "render_errors": errors or None},
                produced={"chart_ids": [c["id"] for c in created]},
                success=True,
            )
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="create_visualization",
                success=True,
                data={"charts": created, "render_errors": errors or None},
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
        # Skip this reuse for explicit multi-chart / new-type intents so LLM specs cannot
        # accidentally chart aggregated one-metric frames as heatmaps.
        if not any(
            k in request.lower()
            for k in (
                "box plot",
                "boxplot",
                "heatmap",
                "heat map",
                "dual axis",
                "dual-axis",
                "dual chart",
                "how numeric columns relate",
            )
        ):
            analysis_specs = self._specs_from_analysis(request, workspace)
            if analysis_specs:
                return analysis_specs + self._extra_distribution_specs(request, schema)

        system = (
            "Return JSON {\"specs\":[...]} where each spec has: "
            "chart_type (bar|line|scatter|histogram|pie|box|heatmap|dual_axis), title, "
            "aggregation optional {group_by, metric_column, agg}, "
            "x, y, y2, color optional. "
            f"Use ONLY these exact column names: {cols}. "
            "Never invent column names like 'revenue' — use total_amount when that exists. "
            "metric_column and group_by must be non-null when aggregation is used. "
            "If the user asks for a pie chart, set chart_type to pie. "
            "If the user asks for a box plot, set chart_type to box. "
            "If the user asks for a heatmap / correlation heat map, set chart_type to heatmap. "
            "If the user asks for dual-axis / two metrics together, set chart_type to dual_axis. "
            "If the user asks for both category revenue and amount distribution, return TWO specs."
        )
        out = self.llm.chat_json(system, f"Request: {request}\nSchema: {schema}")
        heuristic = self._heuristic_specs(request, schema, workspace)
        if isinstance(out.get("specs"), list) and out["specs"]:
            specs = [s for s in out["specs"] if isinstance(s, dict)]
            # Honor explicit pie requests even if the model returned bar
            if self._wants_pie(request):
                for s in specs:
                    if (s.get("chart_type") or "").lower() in {"bar", "", "none"}:
                        s["chart_type"] = "pie"
            # Ensure distribution chart is present when requested even if LLM omitted it
            if self._wants_distribution(request) and not any(
                (s.get("chart_type") or "").lower() == "histogram" for s in specs
            ):
                specs = list(specs) + self._extra_distribution_specs(request, schema)
            # Fill missing explicit toolkit chart types from heuristics
            specs = self._merge_missing_chart_types(request, specs, heuristic)
            return specs
        return heuristic

    @staticmethod
    def _normalize_chart_type_name(chart_type: str | None) -> str:
        ct = (chart_type or "bar").lower().replace("-", "_").replace(" ", "_")
        if ct in {"dualaxis", "dual", "dual_axis_chart"}:
            return "dual_axis"
        if ct in {"heat_map", "corr_heatmap", "correlation_heatmap"}:
            return "heatmap"
        if ct in {"box_plot", "boxplot"}:
            return "box"
        return ct

    @classmethod
    def _is_toolkit_multi_chart(cls, request: str) -> bool:
        """True when the user asked for two+ of box / heatmap / dual-axis in one request."""
        req = request.lower()
        n = 0
        if any(k in req for k in ("box plot", "boxplot", "box chart")):
            n += 1
        if any(k in req for k in ("heatmap", "heat map")):
            n += 1
        if any(k in req for k in ("dual axis", "dual-axis", "dual chart", "two axis")):
            n += 1
        return n >= 2

    @classmethod
    def _dedupe_specs(cls, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep one spec per chart_type (LLM often emits 3× box/heatmap/dual = 9 charts)."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for s in specs:
            ct = cls._normalize_chart_type_name(s.get("chart_type"))
            if ct in seen:
                continue
            seen.add(ct)
            s = dict(s)
            s["chart_type"] = ct
            out.append(s)
        return out

    def _merge_missing_chart_types(
        self,
        request: str,
        specs: list[dict[str, Any]],
        heuristic: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        req = request.lower()
        wanted: set[str] = set()
        if any(k in req for k in ("box plot", "boxplot", "box chart")):
            wanted.add("box")
        if any(k in req for k in ("heatmap", "heat map")):
            wanted.add("heatmap")
        if any(k in req for k in ("dual axis", "dual-axis", "dual chart", "two axis")):
            wanted.add("dual_axis")
        if not wanted:
            return self._dedupe_specs(specs)
        have = {self._normalize_chart_type_name(s.get("chart_type")) for s in specs}
        merged = list(specs)
        for hs in heuristic:
            ct = self._normalize_chart_type_name(hs.get("chart_type"))
            if ct in wanted and ct not in have:
                merged.append(hs)
                have.add(ct)
        return self._dedupe_specs(merged)

    def _specs_from_analysis(self, request: str, workspace: SharedWorkspace) -> list[dict[str, Any]]:
        if not workspace.analysis_results:
            return []
        req = request.lower()
        # Don't reuse analysis bars/pies when the user asked for box / heatmap / dual-axis
        if any(
            k in req
            for k in ("box plot", "boxplot", "box chart", "heatmap", "heat map", "dual axis", "dual-axis", "dual chart")
        ):
            return []
        # Only reuse analysis output for visualization-oriented requests
        if not any(k in req for k in ("visual", "chart", "plot", "graph", "show", "pie", "box", "heatmap", "heat map", "dual")):
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

        # Multi-intent: box + heatmap + dual in one request
        multi_specs: list[dict[str, Any]] = []
        if any(k in req for k in ("box plot", "boxplot", "box chart")) and amount:
            multi_specs.append(
                {
                    "chart_type": "box",
                    "title": f"{amount} by {cat}" if cat else f"Box plot of {amount}",
                    "x": cat,
                    "y": amount,
                    "use_raw_dataset": True,
                }
            )
        if any(k in req for k in ("heatmap", "heat map")):
            multi_specs.append(
                {
                    "chart_type": "heatmap",
                    "title": "How numeric columns relate",
                    "use_raw_dataset": True,
                }
            )
        if any(k in req for k in ("dual axis", "dual-axis", "dual chart", "two axis")) and cat and amount:
            from app.tools.dataset_tools import load_dataset

            df = load_dataset(workspace.dataset) if workspace.dataset else None
            if df is not None and cat in df.columns and amount in df.columns:
                id_col = self._resolve_column(cols, ("order_id", "id")) or amount
                dual = (
                    df.groupby(cat, as_index=False)
                    .agg(**{"revenue": (amount, "sum"), "orders": (id_col, "count")})
                    .to_dict(orient="records")
                )
                multi_specs.append(
                    {
                        "chart_type": "dual_axis",
                        "title": f"Revenue and order count by {cat}",
                        "data_records": dual,
                        "x": cat,
                        "y": "revenue",
                        "y2": "orders",
                    }
                )
        if len(multi_specs) >= 2:
            return multi_specs
        if len(multi_specs) == 1:
            return multi_specs

        # Explicit single new chart types
        if chart_type == "box" and amount:
            specs.append(
                {
                    "chart_type": "box",
                    "title": f"{amount} by {cat}" if cat else f"Box plot of {amount}",
                    "x": cat,
                    "y": amount,
                    "use_raw_dataset": True,
                }
            )
            return specs
        if chart_type == "heatmap":
            specs.append(
                {
                    "chart_type": "heatmap",
                    "title": "How numeric columns relate",
                    "use_raw_dataset": True,
                }
            )
            return specs
        if chart_type == "dual_axis" and cat and amount:
            from app.tools.dataset_tools import load_dataset

            df = load_dataset(workspace.dataset) if workspace.dataset else None
            if df is not None and cat in df.columns and amount in df.columns:
                id_col = self._resolve_column(cols, ("order_id", "id")) or amount
                dual = (
                    df.groupby(cat, as_index=False)
                    .agg(**{"revenue": (amount, "sum"), "orders": (id_col, "count")})
                    .to_dict(orient="records")
                )
                specs.append(
                    {
                        "chart_type": "dual_axis",
                        "title": f"Revenue and order count by {cat}",
                        "data_records": dual,
                        "x": cat,
                        "y": "revenue",
                        "y2": "orders",
                    }
                )
                return specs

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

        chart_type = self._normalize_chart_type_name(out.get("chart_type"))
        out["chart_type"] = chart_type

        # Heatmap / box must use the raw dataset — never LLM-attached one-metric frames.
        # Check chart_type BEFORE the data_records early-return (that path caused
        # "heatmap requires a z column or at least two numeric columns").
        if chart_type == "heatmap":
            out["use_raw_dataset"] = True
            out.pop("aggregation", None)
            out.pop("data_records", None)
            # Drop LLM axis guesses — correlation path owns column_a/b/correlation.
            out.pop("x", None)
            out.pop("y", None)
            out.pop("y2", None)
            out.pop("z", None)
            out.setdefault("title", "How numeric columns relate")
            return out

        if chart_type == "box":
            amount = self._map_column(out.get("y"), cols, _AMOUNT_HINTS) or self._resolve_column(
                cols, _AMOUNT_HINTS
            )
            cat = self._map_column(out.get("x"), cols, _CATEGORY_HINTS) or self._resolve_column(
                cols, _CATEGORY_HINTS
            )
            if not amount:
                return None
            out["chart_type"] = "box"
            out["x"] = cat
            out["y"] = amount
            out["use_raw_dataset"] = True
            out.pop("data_records", None)
            out.pop("aggregation", None)
            out.setdefault("title", f"{amount} by {cat}" if cat else f"Box plot of {amount}")
            return out

        if chart_type == "dual_axis":
            from app.tools.dataset_tools import load_dataset

            # Prefer rebuilding dual-axis from the dataset when available
            cat = self._map_column(out.get("x"), cols, _CATEGORY_HINTS) or self._resolve_column(
                cols, _CATEGORY_HINTS
            )
            amount = self._map_column(out.get("y"), cols, _AMOUNT_HINTS) or self._resolve_column(
                cols, _AMOUNT_HINTS
            )
            if cat and amount and workspace.dataset:
                df = load_dataset(workspace.dataset)
                id_col = self._resolve_column(cols, ("order_id", "id")) or amount
                dual = (
                    df.groupby(cat, as_index=False)
                    .agg(**{"revenue": (amount, "sum"), "orders": (id_col, "count")})
                    .to_dict(orient="records")
                )
                return {
                    "chart_type": "dual_axis",
                    "title": out.get("title") or f"Revenue and order count by {cat}",
                    "data_records": dual,
                    "x": cat,
                    "y": "revenue",
                    "y2": "orders",
                }

        # Already has concrete records — keep if plottable (dual_axis / bar from analysis)
        if out.get("data_records"):
            records = out["data_records"]
            if isinstance(records, list) and records and isinstance(records[0], dict):
                keys = list(records[0].keys())
                x_ref = coerce_column_ref(out.get("x"))
                y_ref = coerce_column_ref(out.get("y"))
                y2_ref = coerce_column_ref(out.get("y2"))
                out["x"] = x_ref if x_ref in keys else keys[0]
                if len(keys) > 1:
                    out["y"] = y_ref if y_ref in keys else (
                        "value" if "value" in keys else keys[1]
                    )
                if y2_ref and y2_ref not in keys:
                    numeric = [k for k in keys if k not in {out.get("x"), out.get("y")}]
                    out["y2"] = numeric[0] if numeric else None
                elif y2_ref:
                    out["y2"] = y2_ref
                out.setdefault("chart_type", "bar")
                out.setdefault("title", "Chart")
                return out
            return None

        if chart_type == "dual_axis":
            # No dataset / columns to rebuild — drop
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
        name: Any,
        cols: list[str],
        hints: tuple[str, ...],
    ) -> str | None:
        name = coerce_column_ref(name)
        if not name:
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
        chart_type = self._normalize_chart_type_name(spec.get("chart_type"))

        # Heatmap / box always need the full dataset — ignore any data_records.
        if chart_type in {"heatmap", "box"} or spec.get("use_raw_dataset"):
            return chart_tools.render_chart(
                chart_type=chart_type if chart_type != "box" else "box",
                title=spec.get("title") or "Chart",
                dataset_meta=workspace.dataset,
                x=spec.get("x"),
                y=spec.get("y"),
                y2=spec.get("y2"),
                z=spec.get("z"),
                color=spec.get("color"),
            )

        if spec.get("data_records") is not None:
            return chart_tools.render_chart(
                chart_type=chart_type or "bar",
                title=spec.get("title") or "Chart",
                data_records=spec["data_records"],
                dataset_meta=workspace.dataset,
                x=spec.get("x"),
                y=spec.get("y"),
                y2=spec.get("y2"),
                z=spec.get("z"),
                color=spec.get("color"),
            )
        aggregation = spec.get("aggregation")
        if aggregation and not aggregation.get("metric_column"):
            raise ValueError("Visualization aggregation is missing metric_column after normalization.")
        return chart_tools.render_chart(
            chart_type=chart_type or "bar",
            title=spec.get("title") or "Chart",
            dataset_meta=workspace.dataset,
            x=spec.get("x"),
            y=spec.get("y"),
            y2=spec.get("y2"),
            z=spec.get("z"),
            color=spec.get("color"),
            aggregation=aggregation,
        )
