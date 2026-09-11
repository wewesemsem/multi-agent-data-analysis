"""Dataset Agent — schema/spec via LLM (or heuristics); rows via deterministic generator."""

from __future__ import annotations

import re
from typing import Any

from app.llm import LLMClient
from app.messages import AgentMessage, AgentResult
from app.state import SharedWorkspace
from app.tools import dataset_tools


class DatasetAgent:
    name = "dataset_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        action = message.action
        try:
            if action in {"create_dataset", "generate_dataset"}:
                return self._create(message, workspace)
            if action == "load_csv":
                return self._load_csv(message, workspace)
            if action == "inspect_schema":
                return self._inspect(workspace, message.task_id)
            if action == "profile_dataset":
                return self._profile(workspace, message.task_id)
            return AgentResult(
                task_id=message.task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action=action,
                success=False,
                error=f"Unknown dataset action: {action}",
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

    def _create(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        params = message.parameters
        user_text = params.get("user_request") or params.get("description") or ""
        spec = params.get("spec") or self._build_spec(user_text, params)

        meta = dataset_tools.generate_from_spec(spec)
        workspace.dataset = meta
        workspace.record(
            agent=self.name,
            action="create_dataset",
            received={"spec": spec},
            produced={"dataset_id": meta["id"], "row_count": meta["row_count"]},
            success=True,
        )
        return AgentResult(
            task_id=message.task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="create_dataset",
            success=True,
            data={"dataset": meta, "spec": spec},
            grounded=True,
            limitations=["Synthetic data only; not real customer records."],
        )

    def _build_spec(self, user_text: str, params: dict[str, Any]) -> dict[str, Any]:
        n_rows = params.get("n_rows")
        if n_rows is None:
            m = re.search(r"(\d[\d,]*)\s*(orders|rows|records|customers)?", user_text.replace(",", ""))
            n_rows = int(m.group(1)) if m else 1000

        system = (
            "You produce JSON dataset specifications for a deterministic generator. "
            "Never invent row values. Return JSON with keys: name, n_rows, template "
            "(ecommerce_orders|generic), seed. "
            "For ecommerce_orders do NOT include a columns field — the generator has a fixed schema. "
            "Only include columns for template=generic, as an object "
            '{"col_name": {"type": "string|int|float|bool"}} — never as a list. '
            "Prefer template=ecommerce_orders for e-commerce / orders / transactions."
        )
        llm_out = self.llm.chat_json(system, f"User request: {user_text}\nHint n_rows={n_rows}")
        if llm_out.get("_offline") or llm_out.get("_fallback") or "template" not in llm_out:
            text_l = user_text.lower()
            template = "ecommerce_orders" if any(
                k in text_l for k in ("e-commerce", "ecommerce", "order", "transaction", "customer")
            ) else "generic"
            return {
                "name": "ecommerce_orders" if template == "ecommerce_orders" else "synthetic_dataset",
                "n_rows": int(n_rows),
                "template": template,
                "seed": 42,
            }

        template = llm_out.get("template") or "ecommerce_orders"
        spec = {
            "name": llm_out.get("name") or "dataset",
            "n_rows": int(llm_out.get("n_rows") or n_rows),
            "template": template,
            "seed": int(llm_out.get("seed") or 42),
        }
        # Ignore columns for ecommerce templates; normalize otherwise
        if llm_out.get("columns") and "ecommerce" not in str(template).lower() and "order" not in str(template).lower():
            from app.tools.dataset_tools import _normalize_columns_spec

            normalized = _normalize_columns_spec(llm_out["columns"])
            if normalized:
                spec["columns"] = normalized
        return spec

    def _load_csv(self, message: AgentMessage, workspace: SharedWorkspace) -> AgentResult:
        path = message.parameters.get("path")
        if not path:
            raise ValueError("load_csv requires parameters.path")
        meta = dataset_tools.load_csv(path, name=message.parameters.get("name"))
        workspace.dataset = meta
        workspace.record(
            agent=self.name,
            action="load_csv",
            received={"path": path},
            produced={"dataset_id": meta["id"]},
            success=True,
        )
        return AgentResult(
            task_id=message.task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="load_csv",
            success=True,
            data={"dataset": meta},
            grounded=True,
        )

    def _inspect(self, workspace: SharedWorkspace, task_id: str) -> AgentResult:
        if not workspace.dataset:
            return AgentResult(
                task_id=task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="inspect_schema",
                success=False,
                error="No dataset in shared workspace.",
                grounded=False,
            )
        return AgentResult(
            task_id=task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="inspect_schema",
            success=True,
            data={"schema": workspace.dataset.get("schema"), "dataset_id": workspace.dataset.get("id")},
            grounded=True,
        )

    def _profile(self, workspace: SharedWorkspace, task_id: str) -> AgentResult:
        if not workspace.dataset:
            return AgentResult(
                task_id=task_id,
                source_agent=self.name,  # type: ignore[arg-type]
                action="profile_dataset",
                success=False,
                error="No dataset in shared workspace.",
                grounded=False,
            )
        df = dataset_tools.load_dataset(workspace.dataset)
        profile = dataset_tools.profile_dataframe(df)
        workspace.dataset["profile"] = profile
        return AgentResult(
            task_id=task_id,
            source_agent=self.name,  # type: ignore[arg-type]
            action="profile_dataset",
            success=True,
            data={"profile": profile},
            grounded=True,
        )
