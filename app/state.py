"""Shared workspace / environment for the multi-agent system.

Agents read and write this state; tools perform the actual computation.
Designed to be replaceable later with BigQuery / Postgres / GCS-backed stores.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent / "workspace"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
CHARTS_DIR = WORKSPACE_ROOT / "charts"


def ensure_workspace() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SharedWorkspace:
    """In-process shared environment for all agents."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset: dict[str, Any] | None = None
    analysis_results: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    task_status: str = "idle"
    agent_history: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    validation_reports: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dataset": self.dataset,
            "analysis_results": self.analysis_results,
            "anomalies": self.anomalies,
            "visualizations": self.visualizations,
            "task_status": self.task_status,
            "agent_history": self.agent_history,
            "plan": self.plan,
            "validation_reports": self.validation_reports,
            "errors": self.errors,
            "final_response": self.final_response,
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def record(
        self,
        *,
        agent: str,
        action: str,
        received: dict[str, Any] | None = None,
        produced: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
        limitations: list[str] | None = None,
    ) -> None:
        entry = {
            "timestamp": utc_now(),
            "agent": agent,
            "action": action,
            "received": received or {},
            "produced": produced or {},
            "success": success,
            "error": error,
            "limitations": limitations or [],
        }
        self.agent_history.append(entry)
        if not success and error:
            self.errors.append({"agent": agent, "action": action, "error": error, "timestamp": utc_now()})
