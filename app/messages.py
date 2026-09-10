"""Structured inter-agent messages (JSON-like objects, not free-text handoffs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentName = Literal[
    "orchestrator",
    "dataset_agent",
    "analysis_agent",
    "anomaly_agent",
    "visualization_agent",
    "validation_agent",
]


class AgentMessage(BaseModel):
    task_id: str
    source_agent: AgentName
    target_agent: AgentName
    action: str
    dataset_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    task_id: str
    source_agent: AgentName
    action: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    limitations: list[str] = Field(default_factory=list)
    grounded: bool = True  # True when numerical/statistical claims come from tools


class PlanStep(BaseModel):
    step_id: str
    agent: AgentName
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExecutionPlan(BaseModel):
    task_id: str
    user_request: str
    steps: list[PlanStep]
    summary: str = ""
