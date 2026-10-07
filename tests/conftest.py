"""Pytest defaults — offline keyword planning without live API calls."""

from __future__ import annotations

import os

import pytest

# Set before imports in case collection imports app.llm (load_dotenv).
os.environ["MAS_ALLOW_OFFLINE_HEURISTICS"] = "1"


@pytest.fixture(autouse=True)
def _force_offline_llm_for_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep orchestrator tests on heuristic planning; do not call live provider APIs.

    `app.llm` calls load_dotenv() on import, which would otherwise re-inject .env keys.
    """
    monkeypatch.setenv("MAS_ALLOW_OFFLINE_HEURISTICS", "1")
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
