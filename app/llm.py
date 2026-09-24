"""Thin LLM client. Used for planning, schema specs, and explanations — never for inventing numbers."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _hydrate_secrets_from_streamlit() -> None:
    """Streamlit Cloud injects secrets via st.secrets, not always via env vars."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if not secrets:
            return
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL", "GOOGLE_API_KEY"):
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
    except Exception:  # noqa: BLE001
        return


_hydrate_secrets_from_streamlit()


class LLMClient:
    """OpenAI-compatible chat client with a deterministic offline fallback for MVP demos."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL")  # optional for compatible providers
        self._client = None
        self.last_error: str | None = None
        if self.api_key and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except Exception as exc:  # noqa: BLE001
                self._client = None
                self.last_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status_label(self) -> str:
        if self._client is None:
            return "offline heuristics (no API key)"
        if self.last_error:
            return f"API key set, last call failed: {self.last_error[:80]}"
        return f"connected ({self.model})"

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict[str, Any]:
        """Ask the LLM for a JSON object. Falls back to heuristic parsing when offline."""
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or "{}"
                self.last_error = None
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001 — MVP: fall through to offline mode
                self.last_error = str(exc)
                return {"_llm_error": str(exc), "_fallback": True}
        return {"_offline": True}

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                self.last_error = None
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                return f"(LLM unavailable: {exc})"
        return ""


def extract_json_block(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
