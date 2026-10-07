"""Thin LLM client. Used for planning, schema specs, and explanations — never for inventing numbers."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Static fallbacks used when a provider API is unreachable (latest 3 chat models).
PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "anthropic": {
        "label": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5",
        "models": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"],
    },
    "google": {
        "label": "Gemini (Google)",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-flash-latest",
        "models": ["gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash"],
        # OpenAI-compatible Gemini endpoint
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

DEFAULT_PROVIDER = "openai"
MAX_MODEL_CHOICES = 3
_MODEL_CACHE_TTL_SEC = 3600.0
_model_cache: dict[str, tuple[float, list[str]]] = {}

# CI / local tests may set this to keep keyword planning without an API key.
# Live demos must leave this unset so missing keys and API failures fail loudly.
_OFFLINE_ENV = "MAS_ALLOW_OFFLINE_HEURISTICS"


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used (missing key or API failure)."""


def offline_heuristics_allowed() -> bool:
    return os.getenv(_OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes"}


def supports_custom_temperature(model: str | None) -> bool:
    """Newer reasoning / GPT-5-class models only accept the API default temperature."""
    name = (model or "").lower()
    if not name:
        return True
    # OpenAI reasoning + GPT-5 family (and preview aliases) reject temperature != default.
    blocked_prefixes = (
        "o1",
        "o3",
        "o4",
        "gpt-5",
        "gpt5",
    )
    if any(name.startswith(p) or f"-{p}" in name or f"/{p}" in name for p in blocked_prefixes):
        return False
    return True


def _is_temperature_unsupported_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "temperature" in text and (
        "unsupported" in text or "does not support" in text or "unsupported_value" in text
    )


# Non-chat / specialty model markers shared across providers.
_EXCLUDE_SUBSTRINGS = (
    "embedding",
    "whisper",
    "transcribe",
    "tts",
    "audio",
    "realtime",
    "image",
    "dall-e",
    "moderation",
    "search",
    "computer-use",
    "veo",
    "lyria",
    "aqa",
    "robotics",
    "deep-research",
    "antigravity",
    "nano-banana",
    "live",
    "codex",
    "davinci",
    "babbage",
    "ada-",
    "curie",
    "sora",
    "omni-moderation",
    "text-similarity",
    "text-search",
    "code-search",
)


def _hydrate_secrets_from_streamlit() -> None:
    """Streamlit Cloud injects secrets via st.secrets, not always via env vars."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if not secrets:
            return
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "LLM_MODEL",
            "LLM_PROVIDER",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
    except Exception:  # noqa: BLE001
        return


_hydrate_secrets_from_streamlit()


def resolve_provider(provider: str | None = None) -> str:
    """Normalize provider id; fall back to env / default."""
    raw = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    aliases = {
        "gpt": "openai",
        "chatgpt": "openai",
        "claude": "anthropic",
        "gemini": "google",
    }
    resolved = aliases.get(raw, raw)
    if resolved not in PROVIDERS:
        return DEFAULT_PROVIDER
    return resolved


def api_key_for(provider: str) -> str | None:
    """Resolve the API key for a provider (incl. GEMINI_API_KEY alias)."""
    provider = resolve_provider(provider)
    key = os.getenv(PROVIDERS[provider]["env_key"])
    if not key and provider == "google":
        key = os.getenv("GEMINI_API_KEY")
    return key or None


def _is_chat_model(model_id: str) -> bool:
    """True only for text chat / completion models usable via chat APIs."""
    mid = model_id.lower().removeprefix("models/")
    if any(token in mid for token in _EXCLUDE_SUBSTRINGS):
        return False
    if mid.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt")):
        # Prefer chat snapshots; skip dated fine-tunes like ft:...
        if mid.startswith("ft:") or "-instruct" in mid:
            return False
        return True
    if mid.startswith("claude-"):
        # Skip Claude specialty / non-Messages API ids if they appear.
        if any(token in mid for token in ("-tools", "-citadel")):
            return False
        return True
    if mid.startswith("gemini-") or mid.startswith("gemma-"):
        # Chat-capable Gemini only (exclude Imagen/Veo-style ids already covered above).
        if any(token in mid for token in ("imagen", "tts", "embedding")):
            return False
        return True
    return False


def _strip_model_prefix(model_id: str) -> str:
    return model_id.removeprefix("models/")


def _version_key(model_id: str) -> tuple[Any, ...]:
    """Heuristic sort key: higher version numbers / 'latest' aliases win."""
    mid = _strip_model_prefix(model_id).lower()
    latest_boost = 1 if mid.endswith("-latest") or "-latest" in mid else 0
    nums = tuple(int(n) for n in re.findall(r"\d+", mid)[:6])
    # Prefer shorter canonical aliases over dated snapshots when versions tie.
    dated = 1 if re.search(r"-\d{8}$", mid) else 0
    return (latest_boost, nums, -dated, -len(mid), mid)


def _fallback_models(provider: str) -> list[str]:
    return list(PROVIDERS[provider]["models"])


def _fetch_openai_models(api_key: str, *, base_url: str | None = None) -> list[tuple[int, str]]:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    scored: list[tuple[int, str]] = []
    for m in client.models.list():
        mid = _strip_model_prefix(getattr(m, "id", "") or "")
        if not mid or not _is_chat_model(mid):
            continue
        created = int(getattr(m, "created", 0) or 0)
        scored.append((created, mid))
    return scored


def _fetch_anthropic_models(api_key: str) -> list[tuple[float, str]]:
    from anthropic import Anthropic
    from datetime import datetime, timezone

    client = Anthropic(api_key=api_key)
    scored: list[tuple[float, str]] = []
    after_id: str | None = None
    while True:
        page = client.models.list(limit=100, after_id=after_id) if after_id else client.models.list(limit=100)
        data = list(getattr(page, "data", []) or [])
        if not data:
            break
        for m in data:
            mid = _strip_model_prefix(getattr(m, "id", "") or "")
            if not mid or not _is_chat_model(mid):
                continue
            created_at = getattr(m, "created_at", None)
            if isinstance(created_at, datetime):
                ts = created_at.replace(tzinfo=created_at.tzinfo or timezone.utc).timestamp()
            else:
                # Anthropic returns newest first; preserve page order with a decaying score.
                ts = float(10_000 - len(scored))
            scored.append((ts, mid))
        if not getattr(page, "has_more", False):
            break
        after_id = getattr(page, "last_id", None) or data[-1].id
        if not after_id:
            break
    return scored


def _fetch_google_models(api_key: str) -> list[tuple[int, str]]:
    base_url = str(PROVIDERS["google"]["base_url"])
    # Gemini's OpenAI-compat list often lacks useful `created`; score via version heuristic.
    raw = _fetch_openai_models(api_key, base_url=base_url)
    ranked = sorted((_strip_model_prefix(mid) for _, mid in raw), key=_version_key, reverse=True)
    ranked = _dedupe(ranked)
    # Pin Google's rolling "latest" aliases first (flash before pro for this app).
    pinned = [m for m in ("gemini-flash-latest", "gemini-pro-latest", "gemini-flash-lite-latest") if m in ranked]
    rest = [m for m in ranked if m not in pinned]
    ordered = pinned + rest
    return [(10_000 - i, mid) for i, mid in enumerate(ordered)]


def _drop_dated_when_alias_exists(model_ids: list[str]) -> list[str]:
    """Prefer canonical aliases over dated snapshots (e.g. gpt-5.5 over gpt-5.5-20260423)."""
    aliases = {m for m in model_ids if not re.search(r"-\d{8}$", m)}
    out: list[str] = []
    for mid in model_ids:
        dated = re.search(r"^(?P<base>.+)-\d{8}$", mid)
        if dated and dated.group("base") in aliases:
            continue
        out.append(mid)
    return out


def list_models(provider: str | None = None, *, force_refresh: bool = False) -> list[str]:
    """Return the latest chat-capable models for a provider (max 3), newest-first."""
    provider = resolve_provider(provider)
    now = time.time()
    if not force_refresh and provider in _model_cache:
        expires_at, cached = _model_cache[provider]
        if now < expires_at and cached:
            return list(cached)

    api_key = api_key_for(provider)
    models: list[str] = []
    try:
        if not api_key:
            models = _fallback_models(provider)
        elif provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL") or None
            scored = _fetch_openai_models(api_key, base_url=base_url)
            scored.sort(key=lambda item: (item[0], _version_key(item[1])), reverse=True)
            models = _drop_dated_when_alias_exists(_dedupe([mid for _, mid in scored]))
        elif provider == "anthropic":
            scored = _fetch_anthropic_models(api_key)
            scored.sort(key=lambda item: item[0], reverse=True)
            models = _drop_dated_when_alias_exists(_dedupe([mid for _, mid in scored]))
        elif provider == "google":
            scored = _fetch_google_models(api_key)
            models = _dedupe([mid for _, mid in scored])
    except Exception:  # noqa: BLE001 — offline / bad key: use fallbacks
        models = _fallback_models(provider)

    if not models:
        models = _fallback_models(provider)

    # Defense in depth: re-filter chat-only, then keep the newest three.
    models = [m for m in models if _is_chat_model(m)][:MAX_MODEL_CHOICES]
    if not models:
        models = _fallback_models(provider)[:MAX_MODEL_CHOICES]

    _model_cache[provider] = (now + _MODEL_CACHE_TTL_SEC, list(models))
    return list(models)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def clear_model_cache(provider: str | None = None) -> None:
    if provider is None:
        _model_cache.clear()
        return
    _model_cache.pop(resolve_provider(provider), None)


def latest_model_for(provider: str | None = None, *, force_refresh: bool = False) -> str:
    """Newest chat model for the provider (first entry of list_models)."""
    provider = resolve_provider(provider)
    models = list_models(provider, force_refresh=force_refresh)
    return models[0] if models else str(PROVIDERS[provider]["default_model"])


def default_model_for(provider: str) -> str:
    """Backward-compatible alias: dynamic latest with static fallback."""
    return latest_model_for(provider)


class LLMClient:
    """Multi-provider chat client. Missing keys / API failures raise unless offline heuristics are allowed."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = resolve_provider(provider)
        meta = PROVIDERS[self.provider]
        self.api_key = api_key or api_key_for(self.provider)
        self.base_url = os.getenv("OPENAI_BASE_URL") or meta.get("base_url")
        self._client: Any = None
        self._backend: str | None = None  # "openai" | "anthropic"
        self.last_error: str | None = None

        available = list_models(self.provider) if self.api_key else _fallback_models(self.provider)
        env_model = os.getenv("LLM_MODEL")
        if model:
            self.model = model
        elif env_model and (env_model in available or not available):
            self.model = env_model
        else:
            self.model = available[0] if available else str(meta["default_model"])

        if not self.api_key:
            return

        try:
            if self.provider == "anthropic":
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self.api_key)
                self._backend = "anthropic"
            else:
                from openai import OpenAI

                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.provider == "google":
                    kwargs["base_url"] = meta["base_url"]
                elif self.provider == "openai" and os.getenv("OPENAI_BASE_URL"):
                    kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
                self._client = OpenAI(**kwargs)
                self._backend = "openai"
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._backend = None
            self.last_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status_label(self) -> str:
        label = PROVIDERS[self.provider]["label"]
        if self._client is None:
            env_key = PROVIDERS[self.provider]["env_key"]
            if offline_heuristics_allowed():
                return f"offline heuristics allowed (no {env_key})"
            return f"LLM unavailable — set {env_key}"
        if self.last_error:
            return f"{label} key set, last call failed: {self.last_error[:80]}"
        return f"{label} · {self.model}"

    def chat_json(self, system: str, user: str, *, temperature: float = 0.1) -> dict[str, Any]:
        """Ask the LLM for a JSON object. Raises LLMUnavailableError when the model cannot be used."""
        if self._client is not None:
            try:
                content = self._complete(system, user, temperature=temperature, json_mode=True)
                self.last_error = None
                parsed = extract_json_block(content) if self._backend == "anthropic" else None
                if parsed is not None:
                    return parsed
                return json.loads(content or "{}")
            except LLMUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                if offline_heuristics_allowed():
                    return {"_llm_error": str(exc), "_fallback": True}
                raise LLMUnavailableError(f"LLM call failed: {exc}") from exc
        if offline_heuristics_allowed():
            return {"_offline": True}
        env_key = PROVIDERS[self.provider]["env_key"]
        raise LLMUnavailableError(
            f"No API key for {PROVIDERS[self.provider]['label']}. "
            f"Set {env_key} in .env (offline heuristics are disabled for live use)."
        )

    def chat_text(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        if self._client is not None:
            try:
                text = self._complete(system, user, temperature=temperature, json_mode=False)
                self.last_error = None
                return (text or "").strip()
            except LLMUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                if offline_heuristics_allowed():
                    return f"(LLM unavailable: {exc})"
                raise LLMUnavailableError(f"LLM call failed: {exc}") from exc
        if offline_heuristics_allowed():
            return ""
        env_key = PROVIDERS[self.provider]["env_key"]
        raise LLMUnavailableError(
            f"No API key for {PROVIDERS[self.provider]['label']}. "
            f"Set {env_key} in .env (offline heuristics are disabled for live use)."
        )

    def _complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        json_mode: bool,
    ) -> str:
        if self._backend == "anthropic":
            return self._complete_anthropic(system, user, temperature=temperature, json_mode=json_mode)
        return self._complete_openai(system, user, temperature=temperature, json_mode=json_mode)

    def _complete_openai(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        json_mode: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Never send temperature on the OpenAI-compatible path (OpenAI + Gemini).
        # GPT-5 / o-series reject any value other than the API default.
        # Gemini's OpenAI-compat endpoint supports json_object; keep it for both.
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Some models reject json_object — retry without it.
            if json_mode and "response_format" in kwargs and (
                "response_format" in str(exc).lower() or "json_object" in str(exc).lower()
            ):
                kwargs.pop("response_format", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ("{}" if json_mode else "")

    def _complete_anthropic(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        json_mode: bool,
    ) -> str:
        sys = system
        usr = user
        if json_mode:
            sys = (
                f"{system}\n\n"
                "Respond with a single valid JSON object only. "
                "No markdown fences, no commentary."
            )
        # anthropic>=1.x Messages.create no longer accepts temperature; omit it.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=sys,
            messages=[{"role": "user", "content": usr}],
        )
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts) or ("{}" if json_mode else "")


def extract_json_block(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
