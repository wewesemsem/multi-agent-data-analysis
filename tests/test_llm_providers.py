"""Unit tests for multi-provider LLM client configuration."""

from __future__ import annotations

import pytest

from app.llm import (
    LLMClient,
    LLMUnavailableError,
    MAX_MODEL_CHOICES,
    PROVIDERS,
    _is_chat_model,
    clear_model_cache,
    latest_model_for,
    list_models,
    resolve_provider,
    supports_custom_temperature,
)


def test_resolve_provider_aliases() -> None:
    assert resolve_provider("claude") == "anthropic"
    assert resolve_provider("gemini") == "google"
    assert resolve_provider("gpt") == "openai"
    assert resolve_provider("unknown-provider") == "openai"


def test_chat_model_filter() -> None:
    assert _is_chat_model("gpt-5.5")
    assert _is_chat_model("claude-sonnet-5-5")
    assert _is_chat_model("gemini-3.8-flash")
    assert not _is_chat_model("text-embedding-3-large")
    assert not _is_chat_model("gpt-image-2")
    assert not _is_chat_model("gemini-embedding-001")
    assert not _is_chat_model("whisper-1")


def test_fallback_models_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    models = list_models("openai", force_refresh=True)
    assert models == PROVIDERS["openai"]["models"]
    assert latest_model_for("openai") == models[0]


def test_list_models_uses_fetched_order(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_fetch_filtered(api_key: str, *, base_url: str | None = None):
        return [(100, "gpt-new"), (50, "gpt-old")]

    monkeypatch.setattr("app.llm._fetch_openai_models", fake_fetch_filtered)
    models = list_models("openai", force_refresh=True)
    assert models[0] == "gpt-new"
    assert "gpt-old" in models


def test_list_models_keeps_latest_three_chat_only(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_fetch(api_key: str, *, base_url: str | None = None):
        return [
            (400, "gpt-newest"),
            (300, "gpt-image-2"),  # non-chat; filtered in real fetch + list_models
            (200, "gpt-mid"),
            (150, "whisper-1"),
            (100, "gpt-older"),
            (50, "gpt-oldest"),
        ]

    monkeypatch.setattr("app.llm._fetch_openai_models", fake_fetch)
    models = list_models("openai", force_refresh=True)
    assert models == ["gpt-newest", "gpt-mid", "gpt-older"]
    assert len(models) == MAX_MODEL_CHOICES
    assert all(_is_chat_model(m) for m in models)
    assert "gpt-image-2" not in models
    assert "whisper-1" not in models
    assert "gpt-oldest" not in models


def test_client_uses_provider_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(
        "app.llm.list_models",
        lambda provider=None, force_refresh=False: ["claude-haiku-4-5", "claude-sonnet-5-5"],
    )
    client = LLMClient(provider="anthropic", model="claude-haiku-4-5")
    assert client.provider == "anthropic"
    assert client.model == "claude-haiku-4-5"
    assert client.api_key == "test-anthropic-key"


def test_supports_custom_temperature() -> None:
    assert supports_custom_temperature("gpt-4o-mini")
    assert supports_custom_temperature("claude-sonnet-4-5")
    assert not supports_custom_temperature("gpt-5")
    assert not supports_custom_temperature("gpt-5.4")
    assert not supports_custom_temperature("o3-mini")
    assert not supports_custom_temperature("o4-mini")


def test_openai_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.llm.list_models",
        lambda provider=None, force_refresh=False: ["gpt-5", "gpt-4o-mini"],
    )

    captured: dict = {}

    class _Msg:
        content = '{"ok": true}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test-key")
    client._client = _Client()
    client._backend = "openai"
    assert client.chat_json("sys", "user") == {"ok": True}
    assert "temperature" not in captured


def test_anthropic_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(
        "app.llm.list_models",
        lambda provider=None, force_refresh=False: ["claude-haiku-4-5"],
    )

    captured: dict = {}

    class _Block:
        text = '{"ok": true}'

    class _Resp:
        content = [_Block()]

    class _Client:
        def messages(self):  # pragma: no cover - attribute access uses nested
            raise AssertionError("use messages.create")

        class messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return _Resp()

    client = LLMClient(provider="anthropic", model="claude-haiku-4-5", api_key="test-anthropic-key")
    client._client = _Client()
    client._backend = "anthropic"
    assert client.chat_json("sys", "user") == {"ok": True}
    assert "temperature" not in captured


def test_offline_when_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("MAS_ALLOW_OFFLINE_HEURISTICS", raising=False)
    client = LLMClient(provider="google")
    assert not client.available
    assert "GOOGLE_API_KEY" in client.status_label
    with pytest.raises(LLMUnavailableError):
        client.chat_json("system", "user")


def test_offline_heuristics_when_env_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MAS_ALLOW_OFFLINE_HEURISTICS", "1")
    client = LLMClient(provider="google")
    assert client.chat_json("system", "user") == {"_offline": True}


def test_env_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert resolve_provider() == "google"


def test_client_defaults_to_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_model_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(
        "app.llm.list_models",
        lambda provider=None, force_refresh=False: ["gpt-newest", "gpt-older"],
    )
    client = LLMClient(provider="openai")
    assert client.model == "gpt-newest"
