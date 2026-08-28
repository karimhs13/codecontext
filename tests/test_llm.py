from __future__ import annotations

import httpx
import pytest

from codecontext.core.llm import (
    LLMError,
    _litellm_model_id,
    check_ollama_available,
    preflight_check,
)
from codecontext.utils.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_litellm_model_id_ollama() -> None:
    assert _litellm_model_id("ollama", "llama3", "http://localhost:11434") == "ollama/llama3"


def test_litellm_model_id_anthropic_adds_prefix_when_missing() -> None:
    assert _litellm_model_id("anthropic", "opus-5", "") == "anthropic/opus-5"


def test_litellm_model_id_anthropic_keeps_claude_prefixed_name() -> None:
    assert (
        _litellm_model_id("anthropic", "claude-3-5-sonnet-20241022", "")
        == "claude-3-5-sonnet-20241022"
    )


def test_litellm_model_id_openai_passthrough() -> None:
    assert _litellm_model_id("openai", "gpt-4o", "") == "gpt-4o"


def test_litellm_model_id_unknown_provider_raises() -> None:
    with pytest.raises(LLMError):
        _litellm_model_id("bogus", "model", "")


def test_check_ollama_available_raises_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, timeout):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(LLMError, match="Could not reach Ollama"):
        check_ollama_available("http://localhost:11434")


def test_preflight_check_unknown_provider() -> None:
    settings = _settings()
    with pytest.raises(LLMError, match="Unknown provider"):
        preflight_check("bogus", settings)


def test_preflight_check_anthropic_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(anthropic_api_key=None)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        preflight_check("anthropic", settings)


def test_preflight_check_openai_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _settings(openai_api_key=None)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        preflight_check("openai", settings)


def test_preflight_check_anthropic_passes_with_key() -> None:
    settings = _settings(anthropic_api_key="sk-test")
    preflight_check("anthropic", settings)  # should not raise
