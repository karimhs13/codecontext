"""LiteLLM-based router supporting Ollama (local), Anthropic, and OpenAI."""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx

from codecontext.utils.config import SUPPORTED_PROVIDERS, Settings


class LLMError(RuntimeError):
    """Raised for provider configuration or connectivity failures."""


def _litellm_model_id(provider: str, model: str, ollama_host: str) -> str:
    if provider == "ollama":
        return f"ollama/{model}"
    if provider == "anthropic":
        return model if model.startswith("claude") else f"anthropic/{model}"
    if provider == "openai":
        return model
    raise LLMError(f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def check_ollama_available(host: str) -> None:
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=3.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach Ollama at {host}. Is `ollama serve` running? ({e})") from e


def preflight_check(provider: str, settings: Settings) -> None:
    """Raise LLMError with a helpful message if the provider isn't usable."""
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMError(
            f"Unknown provider '{provider}'. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    if provider == "ollama":
        check_ollama_available(settings.ollama_host)
    elif provider == "anthropic":
        key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Export it or add it to a .env file.")
    elif provider == "openai":
        key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set. Export it or add it to a .env file.")


def _prepare_env(provider: str, settings: Settings) -> None:
    if provider == "ollama":
        os.environ.setdefault("OLLAMA_API_BASE", settings.ollama_host)
    elif provider == "anthropic" and settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    elif provider == "openai" and settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)


def complete(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    settings: Settings,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Non-streaming completion; returns the full response text."""
    import litellm

    preflight_check(provider, settings)
    _prepare_env(provider, settings)
    model_id = _litellm_model_id(provider, model, settings.ollama_host)
    try:
        response = litellm.completion(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise LLMError(f"LLM call failed ({provider}/{model}): {e}") from e
    return response["choices"][0]["message"]["content"] or ""


def stream_complete(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    settings: Settings,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Streaming completion; yields text deltas as they arrive."""
    import litellm

    preflight_check(provider, settings)
    _prepare_env(provider, settings)
    model_id = _litellm_model_id(provider, model, settings.ollama_host)
    try:
        response = litellm.completion(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"LLM streaming call failed ({provider}/{model}): {e}") from e
