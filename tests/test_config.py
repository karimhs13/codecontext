from __future__ import annotations

from pathlib import Path

import pytest

from codecontext.utils.config import Settings, _ensure_gitignore_entry, provider_env_ok


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_resolved_model_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODECONTEXT_MODEL", "configured-model")
    settings = _settings()
    assert settings.resolved_model("anthropic", override="explicit-model") == "explicit-model"


def test_resolved_model_uses_configured_model_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODECONTEXT_MODEL", "configured-model")
    settings = _settings()
    assert settings.resolved_model("anthropic", override=None) == "configured-model"


def test_resolved_model_falls_back_to_provider_default() -> None:
    settings = _settings()
    assert settings.resolved_model("openai", override=None) == "gpt-4o"
    assert settings.resolved_model("ollama", override=None) == "llama3"


def test_provider_env_ok_missing_anthropic_key() -> None:
    settings = _settings(anthropic_api_key=None)
    ok, msg = provider_env_ok("anthropic", settings)
    assert ok is False
    assert "ANTHROPIC_API_KEY" in msg


def test_provider_env_ok_present_anthropic_key() -> None:
    settings = _settings(anthropic_api_key="sk-test")
    ok, msg = provider_env_ok("anthropic", settings)
    assert ok is True
    assert msg is None


def test_provider_env_ok_ollama_always_ok() -> None:
    settings = _settings()
    ok, msg = provider_env_ok("ollama", settings)
    assert ok is True
    assert msg is None


def test_provider_env_ok_unknown_provider() -> None:
    settings = _settings()
    ok, msg = provider_env_ok("bogus", settings)
    assert ok is False
    assert "Unknown provider" in msg


def test_ensure_gitignore_entry_adds_when_missing(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.log\n")

    _ensure_gitignore_entry(tmp_path)

    content = gitignore.read_text()
    assert ".codecontext/" in content


def test_ensure_gitignore_entry_noop_when_already_present(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.log\n.codecontext/\n")
    before = gitignore.read_text()

    _ensure_gitignore_entry(tmp_path)

    assert gitignore.read_text() == before


def test_ensure_gitignore_entry_noop_when_no_gitignore(tmp_path: Path) -> None:
    _ensure_gitignore_entry(tmp_path)
    assert not (tmp_path / ".gitignore").exists()
