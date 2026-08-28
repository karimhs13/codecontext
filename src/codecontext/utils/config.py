"""Environment and CLI settings for codecontext."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CODECONTEXT_DIR_NAME = ".codecontext"
CHROMA_SUBDIR = "chroma"
COLLECTION_NAME = "codecontext_chunks"

# Reasonable default per provider so users don't have to pass --model every time.
DEFAULT_MODELS = {
    "ollama": "llama3",
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
}

SUPPORTED_PROVIDERS = ("ollama", "anthropic", "openai")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = Field(default="ollama", alias="CODECONTEXT_PROVIDER")
    model: str | None = Field(default=None, alias="CODECONTEXT_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    top_k: int = Field(default=8, alias="CODECONTEXT_TOP_K")

    def resolved_model(self, provider: str, override: str | None = None) -> str:
        if override:
            return override
        if self.model:
            return self.model
        return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])


def get_settings() -> Settings:
    return Settings()


def project_root(start: Path | None = None) -> Path:
    """Return the current working directory as the project root.

    codecontext operates on the directory it is invoked from, mirroring
    tools like git — it does not walk upward searching for a marker file.
    """
    return (start or Path.cwd()).resolve()


def codecontext_dir(root: Path | None = None) -> Path:
    return project_root(root) / CODECONTEXT_DIR_NAME


def chroma_dir(root: Path | None = None) -> Path:
    return codecontext_dir(root) / CHROMA_SUBDIR


def ensure_codecontext_dir(root: Path | None = None) -> Path:
    d = codecontext_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore_entry(project_root(root))
    return d


def _ensure_gitignore_entry(root: Path) -> None:
    gitignore = root / ".gitignore"
    entry = f"{CODECONTEXT_DIR_NAME}/"
    if not gitignore.exists():
        return
    try:
        content = gitignore.read_text(encoding="utf-8")
    except OSError:
        return
    if entry in content or CODECONTEXT_DIR_NAME in content.splitlines():
        return
    with gitignore.open("a", encoding="utf-8") as f:
        if content and not content.endswith("\n"):
            f.write("\n")
        f.write(f"\n# Added by codecontext\n{entry}\n")


def index_exists(root: Path | None = None) -> bool:
    return chroma_dir(root).exists() and any(chroma_dir(root).iterdir())


def provider_env_ok(provider: str, settings: Settings) -> tuple[bool, str | None]:
    """Check whether the required credentials/config exist for a provider."""
    if provider == "anthropic":
        if not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return False, (
                "ANTHROPIC_API_KEY is not set. Export it or add it to a .env "
                "file in this directory."
            )
    elif provider == "openai":
        if not (settings.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            return False, (
                "OPENAI_API_KEY is not set. Export it or add it to a .env file in this directory."
            )
    elif provider == "ollama":
        pass  # checked live via a connection probe in llm.py
    else:
        return False, (
            f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return True, None
