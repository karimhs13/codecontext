"""Shared fixtures for the codecontext test suite."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from git import Repo


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, dependency-free stand-in for FastEmbedFunction in tests."""

    _DIM = 16

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chromadb API name
        return [self._embed_one(text) for text in input]

    def name(self) -> str:
        return "fake-deterministic-embedding"

    @staticmethod
    def get_config() -> dict:
        return {}

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]


@pytest.fixture
def fake_embedding_fn() -> FakeEmbeddingFunction:
    return FakeEmbeddingFunction()


@pytest.fixture
def git_repo(tmp_path: Path) -> Repo:
    """An initialized git repo in a temp dir, with a configured author identity."""
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")
    return repo
