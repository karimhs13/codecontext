from __future__ import annotations

from pathlib import Path

from conftest import FakeEmbeddingFunction

from codecontext.core.db import VectorStore
from codecontext.core.parser import CodeChunk


def _chunk(file_path: str, name: str) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        language="python",
        entity_type="function",
        name=name,
        start_line=1,
        end_line=2,
        code=f"def {name}(): pass",
    )


def test_upsert_and_query_roundtrip(
    tmp_path: Path, fake_embedding_fn: FakeEmbeddingFunction
) -> None:
    store = VectorStore(tmp_path, embedding_fn=fake_embedding_fn)
    chunks = [_chunk("a.py", "foo"), _chunk("b.py", "bar")]

    upserted = store.upsert_chunks(chunks)

    assert upserted == 2
    assert store.count() == 2

    results = store.query("foo", top_k=5)
    assert len(results) == 2
    file_paths = {r["metadata"]["file_path"] for r in results}
    assert file_paths == {"a.py", "b.py"}


def test_delete_by_file_removes_only_that_files_chunks(
    tmp_path: Path, fake_embedding_fn: FakeEmbeddingFunction
) -> None:
    store = VectorStore(tmp_path, embedding_fn=fake_embedding_fn)
    store.upsert_chunks([_chunk("a.py", "foo"), _chunk("b.py", "bar")])

    store.delete_by_file("a.py")

    assert store.count() == 1
    results = store.query("bar", top_k=5)
    assert results[0]["metadata"]["file_path"] == "b.py"


def test_reset_clears_collection(tmp_path: Path, fake_embedding_fn: FakeEmbeddingFunction) -> None:
    store = VectorStore(tmp_path, embedding_fn=fake_embedding_fn)
    store.upsert_chunks([_chunk("a.py", "foo")])
    assert store.count() == 1

    store.reset()

    assert store.count() == 0
