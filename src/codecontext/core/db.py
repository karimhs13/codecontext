"""ChromaDB client and vector operations, backed by local fastembed embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from codecontext.core.parser import CodeChunk
from codecontext.utils.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, chroma_dir

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embedder


class FastEmbedFunction(EmbeddingFunction[Documents]):
    """A chromadb-compatible embedding function backed by fastembed (100% offline)."""

    def __init__(self) -> None:
        self._model = _get_embedder()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chromadb API name
        vectors = list(self._model.embed(list(input)))
        return [v.tolist() for v in vectors]

    def name(self) -> str:  # type: ignore[override]
        return "fastembed-bge-small-en-v1.5"


def _chunk_id(chunk: CodeChunk) -> str:
    key = f"{chunk.file_path}:{chunk.entity_type}:{chunk.name}:{chunk.start_line}:{chunk.end_line}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _chunk_document(chunk: CodeChunk) -> str:
    parts = [
        f"File: {chunk.file_path}",
        f"Type: {chunk.entity_type}",
        f"Name: {chunk.name}",
    ]
    if chunk.docstring:
        parts.append(f"Docstring: {chunk.docstring}")
    parts.append(chunk.code)
    return "\n".join(parts)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection for code chunks."""

    def __init__(
        self,
        root: Path | None = None,
        embedding_fn: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        self.root = root
        self.persist_dir = chroma_dir(root)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = embedding_fn or FastEmbedFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(self, chunks: list[CodeChunk], batch_size: int = 64) -> int:
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if not batch:
                continue
            ids = [_chunk_id(c) for c in batch]
            documents = [_chunk_document(c) for c in batch]
            metadatas: list[dict[str, Any]] = [
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "entity_type": c.entity_type,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in batch
            ]
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            total += len(batch)
        return total

    def delete_by_file(self, file_path: str) -> None:
        self._collection.delete(where={"file_path": file_path})

    def query(self, query_text: str, top_k: int = 8) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict[str, Any]] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists, strict=True):
            score = 1.0 - dist  # cosine distance -> similarity
            results.append({"document": doc, "metadata": meta, "score": score})
        return results
