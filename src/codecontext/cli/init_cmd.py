"""`codecontext init` — scan, parse, and index the current repository."""

from __future__ import annotations

import time

import typer

from codecontext.core.db import VectorStore
from codecontext.core.parser import CodeParser, iter_source_files
from codecontext.utils.config import ensure_codecontext_dir, project_root
from codecontext.utils.display import build_progress, console, index_summary_table, print_error, print_success

def init(
    reset: bool = typer.Option(
        False, "--reset", help="Drop and rebuild the existing index instead of upserting."
    ),
) -> None:
    """Parse the codebase with tree-sitter and index it into local ChromaDB storage."""
    root = project_root()
    ensure_codecontext_dir(root)

    files = iter_source_files(root)
    if not files:
        print_error(
            "No supported source files found (Python, JS/TS, C/C++, Java). "
            "Nothing to index."
        )
        raise typer.Exit(code=1)

    console.print(f"[info]Found {len(files)} source files under[/info] {root}")

    store = VectorStore(root)
    if reset:
        console.print("[warning]--reset passed: clearing existing index...[/warning]")
        store.reset()

    parser = CodeParser()
    languages: dict[str, int] = {}
    files_parsed = 0
    files_skipped = 0
    total_chunks = 0
    start = time.time()

    with build_progress() as progress:
        task = progress.add_task("Parsing & embedding", total=len(files))
        batch: list = []
        for path in files:
            parsed = parser.parse_file(path, root)
            if parsed is None:
                files_skipped += 1
                progress.advance(task)
                continue
            chunks, _graph_info = parsed
            if not chunks:
                files_skipped += 1
                progress.advance(task)
                continue

            rel_path = str(path.relative_to(root))
            store.delete_by_file(rel_path)
            batch.extend(chunks)
            files_parsed += 1
            lang = chunks[0].language
            languages[lang] = languages.get(lang, 0) + 1

            if len(batch) >= 128:
                total_chunks += store.upsert_chunks(batch)
                batch = []

            progress.advance(task)

        if batch:
            total_chunks += store.upsert_chunks(batch)

    elapsed = time.time() - start

    table = index_summary_table(
        files_scanned=len(files),
        files_parsed=files_parsed,
        files_skipped=files_skipped,
        chunks_indexed=total_chunks,
        languages=languages,
        elapsed_seconds=elapsed,
    )
    console.print(table)
    print_success(
        f"Indexed {total_chunks} chunks into {ensure_codecontext_dir(root)} "
        f"(collection now has {store.count()} chunks total)."
    )
