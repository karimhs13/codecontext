"""`codecontext ask` — RAG semantic search over the indexed codebase."""

from __future__ import annotations

import typer

from codecontext.core.db import VectorStore
from codecontext.core.llm import LLMError, stream_complete
from codecontext.utils.config import SUPPORTED_PROVIDERS, get_settings, index_exists, project_root
from codecontext.utils.display import console, print_error, print_info, print_warning, search_results_table

SYSTEM_PROMPT = """You are codecontext, an expert AI pair programmer with access to \
retrieved snippets from the user's local codebase. Answer the user's question using \
ONLY the provided context where relevant. Cite file paths and line ranges (e.g. \
`src/foo.py:12-40`) when referencing specific code. If the context does not contain \
enough information, say so clearly instead of guessing. Format your answer in Markdown."""


def _build_prompt(query: str, results: list[dict]) -> str:
    context_blocks = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        header = (
            f"[{i}] {meta.get('file_path')}:{meta.get('start_line')}-{meta.get('end_line')} "
            f"({meta.get('entity_type')} {meta.get('name')})"
        )
        context_blocks.append(f"{header}\n```{meta.get('language', '')}\n{r['document']}\n```")
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    return f"# Retrieved Context\n\n{context_text}\n\n# Question\n\n{query}"


def ask(
    query: str = typer.Argument(..., help="Natural-language question about the codebase."),
    provider: str = typer.Option(
        None, "--provider", help=f"LLM provider: {', '.join(SUPPORTED_PROVIDERS)}. Defaults to config/env."
    ),
    model: str = typer.Option(None, "--model", help="Model name override for the chosen provider."),
    top_k: int = typer.Option(None, "--top-k", help="Number of context chunks to retrieve."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the retrieved context chunks table before the answer."
    ),
) -> None:
    """Embed the query, retrieve top-k relevant chunks, and stream an LLM answer."""
    root = project_root()
    if not index_exists(root):
        print_error(
            "No index found in this directory. Run `codecontext init` first to build the index."
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    resolved_provider = provider or settings.provider
    resolved_top_k = top_k or settings.top_k
    resolved_model = settings.resolved_model(resolved_provider, model)

    if resolved_provider not in SUPPORTED_PROVIDERS:
        print_error(f"Unknown provider '{resolved_provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}.")
        raise typer.Exit(code=1)

    print_info(f"Searching index for: \"{query}\"")
    store = VectorStore(root)
    try:
        results = store.query(query, top_k=resolved_top_k)
    except Exception as e:
        print_error(f"Vector search failed: {e}")
        raise typer.Exit(code=1) from e

    if not results:
        print_warning("No relevant context found in the index. Answering from general knowledge only.")

    if show_context:
        console.print(search_results_table(results))

    prompt = _build_prompt(query, results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    print_info(f"Querying {resolved_provider}/{resolved_model}...")
    console.rule("[heading]Answer")

    from rich.live import Live
    from rich.markdown import Markdown

    full_text = ""
    try:
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            for delta in stream_complete(resolved_provider, resolved_model, messages, settings):
                full_text += delta
                live.update(Markdown(full_text))
    except LLMError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    if not full_text.strip():
        print_warning("The model returned an empty response.")

    console.rule()
