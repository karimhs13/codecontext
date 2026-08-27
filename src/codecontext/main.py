"""codecontext — local code parsing, semantic search (RAG), git auditing, and code review."""

from __future__ import annotations

import typer

from codecontext import __version__
from codecontext.cli import ask_cmd, graph_cmd, init_cmd, review_cmd
from codecontext.utils.display import console

app = typer.Typer(
    name="codecontext",
    help=(
        "Local-first CLI for code parsing, semantic search (RAG), git auditing, "
        "and AI code review. Indexing and embeddings run 100% offline; LLM calls "
        "route through Ollama, Anthropic, or OpenAI via litellm."
    ),
    add_completion=True,
    no_args_is_help=True,
)

app.command(name="init", help="Scan and index the current repository for semantic search.")(
    init_cmd.init
)
app.command(
    name="ask", help="Ask a question about the codebase using local semantic search + an LLM."
)(ask_cmd.ask)
app.command(
    name="review", help="Audit staged or unstaged git changes with AST-aware AI review."
)(review_cmd.review)
app.command(name="graph", help="Generate a Mermaid.js dependency graph of the codebase.")(
    graph_cmd.graph
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"codecontext {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """codecontext: local-first RAG, git auditing, and AI code review."""


if __name__ == "__main__":
    app()
