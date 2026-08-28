"""`codecontext graph` — Mermaid.js architecture / dependency visualizer."""

from __future__ import annotations

from pathlib import Path

import typer

from codecontext.core.graph_builder import GraphBuilder
from codecontext.utils.config import project_root
from codecontext.utils.display import console, print_error, print_info, print_success


def graph(
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the Mermaid diagram to this Markdown file instead of stdout.",
    ),
) -> None:
    """Extract module imports/calls across the codebase and emit a Mermaid diagram."""
    root = project_root()
    builder = GraphBuilder(root)

    with console.status("[info]Analyzing imports and call graph...[/info]", spinner="dots"):
        infos = builder.analyze()

    if not infos:
        print_error("No supported source files found to graph.")
        raise typer.Exit(code=1)

    mermaid = builder.to_mermaid(infos)
    total_edges = mermaid.count("-->")
    print_info(f"Analyzed {len(infos)} file(s), found {total_edges} dependency edge(s).")

    if output:
        content = f"# Architecture Graph\n\n```mermaid\n{mermaid}\n```\n"
        output.write_text(content, encoding="utf-8")
        print_success(f"Wrote Mermaid diagram to {output}")
    else:
        console.print("```mermaid")
        console.print(mermaid)
        console.print("```")
