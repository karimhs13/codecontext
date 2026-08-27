"""Rich terminal output formatters shared across CLI commands."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

_theme = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "muted": "grey58",
        "heading": "bold magenta",
    }
)

console = Console(theme=_theme)


def print_error(message: str) -> None:
    console.print(f"[error]✖ {message}[/error]")


def print_success(message: str) -> None:
    console.print(f"[success]✔ {message}[/success]")


def print_warning(message: str) -> None:
    console.print(f"[warning]⚠ {message}[/warning]")


def print_info(message: str) -> None:
    console.print(f"[info]ℹ {message}[/info]")


def print_markdown(text: str, title: str | None = None) -> None:
    md = Markdown(text)
    if title:
        console.print(Panel(md, title=title, border_style="cyan", expand=True))
    else:
        console.print(md)


def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def index_summary_table(
    files_scanned: int,
    files_parsed: int,
    files_skipped: int,
    chunks_indexed: int,
    languages: dict[str, int],
    elapsed_seconds: float,
) -> Table:
    table = Table(title="Index Summary", show_header=True, header_style="heading")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files scanned", str(files_scanned))
    table.add_row("Files parsed", str(files_parsed))
    table.add_row("Files skipped", str(files_skipped))
    table.add_row("Chunks indexed", str(chunks_indexed))
    table.add_row("Elapsed", f"{elapsed_seconds:.2f}s")
    for lang, count in sorted(languages.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  {lang} files", str(count))
    return table


def search_results_table(results: list[dict]) -> Table:
    table = Table(title="Retrieved Context", show_header=True, header_style="heading")
    table.add_column("#", width=3)
    table.add_column("File", overflow="fold")
    table.add_column("Symbol", overflow="fold")
    table.add_column("Lines", justify="right")
    table.add_column("Score", justify="right")
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        symbol = f"{meta.get('entity_type', '?')} {meta.get('name', '')}".strip()
        table.add_row(
            str(i),
            escape(str(meta.get("file_path", "?"))),
            escape(symbol),
            f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
            f"{r.get('score', 0.0):.3f}",
        )
    return table


def review_findings_table(findings: list[dict]) -> Table:
    table = Table(title="Code Review Findings", show_header=True, header_style="heading")
    table.add_column("Severity", style="bold")
    table.add_column("Category")
    table.add_column("File:Line", overflow="fold")
    table.add_column("Issue", overflow="fold")
    severity_style = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "cyan",
        "info": "grey58",
    }
    for f in findings:
        sev = str(f.get("severity", "info")).lower()
        style = severity_style.get(sev, "white")
        table.add_row(
            f"[{style}]{escape(sev.upper())}[/{style}]",
            escape(str(f.get("category", "-"))),
            escape(str(f.get("location", "-"))),
            escape(str(f.get("issue", "-"))),
        )
    return table
