from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

from codecontext.utils.display import review_findings_table, search_results_table

# Same theme as codecontext.utils.display.console, so "heading"/"info" etc. resolve.
_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "muted": "grey58",
        "heading": "bold magenta",
    }
)


def _render(table) -> str:
    console = Console(theme=_THEME, record=True, width=120)
    console.print(table)
    return console.export_text()


def test_search_results_table_does_not_execute_link_markup() -> None:
    malicious_path = "[link=https://evil.example.com/phish]https://github.com/real/repo[/link]"
    results = [
        {
            "metadata": {
                "file_path": malicious_path,
                "entity_type": "function",
                "name": "x",
                "start_line": 1,
                "end_line": 2,
            },
            "score": 0.9,
        }
    ]

    output = _render(search_results_table(results))

    # The literal bracket syntax must show up as plain text, not be
    # interpreted as a hyperlink tag hiding a different target URL.
    assert "[link=https://evil.example.com/phish]" in output


def test_review_findings_table_does_not_crash_on_malformed_markup() -> None:
    findings = [
        {
            "severity": "high",
            "category": "sec",
            "location": "some text [/bold] mismatched",
            "issue": "crash test",
        }
    ]

    # Must not raise rich.errors.MarkupError.
    output = _render(review_findings_table(findings))
    assert "mismatched" in output


def test_review_findings_table_escapes_injected_style_tags() -> None:
    findings = [
        {
            "severity": "info][/white][link=https://evil.example.com]click",
            "category": "x",
            "location": "a.py:1",
            "issue": "test",
        }
    ]

    # severity is uppercased for display, so check the escaped tag case-insensitively.
    output = _render(review_findings_table(findings))
    assert "[link=https://evil.example.com]" in output.lower()
