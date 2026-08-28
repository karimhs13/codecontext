"""`codecontext review` — git diff analysis, AST symbol mapping, and LLM audit."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from codecontext.core.git_utils import (
    FileDiff,
    GitError,
    current_branch,
    get_diff,
    open_repo,
    repo_status_summary,
)
from codecontext.core.llm import LLMError, complete
from codecontext.core.parser import CodeChunk, CodeParser, language_for_path
from codecontext.utils.config import SUPPORTED_PROVIDERS, get_settings, project_root
from codecontext.utils.display import (
    console,
    print_error,
    print_info,
    print_markdown,
    print_warning,
    review_findings_table,
)

SYSTEM_PROMPT = """You are codecontext, a senior staff software engineer performing a \
rigorous code review of a git diff. You will be given the diff plus the enclosing \
function/class bodies (resolved via AST parsing) for added lines. Analyze the change for:
1. Security vulnerabilities (OWASP Top 10: injection, auth, secrets, deserialization, SSRF, etc.)
2. Performance issues (N+1 queries, unnecessary loops/allocations, blocking calls)
3. Code smells / maintainability issues (naming, duplication, missing error handling, dead code)

Respond with a single JSON object with exactly two top-level keys:
- "findings": a list of objects, each with keys "severity" (one of critical/high/medium/low/info), \
"category" (one of security/performance/smell), "location" (e.g. "src/foo.py:42"), and "issue" \
(a one-to-two sentence description with a concrete recommendation).
- "pr_description": a Markdown string containing a concise PR title (as an H2) followed by a \
"## Summary" bullet list of what changed and why, and a "## Testing" section describing how \
to verify the change.

Return ONLY the JSON object, no surrounding prose or code fences."""


def _symbols_touching_lines(chunks: list[CodeChunk], added_lines: list[int]) -> list[CodeChunk]:
    if not added_lines:
        return []
    touched = []
    for chunk in chunks:
        if any(chunk.start_line <= ln <= chunk.end_line for ln in added_lines):
            touched.append(chunk)
    return touched


def _build_review_context(root: Path, diffs: list[FileDiff]) -> str:
    parser = CodeParser()
    blocks: list[str] = []
    for fd in diffs:
        blocks.append(f"### Diff: {fd.file_path} ({fd.change_type})\n```diff\n{fd.patch}\n```")

        abs_path = root / fd.file_path
        if fd.change_type != "deleted" and language_for_path(abs_path) and abs_path.exists():
            parsed = parser.parse_file(abs_path, root)
            if parsed:
                chunks, _ = parsed
                touched = _symbols_touching_lines(chunks, fd.added_lines)
                for chunk in touched:
                    blocks.append(
                        f"### Enclosing {chunk.entity_type} `{chunk.name}` in {chunk.file_path} "
                        f"(lines {chunk.start_line}-{chunk.end_line})\n"
                        f"```{chunk.language}\n{chunk.code}\n```"
                    )
    return "\n\n".join(blocks)


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def review(
    staged: bool = typer.Option(
        False,
        "--staged",
        help="Review staged changes (git diff --staged) instead of the working tree.",
    ),
    uncached: bool = typer.Option(
        False, "--uncached", help="Review unstaged working-tree changes (default behavior)."
    ),
    provider: str = typer.Option(
        None, "--provider", help=f"LLM provider: {', '.join(SUPPORTED_PROVIDERS)}."
    ),
    model: str = typer.Option(None, "--model", help="Model name override for the chosen provider."),
) -> None:
    """Analyze the current git diff for security, performance, and code-smell issues."""
    root = project_root()

    try:
        repo = open_repo(root)
    except GitError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    use_staged = staged and not uncached
    try:
        status = repo_status_summary(repo)
        diffs = get_diff(repo, staged=use_staged)
    except GitError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    branch = current_branch(repo)
    scope = "staged" if use_staged else "unstaged (working tree)"
    print_info(f"Branch: {branch} | Reviewing {scope} changes")
    print_info(
        f"Status: {status['staged']} staged, {status['unstaged']} unstaged, "
        f"{status['untracked']} untracked file(s)"
    )

    if not diffs:
        print_warning(
            f"No {scope} changes found. Try `codecontext review --staged` "
            "if you have staged changes, or make some edits first."
        )
        raise typer.Exit(code=0)

    settings = get_settings()
    resolved_provider = provider or settings.provider
    resolved_model = settings.resolved_model(resolved_provider, model)

    if resolved_provider not in SUPPORTED_PROVIDERS:
        print_error(
            f"Unknown provider '{resolved_provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )
        raise typer.Exit(code=1)

    print_info(f"Mapping {len(diffs)} changed file(s) to AST symbols...")
    context_text = _build_review_context(root, diffs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_text},
    ]

    print_info(f"Requesting review from {resolved_provider}/{resolved_model}...")
    try:
        with console.status("[info]Analyzing diff...[/info]", spinner="dots"):
            raw = complete(resolved_provider, resolved_model, messages, settings, max_tokens=4096)
    except LLMError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    parsed = _parse_llm_json(raw)
    findings = parsed.get("findings", [])
    pr_description = parsed.get("pr_description", "")

    if not findings and not pr_description:
        print_warning("Model response could not be parsed as structured JSON. Showing raw output:")
        print_markdown(raw)
        raise typer.Exit(code=0)

    if findings:
        console.print(review_findings_table(findings))
    else:
        print_info("No issues found.")

    if pr_description:
        print_markdown(pr_description, title="Suggested PR Description")

    severities = {f.get("severity", "").lower() for f in findings}
    if "critical" in severities or "high" in severities:
        raise typer.Exit(code=1)
