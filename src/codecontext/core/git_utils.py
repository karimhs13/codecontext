"""Git diff and repository status helpers, built on GitPython."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class GitError(RuntimeError):
    """Raised when the current directory is not a usable git repository."""


@dataclass
class FileDiff:
    file_path: str
    change_type: str  # "added" | "modified" | "deleted" | "renamed"
    added_lines: list[int] = field(default_factory=list)
    patch: str = ""


def open_repo(root: Path | None = None) -> Repo:
    path = root or Path.cwd()
    try:
        repo = Repo(path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as e:
        raise GitError(
            f"'{path}' is not inside a git repository. Run `git init` first."
        ) from e
    if repo.bare:
        raise GitError("The git repository is bare; no working tree to diff.")
    return repo


def repo_status_summary(repo: Repo) -> dict[str, int]:
    try:
        untracked = len(repo.untracked_files)
        diff_unstaged = len(repo.index.diff(None))
        diff_staged = len(repo.index.diff("HEAD")) if repo.head.is_valid() else 0
    except GitCommandError as e:
        raise GitError(f"git status failed: {e}") from e
    return {
        "untracked": untracked,
        "unstaged": diff_unstaged,
        "staged": diff_staged,
    }


def current_branch(repo: Repo) -> str:
    try:
        return repo.active_branch.name
    except TypeError:
        return "(detached HEAD)"


def _parse_unified_diff(diff_text: str) -> dict[str, FileDiff]:
    files: dict[str, FileDiff] = {}
    current_file: str | None = None
    current_line = 0
    change_type = "modified"
    patch_lines: list[str] = []

    def flush() -> None:
        if current_file is not None:
            fd = files.setdefault(
                current_file, FileDiff(file_path=current_file, change_type=change_type)
            )
            fd.patch = "\n".join(patch_lines)

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            flush()
            patch_lines = [line]
            current_file = None
            change_type = "modified"
            continue
        patch_lines.append(line)
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files.setdefault(current_file, FileDiff(file_path=current_file, change_type=change_type))
        elif line.startswith("new file mode"):
            change_type = "added"
        elif line.startswith("deleted file mode"):
            change_type = "deleted"
        elif line.startswith("rename to "):
            change_type = "renamed"
        elif line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(line)
            if m:
                current_line = int(m.group(1))
            else:
                current_line = 0
        elif current_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                files[current_file].change_type = change_type
                files[current_file].added_lines.append(current_line)
                current_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                pass  # removed line, does not consume a line in the new file
            elif not line.startswith("\\"):
                current_line += 1

    flush()
    return files


def get_diff(repo: Repo, staged: bool = False) -> list[FileDiff]:
    try:
        if staged:
            diff_text = repo.git.diff("--staged", "--unified=3")
        else:
            diff_text = repo.git.diff("--unified=3")
    except GitCommandError as e:
        raise GitError(f"git diff failed: {e}") from e

    if not diff_text.strip():
        return []

    parsed = _parse_unified_diff(diff_text)
    return list(parsed.values())


def full_diff_text(repo: Repo, staged: bool = False) -> str:
    try:
        if staged:
            return repo.git.diff("--staged", "--unified=3")
        return repo.git.diff("--unified=3")
    except GitCommandError as e:
        raise GitError(f"git diff failed: {e}") from e
