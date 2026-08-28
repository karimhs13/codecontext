from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

from codecontext.core.git_utils import GitError, get_diff, open_repo


def _commit_all(repo: Repo, message: str) -> None:
    repo.git.add(A=True)
    repo.index.commit(message)


def test_open_repo_raises_git_error_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        open_repo(tmp_path)


def test_get_diff_unstaged_reports_modified_file(git_repo: Repo, tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("line1\nline2\nline3\n")
    _commit_all(git_repo, "initial")

    f.write_text("line1\nCHANGED\nline3\nline4\n")

    diffs = get_diff(git_repo, staged=False)
    assert len(diffs) == 1
    fd = diffs[0]
    assert fd.file_path == "a.py"
    assert fd.change_type == "modified"
    assert 2 in fd.added_lines  # the changed line
    assert 4 in fd.added_lines  # the appended line


def test_get_diff_staged_only_reflects_staged_changes(git_repo: Repo, tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("line1\n")
    _commit_all(git_repo, "initial")

    (tmp_path / "b.py").write_text("new file\n")
    git_repo.git.add("b.py")
    f.write_text("line1\nunstaged change\n")

    staged_diffs = get_diff(git_repo, staged=True)
    assert [d.file_path for d in staged_diffs] == ["b.py"]
    assert staged_diffs[0].change_type == "added"

    unstaged_diffs = get_diff(git_repo, staged=False)
    assert [d.file_path for d in unstaged_diffs] == ["a.py"]
    assert unstaged_diffs[0].change_type == "modified"


@pytest.mark.xfail(
    reason=(
        "Known bug: _parse_unified_diff only recognizes '+++ b/<path>' to set "
        "current_file. Deleted files emit '+++ /dev/null' instead, so current_file "
        "is never set and the deleted file is silently dropped from get_diff()."
    ),
    strict=True,
)
def test_get_diff_deleted_file(git_repo: Repo, tmp_path: Path) -> None:
    f = tmp_path / "gone.py"
    f.write_text("bye\n")
    _commit_all(git_repo, "initial")

    f.unlink()

    diffs = get_diff(git_repo, staged=False)
    assert len(diffs) == 1
    assert diffs[0].file_path == "gone.py"
    assert diffs[0].change_type == "deleted"
    assert diffs[0].added_lines == []


def test_get_diff_empty_when_no_changes(git_repo: Repo, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("line1\n")
    _commit_all(git_repo, "initial")

    assert get_diff(git_repo, staged=False) == []
