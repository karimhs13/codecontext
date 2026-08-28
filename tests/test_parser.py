from __future__ import annotations

from pathlib import Path

import pytest

from codecontext.core.parser import CodeParser, iter_source_files


def test_iter_source_files_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "keep.py").write_text("def a():\n    pass\n")
    ignored_dir = tmp_path / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "skip.py").write_text("def b():\n    pass\n")

    files = iter_source_files(tmp_path)

    names = {f.relative_to(tmp_path) for f in files}
    assert Path("keep.py") in names
    assert not any(str(n).startswith("ignored") for n in names)


def test_iter_source_files_respects_default_excludes(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("def a():\n    pass\n")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "lib.py").write_text("def b():\n    pass\n")

    files = iter_source_files(tmp_path)

    names = {f.relative_to(tmp_path) for f in files}
    assert Path("real.py") in names
    assert not any(".venv" in n.parts for n in names)


@pytest.mark.xfail(
    reason=(
        "Known bug: CodeParser._docstring only recognizes a docstring wrapped in an "
        "'expression_statement' node, but the installed tree-sitter-language-pack "
        "grammar emits a bare 'string' node as the first statement in a block "
        "instead. Docstrings are therefore never extracted for functions or classes."
    ),
    strict=True,
)
def test_parse_file_extracts_function_and_class_with_docstrings(tmp_path: Path) -> None:
    source = '''"""Module docstring."""


class Greeter:
    """Says hello."""

    def greet(self, name):
        """Return a greeting for name."""
        return f"hello {name}"


def standalone(x):
    """Doubles x."""
    return x * 2
'''
    path = tmp_path / "sample.py"
    path.write_text(source)

    parser = CodeParser()
    result = parser.parse_file(path, tmp_path)
    assert result is not None
    chunks, graph_info = result

    by_name = {c.name: c for c in chunks}
    assert "Greeter" in by_name
    assert by_name["Greeter"].entity_type == "class"
    assert by_name["Greeter"].docstring == "Says hello."

    assert "greet" in by_name
    assert by_name["greet"].entity_type == "function"
    assert by_name["greet"].docstring == "Return a greeting for name."

    assert "standalone" in by_name
    assert by_name["standalone"].start_line == 12
    assert by_name["standalone"].end_line == 14

    assert graph_info.file_path == "sample.py"
    assert graph_info.language == "python"
    assert "Greeter" in graph_info.defined_symbols
    assert "standalone" in graph_info.defined_symbols


def test_parse_file_falls_back_to_module_chunk(tmp_path: Path) -> None:
    path = tmp_path / "script.py"
    path.write_text("x = 1\ny = 2\nprint(x + y)\n")

    parser = CodeParser()
    result = parser.parse_file(path, tmp_path)
    assert result is not None
    chunks, _ = result

    assert len(chunks) == 1
    assert chunks[0].entity_type == "module"
    assert chunks[0].name == "script.py"


def test_parse_file_skips_binary_files(tmp_path: Path) -> None:
    path = tmp_path / "binary.py"
    path.write_bytes(b"\x00\x01\x02binarydata")

    parser = CodeParser()
    result = parser.parse_file(path, tmp_path)
    assert result is None


def test_parse_file_returns_none_for_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("just some notes")

    parser = CodeParser()
    result = parser.parse_file(path, tmp_path)
    assert result is None


def test_extract_graph_info_captures_imports_and_calls(tmp_path: Path) -> None:
    source = """import os
from codecontext.core.parser import CodeParser


def run():
    CodeParser()
    os.getcwd()
"""
    path = tmp_path / "caller.py"
    path.write_text(source)

    parser = CodeParser()
    result = parser.parse_file(path, tmp_path)
    assert result is not None
    _, graph_info = result

    assert any("import os" in imp for imp in graph_info.imports)
    assert any("codecontext.core.parser" in imp for imp in graph_info.imports)
    assert "CodeParser" in graph_info.calls
