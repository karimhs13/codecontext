from __future__ import annotations

from pathlib import Path

from codecontext.core.graph_builder import (
    GraphBuilder,
    _module_name_from_import,
    _resolve_local_module,
)


def test_module_name_from_import_keeps_full_dotted_path() -> None:
    # Regression test for the bug fixed today: this used to be truncated to
    # just "pkg", which made _resolve_local_module unable to find submodules.
    text = "from pkg.sub.mod import thing"
    assert _module_name_from_import(text, "python") == "pkg.sub.mod"


def test_module_name_from_import_plain_import() -> None:
    assert _module_name_from_import("import pkg.sub.mod", "python") == "pkg.sub.mod"


def test_resolve_local_module_finds_nested_submodule() -> None:
    all_files = {
        "src/pkg/__init__.py": "python",
        "src/pkg/sub/__init__.py": "python",
        "src/pkg/sub/mod.py": "python",
    }
    resolved = _resolve_local_module("pkg.sub.mod", all_files, "python")
    assert resolved == "src/pkg/sub/mod.py"


def test_to_mermaid_produces_edge_for_local_import(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "mod.py").write_text("def helper():\n    pass\n")
    (pkg_dir / "caller.py").write_text("from pkg.mod import helper\n\nhelper()\n")

    builder = GraphBuilder(tmp_path)
    infos = builder.analyze()
    mermaid = builder.to_mermaid(infos)

    assert "pkg_caller_py --> pkg_mod_py" in mermaid


def test_to_mermaid_ignores_external_library_imports(tmp_path: Path) -> None:
    (tmp_path / "only_stdlib.py").write_text("import os\nimport sys\n")

    builder = GraphBuilder(tmp_path)
    infos = builder.analyze()
    mermaid = builder.to_mermaid(infos)

    assert "-->" not in mermaid


def test_to_mermaid_handles_no_files() -> None:
    builder = GraphBuilder.__new__(GraphBuilder)
    mermaid = builder.to_mermaid([])
    assert "No source files found" in mermaid
