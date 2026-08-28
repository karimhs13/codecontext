"""Tree-sitter based AST extraction for code chunking and graph building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pathspec
from tree_sitter import Node
from tree_sitter_language_pack import get_parser

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".codecontext",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "target",
    ".idea",
    ".vscode",
}


def _load_gitignore_spec(root: Path) -> pathspec.PathSpec:
    patterns: list[str] = []
    gitignore = root / ".gitignore"
    if gitignore.exists():
        try:
            patterns = gitignore.read_text(encoding="utf-8").splitlines()
        except OSError:
            patterns = []
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def iter_source_files(root: Path) -> list[Path]:
    """Walk `root`, respecting `.gitignore` and common noise directories,
    returning paths to files with a recognized source language extension."""
    root = root.resolve()
    spec = _load_gitignore_spec(root)
    results: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            rel = entry.relative_to(root)
            rel_str = str(rel).replace("\\", "/")
            if entry.is_dir():
                if entry.name in DEFAULT_EXCLUDE_DIRS:
                    continue
                if spec.match_file(rel_str + "/"):
                    continue
                walk(entry)
            else:
                if spec.match_file(rel_str):
                    continue
                if language_for_path(entry) is not None:
                    results.append(entry)

    walk(root)
    return results


EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".c": "c",
    ".java": "java",
}

FUNCTION_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "arrow_function"},
    "cpp": {"function_definition"},
    "c": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration"},
}

CLASS_NODE_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "cpp": {"class_specifier", "struct_specifier"},
    "c": {"struct_specifier"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
}

IMPORT_NODE_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "cpp": {"preproc_include"},
    "c": {"preproc_include"},
    "java": {"import_declaration"},
}

CALL_NODE_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "cpp": {"call_expression"},
    "c": {"call_expression"},
    "java": {"method_invocation"},
}

NAME_FIELD_CANDIDATES = ("name", "declarator")


def language_for_path(path: Path) -> str | None:
    return EXTENSION_LANGUAGE_MAP.get(path.suffix.lower())


@dataclass
class CodeChunk:
    file_path: str
    language: str
    entity_type: str  # "function" | "class" | "module"
    name: str
    start_line: int
    end_line: int
    code: str
    docstring: str = ""


@dataclass
class FileGraphInfo:
    file_path: str
    language: str
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    defined_symbols: list[str] = field(default_factory=list)


class CodeParser:
    """Parses a single source file into AST-derived chunks using tree-sitter."""

    def __init__(self) -> None:
        self._parser_cache: dict[str, object] = {}

    def _get_parser(self, language: str):
        if language not in self._parser_cache:
            self._parser_cache[language] = get_parser(language)
        return self._parser_cache[language]

    def parse_file(self, path: Path, root: Path) -> tuple[list[CodeChunk], FileGraphInfo] | None:
        language = language_for_path(path)
        if language is None:
            return None
        try:
            source_bytes = path.read_bytes()
        except OSError:
            return None
        if b"\x00" in source_bytes[:4096]:
            return None  # binary file

        try:
            parser = self._get_parser(language)
            tree = parser.parse(source_bytes)
        except Exception:
            return None

        rel_path = str(path.relative_to(root))
        chunks = self._extract_chunks(tree.root_node, source_bytes, language, rel_path)
        graph_info = self._extract_graph_info(tree.root_node, source_bytes, language, rel_path)

        if not chunks:
            # Fall back to a whole-file module chunk so small/script files are still searchable.
            text = source_bytes.decode("utf-8", errors="replace")
            if text.strip():
                chunks.append(
                    CodeChunk(
                        file_path=rel_path,
                        language=language,
                        entity_type="module",
                        name=path.name,
                        start_line=1,
                        end_line=source_bytes.count(b"\n") + 1,
                        code=text[:4000],
                    )
                )
        return chunks, graph_info

    def _node_text(self, node: Node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _node_name(self, node: Node, source: bytes) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return self._node_text(name_node, source)
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            inner = declarator.child_by_field_name("declarator") or declarator
            return self._node_text(inner, source).split("(")[0].strip()
        return "<anonymous>"

    def _docstring(self, node: Node, source: bytes, language: str) -> str:
        if language != "python":
            return ""
        body = node.child_by_field_name("body")
        if body is None or body.child_count == 0:
            return ""
        first = body.children[0]
        if first.type == "expression_statement" and first.child_count:
            expr = first.children[0]
            if expr.type == "string":
                text = self._node_text(expr, source)
                return text.strip("\"' \n")
        return ""

    def _extract_chunks(
        self, root: Node, source: bytes, language: str, rel_path: str
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        func_types = FUNCTION_NODE_TYPES.get(language, set())
        class_types = CLASS_NODE_TYPES.get(language, set())

        def visit(node: Node) -> None:
            if node.type in class_types:
                name = self._node_name(node, source)
                chunks.append(
                    CodeChunk(
                        file_path=rel_path,
                        language=language,
                        entity_type="class",
                        name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code=self._node_text(node, source)[:6000],
                        docstring=self._docstring(node, source, language),
                    )
                )
            elif node.type in func_types:
                name = self._node_name(node, source)
                chunks.append(
                    CodeChunk(
                        file_path=rel_path,
                        language=language,
                        entity_type="function",
                        name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code=self._node_text(node, source)[:6000],
                        docstring=self._docstring(node, source, language),
                    )
                )
            for child in node.children:
                visit(child)

        visit(root)
        return chunks

    def _extract_graph_info(
        self, root: Node, source: bytes, language: str, rel_path: str
    ) -> FileGraphInfo:
        info = FileGraphInfo(file_path=rel_path, language=language)
        import_types = IMPORT_NODE_TYPES.get(language, set())
        call_types = CALL_NODE_TYPES.get(language, set())
        func_types = FUNCTION_NODE_TYPES.get(language, set())
        class_types = CLASS_NODE_TYPES.get(language, set())

        seen_calls: set[str] = set()

        def visit(node: Node) -> None:
            if node.type in import_types:
                text = self._node_text(node, source).strip().replace("\n", " ")
                info.imports.append(text[:200])
            elif node.type in call_types:
                fn_node = node.child_by_field_name("function") or (
                    node.children[0] if node.children else None
                )
                if fn_node is not None:
                    name = self._node_text(fn_node, source)
                    name = name.split("(")[0].strip()
                    if name and name not in seen_calls:
                        seen_calls.add(name)
                        info.calls.append(name)
            elif node.type in func_types or node.type in class_types:
                info.defined_symbols.append(self._node_name(node, source))
            for child in node.children:
                visit(child)

        visit(root)
        return info
