"""Builds a Mermaid.js dependency graph from module imports and function calls."""

from __future__ import annotations

import re
from pathlib import Path

from codecontext.core.parser import CodeParser, FileGraphInfo, iter_source_files

_PY_IMPORT_MODULE_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import|^\s*import\s+([\w.]+)")
_JS_IMPORT_PATH_RE = re.compile(r"""from\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""")
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;")
_CPP_INCLUDE_RE = re.compile(r'#include\s*[<"]([^">]+)[>"]')


def _module_name_from_import(text: str, language: str) -> str | None:
    if language == "python":
        m = _PY_IMPORT_MODULE_RE.match(text)
        if m:
            return m.group(1) or m.group(2) or None
    elif language in ("javascript", "typescript", "tsx"):
        m = _JS_IMPORT_PATH_RE.search(text)
        if m:
            return m.group(1) or m.group(2)
    elif language == "java":
        m = _JAVA_IMPORT_RE.match(text)
        if m:
            return m.group(1)
    elif language in ("cpp", "c"):
        m = _CPP_INCLUDE_RE.search(text)
        if m:
            return m.group(1)
    return None


def _resolve_local_module(module: str, all_files: dict[str, str], language: str) -> str | None:
    """Best-effort match of an import string to another file in the repo."""
    if language == "python":
        candidate = module.replace(".", "/")
        for rel_path in all_files:
            stem = rel_path.rsplit(".", 1)[0]
            if stem == candidate or stem.endswith("/" + candidate):
                return rel_path
    elif language in ("javascript", "typescript", "tsx"):
        if not module.startswith("."):
            return None
        for rel_path in all_files:
            stem = "/" + rel_path.rsplit(".", 1)[0]
            norm = "/" + module.lstrip("./")
            if stem.endswith(norm):
                return rel_path
    else:
        for rel_path in all_files:
            if rel_path.endswith(module):
                return rel_path
    return None


def _sanitize_id(rel_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", rel_path)


class GraphBuilder:
    """Scans the repository and produces a Mermaid graph of module dependencies."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.parser = CodeParser()

    def analyze(self) -> list[FileGraphInfo]:
        infos: list[FileGraphInfo] = []
        for path in iter_source_files(self.root):
            parsed = self.parser.parse_file(path, self.root)
            if parsed is None:
                continue
            _, graph_info = parsed
            infos.append(graph_info)
        return infos

    def to_mermaid(self, infos: list[FileGraphInfo]) -> str:
        all_files = {info.file_path: info.language for info in infos}
        lines = ["graph LR"]
        edges: set[tuple[str, str]] = set()
        nodes: set[str] = set()

        for info in infos:
            node_id = _sanitize_id(info.file_path)
            nodes.add(node_id)
            for raw_import in info.imports:
                module = _module_name_from_import(raw_import, info.language)
                if not module:
                    continue
                target = _resolve_local_module(module, all_files, info.language)
                if target and target != info.file_path:
                    edges.add((node_id, _sanitize_id(target)))

        if not edges and not nodes:
            lines.append('  empty["No source files found"]')
            return "\n".join(lines)

        label_map = {_sanitize_id(info.file_path): info.file_path for info in infos}
        for node_id in sorted(nodes):
            label = label_map.get(node_id, node_id)
            lines.append(f'  {node_id}["{label}"]')
        for src, dst in sorted(edges):
            lines.append(f"  {src} --> {dst}")

        return "\n".join(lines)

    def build(self) -> str:
        infos = self.analyze()
        return self.to_mermaid(infos)
