from __future__ import annotations

from codecontext.cli.review_cmd import _parse_llm_json, _symbols_touching_lines
from codecontext.core.parser import CodeChunk


def test_parse_llm_json_clean() -> None:
    text = '{"findings": [], "pr_description": "## Title"}'
    assert _parse_llm_json(text) == {"findings": [], "pr_description": "## Title"}


def test_parse_llm_json_fenced_with_language_tag() -> None:
    text = '```json\n{"findings": [], "pr_description": "x"}\n```'
    assert _parse_llm_json(text) == {"findings": [], "pr_description": "x"}


def test_parse_llm_json_extracts_object_from_surrounding_prose() -> None:
    text = 'Sure, here is the review:\n{"findings": [{"severity": "low"}]}\nHope this helps!'
    result = _parse_llm_json(text)
    assert result == {"findings": [{"severity": "low"}]}


def test_parse_llm_json_unparseable_returns_empty_dict() -> None:
    assert _parse_llm_json("not json at all") == {}


def _chunk(name: str, start: int, end: int) -> CodeChunk:
    return CodeChunk(
        file_path="f.py",
        language="python",
        entity_type="function",
        name=name,
        start_line=start,
        end_line=end,
        code="...",
    )


def test_symbols_touching_lines_selects_overlapping_chunks() -> None:
    chunks = [_chunk("inside", 10, 20), _chunk("outside", 30, 40)]
    touched = _symbols_touching_lines(chunks, added_lines=[15])
    assert [c.name for c in touched] == ["inside"]


def test_symbols_touching_lines_empty_added_lines() -> None:
    chunks = [_chunk("a", 1, 5)]
    assert _symbols_touching_lines(chunks, added_lines=[]) == []


def test_symbols_touching_lines_no_overlap() -> None:
    chunks = [_chunk("a", 1, 5)]
    assert _symbols_touching_lines(chunks, added_lines=[100]) == []
