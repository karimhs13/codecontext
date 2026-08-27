# codecontext

A local-first CLI for code parsing, semantic search (RAG), git auditing, and
AI-assisted code review. All indexing and embedding happens on your machine —
only the final prompt to an LLM provider (optionally local via Ollama) leaves
the process.

## Features

- **`codecontext init`** — Parses your repository with `tree-sitter`,
  extracts functions/classes/imports with docstrings, and stores them as
  embedded chunks in a local ChromaDB store under `.codecontext/chroma`.
- **`codecontext ask "<query>"`** — Semantic search over the indexed
  codebase, with retrieved context fed to an LLM (Ollama, Anthropic, or
  OpenAI via `litellm`) and the answer streamed to your terminal as
  Markdown.
- **`codecontext review`** — Reads your `git diff` (staged or unstaged),
  maps changed lines to AST symbols, and asks the LLM for a security /
  performance / code-smell audit plus a PR description.
- **`codecontext graph`** — Builds a Mermaid.js dependency graph of module
  imports and function calls across the codebase.

## Installation

```bash
pip install -e .
```

Requires Python 3.11+.

## Quick start

```bash
cd /path/to/your/project
codecontext init
codecontext ask "How is authentication handled?"
codecontext review --staged
codecontext graph --output architecture.md
```

## LLM Providers

`codecontext` routes LLM calls through [`litellm`](https://github.com/BerriAI/litellm),
so you can use:

- **Ollama** (default, fully local): `--provider ollama --model llama3`
  Requires a running Ollama daemon (`ollama serve`) at `OLLAMA_HOST`
  (default `http://localhost:11434`).
- **Anthropic**: `--provider anthropic --model claude-3-5-sonnet-20241022`
  Requires `ANTHROPIC_API_KEY`.
- **OpenAI**: `--provider openai --model gpt-4o`
  Requires `OPENAI_API_KEY`.

Configure defaults via environment variables or a `.env` file in your
project root:

```bash
CODECONTEXT_PROVIDER=ollama
CODECONTEXT_MODEL=llama3
OLLAMA_HOST=http://localhost:11434
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Storage

All index data lives under `.codecontext/` in the target repository
(ChromaDB persistence directory + config cache). Add `.codecontext/` to
your `.gitignore` — this is done automatically for you if a `.gitignore`
exists.

## Supported languages

Python, JavaScript, TypeScript, C++, and Java (via `tree-sitter-languages`).
Other text files are skipped during parsing but still respected by
`.gitignore` rules.
