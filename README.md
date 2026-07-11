# source-recall

Fast code and document search. Indexes repositories with tree-sitter
parsing, embeds chunks with a local model, and retrieves via hybrid
BM25 + vector search with reciprocal rank fusion.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Requirements

- Python ≥ 3.12 (3.13 recommended)
- [uv](https://docs.astral.sh/uv/) for installs (or any PEP 517 installer)
- [mise](https://mise.jdx.dev/) optional but recommended — pins Python,
  uv, and just to known-good versions (run `mise install` after cloning)

## Install

Full install (with vector search and reranking, ~2 GB model deps):

```bash
uv tool install -e .[embed]
```

FTS-only install (no model download, instant startup, keyword search
only):

```bash
uv tool install -e .
```

This gives you the `sr` CLI globally.

## Quick Start

```bash
# Index a repository
sr index ~/dev/my-project

# Search
sr ask "how does authentication work" ~/dev/my-project

# Start persistent server (typically <100ms warm queries vs ~12s cold)
sr serve ~/dev/my-project
```

## CLI Commands

### `sr index [PATH]`

Build or rebuild the full index. Parses source files with
tree-sitter, splits into semantic chunks, embeds with
CodeRankEmbed, and stores in SQLite.

```bash
sr index .                        # index current directory
sr index ~/dev/project            # index a specific repo
SR_EMBED_ENABLED=false sr index . # skip embeddings (FTS only)
```

### `sr ask QUESTION [PATH]`

Search the index. Returns ranked code chunks.

```bash
sr ask "payment processing"
sr ask "UserModel" ~/dev/api
sr ask --json "auth"              # JSON output
sr ask --plain "config" | llm     # pipe to LLM
sr ask --files "database"         # file paths only
```

### `sr serve [PATHS...]`

Start a persistent HTTP query server. Loads the embedding model
once on startup (~12s), then serves warm queries in <100ms.

```bash
sr serve .                                    # single repo
sr serve ~/dev/api ~/dev/frontend             # multiple repos
sr serve ~/dev/api --port 7249 --host 0.0.0.0 # custom bind
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check (`{ok, repos, uptime_s}`) |
| `GET` | `/repos` | List loaded repos with stats |
| `POST` | `/query` | Search (`{question, top_k?, repo?}`) |
| `GET` | `/status` | Index metrics |
| `POST` | `/refresh` | Incremental re-index |

When serving multiple repos, include `"repo": "name"` in query
requests. With a single repo, the `repo` field is optional.

```bash
# Query the server
curl -X POST http://localhost:7249/query \
  -H "Content-Type: application/json" \
  -d '{"question": "authentication", "top_k": 5}'
```

### `sr status [PATH]`

Show index metrics: file count, chunk count, vector coverage,
database size.

## Configuration

Configuration is resolved in order (first wins):

1. Environment variables (`SR_` prefix)
2. Repo-local `.source-recall.toml`
3. Built-in defaults

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SR_MAX_FILE_SIZE` | `100000` | Skip files larger than this (bytes) |
| `SR_TOP_K` | `8` | Default number of results |
| `SR_CHUNK_MAX_CHARS` | `6000` | Max characters per chunk before sub-chunking |
| `SR_AUTO_REFRESH` | `true` | Auto-refresh stale indexes before queries |
| `SR_EMBED_ENABLED` | `true` | Enable vector embeddings (set `false` for FTS-only). Requires the `embed` install extra. |
| `SR_EMBED_BATCH_SIZE` | `32` | Chunks per embedding batch |
| `SR_RERANK_ENABLED` | `false` | Enable cross-encoder reranking (slower, more accurate) |

### TOML Config

Create `.source-recall.toml` in your repo root:

```toml
[source-recall]
top_k = 10
max_file_size = 200000
embed_enabled = true
rerank_enabled = true
exclude_patterns = ["node_modules/", "dist/", ".git/"]
```

### Config Reference

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `max_file_size` | int | 100,000 | Bytes. Files above this are skipped. |
| `top_k` | int | 8 | Results per query (1–100). |
| `chunk_max_chars` | int | 6,000 | Sub-chunk threshold. |
| `auto_refresh` | bool | true | Re-index changed files before querying. |
| `embed_enabled` | bool | true | `false` disables vector search entirely. Requires the `embed` install extra. |
| `embed_batch_size` | int | 32 | Tune for memory vs speed (1–512). |
| `rerank_enabled` | bool | false | Cross-encoder reranking over top candidates. Adds ~100ms per query but improves relevance. |
| `exclude_patterns` | list | see below | Glob patterns to skip during indexing. |

Default excludes: `node_modules/`, `dist/`, `build/`, `.git/`,
`__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.tox/`,
`.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`.

## Supported Content

| Type | Strategy | Extensions |
|------|----------|------------|
| Python | tree-sitter AST | `.py` |
| TypeScript/JavaScript | tree-sitter AST | `.ts`, `.tsx`, `.js`, `.jsx` |
| Bash | Regex (function boundaries) | `.sh`, `.bash` |
| Markdown | Heading-based sections | `.md`, `.mdx`, `.markdown` |
| PDF | Page-level text extraction (pymupdf) | `.pdf` |
| Plain text | Sentence-boundary sliding window | `.txt`, `.text`, `.rst`, `.log` |
| Everything else | Blank-line splitting | `*` |

## Retrieval Pipeline

```
Query
  → FTS5 BM25 search (top 30)
  → Vector cosine search (top 30)
  → Symbol exact match (if query looks like a symbol)
  → RRF merge (k=15)
  → Cross-encoder rerank (if enabled, top 3x top_k candidates)
  → Graph expansion (top 5 → follow refs → resolve symbols)
  → Return top_k
```

**Reranking** uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M
params, ~90MB). It scores query-document pairs jointly, which is
more accurate than bi-encoder cosine similarity. Adds ~100ms.
Enable with `SR_RERANK_ENABLED=true`.

**Graph expansion** follows cross-references (imports,
inheritance, decorators) from the top results and pulls in
the referenced definitions. This surfaces related code without
the user explicitly asking for it.

## Embedding Model

**CodeRankEmbed** (`nomic-ai/CodeRankEmbed`, 137M params, 768d,
MIT license). Runs locally via sentence-transformers + ONNX
Runtime. Downloads ~522MB on first run (cached in
`~/.cache/huggingface/`).

Disable with `SR_EMBED_ENABLED=false` for pure FTS keyword
search (no model download, instant startup).

## Auto-Start (macOS)

The daemon can run as a launchd service that auto-starts at login and
restarts on crash. `sr daemon start` generates a plist dynamically
(with resolved paths from your environment) and loads it — no manual
editing required:

```bash
# Configure which repos the daemon serves
sr add ~/dev/my-project

# Generate the plist and start the service
sr daemon start

# Check status
sr daemon status

# View logs
sr daemon logs

# Stop
sr daemon stop
```

Other platforms (systemd on Linux, Task Scheduler on Windows) are not
yet automated; run `sr daemon run` in your platform's service manager.

## Justfile

```bash
just test          # run tests
just t             # run tests (short output)
just lint          # check lint + format
just fix           # auto-fix lint + format
just serve         # start server (current dir)
just query "auth"  # query running server
just health        # server health check
just repos         # list served repos
just status        # server index status
just install       # install sr globally
```

## Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| sentence-transformers | ~2GB (includes torch) | Embedding + reranking models |
| sqlite-vec | ~5MB | Vector search in SQLite |
| pymupdf | ~15MB | PDF text extraction |
| tree-sitter + grammars | ~10MB | AST parsing |
| fastapi + uvicorn | ~5MB | Query server |
| apsw | ~5MB | SQLite extension loading on macOS |

## Development

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full guide. The
short version:

```bash
# Setup
mise install
uv sync

# Run tests (fast, skips model-loading tests)
uv run pytest -m "not slow"

# Run all tests including slow model tests
uv run pytest

# Lint
uv run ruff check src/ tests/
```

This repo uses mise to pin the local Python, uv, and just versions.
Run `mise install` after cloning, then use the existing `uv` and `just`
commands normally.

## Support

- **Bugs and feature requests:** [GitHub Issues][issues]
- **Security vulnerabilities:** see [`SECURITY.md`](./SECURITY.md) — do
  not open a public issue; use a private GitHub Security Advisory.
- **Code of conduct:** [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)

[issues]: https://github.com/dungle-scrubs/source-recall/issues
