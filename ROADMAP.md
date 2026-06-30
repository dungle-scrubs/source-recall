# Roadmap

## Done

### Phase 1 — FTS5 keyword search
Tree-sitter chunking, SQLite FTS5, CLI (`sr index`, `sr ask`,
`sr status`).

### Phase 1b — Local vector search
CodeRankEmbed embeddings, sqlite-vec, RRF hybrid retrieval,
BagOfWordsEmbedder test double.

### Server + multi-repo
`sr serve` with persistent model, /query /status /refresh /health
/repos endpoints, multi-repo support, justfile, launchd plist.

### Prose chunking
Markdown (heading-based), PDF (pymupdf page extraction),
plain text (sentence-boundary sliding window).

### Phase 1c — Reranking + cross-references
Local cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`),
import/inherit/decorator/import ref extraction in `chunker.py`,
`refs` + `symbol_lookup` tables (schema migration v3), store CRUD
with FK cascade, score-gated graph expansion in `querier.py`.
Off by default; enable with `SR_RERANK_ENABLED=true`. Adds ~100ms
per query. Validated by `tests/test_refs.py` (11 tests) and
`tests/test_reranker.py` (3 tests, one marked `slow`).

## Planned

### Phase 2 — MCP server
Model Context Protocol server so LLM tools (Claude, Cursor, etc.)
can use source-recall as a retrieval backend. Separate from the
HTTP query server — MCP uses stdio transport with JSON-RPC.

Tools to expose:

- `search_code` — query with natural language
- `lookup_symbol` — exact symbol lookup
- `get_file_context` — retrieve a file's chunks with refs
- `index_status` — check index health

The MCP server wraps the same `Index` API that `sr serve` uses.
Difference is transport (stdio vs HTTP) and protocol (MCP vs REST).
