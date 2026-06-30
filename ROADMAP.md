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

## In Progress

### Phase 1c — Reranking + cross-references
See `PHASE1C.md`. Local cross-encoder reranking, import/call/type
ref extraction, symbol_lookup table, score-gated graph expansion.

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
