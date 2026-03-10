# War Counsel: Source-Recall Plan Audit

Four adversarial critics examined this plan. Here's the consolidated verdict, organized by severity.

## 🔴 Critical (Fix Before Writing Code)

### 1. `PRAGMA foreign_keys = ON` is never set

**Storage critic**: The entire `ON DELETE CASCADE` design on the refs table silently does nothing. SQLite disables foreign keys by default — you need this pragma on every connection. Without it, deleting a chunk leaves orphaned refs, which is the exact bug the cascade was designed to prevent.

### 2. `FakeEmbedder` doesn't test retrieval quality

**Architecture critic**: Hash-derived deterministic vectors have *random* cosine similarity. Your integration tests can't catch: vector search being disabled, query vectors inverted, RRF weights broken, or the vec_chunks join failing. A bag-of-words fake embedder (where shared vocabulary produces genuine similarity) would actually validate the retrieval contract.

### 3. Phase 1 scope is an entire product, not a phase

**Architecture critic**: You've bundled 6 independently shippable systems (chunking, FTS, vectors, reranking, cross-refs, incremental refresh) into "Phase 1." The realistic line count is 3,000–4,000 lines, not 2,000–2,500. Ship FTS + chunking + CLI first (Phase 1a), add vectors (1b), then reranking + cross-refs (1c).

## 🟠 High Severity

### 4. Sync-first SDK ages poorly

**Architecture critic**: The CPU-bound justification is wrong for queries — Voyage (~200-500ms) and Cohere (~300-800ms) are pure I/O. Every async consumer (MCP, FastAPI, LangChain, Jupyter) needs `run_in_executor` boilerplate. Build async-first core with a sync shim for CLI.

### 5. No local embedding fallback

**Search critic**: "FTS-only when Voyage is down" removes the system's core value proposition (semantic search). A local ONNX embedder (Nomic Embed Code, 768d, Apache 2.0, runs on a MacBook) would be a dramatically better degradation path. FTS-only should be the last resort.

### 6. sqlite-vec pinned as range, not exact version

**Storage critic**: `>=0.1.6,<0.2` allows untested patch releases that could break vec0 semantics. Pre-1.0 library + range pin = segfaults in production. Pin `==0.1.6`.

### 7. vec_chunks rowid instability

**Storage critic**: SQLite rowids aren't stable across `VACUUM` or table rebuilds. If `chunks` is dropped and rebuilt during migration, vec_chunks has dangling references pointing at wrong chunks — silent correctness bug, not errors. Use an explicit `INTEGER PRIMARY KEY`, not implicit rowid.

### 8. WAL file confusion after atomic rename

**Storage critic**: The WAL checkpoint runs on the *temp* file's WAL. After rename, a stale `index.db-wal` from the *old* index's readers could be replayed against the new file. Fix: unlink `index.db-wal` and `index.db-shm` before rename.

### 9. Cross-reference extraction is severely under-specified

**Parsing critic**: "Call" extraction has no specification at all. Type references (TypeScript annotations), decorators, and dynamic imports are entirely unhandled. The refs graph has systematic blind spots in exactly the places that matter.

### 10. symbol_lookup resolution is backwards

**Parsing critic**: Same-directory preference is wrong — you typically import from *other* directories. Barrel files break the algorithm completely (re-export file ranks above the actual implementation). Aliased imports (`import numpy as np`) always resolve to NULL. Use import-graph-aware resolution instead of directory proximity.

### 11. `Index` is a god object with 10 responsibilities

**Architecture critic**: Config resolution, connection lifecycle, file locking, staleness detection, refresh orchestration, build orchestration, backfill, query, status, and temp file cleanup. Decompose into `IndexStore` + `IndexBuilder` + `IndexQuerier` behind a thin facade.

### 12. CLI has no `--json` flag

**Architecture critic**: The plan says "users pipe results to an LLM" but Rich formatting outputs ANSI escape codes and decorative rules. `sr ask "..." | llm` gets garbage. `--json` is mandatory for the "retrieval tool" use case.

### 13. No query rewriting is the biggest retrieval quality gap

**Search critic**: "How does authentication work" fails on FTS (no function named `authentication`) and may fail on vectors too. Vocabulary mismatch is a silent, unacknowledged failure mode. At minimum, add a simple vocabulary expansion map. For bigger gains, consider HyDE (hypothetical document embeddings).

## 🟡 Medium Severity

### 14. RRF k=60 is wrong for short lists

**Search critic**: k=60 was designed for lists of hundreds of documents. At 20-30 results, it under-differentiates. Use k=10-20 and add query-class-aware weights (symbol queries boost FTS; semantic queries boost vectors).

### 15. FTS5 storage wildly underestimated

**Storage critic**: The plan claims "~50MB overhead" — realistic figure is 150-300MB for large repos. FTS5 external-content with triggers would save this storage and be enforced by schema, not convention.

### 16. 1024d embeddings unjustified

**Search critic**: Voyage Code 3 supports Matryoshka truncation to 512d with ~3% quality loss. That halves vector storage (~200MB savings for large repos). Should be the default with 1024d as opt-in.

### 17. React component detection misses 40-60% of real patterns

**Parsing critic**: The detection only catches `export const X = () => <jsx>`. It misses: `forwardRef`, `memo`, `lazy`, function declarations, non-exported components, class components. Use PascalCase + JSX-anywhere-in-body as a two-signal heuristic.

### 18. tree-sitter ABI pinning is reactive, not proactive

**Parsing critic**: The symptom of an ABI mismatch is a segfault, not a graceful error. Pin exact `==` versions and add a CI smoke test that actually parses code on import. Add error-node density check (>10% ERROR nodes → demote to `partial_ast`).

### 19. Migrations not wrapped in savepoints

**Storage critic**: A migration that crashes halfway leaves the schema in a partial state. Next startup tries the same migration, which fails because the column already exists. User is stuck. Wrap each migration in a savepoint.

### 20. 5-level config resolution is too complex

**Architecture critic**: Constructor > repo-local > global > env > defaults. Constructor-beats-env-var inverts the standard mental model (Docker/K8s/Lambda all use env vars as highest-priority override). Cut to 3 levels: env vars > repo-local TOML > defaults. Add `sr config show`.

### 21. Shallow clones break repo identity

**Storage critic**: `git clone --depth=1` returns the shallow boundary commit, not the original root. Full clone of the same repo gets a different root commit → `IndexIdentityError` → forced re-index of the same repo. Fallback to remote URL hash for identity.

### 22. Graph expansion too shallow and unranked

**Search critic**: 1-level expansion misses 3-level call chains. 15 expanded chunks are appended without reranking — the LLM gets unranked filler. Score-gate expanded chunks with vector similarity >0.65 to the query.

### 23. Sub-chunks have no parent_chunk_id

**Parsing critic**: "What does `process_checkout` call?" only retrieves refs from sub-chunk 1, not sub-chunks 2-N. There's no structural link between sub-chunks. Add `parent_chunk_id` and `sub_chunk_index` columns.

### 24. 6000-char cap underestimates TypeScript token density

**Parsing critic**: TS generics/decorators tokenize at ~2.5 chars/token, not 4. A 6000-char TS chunk is ~2400 tokens, not 1500. Cost estimates are 60% off for TS-heavy repos.

## 🟢 Low Severity (But Worth Noting)

- **Bash is the wrong third language** — Go serves 10x more users for similar effort
- **6-char path hash** — collision at ~4600 repos, use 12 chars
- **Chunk ID separator collision** — colon-concatenated fields produce collisions on paths containing colons
- **No `CHECK` constraint on `parse_mode`** — a typo like `'AST'` goes undetected
- **No fsync before os.rename** — power loss between rename and flush corrupts the new index
- **No vector coverage in `sr status`** — users won't know they're running FTS-only after a Voyage outage
- **Error hierarchy exports internal exceptions** — `EmbeddingKeyMissingError` is never raised publicly
- **Text-fallback blocks compete equally with AST chunks** — add a 0.7x quality multiplier in RRF

---

## Top 5 "Change This Decision" Recommendations

| # | Current Decision | Better Alternative | Why |
|---|---|---|---|
| 1 | Ship everything as Phase 1 | Split into 1a (FTS+CLI), 1b (vectors), 1c (rerank+refs) | 3,000+ lines in a single phase is a multi-month project, not a phase |
| 2 | Sync-first SDK | Async-first core + sync shim | Query path is 60-70% network I/O; every MCP/async consumer pays for this |
| 3 | FTS-only degradation | Local ONNX embedder (Nomic Embed Code) as fallback | FTS-only removes the product's core value proposition |
| 4 | Directory-proximity symbol resolution | Import-graph-aware resolution | Current algorithm gives wrong answers for the most common case |
| 5 | Hash-derived FakeEmbedder | Bag-of-words FakeEmbedder | Your test suite literally cannot catch a broken search engine |

---

## Full Individual Critiques

The detailed per-domain critiques with concrete code examples, alternative implementations, and tradeoff analysis are in:

- `critiques/storage.md` — Storage & data model (15 issues)
- `critiques/search.md` — Search & retrieval pipeline (10 issues)
- `critiques/parsing.md` — Parsing & indexing strategy (10 issues)
- `critiques/architecture.md` — Architecture & API design (11 issues)
