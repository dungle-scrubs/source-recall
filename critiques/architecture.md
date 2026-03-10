# Architecture & API Design — Adversarial Critique

---

## 1. Sync-First Python SDK — "Hot Path is CPU-Bound"

**The Decision**: Sync-only API justified by claiming tree-sitter parsing and SQLite are CPU-bound. Voyage and Cohere are "batched and infrequent."

**What's Wrong**: The CPU-bound justification is mostly true for *indexing* but completely wrong for *querying*. Decompose a single `query()` call:

- Embed query via Voyage → **pure I/O, ~200–500ms**, blocks the thread
- SQLite vector/FTS/symbol search → CPU + I/O, ~5–20ms, negligible
- Cohere rerank → **pure I/O, ~300–800ms**, blocks the thread
- Ref expansion → fast SQLite read

The two dominant latency contributors are both network I/O. A sync thread idles for **500ms–1.3s of pure network wait** on every query.

The "batched and infrequent" claim is wrong at query time too: query embedding is a single text, not a batch. There is no batch benefit on the hot path.

**The DX problem is worse than the latency problem.** Every async-native consumer — FastAPI, async MCP frameworks, LangChain/LangGraph, Jupyter — must use either `asyncio.run()` (errors if there's already a running loop) or `loop.run_in_executor()` boilerplate. The plan acknowledges the MCP server needs `run_in_executor` anyway. That's the cost of the sync choice manifesting immediately, in the design document, before any code exists.

**Concrete Alternative**: Async-first core with a sync shim for the CLI:

```python
# Async-first core (primary API)
class Index:
    async def build(self) -> None: ...
    async def query(self, question: str, ...) -> list[QueryResult]: ...

# Sync shim (CLI and scripts only)
class SyncIndex:
    def build(self) -> None:
        asyncio.run(self._index.build())
    def query(self, question: str, ...) -> list[QueryResult]:
        return asyncio.run(self._index.query(question, ...))
```

SQLite operations stay sync inside `asyncio.to_thread()`. The Typer CLI uses `SyncIndex`. The MCP server uses `Index` directly with `await`.

**Tradeoffs**: `aiosqlite` adds a dependency; async code is harder to debug; thread-safety semantics get trickier. But the sync shim is a thin wrapper (~30 lines), and the cost of retrofitting async after real consumers exist is enormous.

**Verdict**: The sync-first choice is a bet that the MCP server stays sequential forever and async consumers never arrive. Given the plan's own acknowledgment that dispatch might go concurrent, and given LLM tooling's async-native trend, this decision ages poorly.

---

## 2. `Index` Class as Main Entry Point — God Object

**The Decision**: Single `Index` class exposes `build()`, `refresh()`, `backfill()`, `query()`, and `status()`.

**What's Wrong**: Count the responsibilities `Index` actually owns:
1. Config resolution (5 levels)
2. Database connection lifecycle
3. Advisory file lock acquisition and retry
4. Staleness detection + caching (`index.db.stale_check`)
5. Incremental refresh orchestration (5-step algorithm)
6. Full build orchestration (temp file, WAL checkpoint, atomic rename)
7. Backfill orchestration
8. Query orchestration (embed → retrieve → rerank → expand)
9. Status reporting
10. Temp file cleanup from crashed prior builds

That's **10 distinct responsibilities**. The plan estimates `index.py` at 150–200 lines. That estimate is almost certainly wrong (see §11), and even if correct, those 200 lines are the most heavily mutated file in the project — every new feature touches `Index`.

The maintenance consequence: adding MCP progress streaming requires `build()` callbacks → `Index` grows. Watch mode needs streaming refresh events → `Index` grows. Phase 3 migration needs different config surface → `Index` constructor grows. The query path also pays for indexing machinery it never uses — an MCP server that only calls `query()` still instantiates a class that knows about temp files and file locks.

**Concrete Alternative**: Decompose along operation boundaries, expose a thin façade:

```python
class IndexStore:       # connection + schema lifecycle
class IndexBuilder:     # build + refresh + backfill
class IndexQuerier:     # query + status

# Thin public façade — unchanged API surface
class Index:
    def __init__(self, repo_path: str, **kwargs):
        config = resolve_config(repo_path, **kwargs)
        store = IndexStore.open(config.db_path)
        self._builder = IndexBuilder(store, ...)
        self._querier = IndexQuerier(store, ...)
    def build(self): self._builder.build()
    def query(self, ...): return self._querier.query(...)
```

The MCP server, which is query-only, instantiates `IndexQuerier` directly. Tests for query behavior don't touch build infrastructure.

**Tradeoffs**: More classes to navigate. The façade still exists so public API is unchanged. The decomposition is implementation-level and requires discipline to enforce.

**Verdict**: The God Object will cause real maintenance pain by Phase 2. Decompose before writing the first production line.

---

## 3. Thread Safety Model — "No Sharing, `check_same_thread=True`"

**The Decision**: `Index` instances are not thread-safe. Each thread creates its own instance with its own SQLite connection. `check_same_thread=True` (SQLite default) raises immediately on cross-thread misuse.

**What's Wrong**:

**Problem A: Connection overhead at concurrent load.** Each `Index` instantiation opens a SQLite connection, runs two PRAGMAs, checks `schema_version`, validates `repo_root_commit`, and validates `embedding_dimensions`. ~5–10ms per instantiation. Under `run_in_executor` (concurrent requests), N requests = N full cold-start connection setups simultaneously, with zero sharing.

**Problem B: The per-repo `Index` cache in the MCP server silently loses its benefit under concurrency.** The cache is only safe to read/write from one thread (no lock). Under `run_in_executor`, each thread creates a *new* `Index` instance rather than using the cache. The 5–10ms cache benefit the plan explicitly documents evaporates entirely in the concurrent path. This is a latent correctness+performance bug: the code looks like it benefits from the cache in all cases but silently doesn't under the scenario it was designed for.

**Problem C: Hidden race on the MCP cache.** "Since dispatch is sequential, no locking is needed." That's only correct while the sequential guarantee holds. If it ever breaks (tool-proxy upgrade, partial restart, signal handlers), the cache dict is mutated from multiple threads. Dict operations in Python are GIL-protected for atomic single-step operations, but TTL eviction + LRU replacement involves multi-step mutations that aren't atomic.

**Problem D: `check_same_thread=True` gives an opaque error.** `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread` doesn't identify which `Index` instance crossed threads or how to fix it.

**Concrete Alternative**: Thread-local connections with lazy open:

```python
class IndexStore:
    _local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = self._open_connection()  # pragma setup here
        return self._local.conn
```

One connection per thread, opened lazily on first access. The `Index` object itself is shareable across threads (read path); write operations still acquire the file lock. A single `Index` per repo path can be cached in the MCP server and safely shared across executor threads.

**Tradeoffs**: Thread-local connections aren't explicitly closed (they leak until thread death — acceptable for daemon threads). The lazy-open adds a branch per operation. But it eliminates per-request cold-start cost and makes the cache semantics honest.

**Verdict**: The current model is safe and simple but requires a painful rewrite the moment the sequential dispatch assumption breaks. Use thread-local connections now.

---

## 4. MCP Server Concurrency — Sequential Assumption + `run_in_executor` Fallback

**The Decision**: Sequential dispatch assumed; 60-second TTL LRU cache for `Index` instances; if dispatch goes concurrent, wrap handlers in `asyncio.get_event_loop().run_in_executor(None, handler)`.

**What's Wrong**:

**The fallback is architecturally inconsistent with the cache.** Under `run_in_executor`, each thread gets a *new* `Index` instance (cache isn't thread-safe, as noted in §3). The cache benefit is zero in the concurrent path. The code looks correct for both paths but is only correct for one.

**The stale-index window is undocumented.** When `sr index` runs from CLI while the MCP server has a cached `Index`, the MCP server holds an fd to the pre-rename `index.db`. After `os.rename()`, the MCP server's connection still reads the old file (the fd stays valid until it's closed). For up to 60 seconds, the MCP server serves results from a pre-rebuild index. This is probably acceptable but must be documented, not discovered in production.

**`asyncio.get_event_loop()` is deprecated.** Deprecated in Python 3.10+, may raise or behave unexpectedly in 3.12 in contexts without a running loop. Should be `asyncio.get_running_loop()`.

**Unbounded `run_in_executor` thread pool.** Default `ThreadPoolExecutor` limit: `min(32, cpu_count + 4)`. Each thread doing Voyage + Cohere holds the thread for ~1.5s of network wait. On an 8-core machine: 12 threads max, 12 concurrent queries saturate the pool, request 13 queues silently. Failure mode is invisible latency spikes.

**Concrete Alternative**:

```python
# Thread-safe cache (Day 1, not "if dispatch changes")
_cache_lock = threading.Lock()
_cache: dict[str, tuple[Index, float]] = {}

# Bounded, named thread pool
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="mcp-sr"
)

# Correct API
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(_executor, handler)
```

Document the 60-second stale-index window explicitly. Add `source_recall:status` as the recommended "check freshness" call before trusting `ask` results.

**Verdict**: Treat concurrency as a Day 1 correctness requirement, not a future contingency. The sequential assumption is a time bomb with a documented fuse.

---

## 5. Error Hierarchy — Depth, Granularity, Internal-Only Exceptions in Public API

**The Decision**: 3-level hierarchy under `SourceRecallError`. `EmbeddingKeyMissingError` and `RerankKeyMissingError` exist in the public hierarchy but are caught internally and never propagate to users.

**What's Wrong**:

**Internal-only exceptions in the public hierarchy is misleading.** A user who reads the exception docs might write `except EmbeddingKeyMissingError` and never see it triggered. These are implementation details of `VoyageEmbedder`/`CohereReranker` that the public API absorbs via graceful degradation. Exporting them creates false expectations.

**`FileDiscoveryError` is under-specified.** It covers "git not installed," "path doesn't exist," and "path isn't a directory." These have different recovery actions. Consumers can't programmatically distinguish them without parsing message strings.

**`ConfigError` is too coarse.** Malformed TOML, invalid env var value, and missing required field need different user actions. A single class isn't actionable.

**The hierarchy is too deep for a library.** `SourceRecallError → IndexingError → IndexLockError` requires consumers to understand a 3-level taxonomy to catch correctly. `IndexLockError` means "retry"; `IndexNotFoundError` means "run sr index"; `SchemaVersionError` means "upgrade the package." They need fundamentally different handling but share the same `IndexingError` parent.

**Exceptions lack structured attributes.** `IndexIdentityError` has no `old_commit` or `new_commit` attributes — consumers must parse message strings to get machine-readable data.

**Concrete Alternative**: Flat(ter) public hierarchy with structured attributes:

```python
# Public (exported) — distinct recovery actions
class SourceRecallError(Exception): ...
class IndexNotFoundError(SourceRecallError):
    repo_path: str
class IndexLockError(SourceRecallError):
    lock_path: str
    retry_after: float  # seconds
class IndexStalenessError(SourceRecallError): ...
class SchemaVersionError(SourceRecallError):
    on_disk: int
    expected: int
class ConfigError(SourceRecallError):
    field: str
    value: Any

# Private (not exported from __init__.py)
class _EmbeddingAPIError(Exception): ...   # caught internally
class _RerankAPIError(Exception): ...      # caught internally
```

**Tradeoffs**: Flattening loses `except EmbeddingError` for catching all Voyage issues — but since those never surface publicly, it's not a real loss. Structured attributes require more code to construct.

**Verdict**: Fix this before the public API is documented. Internal exceptions in the public hierarchy are a DX anti-pattern.

---

## 6. Config Resolution — 5 Levels, Constructor > Env Vars

**The Decision**: Constructor args > repo-local `.source-recall.toml` > global `~/.config/source-recall/config.toml` > `SR_` env vars > defaults.

**What's Wrong**:

**Constructor arguments beating environment variables inverts the standard mental model.** Docker, AWS Lambda, Kubernetes, Heroku — all use env vars as the highest-priority runtime override. The plan's ordering means `Index(".", max_file_size=200_000)` in code *cannot* be overridden at deploy time via `SR_MAX_FILE_SIZE`. This is a real operational constraint for CI, containerized MCP servers, and any environment where runtime config comes from env vars rather than code.

**Two config files is a debugging nightmare.** `.source-recall.toml` + `~/.config/source-recall/config.toml` + env vars + constructor kwargs = a 4-dimensional space users must reason about. "Why is `top_k` 5 instead of 8?" Answer: the global config from six months ago. This is the kind of surprise that generates support issues.

**No observability into resolved config.** With 5 resolution levels and no `sr config show` command, users have no way to see the effective config at runtime. This is mandatory, not optional.

**Pydantic Settings doesn't naturally handle two TOML files.** It resolves env vars → dotenv file → init values. Two TOML files at different paths require custom merge logic *before* passing to pydantic-settings, defeating the purpose of using pydantic-settings for config management.

**Concrete Alternative**: Reduce to 3 levels:

```
env vars (highest: SR_MAX_FILE_SIZE, etc.)
  ↓
.source-recall.toml in repo root
  ↓
defaults in code
```

Eliminate the global `~/.config/` file — it's the source of most config confusion. Constructor arguments are treated as explicit overrides equivalent to env vars (both are "caller intent"). Add `sr config show` as a Phase 1 CLI command that prints the resolved config as TOML.

**Tradeoffs**: Removing global config means users who want a consistent `top_k` across all repos lose that convenience. Mitigation: a `~/.source-recall.toml` in the home directory (lower priority than project config) is simpler to explain than a separate `~/.config/` directory.

**Verdict**: 5 levels is 2 too many. Cut the global config in Phase 1. Fix the constructor-beats-env-var ordering. Make `sr config show` mandatory.

---

## 7. Protocol-Based Test Doubles — `FakeEmbedder` with Hash-Derived Vectors

**The Decision**: `FakeEmbedder` returns deterministic vectors based on text hash. Integration tests run `full index → query on fixture repos` using `FakeEmbedder`.

**What's Wrong**: This is the most subtle and dangerous flaw in the plan. **Hash-derived vectors are not semantically meaningful.** They are deterministic random floats. When `test_index_to_query.py` asserts that `sr ask "how does auth work"` returns `AuthService.validate` in the top results, it passes for one of two reasons:

1. FTS5 keyword matching returned it (no embeddings involved)
2. The test asserts `chunk_id in results` where the test planted the chunk — confirming plumbing, not retrieval

The cosine similarity between `hash("how does auth work")` and `hash("AuthService.validate method body")` is **random**. Integration tests with `FakeEmbedder` cannot catch:
- Vector search being accidentally disabled
- Query vector being inverted (multiplied by -1)
- RRF weighting being changed to deprioritize vector results
- The vec_chunks join being broken

The Protocol-based approach is correct. The `FakeEmbedder` *implementation* is wrong.

**Concrete Alternative**: Bag-of-words `FakeEmbedder` with genuine cosine similarity:

```python
class FakeEmbedder:
    """Fake embedder where shared vocabulary produces higher similarity."""
    VOCAB = ["auth", "user", "payment", "validate", "process", "query", "token", ...]
    DIM = len(VOCAB)

    def _embed(self, text: str) -> list[float]:
        words = set(text.lower().split())
        vec = [1.0 if w in words else 0.0 for w in self.VOCAB]
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
```

Now `"how does auth work"` genuinely scores higher against `"AuthService.validate authenticates user tokens"` than against `"PaymentProcessor.run processes charges"`. Integration tests can assert: *the semantically relevant result ranks above the irrelevant one.* This catches real retrieval bugs.

Add a **retrieval quality assertion tier**: not just "some results returned" but "the correct result is ranked above clearly irrelevant results."

**Tradeoffs**: Slightly slower (vector math vs hash lookup); VOCAB must be maintained and kept representative of fixture content. But tests actually validate the retrieval contract.

**Verdict**: A search library where the test suite cannot catch a broken search is dangerous. This is a Day 1 fix before any integration tests are written.

---

## 8. Phasing Strategy — Phase 1 Is Everything At Once

**The Decision**: Phase 1 ships: tree-sitter chunking (3 languages) + text fallback + SQLite + FTS5 + WAL + advisory locking + sqlite-vec + Voyage embedding + batching + retry + Cohere reranking + graceful degradation + hybrid retrieval + cross-reference extraction (3 languages) + refs table + graph expansion + incremental refresh + full CLI (4 commands) + protocol test doubles + schema migrations.

**What's Wrong**: This is not a phase. It's an entire product. There are at least 6 independently shippable components bundled:

1. Chunking pipeline (tree-sitter + fallback)
2. FTS5 + keyword search
3. Vector search (sqlite-vec + Voyage)
4. Reranking (Cohere)
5. Cross-reference graph (refs table + symbol_lookup + resolution pass + query expansion)
6. Incremental refresh

By bundling all 6, the first working binary requires all 6 to be correct simultaneously. The riskiest component — sqlite-vec (pre-1.0, with the vec0 integer PK constraint the plan already flags as a landmine) — must work on Day 1. The most complex component — cross-reference resolution (two-pass, 3 languages, disambiguation algorithm) — also ships Day 1.

The justification "keyword-only search is marginally better than `rg`" is product reasoning, not engineering reasoning. FTS-only would validate: chunking quality, incremental refresh correctness, config resolution, CLI UX, and schema migrations — 5 of 7 subsystems — without touching sqlite-vec.

**Concrete Alternative**:

```
Phase 1a (~2 weeks): Chunking + FTS5 + CLI (index/ask/status)
  No vector store, no embeddings, no reranking, no refs.
  sr ask returns BM25 keyword results.
  Validates: chunking, store schema, incremental refresh, CLI, config.

Phase 1b (~1 week): Vector search
  sqlite-vec integration, Voyage embeddings, graceful degradation.
  Validates: vec0 PK constraints, embedding batching, hybrid merge.

Phase 1c (~1 week): Reranking + cross-references
  Cohere rerank, refs table, query expansion.
  These are additive to the 1a/1b pipeline.
```

**Tradeoffs**: Users (and cheater) can't use semantic search until 1b. First demo is less impressive. But each phase ships to a real user and generates feedback before the next phase is built.

**Verdict**: The "ship value, not features" philosophy is undermined by Phase 1's own scope. Ship a working FTS tool first.

---

## 9. The Cheater Dependency — Phase 3 Migration Without Interface Contract

**The Decision**: Phase 3 replaces cheater's `context/chunkers/`, `context/codebase.py`, and `context/lexical_index.py` with source-recall SDK. Cheater becomes a consumer: `from source_recall import Index`.

**What's Wrong**:

**The migration target is undefined.** The plan doesn't specify: What does cheater's query API return today? Does cheater expect sync or async? What fields does it consume from query results? `QueryResult` is being designed without knowing cheater's consumption contract. This creates two possible outcomes:
- `QueryResult` matches cheater's needs exactly (unlikely without reading cheater's interface)
- Phase 3 requires changing both source-recall's API *and* cheater simultaneously — two codebases, one migration, double the blast radius

**No API stability policy.** When cheater pins `source-recall>=0.1`, what does that guarantee? Phase 1–2 will evolve `Index`, `QueryResult`, and config formats. Without a semver policy, any breaking change in source-recall silently degrades cheater.

**The migration itself has no rollback strategy.** "Replace cheater's internals" is an aspiration, not a plan. Can old and new code paths run simultaneously during migration? What's the acceptance test? The plan has no answer.

**Concrete Alternative**:

1. **Read cheater's interface contract now** — before writing source-recall code. Document what `context/codebase.py` accepts and returns. Design `QueryResult` to fit.

2. **Add an adapter layer in cheater**, not in source-recall:
   ```python
   # cheater/context/source_recall_adapter.py
   from source_recall import Index

   def query_codebase(repo_path: str, question: str) -> list[CheaterChunk]:
       index = Index(repo_path)
       results = index.query(question)
       return [_to_cheater_chunk(r) for r in results]
   ```
   The adapter contains the impedance mismatch. source-recall's API stays clean. The migration is atomic and reversible.

3. **Set a semver policy at Phase 1**: `0.x` = no stability; `1.0` = stable. Cheater pins exact version during Phase 1–2.

4. **Run cheater's existing test suite against the adapter** as the migration acceptance test.

**Verdict**: The migration plan is aspirational. Without reading cheater's interface contract first, Phase 3 is a rewrite risk dressed as a "migration."

---

## 10. CLI Design — `sr ask` Returns Raw Chunks, No Output Modes

**The Decision**: `sr ask` prints Rich-formatted blocks with ANSI color, horizontal rules, scores, and syntax highlighting. No LLM synthesis. "Users pipe results to an LLM if they want a synthesized answer."

**What's Wrong**:

**Rich formatting is hostile to piping.** The plan simultaneously says "users pipe results to an LLM" and "Rich provides syntax highlighting." These goals are in direct conflict. `sr ask "how does auth work" | llm` receives ANSI escape codes, `─────` horizontal rules, score annotations, and file paths interleaved with code. The LLM gets garbage. The user needs to add `--no-color --plain` flags (not in the plan) or use the SDK instead of CLI.

**There's no `--json` flag.** JSON output is the primary integration interface for shell scripting, piping to `jq`, and calling `llm`. Without it, the CLI is usable only for visual inspection — not for the "pipe to an LLM" workflow the plan explicitly describes.

**Missing output modes break common developer workflows:**
- `sr ask --json "..."` → JSON array (for `jq`, `llm`)
- `sr ask --files "..."` → just file paths (for `vim $(sr ask --files "...")`)
- `sr ask --plain "..."` → content only, no ANSI

**`sr ask` with raw chunks is a power-user debugging tool.** A developer asking "how does the payment flow work" doesn't want 8 code blocks — they want an answer. Raw chunks serve developers debugging the retrieval pipeline, not end users. The CLI's primary value is as a debugging/inspection tool; that's fine, but the output format must support both human review and programmatic consumption.

**Concrete Alternative**:

```bash
sr ask "how does auth work"             # default: Rich (human review)
sr ask --json "how does auth work"      # JSON array of QueryResult
sr ask --plain "how does auth work"     # no scores/metadata, content only
sr ask --files "how does auth work"     # file paths only (for shell integration)
```

`--json` emits: `[{"file_path": "...", "symbol_name": "...", "score": 0.94, "content": "..."}]`. Add `--pipe` as an alias for `--plain --no-color`. This should be **Phase 1** — the JSON flag is the primary integration point.

**Verdict**: The current design optimizes for neither goal. `--json` is mandatory for a "retrieval tool." Ship it in Phase 1 or the CLI is usable only for visual inspection.

---

## 11. Project Scope Estimate — ~2000–2500 Lines

**The Decision**: Total production code estimated at 2000–2500 lines. `chunker.py`: 400–500; `store.py`: 300–400; `index.py`: 150–200.

**What's Wrong**: The estimate is 40–60% too low. Auditing against the plan's own descriptions:

**`chunker.py` (claimed 400–500 lines):**
- TypeScript: 7+ node types, React component detection, barrel file detection, type-only import handling
- Python: 4 node types, Pydantic/dataclass detection, module-level grouping, decorator handling
- Bash: function detection, 3-tier fallback heuristics, env_var extraction, heredoc truncation
- All languages: sub-chunk attribution by line range, token-count splitting, signature prepending, 6000-char cap

Realistically: **600–800 lines** for rules alone, not counting shared infrastructure.

**`store.py` (claimed 300–400 lines):**
- DDL for 4 tables
- WAL + busy_timeout setup
- Advisory file lock with retry (fcntl)
- Schema migration runner
- `build()` atomic temp file + WAL checkpoint + rename
- 9 mutation paths (insert/delete/update × chunks + FTS5 + vec_chunks)
- Stale-check file read/write/invalidation
- Repo root commit + embedding dimension verification

Realistically: **500–700 lines**.

**`refs.py` (claimed 150–200):** Three languages of ref extraction, relative import resolution, symbol normalization. Realistically **250–350 lines**.

**`index.py` (claimed 150–200):** Orchestrates build (temp file lifecycle, WAL checkpoint), refresh (5-step algorithm with 500-commit fallback), backfill, query (auto-refresh + staleness check), status, cleanup. 150 lines means 15 lines per function with no error handling in the orchestration layer. Realistically **200–300 lines**.

| Module | Claimed | Realistic |
|--------|---------|-----------|
| chunker.py | 400–500 | 600–800 |
| store.py | 300–400 | 500–700 |
| retriever.py | 200–250 | 250–350 |
| refs.py | 150–200 | 250–350 |
| index.py | 150–200 | 200–300 |
| cli.py | 100–150 | 150–200 |
| models.py | 80–100 | 100–150 |
| config.py | 50–80 | 80–120 |
| watcher.py | 50–80 | 80–120 |
| **Total** | **1,480–1,960** | **2,210–3,090** |

Plus test code (~800–1,200 lines for 4 fixture repos × 3 languages × multiple query types) not counted in the estimate at all.

**Why this matters:** The estimate is used implicitly to justify "Phase 1 is everything." If realistic scope is 3,000+ lines, Phase 1 is a multi-month effort, not a phase. A 50% underestimate that isn't caught until 60% of the way through is the most common cause of project slippage.

**Verdict**: Re-estimate with a 1.5× complexity buffer. Use the realistic number to justify cutting cross-reference extraction from Phase 1 (see §8). The underestimate will cause Phase 1 to drag unless scope is cut explicitly.

---

## Summary Severity Table

| Issue | Severity | When to Fix |
|-------|----------|-------------|
| FakeEmbedder doesn't test retrieval quality | **Critical** | Before writing integration tests |
| Sync-first SDK fails async consumers | **High** | Before Phase 2 (MCP) |
| `Index` god object maintenance burden | **High** | Before first production line |
| Thread-safe cache not implemented until "needed" | **High** | Before Phase 2 |
| `run_in_executor` silently bypasses the Index cache | **High** | Before Phase 2 |
| CLI has no `--json` flag | **High** | Phase 1 |
| Phase 1 scope is too large | **High** | Now (cut scope) |
| Line count underestimate masks Phase 1 risk | **High** | Now |
| Error hierarchy exports internal exceptions | **Medium** | Before 1.0 |
| Constructor-beats-env-var config ordering | **Medium** | Before 1.0 |
| No `sr config show` observability | **Medium** | Phase 1 |
| Phase 3 migration lacks interface contract | **Medium** | Before Phase 3 design |
| MCP stale-index window undocumented | **Low** | Before Phase 2 docs |
| Global config file adds confusion | **Low** | Phase 1 cut candidate |
