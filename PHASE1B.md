# Phase 1b: Local Vector Search

Phase 1b adds hybrid retrieval: BM25 keyword search (Phase 1) merged
with cosine similarity from a local code embedding model.

**What ships:** `sr index` now embeds chunks during build. `sr ask`
uses BM25 + vector cosine → RRF merge. `sr status` shows vector
coverage. No new CLI commands.

**What it validates:** embedding quality for code search, RRF merge
tuning, sqlite-vec integration, local inference performance.

**What it defers:** API embedders like Voyage (Phase 1b+ or optional
provider), reranking (Phase 1c), cross-refs (Phase 1c).

---

## Key Decisions

### Local-first, not API-first

The primary embedder is **CodeRankEmbed** (Nomic, 137M params, MIT).
It runs locally via `sentence-transformers` + ONNX Runtime. No API
key, no latency, no cost, offline-capable.

Rationale: Voyage Code 3 adds 200-500ms per embed call, requires
an API key, costs money, and fails when the network is down. A local
model at ~50ms/chunk is fast enough to embed during `sr index`
without a separate backfill step.

Voyage becomes an optional future provider, not Phase 1b scope.

### No async refactor

Phase 1b adds one inference boundary: local ONNX model, which is
CPU-bound. The only network I/O would be an API embedder, which
is deferred. Async refactor pays off when MCP server needs
concurrent queries (Phase 2). Premature now.

### No separate backfill command

With local inference at ~50ms/chunk, embedding happens inline
during `sr index`. The `sr backfill` concept only makes sense when
embedding is slow/expensive (API calls). Eliminated.

### Embedder protocol for future extensibility

Even though Phase 1b ships only CodeRankEmbed, the embedder is
behind a Protocol so API providers can be added later without
touching the retrieval pipeline.

---

## Embedding Model

**CodeRankEmbed** (`nomic-ai/CodeRankEmbed`)

| Property | Value |
|----------|-------|
| Parameters | 137M |
| Dimensions | 768 |
| Context length | 8192 tokens |
| License | MIT |
| Model size | ~522 MB |
| Inference | sentence-transformers + ONNX Runtime |
| CSN MRR | 77.9 (SOTA for size) |
| Languages | Python, JavaScript, Java, Go, PHP, Ruby |

### Query prefix requirement

CodeRankEmbed requires a task instruction prefix on queries:

```
"Represent this query for searching relevant code: <query>"
```

Code chunks are embedded as-is (no prefix). This asymmetry is
handled inside the embedder implementation, not exposed to callers.

### First-run model download

The model downloads on first `sr index` (~522 MB). Stored in
the HuggingFace cache (`~/.cache/huggingface/`). Subsequent runs
load from cache. Display a progress bar via Rich during download.

---

## Data Model Changes

### New: `vec_chunks` table (sqlite-vec)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
);
```

Uses `chunk_id` as explicit TEXT primary key — not implicit rowid.
This avoids the rowid instability bug flagged in PLAN.md §7:
rowids aren't stable across VACUUM or table rebuilds, so a
vec_chunks table keyed on rowid would silently point at wrong
chunks after migrations.

### Schema migration (version 2)

```sql
-- Migration 2: add vec_chunks for vector search
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
);
```

Added to `_MIGRATIONS` in `store.py`. Existing Phase 1 indexes
auto-migrate on next `sr index`. `_SCHEMA_VERSION` bumps to 2.

### New meta keys

| Key | Value |
|-----|-------|
| `embed_model` | `nomic-ai/CodeRankEmbed` |
| `embed_dimensions` | `768` |
| `embed_coverage` | `1.0` (fraction of chunks with vectors) |

---

## Architecture

### New module: `embedder.py`

```python
class Embedder(Protocol):
    """Embedding provider protocol."""

    @property
    def dimensions(self) -> int: ...

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed code chunks (no prefix)."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (with task prefix)."""
        ...


class CodeRankEmbedder:
    """Local CodeRankEmbed via sentence-transformers."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            "nomic-ai/CodeRankEmbed",
            trust_remote_code=True,
        )

    @property
    def dimensions(self) -> int:
        return 768

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"Represent this query for searching relevant code: {query}"
        return self._model.encode([prefixed])[0].tolist()
```

### Modified: `store.py`

New methods on `IndexStore`:

- `insert_vectors(chunk_ids: list[str], embeddings: list[list[float]])` —
  batch insert into `vec_chunks`
- `search_vectors(query_embedding: list[float], top_k: int) -> list[tuple[str, float]]` —
  returns `(chunk_id, distance)` pairs
- `get_vector_count() -> int` — for status/coverage
- `delete_vectors_by_file(file_path: str)` — delete vec_chunks
  rows matching chunk_ids from that file

sqlite-vec loaded via:

```python
import sqlite_vec
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

### Modified: `builder.py`

`IndexBuilder.build()` and `refresh()` now:

1. Chunk the file (existing)
2. Insert chunks into `chunks` table (existing)
3. Batch-embed chunk contents via `Embedder.embed_chunks()`
4. Insert vectors into `vec_chunks` via `store.insert_vectors()`

Batching: 32 chunks per embed call (balances memory vs overhead).

Lazy model loading: the `CodeRankEmbedder` is instantiated once
per build, not per file. The `SentenceTransformer` constructor
handles model download + ONNX setup.

### Modified: `querier.py`

New retrieval pipeline:

```
user query
  → FTS5 BM25 search (top 30)
  → vector cosine search (top 30)
  → symbol exact match (if PascalCase/snake_case detected)
  → RRF merge (k=15)
  → apply quality multiplier (0.7x for text_fallback)
  → return top_k (default 8)
```

**RRF (Reciprocal Rank Fusion):**

```python
score = sum(1 / (k + rank) for each result list containing the chunk)
```

With `k=15` (not 60 — optimized for short lists per PLAN.md §14).

When vectors are unavailable (pre-1b index, migration pending),
falls back to FTS-only (Phase 1 behavior). No crash, no error.

### Modified: `__init__.py` (facade)

`Index.__init__` accepts optional `embedder: Embedder | None`.
Defaults to `CodeRankEmbedder()`. Pass `None` to disable vectors
(FTS-only mode).

### Modified: `cli.py`

- `sr index` shows embedding progress (chunk count + time)
- `sr status` shows vector coverage:

```
Vectors:     4,821/4,821 (100%)
Embed model: nomic-ai/CodeRankEmbed (768d)
```

### Modified: `config.py`

New config fields:

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `embed_enabled` | `SR_EMBED_ENABLED` | `true` | Enable vector embeddings |
| `embed_batch_size` | `SR_EMBED_BATCH_SIZE` | `32` | Chunks per embed call |

---

## Dependencies

New dependencies:

```toml
dependencies = [
    # ... existing ...
    "sqlite-vec==0.1.6",
    "sentence-transformers>=3.0.0",
    "onnxruntime>=1.18.0",
]
```

`sqlite-vec` pinned exactly (pre-1.0, same rationale as
tree-sitter). `sentence-transformers` and `onnxruntime` use
range pins — they're stable libraries with semver.

Note: `sentence-transformers` pulls in `torch`. This is a heavy
dependency (~2 GB). Documented in README. Future optimization:
switch to pure ONNX Runtime inference without torch (Phase 2
consideration — requires custom tokenizer + model loading code).

---

## Testing Strategy

### Unit tests

- `test_embedder.py`:
  - `BagOfWordsEmbedder` produces genuine similarity (shared vocab
    → high cosine, disjoint vocab → low cosine)
  - `CodeRankEmbedder.embed_query` prepends task prefix
  - `CodeRankEmbedder.dimensions` returns 768
  - Batch embedding preserves order

- `test_store.py` additions:
  - `vec_chunks` creation via migration
  - `insert_vectors` / `search_vectors` round-trip
  - `delete_vectors_by_file` removes correct rows
  - `get_vector_count` accuracy

- `test_querier.py` additions:
  - RRF merge produces correct ranking
  - FTS-only fallback when no vectors exist
  - Quality multiplier applied after RRF

### Integration tests

- `test_integration.py` additions:
  - Full build with embeddings → query returns relevant results
  - Vector search finds semantically similar code that FTS misses
  - Incremental refresh updates vectors for changed files
  - Status shows vector coverage

### Test embedder: BagOfWordsEmbedder

For unit tests, use a `BagOfWordsEmbedder` (not hash-derived
random vectors). Bag-of-words produces genuine similarity from
shared vocabulary. This catches real retrieval bugs:

```python
class BagOfWordsEmbedder:
    """Test embedder with genuine similarity properties."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dim = dimensions

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        return [self._bow(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._bow(query)

    def _bow(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            idx = hash(word) % self._dim
            vec[idx] += 1.0
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
```

Integration tests that need real model quality use
`CodeRankEmbedder` directly (marked `@pytest.mark.slow`).

---

## File-by-file implementation order

| Order | File | Changes | Dependencies |
|-------|------|---------|--------------|
| 1 | `embedder.py` | New — Protocol + CodeRankEmbedder + BagOfWordsEmbedder | sentence-transformers |
| 2 | `store.py` | Add vec_chunks DDL, migration, CRUD methods | sqlite-vec |
| 3 | `builder.py` | Embed during build/refresh, batch processing | embedder, store |
| 4 | `querier.py` | Vector search, RRF merge, fallback logic | store |
| 5 | `config.py` | `embed_enabled`, `embed_batch_size` fields | — |
| 6 | `__init__.py` | Accept `embedder` param in facade | embedder |
| 7 | `cli.py` | Embedding progress, vector status display | — |
| 8 | `models.py` | Update `IndexStatus` with vector fields | — |

Plus tests for each.

---

## Graceful degradation

| Scenario | Behavior |
|----------|----------|
| Fresh install, first `sr index` | Downloads model, embeds everything |
| Phase 1 index exists, no vectors | `sr index` migrates schema, embeds all chunks |
| `SR_EMBED_ENABLED=false` | FTS-only, no model download, no vec_chunks |
| `sr ask` on index without vectors | Falls back to FTS-only, no error |
| Model download fails (no network) | `sr index` still builds FTS index, warns about missing vectors |
| sqlite-vec import fails | Same as above — FTS-only with warning |

---

## What Phase 1c adds (reranking + refs)

- Cohere rerank (or local CodeRankLLM via ONNX)
- Cross-reference extraction (imports, type_refs, decorators)
- `refs` table + `symbol_lookup` table
- Import-graph-aware symbol resolution
- Graph expansion (2-level, score-gated at cosine > 0.65)
- RRF with query-class-aware weights
