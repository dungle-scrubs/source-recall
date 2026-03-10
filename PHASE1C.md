# Phase 1c: Reranking + Cross-References

Phase 1c improves retrieval quality via two additions:

1. **Local reranking** — reorder RRF results using a cross-encoder
2. **Cross-reference graph** — extract imports/refs, expand results
   along the dependency graph

## Reranking

### Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Property | Value |
|----------|-------|
| Parameters | 22M |
| Size | ~90 MB |
| Inference | sentence-transformers CrossEncoder |
| Task | Query-document relevance scoring |
| Latency | ~5ms per pair on CPU |

This is a cross-encoder — it scores (query, document) pairs
jointly, which is more accurate than bi-encoder cosine similarity.
It runs after RRF merge on the top candidates (not all chunks).

### Pipeline change

```
FTS top-30 + vector top-30
  → RRF merge
  → take top 20 candidates
  → cross-encoder rerank(query, candidate.content)
  → return top_k
```

Reranking adds ~100ms for 20 candidates. Acceptable for the
30ms baseline query time.

### Graceful degradation

If the cross-encoder model fails to load, skip reranking and
return RRF results directly. Same behavior as pre-1c.

## Cross-References

### What gets extracted

| Ref type | Example | Source |
|----------|---------|-------|
| `import` | `from foo.bar import Baz` | Python import statements |
| `import` | `import { Foo } from './bar'` | TS/JS import statements |
| `type_ref` | `def f(x: SomeType)` | Type annotations |
| `call` | `some_function(args)` | Function/method calls |
| `inherits` | `class Foo(Bar)` | Class inheritance |
| `decorator` | `@some_decorator` | Decorator usage |

### New tables

```sql
CREATE TABLE IF NOT EXISTS refs (
    source_chunk_id TEXT NOT NULL,
    target_symbol   TEXT NOT NULL,
    ref_type        TEXT NOT NULL,  -- import|type_ref|call|inherits|decorator
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(chunk_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_refs_target ON refs(target_symbol);
CREATE INDEX idx_refs_source ON refs(source_chunk_id);

CREATE TABLE IF NOT EXISTS symbol_lookup (
    symbol_name TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_symbol_name ON symbol_lookup(symbol_name);
```

### Graph expansion

After reranking, expand top results along the ref graph:

1. For each top-5 result, look up its `refs` (outgoing edges)
2. Resolve target symbols via `symbol_lookup`
3. Score-gate: only include expanded chunks with vector cosine
   > 0.5 to the query
4. Deduplicate against existing results
5. Append expanded chunks (max 10 expansion slots)

This answers "what does X call?" and "what calls X?" without
the user explicitly asking.

### Ref extraction via tree-sitter

Extract refs during chunking (in `chunker.py`), store during
build (in `builder.py`). Refs are per-chunk, not per-file.

Python extraction patterns:
- `import_from_statement` → target = module + imported name
- `typed_parameter` / `type` nodes → type_ref
- `call` expression → function name
- `class_definition` → superclasses = inherits
- `decorator` → decorator name

TypeScript extraction patterns:
- `import_statement` → imported identifiers
- `type_annotation` → type_ref
- `call_expression` → function name
- `class_heritage` → inherits

## Implementation Order

| Order | File | Changes |
|-------|------|---------|
| 1 | `models.py` | `RefData` dataclass, `RefType` enum |
| 2 | `store.py` | `refs` + `symbol_lookup` tables, CRUD, migration v3 |
| 3 | `chunker.py` | Extract refs during chunking (return with chunks) |
| 4 | `builder.py` | Store refs + symbol_lookup during build |
| 5 | `reranker.py` | New module — `Reranker` protocol + `CrossEncoderReranker` |
| 6 | `querier.py` | Rerank after RRF, graph expansion |
| 7 | `config.py` | `rerank_enabled`, `graph_expansion` fields |

## Configuration

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `rerank_enabled` | `SR_RERANK_ENABLED` | `false` | Enable cross-encoder reranking |

Reranking is opt-in because it adds ~100ms per query and
downloads a ~90MB model on first use. Enable it when retrieval
quality matters more than latency.

## Dependencies

```toml
# cross-encoder uses sentence-transformers (already installed)
# No new dependencies required
```

## What Phase 2 adds

MCP server for LLM tool integration — separate from HTTP server.
