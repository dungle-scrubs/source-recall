# Phase 1: FTS + Chunking + CLI

Phase 1 delivers a working code search tool with keyword retrieval.
No vectors, no embeddings, no reranking, no cross-references.

**What ships:** `sr index`, `sr ask`, `sr status`, `sr clean` —
BM25 keyword search over tree-sitter-chunked code.

**What it validates:** chunking quality, FTS5 retrieval, incremental
refresh, schema migrations, config resolution, CLI UX, store
correctness.

**What it defers:** vectors (Phase 1b), reranking + cross-refs
(Phase 1c), MCP server (Phase 2).

---

## Scope

| In scope | Out of scope |
|----------|-------------|
| tree-sitter chunking (TS/JS, Python) | Bash AST chunking (regex fallback only) |
| Text fallback for unsupported languages | Vector embeddings (Voyage) |
| FTS5 external-content with triggers | sqlite-vec / ANN search |
| BM25 keyword retrieval | Cohere reranking |
| Incremental refresh via git diff | Cross-reference extraction |
| CLI: index, ask, status, clean | Graph expansion |
| `--json` output on all commands | MCP server |
| Config: env vars > repo-local TOML > defaults | Global `~/.config/` config file |
| Advisory PID-file locking | Async SDK (sync-first for Phase 1) |
| Schema migrations with savepoints | |
| `sr config show` | |

---

## Architecture

```
source_recall/
├── __init__.py          # Public API: Index, SyncIndex (thin facade)
├── config.py            # 3-level config resolution
├── models.py            # Pydantic models, error hierarchy
├── chunker.py           # tree-sitter chunking + text fallback
├── store.py             # IndexStore: connection, schema, CRUD
├── builder.py           # IndexBuilder: build, refresh, backfill
├── querier.py           # IndexQuerier: FTS query, status
├── cli.py               # Typer CLI
└── _version.py          # Version
```

**Key decomposition** (from architecture critique §2): No god object.
`IndexStore` owns connections + schema. `IndexBuilder` owns
build/refresh. `IndexQuerier` owns queries. `Index` is a thin facade.

---

## Data Model

### Schema

```sql
-- Metadata
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Keys: schema_version, repo_root_commit, repo_remote_url_hash,
--        repo_path, indexed_at, last_commit

-- Chunks
CREATE TABLE chunks (
    id            TEXT PRIMARY KEY,  -- sha256(len:path|len:symbol|hash)[:32]
    file_path     TEXT NOT NULL,
    symbol_name   TEXT NOT NULL DEFAULT '',
    symbol_type   TEXT NOT NULL DEFAULT 'block',
    content       TEXT NOT NULL,
    start_line    INTEGER NOT NULL DEFAULT 0,
    end_line      INTEGER NOT NULL DEFAULT 0,
    parent_chunk_id   TEXT,          -- links sub-chunks to parent
    sub_chunk_index   INTEGER,       -- ordering within parent
    search_quality    TEXT NOT NULL DEFAULT 'ast'
        CHECK (search_quality IN ('ast', 'regex', 'text_fallback')),
    FOREIGN KEY (parent_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id)
    WHERE parent_chunk_id IS NOT NULL;

-- FTS5 external-content (storage-efficient, trigger-enforced)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content, file_path, symbol_name,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers enforce FTS consistency at schema level
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
    VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
    INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
    VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
END;

-- File change tracking
CREATE TABLE file_hashes (
    file_path    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,       -- sha256 hex
    parse_mode   TEXT NOT NULL DEFAULT 'ast'
        CHECK (parse_mode IN ('ast', 'text_fallback', 'regex')),
    mtime_ns     INTEGER              -- stat mtime for non-git fast path
);

-- Schema migrations
CREATE TABLE schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    description TEXT NOT NULL
);
```

### Critical fixes incorporated

| Critique | Fix |
|----------|-----|
| `PRAGMA foreign_keys = ON` missing (storage §5) | Set on every connection alongside WAL + busy_timeout |
| FTS5 standalone wastes 150-300MB (storage §3) | External-content with triggers |
| No `CHECK` on parse_mode (storage §5) | CHECK constraints on `parse_mode` and `search_quality` |
| Chunk ID separator collision (storage §5) | Length-prefixed: `sha256(f"{len(path)}:{path}\|{len(sym)}:{sym}\|{hash}")` |
| No parent_chunk_id for sub-chunks (parsing §8) | `parent_chunk_id` + `sub_chunk_index` columns |
| Migrations not wrapped in savepoints (storage §6) | `schema_migrations` table + savepoint per migration |
| 6-char path hash collision (storage §7) | 12-char hash (48-bit) |
| WAL file confusion after rename (storage §8) | Unlink stale `-wal` and `-shm` before rename |
| Shallow clone breaks identity (storage §9) | Fallback to remote URL hash |

---

## Connection Setup

Every connection executes:

```python
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;
```

---

## Config Resolution (3 levels)

```
env vars (highest)    SR_MAX_FILE_SIZE, SR_TOP_K, etc.
  ↓
.source-recall.toml   in repo root
  ↓
defaults              in code
```

No global `~/.config/` file. Constructor kwargs are equivalent to
env vars (both are "caller intent").

### Config fields (Phase 1)

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `max_file_size` | `SR_MAX_FILE_SIZE` | 100000 | Skip files larger than N bytes |
| `top_k` | `SR_TOP_K` | 8 | Results returned by `ask` |
| `chunk_max_chars` | `SR_CHUNK_MAX_CHARS` | 6000 | Sub-chunk threshold |
| `auto_refresh` | `SR_AUTO_REFRESH` | true | Refresh before queries |
| `exclude_patterns` | — | see below | Glob patterns to exclude |

Default excludes: `node_modules/`, `__pycache__/`, `.git/`, `dist/`,
`build/`, `*.min.js`, `*.map`, `*.lock`, `package-lock.json`,
`*.pyc`, `.env`, `venv/`, `.venv/`.

---

## Chunking

### Languages with AST support (Phase 1)

| Language | Extensions | Chunk boundaries |
|----------|-----------|------------------|
| TypeScript/JS | `.ts`, `.tsx`, `.js`, `.jsx` | functions, arrow functions, classes (shell), methods, interfaces, type aliases, enums |
| Python | `.py` | functions, classes (shell), methods, module-level assignments |

### React component detection (parsing critique §10)

Two-signal approach:
1. PascalCase exported name in `.tsx`/`.jsx`
2. JSX node anywhere in function body (including nested)

Plus explicit detection of `React.memo(...)`, `React.forwardRef(...)`,
`React.lazy(...)` call expressions → `symbol_type = "component"`.

### Bash (regex only, not full AST)

- Regex: split on `^function\s+\w+|^\w+\s*\(\)` boundaries
- Extract `source` includes and env var assignments
- `search_quality = 'regex'`

### Text fallback

- Blank-line splitting for all other extensions
- `search_quality = 'text_fallback'`
- **Quality multiplier**: 0.7x score in BM25 ranking for
  text_fallback chunks (prevents noisy blocks from ranking above
  AST-parsed functions)

### Sub-chunking

- Soft target: ~1500 tokens (~6000 chars for Python, ~4000 for TS)
- Prefer AST-aware boundaries (try/except, if/else, comment sections)
- **5-10 line overlap** between consecutive sub-chunks
- Function signature prepended to each sub-chunk
- `parent_chunk_id` links back to root chunk
- `sub_chunk_index` preserves ordering

### Chunk ID

Length-prefixed to avoid separator collision:

```python
raw = f"{len(file_path)}:{file_path}|{len(symbol_name)}:{symbol_name}|{content_hash}"
chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
```

### tree-sitter pinning (parsing critique §1)

```toml
dependencies = [
    "tree-sitter==0.24.4",
    "tree-sitter-language-pack==0.24.4",
]
```

Exact `==` pins from day one. CI smoke test that imports and parses
known code. Error-node density check: >10% ERROR nodes → demote to
`parse_mode = 'partial_ast'`, fall back to text chunking for that file.

---

## Retrieval (FTS-only for Phase 1)

### Query pipeline

```
user query
  → FTS5 BM25 search (top 30)
  → symbol exact match (if PascalCase/snake_case detected)
  → merge + deduplicate
  → apply quality multiplier (0.7x for text_fallback)
  → rank by adjusted BM25 score
  → return top_k (default 8)
```

### Symbol detection heuristic

Multi-signal scoring (search critique §5):
- +0.3 for PascalCase tokens
- +0.2 for snake_case tokens
- +0.2 for dot-qualified tokens (`foo.bar`)
- +0.1 for camelCase tokens
- −0.1 for question words (how, why, what, where)

If symbol_weight > 0.3 → include symbol exact match in results.
Symbols matched by name get a rank-1 bonus.

### QueryResult model

```python
@dataclass(frozen=True)
class QueryResult:
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    content: str
    score: float
    start_line: int
    end_line: int
    search_quality: str      # ast | regex | text_fallback
    match_reason: str         # "bm25", "symbol_exact", "bm25+symbol"
```

---

## Incremental Refresh

### Algorithm

1. Read `last_commit` from meta
2. Verify ancestor: `git merge-base --is-ancestor <last> HEAD`
   - If fails (rebase, filter-repo): skip git diff, use file hash
     comparison only
3. Count changed files: `git diff --name-only <last> | wc -l`
   - If > 500 files: full rebuild (not commit-count threshold)
4. For each changed file:
   - Read content, compute sha256 hash
   - Compare to stored hash in `file_hashes`
   - If different: re-chunk, delete old chunks, insert new chunks
   - FTS updated automatically via triggers
5. Update `last_commit` and `indexed_at` in meta

### Non-git mode

- `mtime_ns` column in `file_hashes` enables fast staleness check
- Only re-hash files whose mtime changed (O(n) stat, not O(n) read)
- Fall back to full hash if mtime unavailable

---

## Build Pipeline

### Full build (`sr index`)

1. Acquire PID-file lock (`index.db.lock`)
   - Write `{"pid": <pid>, "started": "<iso>"}` to lock file
   - If lock exists, check if PID alive (`os.kill(pid, 0)`)
   - If dead → steal lock; if alive → skip or wait
2. Create temp DB: `index.db.tmp.<pid>`
3. Apply schema DDL
4. Discover files (respecting excludes)
5. For each file:
   - Read content, compute sha256
   - Parse with tree-sitter (or regex/text fallback)
   - Chunk → insert into `chunks` table
   - FTS populated automatically via trigger
   - Store hash in `file_hashes`
6. Write meta (schema_version, repo identity, timestamps)
7. `PRAGMA wal_checkpoint(TRUNCATE)` on temp DB
8. `os.fsync(fd)` on temp file
9. Close connection
10. Unlink stale `index.db-wal` and `index.db-shm` if present
11. `os.rename(tmp, index.db)`
12. Release PID-file lock

### Index directory

```
~/.local/share/source-recall/<repo-name>-<sha256(realpath)[:12]>/
```

12-char hash (48-bit) — collision-safe for any realistic use.

### Repo identity

Stored in meta:
- `repo_root_commit`: `git rev-list --max-parents=0 HEAD`
  (comma-separated if multiple roots)
- `repo_remote_url_hash`: `sha256(git remote get-url origin)`
  (fallback for shallow clones where root commit is the shallow
  boundary)

Identity check on open: if root commit doesn't match AND remote URL
hash doesn't match → `IndexIdentityError`.

---

## Error Hierarchy

Flat public hierarchy with structured attributes (architecture
critique §5). Internal exceptions are private.

```python
class SourceRecallError(Exception): ...

class IndexNotFoundError(SourceRecallError):
    repo_path: str

class IndexLockError(SourceRecallError):
    lock_path: str
    retry_after: float

class IndexIdentityError(SourceRecallError):
    stored_commit: str
    current_commit: str

class SchemaVersionError(SourceRecallError):
    on_disk: int
    expected: int

class ConfigError(SourceRecallError):
    field: str
    value: Any

class FileDiscoveryError(SourceRecallError):
    reason: str  # "git_not_installed" | "path_not_found" | "not_a_directory"

# Private (not exported from __init__.py)
class _ParseError(Exception): ...
```

---

## CLI

### Commands

```
sr index [PATH]         Build/rebuild index for repo at PATH (default: .)
sr ask QUESTION [PATH]  Search the index
sr status [PATH]        Show index status
sr clean                Remove orphaned indexes
sr config show [PATH]   Print resolved config as TOML
```

### Output modes (architecture critique §10)

Every data-outputting command supports:

```
sr ask "query"              # default: Rich human-readable
sr ask --json "query"       # JSON array of QueryResult
sr ask --plain "query"      # content only, no ANSI/scores
sr ask --files "query"      # file paths only (for shell integration)
```

`--json` is the primary integration point. When stdout is not a TTY,
auto-switch to `--plain` (no ANSI codes).

### `sr status` output

```
Repository:  /Users/kevin/dev/myrepo
Index:       ~/.local/share/source-recall/myrepo-a1b2c3d4e5f6/index.db
Size:        12.4 MB
Indexed at:  2026-03-10 09:00:00
Last commit: abc1234
Files:       342 (287 AST, 12 regex, 43 text fallback)
Chunks:      4,821
Stale files: 3
```

---

## Dependencies

```toml
[project]
name = "source-recall"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "tree-sitter==0.24.4",
    "tree-sitter-language-pack==0.24.4",
    "typer>=0.15.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-tmp-files>=0.0.2",
    "ruff>=0.9.0",
]

[project.scripts]
sr = "source_recall.cli:app"
```

No sqlite-vec. No Voyage. No Cohere. No aiohttp. Minimal dependency
surface.

---

## Testing Strategy

### Fixture repos

Small synthetic repos in `tests/fixtures/`:

| Fixture | Purpose |
|---------|---------|
| `ts-app/` | TypeScript with components, barrel exports, classes |
| `py-app/` | Python package with classes, decorators, imports |
| `mixed/` | Multi-language with unsupported files for text fallback |

### Test tiers

1. **Unit tests** — chunker (AST boundaries, sub-chunking, overlap),
   store (CRUD, triggers, migrations), config resolution
2. **Integration tests** — full index → query on fixture repos,
   incremental refresh, identity checks
3. **CLI tests** — `sr index`, `sr ask --json`, `sr status`,
   `sr clean` via Typer's test runner

### Key assertions

- Chunker: function boundaries match expected line ranges
- Chunker: sub-chunks have overlap, parent_chunk_id set
- Chunker: React component detection catches `forwardRef`, `memo`,
  function declarations
- Chunker: error-node density >10% → text fallback
- Store: FTS triggers fire on insert/update/delete
- Store: `PRAGMA foreign_keys = ON` verified
- Store: migrations wrapped in savepoints; partial failure rolls back
- Store: atomic swap doesn't leave stale WAL files
- Retrieval: BM25 returns relevant results for keyword queries
- Retrieval: text_fallback chunks score lower than AST chunks
- Retrieval: symbol exact match ranks at top
- Refresh: only changed files re-indexed
- Refresh: rebase detected, falls back to hash comparison
- CLI: `--json` produces valid JSON with no ANSI codes
- CLI: `--files` produces one path per line

---

## File-by-file implementation order

| Order | File | Est. lines | Dependencies |
|-------|------|-----------|--------------|
| 1 | `models.py` | 100-150 | — |
| 2 | `config.py` | 80-120 | models |
| 3 | `store.py` | 500-700 | models, config |
| 4 | `chunker.py` | 500-650 | models |
| 5 | `builder.py` | 200-300 | store, chunker, config |
| 6 | `querier.py` | 150-200 | store, models |
| 7 | `__init__.py` | 60-80 | builder, querier (facade) |
| 8 | `cli.py` | 150-200 | facade |
| **Total** | | **1740-2400** | |

Plus ~800-1000 lines of tests.

---

## What Phase 1b adds (vectors)

- `sqlite-vec==0.1.6` dependency
- `vec_chunks` table with explicit INTEGER PK (not rowid)
- Voyage Code 3 embedder (512d default, 1024d opt-in)
- Hybrid retrieval: BM25 + vector cosine → RRF merge
- `VectorStore` protocol for future hnswlib swap
- `sr backfill` command
- Vector coverage in `sr status`

## What Phase 1c adds (reranking + refs)

- Cohere rerank (or local Jina Reranker v2 via ONNX)
- Cross-reference extraction (imports, type_refs, decorators)
- `refs` table + `symbol_lookup` table
- Import-graph-aware symbol resolution
- Graph expansion (2-level, score-gated)
- RRF with query-class-aware weights (k=15)
