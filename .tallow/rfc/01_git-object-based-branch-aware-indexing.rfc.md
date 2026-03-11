---
number: 01
title: "Git-Object-Based Branch-Aware Indexing"
type: migration
status: Draft
author: kevin
date: 2026-03-11
---

# RFC-01: Git-Object-Based Branch-Aware Indexing

## Abstract

source-recall's index is branch-unaware: one `index.db` per repo
path, keyed to whatever HEAD was at build time. Switching git
branches forces a full hash comparison and re-embeds chunks that
were already computed on the previous branch. This RFC proposes
git-object-based chunking — reading file content from git blob
objects instead of the working tree — to enable branch-aware
indexing with automatic cross-branch deduplication. The change
adds one column to `chunks`, one column to `file_hashes`, two
indexes, and a ~126-LOC implementation.

## Introduction

### Problem

When a developer switches branches:

1. `refresh()` runs `git merge-base --is-ancestor <last_commit>
   HEAD`. If the old HEAD isn't an ancestor (the common case for
   branch switches), this fails.
2. The system falls back to `_hash_diff()`, which re-hashes every
   tracked file against stored content hashes.
3. All chunks from files that differ are deleted and re-created.
4. All embeddings for those chunks are re-computed, even if
   switching back to the previous branch would need the exact same
   embeddings again.

For a repo with 500 files where 200 differ between branches, each
switch costs ~2s of chunking + ~3s of embedding. Switching back
repeats the same work. Over a day of branch-hopping, this adds up
to minutes of wasted local compute.

### Scope

**In scope:**

- Schema migration (v3 → v4) to add branch tracking
- Modified `build()` and `refresh()` to read from git objects
- Working-tree fallback for uncommitted files
- Branch-filtered queries
- Shallow clone detection and graceful degradation

**Out of scope:**

- Multi-repo / monorepo deduplication
- Git worktree support (separate index per worktree is preserved)
- Remote embedding APIs (local CodeRankEmbed only today)
- Orphan branch handling (existing `IndexIdentityError` behavior
  is intentionally preserved per `critiques/storage.md:226`)

## Terminology

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD,
SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to
be interpreted as described in RFC 2119.

- **blob SHA**: Git's content-addressed hash for a file object.
  Computed as `sha1("blob " + filesize + "\0" + content)`. Two
  files with identical content produce the same blob SHA regardless
  of filename or branch.
- **working tree**: The checked-out files on disk, which may
  contain uncommitted changes not yet in git's object store.
- **chunk**: A function-level or block-level code fragment produced
  by tree-sitter or regex chunking (`chunker.py`).
- **chunk ID**: The existing 32-char hex digest from
  `ChunkData.chunk_id`, derived from
  `sha256(file_path + symbol_name + content_hash)`. Identical
  content at the same path with the same symbol name produces the
  same chunk ID across branches.
- **branch CSV**: A comma-separated list of branch names stored in
  `chunks.branches`, indicating which branches contain this chunk.

## Current State

### Schema (v3)

```sql
-- Key tables (from store.py _DDL)
CREATE TABLE chunks (
    id              TEXT PRIMARY KEY,
    file_path       TEXT NOT NULL,
    symbol_name     TEXT NOT NULL DEFAULT '',
    symbol_type     TEXT NOT NULL DEFAULT 'block',
    content         TEXT NOT NULL,
    start_line      INTEGER NOT NULL DEFAULT 0,
    end_line        INTEGER NOT NULL DEFAULT 0,
    parent_chunk_id TEXT,
    sub_chunk_index INTEGER,
    search_quality  TEXT NOT NULL DEFAULT 'ast'
);

CREATE TABLE file_hashes (
    file_path    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    parse_mode   TEXT NOT NULL DEFAULT 'ast',
    mtime_ns     INTEGER
);

-- FTS5 external-content (triggers: chunks_ai, chunks_ad, chunks_au)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content, file_path, symbol_name,
    content='chunks', content_rowid='rowid'
);

-- Vec (when sqlite-vec available)
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
);
```

### Build flow (`builder.py`)

1. `_discover_files()` → `git ls-files` or directory walk
2. `_index_file()` → `Path(rel_path).read_text()` → `chunk_file_with_refs()` → `store.insert_chunks()`
3. Batch embed → `store.insert_vectors()`
4. Write meta (`last_commit`, `repo_root_commit`, etc.)
5. Atomic swap temp file → `index.db`

### Refresh flow (`builder.py`)

1. `_detect_changes()` → `git diff --name-only <last_commit>` (or
   full `_hash_diff()` if ancestor check fails)
2. For each changed file: delete old chunks/vectors/refs, re-index
3. Update `last_commit` meta

### Change detection failure on branch switch

```
main HEAD:    abc123
feature HEAD: def456 (not a descendant of abc123)

_git_diff_files("abc123"):
  git merge-base --is-ancestor abc123 def456 → exit 1
  return None  →  fall back to _hash_diff()
```

`_hash_diff()` reads every file on disk, computes SHA-256, and
compares against stored hashes. Correct but O(all files), and all
differing chunks are destroyed and re-created.

## Target State

### Schema (v4)

Two columns added. No new tables.

```sql
ALTER TABLE file_hashes ADD COLUMN branch TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN branches TEXT NOT NULL DEFAULT '';

CREATE INDEX idx_file_hashes_branch ON file_hashes(branch, file_path);
CREATE INDEX idx_chunks_branches ON chunks(branches);
```

After migration, existing rows get `branch = ''` and
`branches = ''`, representing the legacy single-branch state.
The first `build()` or `refresh()` on v4 MUST populate these
with the current branch name.

### Build flow (modified)

1. `_discover_files_git()` → `git ls-tree -r HEAD` returns
   `(file_path, blob_sha)` pairs
2. `_read_content(rel_path, blob_sha)` → `git cat-file blob
   <sha>` for committed files; working-tree read for uncommitted
3. `chunk_file_with_refs()` → same chunking logic (unchanged)
4. `store.insert_or_update_branch(chunks, branch)` →
   - If chunk ID already exists: append branch to `branches` CSV
   - If chunk ID is new: insert with `branches = <branch>`
5. Batch embed only chunks whose ID is not already in `vec_chunks`
6. Record `active_branch` and `last_commit` in meta

### Refresh flow (modified)

1. Detect current branch via `git branch --show-current`
2. If branch unchanged since last refresh: use existing
   `git diff --name-only` fast path (no change)
3. If branch changed: use `git ls-tree -r HEAD` to enumerate
   files on new branch, diff against stored `file_hashes` for
   that branch
4. For each changed file:
   - Read content from git blob (or working tree if uncommitted)
   - Chunk → produce chunk IDs
   - Existing chunk IDs: update `branches` CSV
   - New chunk IDs: insert + embed
5. Update meta: `last_commit`, `active_branch`

### Deduplication properties

Same content at the same path with the same symbol name →
same `chunk_id` (per existing `ChunkData.chunk_id` formula).
This holds across branches because the formula is deterministic
on `(file_path, symbol_name, content_hash)`.

When branch `feature` has an identical copy of `src/utils.py`
as `main`:

- `git ls-tree` on `feature` → same blob SHA → same content
- Same content → same chunks → same chunk IDs
- `INSERT OR IGNORE` skips re-inserting content
- Embedding already exists in `vec_chunks` → no re-embedding
- Only the `branches` CSV is updated: `'main'` → `'main,feature'`

### Query flow (modified)

```python
def query(self, question, *, branch=None, top_k=None):
    if branch is None:
        branch = store.get_meta("active_branch") or ""

    fts_results = store.fts_search(question, limit=30)
    vec_results = store.search_vectors(query_vec, top_k=30)

    # Post-filter by branch (when branch is set)
    if branch:
        fts_results = _filter_by_branch(fts_results, branch)
        vec_results = _filter_by_branch(vec_results, branch)

    # ... RRF merge, symbol search, graph expansion (unchanged)
```

Branch filtering is a post-filter on already-retrieved results,
not a WHERE clause, because:

1. FTS5 external-content queries cannot be joined with arbitrary
   predicates efficiently
2. `vec_chunks` uses `vec0` which has its own query interface
3. The filter is cheap — we're filtering ≤30 results

When `branch` is empty string or None, no filtering is applied
(search across all branches — backward-compatible default).

## Migration Strategy

### Phase 1: Schema migration (savepoint-wrapped)

Add to `_MIGRATIONS` in `store.py`:

```python
(
    4,
    "add branch-aware columns for git-object chunking",
    """
    ALTER TABLE file_hashes
        ADD COLUMN branch TEXT NOT NULL DEFAULT '';
    ALTER TABLE chunks
        ADD COLUMN branches TEXT NOT NULL DEFAULT '';
    CREATE INDEX IF NOT EXISTS idx_file_hashes_branch
        ON file_hashes(branch, file_path);
    CREATE INDEX IF NOT EXISTS idx_chunks_branches
        ON chunks(branches);
    """,
)
```

Bump `_SCHEMA_VERSION` to 4.

Existing rows retain empty-string defaults. The next `build()` or
`refresh()` populates them with the detected branch name. This is
safe because empty-string branches pass through the branch filter
(they match nothing, but the first refresh overwrites them).

### Phase 2: Builder changes

1. Add `_discover_files_git()` using `git ls-tree -r HEAD`
2. Add `_read_content(rel_path, blob_sha)` with working-tree
   fallback
3. Modify `_index_file()` to accept `blob_sha` parameter
4. Add `_get_current_branch()` helper
5. Add `_is_shallow_clone()` detection
6. Modify `build()` and `refresh()` to use branch-aware paths

### Phase 3: Store changes

1. Add `insert_or_update_branch()` method
2. Modify `insert_chunks()` to handle `branches` column
3. Add `get_file_hashes_for_branch()` method

### Phase 4: Querier changes

1. Add `branch` parameter to `query()`
2. Add `_filter_by_branch()` helper
3. Read `active_branch` from meta as default

### Phase 5: FTS5 trigger update

The existing `chunks_au` trigger already handles UPDATE correctly:

```sql
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, ...)
    VALUES ('delete', old.rowid, old.content, ...);
    INSERT INTO chunks_fts(rowid, ...)
    VALUES (new.rowid, new.content, ...);
END;
```

When we UPDATE the `branches` column on an existing chunk, this
trigger fires and re-indexes the FTS entry. The content, file_path,
and symbol_name are unchanged, so the FTS index stays correct.
No trigger changes are needed.

## Backward Compatibility

- **Schema v3 indexes**: The v4 migration is additive (ALTER TABLE
  ADD COLUMN). Existing indexes continue to work. The empty-string
  defaults are functionally equivalent to "no branch info" and
  queries without a branch filter return all results.

- **CLI users**: `sr build` and `sr query` work identically. The
  `--branch` flag is new and OPTIONAL.

- **MCP server**: The `query` tool gains an optional `branch`
  parameter. Omitting it searches all branches (current behavior).

- **FTS-only mode** (`--no-embed`): Branch tracking works the same
  way — it's independent of vector embeddings.

- **Non-git repos**: `_discover_files_git()` returns None → falls
  back to existing `_discover_files()` with directory walk. Branch
  tracking is silently disabled (branches column stays empty).

## Rollback Plan

### If migration fails mid-flight

Savepoint-wrapped: automatic rollback to schema v3. The index
remains fully functional at v3.

### If v4 behavior is buggy after deployment

1. The `branches` column is additive — it does not replace any
   existing column.
2. Queries without `branch` filter ignore the column entirely.
3. A full `sr build` at v4 overwrites all data (atomic swap), so
   the rollback is: downgrade code, `sr build` to rebuild at v3.

### If we need to revert the migration

Schema v4 → v3 is not automatically reversible (SQLite doesn't
support DROP COLUMN before 3.35.0, and ALTER TABLE DROP COLUMN
was only added in 3.35.0). If reverting, the simplest path is:

1. Delete `index.db`
2. Downgrade to v3 code
3. `sr build` (full rebuild)

This is acceptable because source-recall indexes are derived
data — the source of truth is the git repository.

## Risk Assessment

### R1: Shallow clones missing blob objects — MEDIUM/LOW

`git cat-file blob <sha>` fails if the blob isn't in the local
pack. Shallow clones may lack blobs for older commits.

**Mitigation**: Detect via `git rev-parse --is-shallow-repository`.
If true, fall back to working-tree reads and log a warning. Branch
deduplication is degraded (hashes computed from content, not blob
SHAs) but indexing still works.

### R2: Uncommitted working-tree changes — MEDIUM/MEDIUM

Files edited but not committed are invisible to `git ls-tree HEAD`.

**Mitigation**: Hybrid approach. After `git ls-tree`, also check
for dirty files via `git status --porcelain`. For dirty files, read
content from the working tree and compute a synthetic blob SHA via
`git hash-object --stdin` (computes what the blob SHA would be
without writing to `.git/objects/`). This gives dedup parity with
committed blobs.

### R3: Comma-separated `branches` has false-positive matching — LOW/MEDIUM

Naive `LIKE '%feature%'` matches `feature`, `feature-old`, and
`my-feature`. 

**Mitigation**: Use delimited matching:
```python
def _branch_in_csv(branches_csv: str, target: str) -> bool:
    return target in branches_csv.split(',')
```

Since filtering happens in Python on ≤30 results (post-filter, not
SQL WHERE), this is both correct and fast.

### R4: `git cat-file` subprocess overhead — LOW/LOW

One `git cat-file blob <sha>` per file per build. For 500 files,
that's 500 subprocess spawns.

**Mitigation**: Use `git cat-file --batch` with stdin piping to
batch all blob reads into a single subprocess. This reduces 500
spawns to 1. The batch protocol sends `<sha>\n` on stdin and reads
`<sha> blob <size>\n<content>\n` from stdout.

### R5: Branch accumulation in `branches` CSV — LOW/LOW

Long-lived repos with many branches accumulate comma-separated
branch names on shared chunks. A chunk shared across 50 branches
has a ~500-byte `branches` value.

**Mitigation**: Acceptable. At 10,000 chunks × 500 bytes = 5MB
overhead in the worst case. `sr clean-branches` could prune
deleted branches in the future (out of scope for this RFC).

### R6: Detached HEAD state — LOW/LOW

`git branch --show-current` returns empty string in detached HEAD.

**Mitigation**: Fall back to `detached-<sha[:8]>` as the branch
name. This is a synthetic identifier that prevents detached HEAD
state from corrupting the branch index.

## Validation & Testing

All tests follow TDD (RED → GREEN → REFACTOR) per AGENTS.md.

### Unit tests (store.py)

1. `test_migration_004_adds_columns` — schema migration creates
   `branches` column on `chunks` and `branch` column on
   `file_hashes`
2. `test_insert_or_update_branch_new_chunk` — new chunk gets
   `branches = '<branch>'`
3. `test_insert_or_update_branch_existing_chunk` — existing chunk
   appends branch to CSV
4. `test_fts_trigger_fires_on_branches_update` — FTS index stays
   correct when `branches` column is updated

### Unit tests (builder.py)

5. `test_discover_files_git_ls_tree` — parses `git ls-tree` output
   correctly
6. `test_read_content_committed_file` — reads via `git cat-file`
7. `test_read_content_uncommitted_fallback` — reads from working
   tree when file not in HEAD
8. `test_shallow_clone_detection` — detects shallow clone and falls
   back
9. `test_branch_switch_no_reembed` — switching branches with
   unchanged files does NOT call embedder
10. `test_branch_switch_selective_reembed` — switching branches
    re-embeds only genuinely new chunks

### Unit tests (querier.py)

11. `test_query_filters_by_branch` — results scoped to requested
    branch
12. `test_query_no_branch_returns_all` — omitting branch returns
    results from all branches

### Integration tests

13. `test_build_refresh_branch_cycle` — build on main, switch to
    feature, refresh, switch back, refresh. Verify embedding count
    is minimal.

## Timeline

### Day 1: Schema + store + FTS validation

- Migration DDL (Phase 1)
- `insert_or_update_branch()` (Phase 3)
- Tests 1–4
- Verify FTS triggers fire correctly on UPDATE

### Day 2: Builder + querier + integration

- `_discover_files_git()`, `_read_content()`, `_get_current_branch()`,
  `_is_shallow_clone()` (Phase 2)
- Modified `build()` and `refresh()` (Phase 2)
- `_filter_by_branch()` and `branch` parameter (Phase 4)
- Tests 5–13

### Go/no-go criteria

- All 13 tests pass
- `sr build` on source-recall's own repo produces identical query
  results to v3 (regression check)
- Branch switch + refresh re-embeds <5% of chunks for a repo with
  <5% file delta between branches

## Open Questions

### Q1: Should branch filtering happen in SQL or Python?

**Options:**

A. SQL WHERE clause on `branches` column (requires careful LIKE or
   json_each pattern, interacts with FTS5 ranking)
B. Python post-filter on retrieved results (simpler, ≤30 items)
C. Hybrid: SQL pre-filter for `file_hashes`, Python post-filter
   for query results

**Current recommendation:** B (Python post-filter). The result sets
are small (≤30 from FTS, ≤30 from vec), and post-filtering avoids
FTS5/vec0 query complexity. Revisit if result set sizes grow.

### Q2: Should `branches` be CSV or JSON array?

**Options:**

A. Comma-separated text (`'main,feature-auth,develop'`)
   - Simpler, smaller, no JSON parsing overhead
   - Requires careful split-based matching

B. JSON array (`'["main","feature-auth","develop"]'`)
   - SQLite has `json_each()` for proper querying
   - Larger on disk, parsing overhead

**Current recommendation:** A (CSV). Branch names cannot contain
commas (git forbids them). Split-based matching in Python is
sufficient for ≤30 result post-filtering.

### Q3: Should we batch blob reads with `git cat-file --batch`?

For the initial implementation, individual `git cat-file blob <sha>`
calls are simpler. Batching via `--batch` is an optimization for
repos with 1000+ files. Defer unless profiling shows subprocess
overhead is significant.

### Q4: How should branch deletion be handled?

When a git branch is deleted locally, its chunks remain in the
index with stale `branches` CSV entries. Options:

A. Lazy cleanup: on next `refresh()`, detect deleted branches and
   strip them from CSV. Remove chunks with empty `branches`.
B. Explicit `sr prune-branches` command
C. Ignore — stale branch names in CSV are harmless for queries

**Current recommendation:** C for initial implementation. Stale
branch names only cause a chunk to appear in queries when that
branch is explicitly requested (which won't happen for a deleted
branch). Revisit when storage pressure warrants cleanup.

## References

### Normative

- [AGENTS.md](../../AGENTS.md) — Project constraints: TDD mandate,
  tree-sitter pins, FTS5 trigger requirements, atomic swap builds,
  schema migration patterns
- [store.py](../../src/source_recall/store.py) — Schema DDL,
  migrations, FTS5 triggers
- [builder.py](../../src/source_recall/builder.py) — Build/refresh
  flow, git plumbing, change detection
- [querier.py](../../src/source_recall/querier.py) — Query flow,
  RRF merge, result construction

### Informative

- [critiques/storage.md](../../critiques/storage.md) — Adversarial
  review of storage design; discusses worktree behavior (line 187),
  WAL race conditions (line 203), orphan branches (line 226)
- [critiques/architecture.md](../../critiques/architecture.md) —
  Complexity cost analysis
- War counsel debate artifacts:
  - `/tmp/war-counsel-dedup.md` — Content-addressed dedup argument
  - `/tmp/war-counsel-gitobj.md` — Git-object chunking argument
  - `/tmp/war-counsel-verdict.md` — Judge's verdict and scoring
