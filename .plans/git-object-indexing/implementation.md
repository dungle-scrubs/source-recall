# Git-Object-Based Indexing — Implementation Plan

## Architecture

### What's already implemented

- Schema v4: `branches` column on `chunks`, `branch` column on
  `file_hashes`, indexes, migration
- Schema v5: narrowed `chunks_au` FTS trigger
- Store: `insert_chunks(branch=)` with CSV append/dedup,
  `get_all_file_hashes(branch=)`, `upsert_file_hash(branch=)`
- Querier: `_filter_by_branch()`, `branch` param on `query()`,
  `active_branch` meta default
- Builder: `_get_current_branch()` with detached HEAD fallback,
  branch passed through build/refresh flows
- Tests: schema migration, branch insert/append/dedup, FTS
  trigger correctness, branch filtering, build-refresh-branch cycle

### What's missing (this plan)

The core optimization: reading file content from git objects
instead of the working tree, so that:

1. Files with identical blob SHAs across branches are not re-read
2. Chunks that already exist are not re-inserted or re-embedded
3. Only the `branches` CSV column is updated for shared chunks

### Constraints

- <!-- D-003 --> Individual `git cat-file blob` per file first;
  batch `--batch` mode deferred to profiling
- <!-- D-001 --> Branch filter stays as Python post-filter
- <!-- D-002 --> Branches stored as CSV (git forbids commas)
- tree-sitter pins are exact (0.25.2 / 0.13.0) — no changes
- FTS5 triggers are load-bearing — no changes needed
- Atomic swap for full builds — unchanged
- PID-file lock for concurrent builds — unchanged

### Key design

```
git ls-tree -r --format='%(objectmode) %(objecttype) %(objectname)	%(path)' HEAD
  → [(blob_sha, file_path), ...]

For each file:
  blob_sha matches stored file_hash.content_hash?
    YES → skip (content identical, just update branches CSV)
    NO  → git cat-file blob <sha> → chunk → insert/embed
           (or read from working tree if file is dirty)
```

The critical change: `content_hash` in `file_hashes` becomes the
**git blob SHA** instead of `sha256(content)`. This lets us compare
against `git ls-tree` output without reading file content at all.

For uncommitted (dirty) files: `git hash-object --stdin` computes
the blob SHA from working-tree content without writing to git's
object store.

---

## Phase 1: Git-object helpers (builder.py)

### M1.1: `_discover_files_git_objects()`
- **Dependencies:** none
- **Effort:** S (1-2d)
- **Test spec:** Parses `git ls-tree -r HEAD` output into
  `(file_path, blob_sha)` pairs. Falls back to None for non-git
  repos. Handles empty repos.
- **Tasks:**
  1. RED: Test `_discover_files_git_objects()` returns list of
     `(path, blob_sha)` tuples for a git repo with committed files
  2. GREEN: Implement using `subprocess.run(["git", "ls-tree", "-r",
     "--format=...", "HEAD"])`, parse output
  3. RED: Test returns None for non-git directory
  4. GREEN: Return None on non-zero exit code
  5. RED: Test excludes files matching `_is_excluded()` patterns
  6. GREEN: Filter output through existing exclusion logic

### M1.2: `_read_git_blob()`
- **Dependencies:** none
- **Effort:** S (1d)
- **Test spec:** Reads committed file content via `git cat-file blob
  <sha>`. Returns content string. Handles missing blob gracefully.
- **Tasks:**
  1. RED: Test `_read_git_blob(sha)` returns file content for a
     committed file
  2. GREEN: Implement via `subprocess.run(["git", "cat-file", "blob",
     sha], capture_output=True, text=True)`
  3. RED: Test returns None for nonexistent blob SHA
  4. GREEN: Return None on non-zero exit code

### M1.3: `_detect_dirty_files()`
- **Dependencies:** none
- **Effort:** S (1d)
- **Test spec:** Uses `git status --porcelain` to find uncommitted
  files. Computes synthetic blob SHA via `git hash-object --stdin`
  for dirty files.
- **Tasks:**
  1. RED: Test detects modified-but-uncommitted file
  2. GREEN: Implement via `git status --porcelain`, parse M/A/? flags
  3. RED: Test computes correct blob SHA for dirty file content
  4. GREEN: Pipe content to `git hash-object --stdin`, capture output
  5. RED: Test returns empty dict for clean working tree
  6. GREEN: Handle clean case

### M1.4: `_is_shallow_clone()`
- **Dependencies:** none
- **Effort:** S (0.5d)
- **Test spec:** Detects shallow clone via `git rev-parse
  --is-shallow-repository`. Returns bool.
- **Tasks:**
  1. RED: Test returns False for a normal repo
  2. GREEN: Implement via subprocess, parse `true`/`false` output
  3. RED: Test returns True for a shallow clone
  4. GREEN: Create shallow clone fixture in test

### Gate 1→2

- [ ] `_discover_files_git_objects()` parses ls-tree correctly
- [ ] `_read_git_blob()` reads committed content via cat-file
- [ ] `_detect_dirty_files()` finds uncommitted changes with synthetic blob SHAs
- [ ] `_is_shallow_clone()` detects shallow repos
- [ ] All existing tests still pass

---

## Phase 2: Modified build/refresh (builder.py)

### M2.1: `_index_file` with blob-SHA fast path
- **Dependencies:** M1.1, M1.2, M1.3
- **Effort:** M (2-3d)
- **Test spec:** When `blob_sha` matches stored `content_hash`,
  file is skipped (no read, no chunk, no embed). Only `branches`
  CSV is updated. When `blob_sha` differs or file is new, content
  is read from git blob (or working tree if dirty) and chunked.
- **Tasks:**
  1. RED: Test that building with identical blob_sha skips re-read
     (mock subprocess to verify `git cat-file` not called)
  2. GREEN: Add `blob_sha` parameter to `_index_file()`. Before
     reading content, check if `stored_hash == blob_sha` and chunk
     IDs already exist → skip, just update branches
  3. RED: Test that changed blob_sha triggers re-chunk
  4. GREEN: When hash differs, read via `_read_git_blob()`, chunk,
     insert, embed
  5. RED: Test that dirty file reads from working tree, not blob
  6. GREEN: If file is in dirty set, use `full.read_text()` instead
     of `_read_git_blob()`
  7. REFACTOR: Extract the "skip or re-index" decision into a
     clear helper

### M2.2: Modified `build()` flow
- **Dependencies:** M2.1
- **Effort:** S (1-2d)
- **Test spec:** Full build uses `git ls-tree` for file discovery
  when available. Falls back to existing `git ls-files` + filesystem
  for non-git and shallow clones. Stores blob SHA as `content_hash`.
- **Tasks:**
  1. RED: Test full build stores blob SHA (not sha256 of content)
     as `content_hash` in `file_hashes`
  2. GREEN: In `_build_locked()`, call `_discover_files_git_objects()`
     first. If available, use blob SHAs for content_hash. If None
     (non-git), fall back to existing path.
  3. RED: Test shallow clone falls back to filesystem reads
  4. GREEN: Check `_is_shallow_clone()`, if True use existing
     filesystem path with sha256 hashes
  5. RED: Test dirty files are included in build with working-tree
     content
  6. GREEN: Merge `_detect_dirty_files()` output with ls-tree output

### M2.3: Modified `refresh()` flow
- **Dependencies:** M2.1, M2.2
- **Effort:** M (2-3d)
- **Test spec:** On branch switch, refresh uses ls-tree diff against
  stored file_hashes to detect changes. Chunks that already exist
  (same chunk_id) are NOT re-embedded — only branches CSV updates.
  Genuinely new chunks are embedded.
- **Tasks:**
  1. RED: Test branch switch with identical files does NOT call
     embedder (zero new embeddings)
  2. GREEN: In refresh, when branch changes, enumerate files via
     `_discover_files_git_objects()`, diff blob SHAs against stored
     `content_hash`. Skip files with matching hash — only update
     branches CSV.
  3. RED: Test branch switch with some changed files re-embeds only
     the changed ones
  4. GREEN: For files with different blob SHAs, re-index normally.
     For existing chunk IDs, `insert_chunks(branch=)` already
     handles the CSV append without re-insert.
  5. RED: Test switching back to original branch re-embeds zero
     chunks (all chunk IDs already exist with vectors)
  6. GREEN: Before calling `_flush_vectors()`, filter out chunk IDs
     that already exist in `vec_chunks`. Only embed genuinely new
     chunk IDs.

### Gate 2→3

- [ ] Full build uses blob SHAs from `git ls-tree`
- [ ] Branch switch with identical files: 0 re-reads, 0 re-embeds
- [ ] Branch switch with N changed files: only N files re-chunked
- [ ] Switch back to original branch: 0 re-embeds
- [ ] Shallow clone degrades gracefully to filesystem reads
- [ ] Dirty files included with working-tree content
- [ ] All existing tests still pass

---

## Phase 3: Skip-existing-vectors optimization (store.py)

### M3.1: `get_existing_vector_ids()`
- **Dependencies:** M2.3
- **Effort:** S (1d)
- **Test spec:** Given a list of chunk IDs, returns the subset
  that already have vectors in `vec_chunks`. Used by builder to
  skip redundant embedding.
- **Tasks:**
  1. RED: Test returns empty set for chunk IDs with no vectors
  2. GREEN: Implement via apsw query `SELECT chunk_id FROM
     vec_chunks WHERE chunk_id IN (...)`
  3. RED: Test returns correct subset for mixed present/absent IDs
  4. GREEN: Handle batching for >500 IDs (SQLite variable limit)
  5. REFACTOR: Use in `_flush_vectors` to filter before embedding

### Gate 3→done

- [ ] `get_existing_vector_ids()` correctly identifies pre-existing vectors
- [ ] Builder skips embedding for chunks that already have vectors
- [ ] Full branch-switch cycle: build main → switch feature → refresh → switch main → refresh — embedding count matches changed-file count, not total-file count

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| `git cat-file` subprocess overhead for large repos | medium | <!-- D-003 --> Individual calls first; batch `--batch` deferred |
| Shallow clone missing blobs | medium | `_is_shallow_clone()` detection → filesystem fallback |
| Uncommitted file content diverges from blob SHA | medium | `_detect_dirty_files()` + synthetic blob SHA via `git hash-object` |
| Changing `content_hash` from sha256 to blob SHA breaks existing indexes | medium | First build after upgrade re-indexes everything (atomic swap); subsequent builds benefit |
| `git ls-tree` unavailable (non-git repo) | low | Falls back to existing `_discover_files()` + filesystem reads |

## Open Questions (all resolved)

All RFC open questions have been decided:

- **Q1 (filter strategy):** <!-- D-001 --> Python post-filter ✓
- **Q2 (branches format):** <!-- D-002 --> CSV ✓
- **Q3 (batch blob reads):** <!-- D-003 --> Deferred ✓
- **Q4 (branch deletion):** <!-- D-004 --> Ignore stale ✓

## References

- [RFC-01](.tallow/rfc/01_git-object-based-branch-aware-indexing.rfc.md) — full specification
- [AGENTS.md](AGENTS.md) — TDD mandate, tree-sitter pins, FTS5 constraints
