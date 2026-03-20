# RFC-01 Review: Git-Object-Based Branch-Aware Indexing

## Implementation Status

The RFC has 5 phases. Phases 1, 3, 4 are fully implemented.
Phases 2 and 5 are partially or not implemented.

| RFC Phase | Status | Evidence |
|-----------|--------|----------|
| 1: Schema migration v3→v4 | ✅ Done | `_MIGRATIONS` has v4 entry, v5 narrows FTS trigger |
| 2: Builder changes (git-object reads) | ❌ Not done | No `git ls-tree`, `git cat-file`, `_discover_files_git()`, `_read_content()`, `_is_shallow_clone()` |
| 3: Store changes | ✅ Done | `insert_chunks` handles branch CSV, `get_all_file_hashes(branch=)`, `upsert_file_hash(branch=)` |
| 4: Querier changes | ✅ Done | `_filter_by_branch()`, `branch` param on `query()`, `active_branch` meta |
| 5: FTS trigger update | ✅ Done | Migration v5 narrows `chunks_au` to content/file_path/symbol_name only |

### RFC Test Matrix

| # | RFC Test | Implemented? |
|---|---------|-------------|
| 1 | `test_migration_004_adds_columns` | ✅ `test_migration_004_adds_branch_columns` |
| 2 | `test_insert_or_update_branch_new_chunk` | ✅ `test_insert_chunks_with_branch` |
| 3 | `test_insert_or_update_branch_existing_chunk` | ✅ `test_insert_chunks_appends_branch` |
| 4 | `test_fts_trigger_fires_on_branches_update` | ✅ `test_fts_correct_after_branch_update` |
| 5 | `test_discover_files_git_ls_tree` | ❌ Not implemented |
| 6 | `test_read_content_committed_file` | ❌ Not implemented |
| 7 | `test_read_content_uncommitted_fallback` | ❌ Not implemented |
| 8 | `test_shallow_clone_detection` | ❌ Not implemented |
| 9 | `test_branch_switch_no_reembed` | ❌ Not implemented |
| 10 | `test_branch_switch_selective_reembed` | ❌ Not implemented |
| 11 | `test_query_filters_by_branch` | ✅ `test_filters_to_matching_branch` |
| 12 | `test_query_no_branch_returns_all` | ✅ `test_empty_branches_passes_through` |
| 13 | `test_build_refresh_branch_cycle` | ✅ (exists but doesn't verify embedding counts) |

## Structural Validation

- Abstract ✓ — clear, scoped
- RFC 2119 keywords used consistently ✓
- Error handling via Risk Assessment ✓
- Open Questions are specific with recommendations ✓
- Migration strategy includes rollback ✓
- Timeline is concrete (2 days) ✓

No structural issues.

## Internal Consistency

1. **Migration says v3→v4, schema is now at v5.** The migration
   shipped as v4, then v5 was added to narrow the FTS trigger.
   The RFC doesn't know about v5. The implementation is correct
   — v5 is additive, not a conflict. Just stale RFC text.

2. **Open Questions are all answered.** Q1 (Python post-filter) —
   implemented as B. Q2 (CSV vs JSON) — implemented as A. Q3
   (batch blob reads) — deferred, correct per RFC. Q4 (branch
   deletion) — implemented as C (ignore). These should be marked
   decided.

## Codebase Alignment

1. **Builder still reads from filesystem, not git objects.** The
   RFC's core optimization (Phase 2) — `git ls-tree` for file
   enumeration, `git cat-file blob` for content reads — is entirely
   missing. `_index_file()` still calls `full.read_text()`. This
   means branch switches still re-hash everything via `_hash_diff`.

2. **`_discover_files()` still uses `git ls-files`, not `git ls-tree`.**
   `git ls-files` returns the working tree's tracked files.
   `git ls-tree -r HEAD` returns the committed tree, which is what
   enables blob-SHA-based dedup. This is the critical difference.

3. **No `_is_shallow_clone()` detection.** The RFC specifies
   graceful degradation for shallow clones. Not implemented.

4. **No `git cat-file --batch` optimization.** Expected — RFC
   deferred this to post-profiling (Q3).

5. **No uncommitted file handling via `git status --porcelain` +
   `git hash-object`.** The hybrid approach for dirty working tree
   files is not implemented.

## Adversarial Review

### High

1. **The entire performance benefit is missing.** The RFC's
   motivation is eliminating redundant re-chunking and re-embedding
   on branch switches. Without git-object reads, switching branches
   still falls back to `_hash_diff()` which re-reads every file,
   and chunks with changed content are destroyed and recreated
   (with their embeddings). The `branches` CSV column exists but
   provides no dedup benefit because chunks are identified by
   content — and content is always re-read from disk, not from
   git objects where blob-SHA comparison would skip identical files.

   **Severity: high** — the schema infrastructure shipped but the
   feature that uses it didn't.

### Medium

2. **`_hash_diff` reads files that haven't changed.** For a branch
   switch where `merge-base --is-ancestor` fails, all files are
   re-read and hashed. With git-object reads, files with the same
   blob SHA on both branches would be skipped entirely (blob SHA
   is free from `git ls-tree` output).

3. **Embeddings are always recomputed for "new" chunks.** The
   builder's `_flush_vectors` doesn't check if a chunk_id already
   has a vector in `vec_chunks`. On branch switch, identical chunks
   get new IDs (same content → same ID), but the builder deletes
   the old chunks first (`delete_chunks_for_file`), then re-inserts
   and re-embeds. With the RFC's approach, chunks that already exist
   would skip insertion entirely — only the `branches` CSV updates.

### Low

4. **`test_build_refresh_branch_cycle` doesn't verify embedding
   efficiency.** The test exists but doesn't assert that embedding
   count is minimal after switching back. It tests correctness
   (chunks exist) but not the performance property the RFC promises.

## Summary

- **Structural issues**: 0
- **Inconsistencies**: 2 (stale version reference, unanswered OQs)
- **Alignment issues**: 5 (all from the unimplemented Phase 2)
- **Adversarial findings**: 4 (1 high, 2 medium, 1 low)

**Bottom line:** The plumbing shipped (schema, store API, querier
filter, tests for the plumbing). The feature didn't (git-object
reads). Completing the RFC means implementing Phase 2: `git ls-tree`,
`git cat-file blob`, shallow clone detection, uncommitted file
handling, and the skip-if-exists optimization in build/refresh.
