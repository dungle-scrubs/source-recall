# AGENTS.md

This file provides guidance to AI coding agents working
with code in this repository.

## TDD is mandatory

Every change follows RED → GREEN → REFACTOR. One failing
test, then the minimum implementation, then cleanup. Never
write more than one test before implementing. Never skip
running the full suite between phases. If a new test passes
immediately, stop and understand why before moving on.

Skip TDD only for exploratory spikes or trivial glue code.

## tree-sitter pins are exact — do not upgrade

`tree-sitter==0.25.2` and `tree-sitter-language-pack==0.13.0`
are pinned exactly. The language-pack C ABI breaks across
minor versions. Upgrading either requires verifying every
chunker code path and the `get_parser()` call site.

## FTS5 is external-content — triggers are load-bearing

`chunks_fts` uses `content='chunks'` (external-content mode).
The `chunks_ai`, `chunks_ad`, `chunks_au` triggers in
`store.py` keep it in sync. If you add columns to `chunks`
or change the schema, you must update the triggers and DDL
together or FTS silently returns stale data.

## Schema migrations use savepoints

Each migration in `_MIGRATIONS` runs inside a savepoint.
Bump `_SCHEMA_VERSION`, add the migration tuple, and test
with `SchemaVersionError` for forward-version rejection.
Never use bare `ALTER TABLE` outside the migration list.

## Full builds use atomic swap

`builder.py` writes to a `.tmp.<pid>` file, then calls
`IndexStore.atomic_swap()` which does WAL checkpoint →
fsync → sidecar cleanup → `os.rename`. Do not write
directly to `index.db` during builds.

## Identity verification checks path first

`_verify_identity` compares stored `repo_path` before
root commit hashes. This catches forks/clones sharing
a root commit and 48-bit index-dir hash collisions.
The `repo_path` meta key written by `build()` is the
source of truth.

## Tests monkeypatch `get_index_dir`

The `clean_index_dir` autouse fixture in `conftest.py`
redirects all index storage to `tmp_path`. Tests never
touch `~/.local/share/source-recall/`. If you add a new
code path that calls `get_index_dir` or `get_db_path`,
it will automatically use the monkeypatched version.

## Chunk IDs are length-prefixed

`ChunkData.chunk_id` uses
`sha256(f"{len(path)}:{path}|{len(sym)}:{sym}|{hash}")`.
The length prefixes prevent separator collision between
paths and symbol names containing `|` or `:`. Don't
change the format without migrating existing indexes.

## Adding a new language to the chunker

1. Add extensions to a `_FOO_EXTENSIONS` frozenset
2. Add a `_chunk_foo()` function following the
   `_chunk_typescript` / `_chunk_python` pattern
3. Wire it into `chunk_file()` dispatch
4. tree-sitter languages use `get_parser("language_name")`
   from `tree_sitter_language_pack`
5. If error-node density exceeds 10%, the file falls
   back to text chunking automatically
6. Compound extensions (`.blade.php`) cannot go in the
   extension table: `Path.suffix` only yields `.php`.
   They are matched by `_is_blade`-style basename checks
   that run *before* the extension dispatch, or the more
   specific type silently loses to the shorter suffix.

## `docs/history/critiques/` is design rationale, not code

The four markdown files in `docs/history/critiques/` document
adversarial review of early design decisions. They drove
the Phase 1 architecture. Read them before proposing
structural changes.
