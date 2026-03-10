## Adversarial Critique: Storage & Data Model Decisions

---

### 1. SQLite Single-File-Per-Repo

**The decision**: One `index.db` per repo, stored at `~/.local/share/source-recall/<repo-hash>/`. No shared database across repos.

**What's wrong or risky**:

The single-file-per-repo model is actually defensible for isolation — a corrupted index for repo A doesn't kill repo B. But several second-order problems aren't addressed:

- **Directory proliferation without lifecycle management**: `sr clean` only catches repos where the *path no longer exists*. But it explicitly admits: "if repo A at /path/x is deleted and repo B is created at /path/x, clean won't detect the stale index." Over time, a developer with 40 repos who reorganizes their `~/dev/` directory will accumulate ghost indexes with no automatic reclamation. The `sr clean --all` nuclear option is the only escape hatch. There's no TTL-based expiry, no LRU eviction of stale indexes, no `sr clean --older-than 90d`.

- **No cross-repo deduplication**: If you have a monorepo checked out at two paths (worktree or clone), you get two full 400MB+ indexes with nearly identical vector data. For embedding-heavy repos this is significant.

- **WAL files in the index directory are invisible to `clean`**: If `build()` crashes between WAL checkpoint and rename, you get a stale `.tmp.{pid}` file (the plan handles this) — but you also could have `-wal` and `-shm` sidecar files from a previously-completed index that a reader has open. The `clean` command presumably does an `os.unlink()` on `index.db`. If a reader has it open via WAL and you unlink the main file, the reader gets a broken reference. SQLite on Linux will keep the file alive (unlink semantics), but the WAL sidecars are now orphaned and a subsequent `open()` of a new `index.db` at the same path will look for `index.db-wal` — which still exists — and try to replay it against the new file, potentially corrupting the new index silently.

- **The "shared database" alternative is actually better for `sr status --all`** and cross-repo graph analysis. If you ever want "what repo exports a symbol that this repo imports," you need cross-repo data and the single-file model makes that impossible without opening N databases.

**Concrete alternative**: Keep single-file-per-repo but add a lightweight **registry database** at `~/.local/share/source-recall/registry.db` with a `repos` table: `(path_hash, repo_path, repo_name, last_indexed_at, db_size_bytes, schema_version)`. The `clean` command queries the registry, not the filesystem. `sr status --all` is a single query. Lifecycle: entries older than N days with non-existent paths get auto-purged. Tradeoff: another file to keep in sync, but it's tiny and append-mostly. The registry becomes the source of truth for which indexes exist rather than relying on directory enumeration.

**Alternative 2 (rejected for good reason)**: DuckDB. DuckDB has native HNSW vector indexes (`CREATE INDEX ... USING HNSW`), excellent analytical query performance, and Parquet export. But it's OLAP-oriented and does not support concurrent writers *or* concurrent readers with writers reliably — it's single-writer, single-reader by design for embedded use. For this tool's concurrency model (MCP server + CLI potentially running simultaneously), DuckDB would be strictly worse than SQLite+WAL. Also, DuckDB doesn't have FTS5 — you'd need to implement BM25 manually or vendor it. Correct to reject.

**Alternative 3**: Postgres for "larger deployments" — this is a category error. The tool's value proposition is *zero-infrastructure*, single-binary-style operation. The moment you need a Postgres server, you've broken the "just works" contract. If someone has a 10M-line monorepo and needs Postgres, they have different problems. Not worth planning for.

---

### 2. sqlite-vec (~0.1.6, Pre-1.0)

**The decision**: Use sqlite-vec as a loadable SQLite extension for vector storage and ANN search. Pin to `>=0.1.6,<0.2`. The vec0 virtual table stores `float[1024]` embeddings, joined to the chunks table via rowid.

**What's wrong or risky**:

- **Pre-1.0 means the API surface is contractually unstable.** The plan acknowledges this and says "pin minor version, test on upgrade." But the dependency pin is `>=0.1.6,<0.2` — that's a *range*, not a pin. If sqlite-vec releases 0.1.7 and breaks something in vec0's query syntax or rowid semantics, uv will happily pull it in on the next `uv sync`. "Pin minor version" in the prose contradicts the `<0.2` ceiling in the TOML. This should be `==0.1.6` until there's a tested upgrade path.

- **The join-via-rowid design is fragile**: The plan correctly identifies that vec0 only supports integer PKs, so they use `chunks.rowid` as the shared key. But SQLite rowids are *not* stable across vacuum operations (`VACUUM` reassigns rowids). More importantly: the plan's migration strategy explicitly states some migrations "drop and rebuild" tables. If `chunks` is dropped and rebuilt, all rowids change. The `vec_chunks` table would then be left with dangling rowid references pointing at wrong chunks. This is a silent correctness bug — queries would return wrong chunks, not errors. The plan mentions "JOIN via chunks.rowid = vec_chunks.rowid" but nowhere mentions this VACUUM/rebuild hazard.

- **ANN quality is unknown**: vec0 uses brute-force exact search (no HNSW or IVF index) for the sizes documented. At 100k chunks × 1024 dimensions, a full cosine scan over 400MB of floats is slow — potentially 200-500ms per query depending on memory. The plan claims "top 30" vector results, but doesn't acknowledge the scan time. hnswlib at 100k vectors returns top-30 in <5ms with 95%+ recall. This could be the dominant query latency.

- **sqlite-vec vs sqlite-vss**: sqlite-vss (the predecessor, now deprecated) used Faiss under the hood and had HNSW. sqlite-vec deliberately *removed* HNSW in favor of simplicity. The plan picked the *simpler but slower* tool, which may be fine at 10k chunks but starts hurting at 100k+.

- **The "contingency: hnswlib sidecar" is described as a fallback but is arguably the better primary**: hnswlib gives you HNSW with configurable ef/M parameters, is pure-C++ with Python bindings, serializes to a single binary file, and is 20-50× faster at query time for large indexes. The sidecar management (keeping hnswlib file in sync with SQLite) is non-trivial but the plan's `build()` already does an atomic swap — the hnswlib file could be built alongside the temp db and renamed atomically at the same time. The plan dismisses this as a contingency but the performance argument for making it primary is real.

**Concrete alternative**: Keep sqlite-vec for Phase 1 (simplicity wins when the codebase is small), but build the vector abstraction layer cleanly enough that swapping to hnswlib requires changing only the `store.py` vector methods. Specifically:
  1. Tighten pin to `==0.1.6` immediately.
  2. Never use `chunks.rowid` as the join key for vec_chunks — instead, assign an explicit `INTEGER PRIMARY KEY` to chunks (rename the existing TEXT `id` to `chunk_sha` and add `rowid_pk INTEGER PRIMARY KEY` auto-assigned). This decouples vec_chunks from content-addressed IDs and makes rowid stability explicit.
  3. Add a `VectorStore` protocol with `insert(rowid, embedding)`, `search(query_vec, top_k) -> list[rowid]`, `delete(rowid)` — concrete impls: `SqliteVecStore` and `HnswlibStore`. Migration is then a config flag, not a rewrite.

---

### 3. FTS5 Standalone (Not External-Content)

**The decision**: Use a standalone FTS5 table (`chunks_fts`) that stores its own copy of `content`, `file_path`, and `symbol_name`. This duplicates data. The rationale: external-content tables can silently corrupt on mismatched deletes.

**What's wrong or risky**:

- **The "~50MB overhead is negligible" claim is hand-wavy**: The plan compares it to "~400MB" for the vector table. But:
  - The vector table at 100k chunks × 1024 floats × 4 bytes = 409MB. That's accurate.
  - The chunks table itself stores full content. At 100k chunks × avg 2KB content = ~200MB.
  - FTS5 standalone adds another ~200MB (content + posting lists + token indexes). FTS5 posting lists for code are large because code has many unique tokens (mangled names, hex literals, etc.). The "~50MB" estimate is wildly optimistic — FTS5 on 200MB of code text will consume 150-300MB easily.
  - Total: the database could easily reach 800MB-1GB for a 100k-chunk repo. Not negligible on a developer laptop where storage is constrained. The plan says this is "negligible compared to the vector table" but that's the wrong comparison — you should compare it against the benefit of a simpler content table design.

- **The external-content sync concern is real but solvable**: The plan says external-content's DELETE requires re-providing the old content, creating a sync risk. But this is only a problem if you're issuing raw DELETE statements outside a transaction. If the store layer wraps DELETE+FTS-delete in a single transaction (which the plan already mandates for the standalone case), you can do the same for external-content with triggers. SQLite's FTS5 external-content table supports `content_rowid` and automatic trigger creation via `CREATE VIRTUAL TABLE ... USING fts5(content='chunks', content_rowid='rowid')`. The triggers handle consistency automatically without any application code managing it.

- **The duplicate copy creates an integrity gap during migrations**: When a migration "drops and rebuilds" the chunks table, the FTS5 standalone table must also be dropped and rebuilt. The migration list doesn't mention this. If a migration drops `chunks` and recreates it (to add a column, change a constraint), the developer must remember to also drop+rebuild `chunks_fts`. With external-content, dropping chunks also invalidates FTS (same problem) but at least the relationship is explicit via the table definition, not implicit via INSERT/DELETE ordering.

- **The "atomically wrap both tables in BEGIN...COMMIT" requirement is a latent bug magnet**: Every future developer touching the store layer must know they can never touch `chunks` without touching `chunks_fts` in the same transaction. This isn't enforced by the schema — it's enforced by convention. In a 300-400 line `store.py`, this is manageable. In a maintained codebase with multiple contributors, it's a footgun.

**Concrete alternative**: Use FTS5 external-content with explicit triggers. The schema becomes:
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content, file_path, symbol_name,
    content='chunks',
    content_rowid='rowid'
);
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
```
The triggers are in the schema DDL, enforced by SQLite itself. Storage savings: 150-300MB for large repos. The feared "silent corruption on mismatched DELETE" only occurs if you use the FTS5 delete shadow table API directly (`INSERT INTO chunks_fts(chunks_fts, ...) VALUES ('delete', ...)`), which the trigger handles correctly. Tradeoff: triggers are invisible to application developers; debugging FTS bugs requires knowing the trigger exists.

---

### 4. WAL Mode + Advisory File Locking

**The decision**: All connections use `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`. Write exclusivity is enforced by `fcntl.flock()` on `index.db.lock`. Readers never acquire the lock.

**What's wrong or risky**:

- **`fcntl.flock()` is not cross-platform**: flock is a POSIX call unavailable on Windows and unreliable on NFS-mounted filesystems. The plan targets macOS/Linux developer machines, so Windows isn't urgent — but if this tool ever needs to run on WSL2, flock behaves differently (it works within WSL2 but doesn't cross the WSL boundary). NFS home directories (common in corporate environments) will silently not lock. `fcntl.lockf()` has different semantics (process-scoped, not fd-scoped). The plan picks flock without justifying the choice over lockf or a lockfile-with-PID approach.

- **The advisory lock and WAL mode are partially redundant, partially complementary, and the interaction is underdocumented**: WAL mode allows one writer + N readers concurrently *within SQLite's own locking*. But two Python processes both trying to `INSERT INTO chunks` will *not* deadlock — SQLite's write lock (not flock) serializes them, with one getting `SQLITE_BUSY` after `busy_timeout`. The flock advisory lock is *on top of* this to prevent *application-level* invariants (like the two-table INSERT to chunks + chunks_fts + vec_chunks) from interleaving. But the plan doesn't state this clearly, and future developers might assume flock alone is sufficient and bypass it. The plan says "Reads never need the lock — WAL allows concurrent readers" — this is correct for WAL's SQLite-level write lock, but if a reader opens the DB while a migration is in progress (which does DDL), the reader could see a schema mismatch mid-migration that busy_timeout can't protect against (DDL changes schema_version after creating tables — a reader that opens between the table creation and the schema_version update sees the wrong version).

- **The lock-skip-on-contention behavior during auto-refresh is a consistency hole**: "If the lock is already held (another process is refreshing), it skips the refresh and queries the existing index." This means a long-running refresh (30 files, say 10 seconds) causes all concurrent MCP queries to run against a stale index for 10 seconds. The plan presents this as a feature ("avoiding redundant work") but doesn't acknowledge the resulting consistency window. For a code Q&A tool this is acceptable, but it should be documented as a known limitation, not presented as obviously correct behavior.

- **The stale_check cache race condition**: The plan says "The cache stores both a timestamp and the HEAD commit hash at check time — if HEAD has moved since the cached check, the cache is invalid regardless of age." But this only covers HEAD movement. What about:
  - Working tree changes (uncommitted edits) that don't move HEAD?
  - The MCP server and CLI processes having different views of the filesystem (one has the file open, the other has just written it)?
  The 10-second cache window means working tree edits won't trigger a refresh within that window even with `auto_refresh=True`. For a developer actively editing code and querying in parallel, this is a stale-result period of up to 10 seconds per file.

**Concrete alternative**: Replace the two-tier locking (flock + SQLite WAL) with a cleaner model:
  1. Keep WAL mode for read concurrency — it's correct.
  2. Replace flock with a **PID-file lock**: write `{"pid": os.getpid(), "started": timestamp}` to `index.db.lock`. On acquiring, check if the PID is alive (`os.kill(pid, 0)`). If the process is dead, steal the lock. This survives process crashes without a stale lock and is portable via Python's `os` module. Tradeoff: two syscalls instead of one flock call.
  3. For the stale check cache, add working tree stat mtime to the cache key alongside HEAD — `max(mtime of all indexed file_paths in file_hashes)` is expensive, but a cheaper proxy is `stat(repo_root/.git/index)` which changes on every `git add` or working tree modification.

---

### 5. Schema Design

**The decision**: chunk ID is `sha256(file_path + ":" + symbol_name + ":" + content_hash)[:32]` (128 bits as hex = 32 chars). The refs table uses composite PK `(source_chunk, target_symbol, ref_type)`. symbol_lookup maps symbol keys to chunk IDs. file_hashes tracks per-file content hashes.

**What's wrong or risky**:

**Chunk ID scheme**:
- `sha256(file_path + ":" + symbol_name + ":" + content_hash)[:32]` — the input is three fields concatenated with colons. This means `file_path="a:b", symbol_name="c"` and `file_path="a", symbol_name="b:c"` produce the same hash input. This is a classic separator-collision bug. The fix is to length-prefix each component: `sha256(f"{len(file_path)}:{file_path}:{len(symbol_name)}:{symbol_name}:{content_hash}")` or use a canonical encoding like JSON array serialization. At 100k chunks this collision is extremely unlikely but the *correctness guarantee* the plan asserts ("collisions indicate a bug in input, not hash collision") is undermined — you could have two legitimately different chunks with the same ID if file paths contain colons (common: Windows paths, URLs in config, Docker image references in shell scripts).

- The ID is stored as `TEXT PRIMARY KEY`. That means it's used as the FK in refs (`source_chunk TEXT`, `resolved_chunk TEXT`) and symbol_lookup (`chunk_id TEXT`). SQLite stores TEXT PKs inline in every index that references them — 32-char hex PKs are 3× larger than a 4-byte integer rowid. For 100k chunks with O(chunks) refs and O(chunks × forms) symbol_lookup entries, this is meaningfully larger than a rowid-based design. Not a killer, but a choice that compounds across indexes.

**refs table**:
- `PRIMARY KEY (source_chunk, target_symbol, ref_type)` — this PK is stored as a B-tree. `target_symbol` is a freeform string (can be up to `len(module_path) + "." + len(symbol_name)` — could be 100+ characters for deeply nested Python packages). The composite PK index size for 500k refs would be substantial. More importantly: the FK `FOREIGN KEY (source_chunk) REFERENCES chunks(id) ON DELETE CASCADE` is declared but **SQLite foreign keys are disabled by default**. `PRAGMA foreign_keys = ON` must be executed on every connection. The plan mentions `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout` in the connection setup but never mentions `PRAGMA foreign_keys = ON`. If this pragma is missing, the CASCADE deletes won't happen, and deleting a chunk will leave orphaned refs — exactly the bug they designed the cascade to prevent.

- `idx_refs_resolved ON refs(resolved_chunk)` — this index will be heavily fragmented during refresh (many NULL → non-NULL transitions as resolution populates the column). A partial index `WHERE resolved_chunk IS NOT NULL` would be smaller and faster for the "find all refs to this chunk" use case. The current index includes the NULL rows, which are noise for lookups.

**symbol_lookup**:
- "A symbol may appear in multiple forms: 'UserService.validate', 'validate', 'user_service.UserService'" — the indexer inserts all forms for each chunk. This is a design smell: the disambiguation logic ("same directory preferred") is in application code, not in the schema. There's no uniqueness constraint that says one canonical form is preferred. During resolution, if `symbol_key='validate'` matches 50 chunks (common with method names like `run`, `start`, `execute`), the disambiguation algorithm runs 50 comparisons in Python. This could be pre-computed: add a `priority` column (1=full-qualified, 2=module.symbol, 3=bare) so resolution prefers higher-priority matches without Python-side comparisons.

- No index on `symbol_lookup.file_path`. The disambiguation step ("same directory as source chunk's file") requires filtering by `file_path`, but the only index is on `(symbol_key, chunk_id)`. This means disambiguation for ambiguous symbols triggers a full scan of the matching symbol_key rows. For symbols like `validate` in a large Django app (dozens of matches), this is an O(n) scan.

**file_hashes**:
- The `content_hash` column stores the hash used for change detection. But what hash function? SHA256? MD5? xxHash? The plan uses "content_hash" in both chunk IDs and file_hashes but doesn't specify the algorithm. If it's SHA256 (32 bytes as hex = 64 chars), that's fine. If it's something faster like xxHash64, it needs to be documented as a stability requirement (changing hash functions across versions invalidates all stored hashes). This is a schema-level omission.

- `parse_mode TEXT NOT NULL DEFAULT 'ast'` — only two values. This would be better as a schema-enforced CHECK constraint: `CHECK (parse_mode IN ('ast', 'text_fallback'))`. Without it, a code bug that writes `parse_mode = 'AST'` (wrong case) goes undetected and `sr status` shows incorrect counts.

**Concrete alternative for the biggest issue (missing foreign_keys pragma)**:
Add `PRAGMA foreign_keys = ON` to the connection setup alongside WAL and busy_timeout. It's one line but its absence silently breaks the entire cascade-delete design. Additionally, add a CHECK constraint to `file_hashes.parse_mode` and a partial index on `refs.resolved_chunk WHERE resolved_chunk IS NOT NULL`.

---

### 6. Migration Strategy

**The decision**: Forward-only migrations as plain Python functions in a list. `migrate_001_...`, `migrate_002_...` etc. No rollback. On version mismatch (downgrade), raise `SchemaVersionError`.

**What's wrong or risky**:

- **No transaction wrapping of migrations**: The plan shows a list of migration functions but doesn't specify whether each migration runs in a transaction. SQLite DDL (CREATE TABLE, ALTER TABLE, DROP TABLE) is transactional in SQLite — you can roll back a CREATE TABLE inside a BEGIN block. If a migration function crashes halfway through (e.g., adds a column to `chunks` but fails before adding an index on it), the schema is left in a partial state. On the next startup, `schema_version` is still at the old value (the migration didn't complete), so the same migration runs again — which now fails immediately because the column already exists. The developer is stuck: they can't open the index at all without manually editing the DB.

- **"Drop and rebuild is acceptable because re-indexing is cheap (~30s)"**: This is a planning fallacy. Re-indexing costs $0.03 in Voyage API credits and ~30 seconds for a 100k-line repo. If a user has 10 repos indexed and a migration fires, that's $0.30 and 5 minutes. More importantly, if the user has a Voyage API outage at the time of migration, the rebuild will produce an incomplete index (FTS-only, no vectors) and the user won't know. A migration that drops vec_chunks and rebuilds it could be done without a rebuild: `INSERT INTO new_vec_chunks SELECT rowid, embedding FROM old_vec_chunks` with the new schema.

- **No forward-compatibility shim**: If a user runs source-recall v1.5 which creates schema_version=5, then rolls back to v1.3 (which knows about version 3), they get `SchemaVersionError` and must re-index. That's the intended behavior. But what if the MCP server is on v1.3 and the user runs `sr index` from v1.5? They can't use MCP until they upgrade it. The plan has no mechanism for the MCP server to auto-detect that it needs upgrading — it just fails with `SchemaVersionError`. This is a real operational hazard when the tool is used across multiple entry points (CLI, SDK, MCP server) that may be at different versions.

- **The migration list ordering is the schema version**: `len(MIGRATIONS)` implicitly is the current schema version. Adding a migration in the middle (e.g., to backport a fix between two already-released versions) would renumber all subsequent migrations and corrupt existing databases. This needs to be explicit: each migration must carry its own version number as metadata, not rely on list position.

**Concrete alternative**: Use a proper migration table:
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);
```
Each migration function registers its own version number. The runner applies `WHERE version > MAX(applied_at version)`, wrapped in a savepoint per migration so partial failures are rolled back. On startup, check `MAX(version)` — if higher than code's max known version, raise SchemaVersionError. This makes migrations idempotent, order-safe, and debuggable. Tradeoff: 20 more lines in `store.py`, explicit version registration per migration.

---

### 7. Path Hashing for Index Directories

**The decision**: Index directory name is `<repo-name>-<sha256(realpath)[:6]>`, computed from `os.path.realpath()` of the repo root.

**What's wrong or risky**:

- **Moved repos silently create a new index, leaving the old one orphaned**: If you `mv ~/dev/myrepo ~/dev/projects/myrepo`, the path hash changes. `sr index ~/dev/projects/myrepo` creates `myrepo-<new-hash>/index.db`. The old `myrepo-<old-hash>/index.db` (potentially 500MB) is never cleaned up because `sr clean` checks if the *stored* `repo_path` still exists — and `~/dev/myrepo` no longer does, so it would be cleaned. Actually wait — this is one case where clean *would* work. But the subtlety: the old index has `repo_path = "/Users/kevin/dev/myrepo"` in meta, which no longer exists → clean marks it orphaned → correct.

  But what about **symlinked repos**? If `~/dev/foo` is a symlink to `/data/projects/foo`, `realpath()` gives `/data/projects/foo`. If the symlink is deleted and recreated pointing to `/data/projects/foo-v2`, `realpath()` now gives a different hash. The old index is orphaned but `sr clean` checks if the stored path (`/data/projects/foo`) still exists — it does (v2 just replaced it). So `sr clean` doesn't clean it up. The user now has two indexes for what appears to be the same path. `repo_root_commit` verification would catch this if they try to query the old one with the new repo — but only if they explicitly open the old one, which they won't because it's now a different path hash.

- **Multiple git worktrees**: `git worktree add ../myrepo-feature feature-branch` creates a new working tree at a different path. `realpath()` of the worktree gives a different path than the main worktree. So two worktrees of the same repo get two indexes. This may be intentional (they have different file states), but it's undocumented. More critically: the two worktrees share `.git/` — so `repo_root_commit` is identical for both. The `repo_root_commit` identity check won't differentiate them, which is correct, but it means you can't use the main worktree's index for queries on the feature worktree without building a new one. The plan doesn't address this at all.

- **6-char hash prefix for human readability creates collision probability**: 6 hex chars = 24 bits = 16.7M possible values. Birthday collision at 50% probability at ~4600 repos. A developer with many repos (a sysadmin, an open-source contributor with hundreds of clones) will eventually get a collision, creating `myrepo-a1b2c3/` where two different repos share a directory. The `repo_root_commit` check would catch this — but only after the user wonders why their index was "corrupted." The fix is trivial: use 8 chars (32 bits) or 12 chars (48 bits).

- **The directory name `<repo-name>-<hash>` uses the repo *directory name*, not a canonical identifier**: A repo named `src` at `/Users/kevin/clients/acme/src` and `/Users/kevin/clients/betacorp/src` would both get directories named `src-<different-hashes>`. Human-readable in isolation but ambiguous in `ls`. Not a correctness bug, just an UX irritant.

**Concrete alternative**: Keep realpath-based hashing but use 12 hex chars (48-bit hash, ~280 trillion possible values, collision-safe for any realistic use). Add a `repos` registry as described in point 1 to make cleanup reliable. Document the worktree behavior explicitly: each worktree gets its own index, which is correct but means 2× storage for worktrees.

---

### 8. Atomic Swap via os.rename on Temp DB

**The decision**: `build()` writes to `index.db.tmp.{os.getpid()}`. Before renaming, runs `PRAGMA wal_checkpoint(TRUNCATE)` to flush WAL. Closes connection. Then `os.rename()`. Stale tmp files are cleaned at the next `build()` start.

**What's wrong or risky**:

- **The checkpoint-then-close-then-rename sequence has a race condition with existing readers**: If a reader opens `index.db` (old version) and holds a WAL read transaction open, the new `build()` renames the new `index.db.tmp.{pid}` into place. The reader still has the old `index.db` open via its file descriptor. On Linux/macOS, `rename()` is atomic at the filesystem level — the directory entry is swapped atomically. The old reader continues reading from its open file descriptor (now unlinked from the directory but still valid). This is correct. BUT: the old reader's `-wal` file is `index.db-wal` (associated with the old `index.db` by filename, not fd). After rename, the old `index.db-wal` is still present. The new `index.db` also looks for `index.db-wal` on open — if it finds one, SQLite tries to replay it against the new file. **The pre-rename WAL checkpoint on the temp file runs on `index.db.tmp.{pid}-wal`** (associated with the temp file). After rename, the tmp file becomes `index.db`, and SQLite looks for `index.db-wal`. If there's a stale `index.db-wal` from a previous reader that had the old file open, the new readers will try to apply it. The TRUNCATE checkpoint should prevent the new index from having a WAL file at all — but the *old* WAL from readers of the *previous* `index.db` could be `index.db-wal` and it won't be cleaned up. The plan says "flush all WAL pages back into the main file and truncate the -wal and -shm sidecars" — but this applies to the TEMP file's WAL, not the *old* index.db's WAL. If there are concurrent readers of the old index, the old `index.db-wal` could still be present when the new `index.db` lands.

  The fix: after checkpoint and close on the temp file, also `os.unlink('index.db-wal')` and `os.unlink('index.db-shm')` if they exist before rename. But this requires careful coordination: if a reader is actively using the old WAL, unlinking it is safe on POSIX (open FDs keep it alive) but the new `index.db` won't try to replay a deleted file.

- **Per-PID temp filenames don't prevent OS PID reuse**: If `build()` crashes, leaving `index.db.tmp.12345`, and later an unrelated process gets PID 12345, then a new `build()` call starts and sees `index.db.tmp.12345` — it cleans it up assuming it's stale. This is correct behavior, but if the "unrelated" process is actually a different `build()` call that's still running with that PID... wait, that's not possible since our `build()` uses the current process's PID. Actually this is fine — the cleanup at the start of build() only deletes files named `index.db.tmp.*` that are NOT the current PID. The current process's own tmp file is created fresh. The only risk is if PID reuse causes two concurrent build() calls to use the same PID — impossible since PIDs are unique per running process.

- **No fsync before rename**: `os.rename()` is atomic *in the directory entry* but the new file's data pages may not be flushed to disk. If the machine loses power between the rename and the OS flushing dirty pages, the new `index.db` could be corrupt on next open. The fix is `os.fsync(fd)` on the temp file's fd before closing it (after the WAL checkpoint). This adds ~100ms on spinning rust but is near-zero on NVMe. For a developer tool this is arguably acceptable to skip, but it should be a documented risk, not an unexamined assumption.

**Concrete alternative**: The plan's approach is mostly correct. The specific fix needed is:
1. After `PRAGMA wal_checkpoint(TRUNCATE)` on the temp DB, also explicitly unlink `index.db-wal` and `index.db-shm` (if they exist) before rename, preventing WAL replay confusion.
2. Add `os.fsync()` on the temp file fd before closing.
3. Document that this atomic swap is POSIX-only and not safe on non-POSIX filesystems (FAT32, some network mounts).

---

### 9. repo_root_commit for Identity Verification

**The decision**: Store the hash of the repository's initial commit (`git rev-list --max-parents=0 HEAD`) in meta. On every open, verify it matches the current repo's root commit. Mismatch → raise `IndexIdentityError` → force re-index.

**What's wrong or risky**:

- **Shallow clones break this entirely**: `git clone --depth=1` (common in CI, common for large repos) produces a shallow clone where `git rev-list --max-parents=0 HEAD` returns the shallow boundary commit, not the original root commit. If the user indexes a shallow clone of repo A, then later replaces it with a full clone of repo A, the root commits differ — `IndexIdentityError` is raised and a full re-index is forced, even though it's the same repo. The plan mentions "NULL for non-git repos" but doesn't mention shallow clones. This is a real-world scenario: a developer clones a large repo with `--depth=1` to save time, indexes it, then later fetches full history. Their index is invalidated.

- **Orphan branches have no initial commit in common**: `git checkout --orphan new-root` creates a branch with a completely different root commit. If the user indexes the main branch (root commit A), then switches to an orphan branch (root commit B), the open raises `IndexIdentityError`. This is arguably correct behavior (different history = different repo, effectively). But for a code Q&A tool, an orphan branch that happens to share most of the same files is still useful to query. The identity check is too strict.

- **Repos with rewritten history**: `git filter-branch` or `git filter-repo` to strip secrets or large files produces a repo where the root commit hash changes but it's logically the same repo at the same path. Every user who runs `git filter-repo` on an indexed repo must full re-index. Not a bug, but worth documenting.

- **`git rev-list --max-parents=0 HEAD` can return multiple commits**: A repo with multiple independent root commits (e.g., after `git merge --allow-unrelated-histories`) returns multiple root commit hashes. The plan doesn't say how these are stored or compared. Storing as a newline-separated string in meta and checking set membership is the obvious implementation, but it's not specified.

- **The identity check happens "on open"**: Opening the index and querying it are separate operations. A user opens `Index("~/dev/foo")`, passes identity check, gets results, closes. Meanwhile, someone replaces `~/dev/foo` with a different repo (unrelated path reuse). The next open catches it. But if the same `Index` object is kept alive (e.g., MCP server's 60-second TTL cache), the identity check only ran at construction time. A repo that was replaced mid-cache-lifetime will return results from the wrong database for up to 60 seconds. The plan describes this per-TTL cache but doesn't mention that the identity check should be re-run on cache reuse, not just at construction.

**Concrete alternative**:
1. For shallow clones, fallback to a different identity signal: the remote URL (`git remote get-url origin`) hashed, stored as `repo_remote_url_hash` in meta. If root commit is NULL (shallow boundary, not a true root), use remote URL hash instead. This survives shallow-to-full transitions for the same remote.
2. For the MCP cache lifetime issue: re-run a lightweight identity check on cache hit — just re-read `repo_root_commit` from the DB meta and compare it to the current `git rev-list --max-parents=0 HEAD`. This costs ~5ms and prevents stale cache identity bugs.
3. Document shallow clone behavior explicitly in the README/PLAN as a known limitation.
4. For the multiple-root-commit case: store as a comma-separated list, verify with set intersection (any match passes).

---

## Summary: Severity Rankings

| Issue | Severity | Category |
|-------|----------|----------|
| Missing `PRAGMA foreign_keys = ON` — CASCADE deletes silently broken | **Critical** | Schema |
| vec_chunks rowid instability during migrations | **High** | Schema + sqlite-vec |
| sqlite-vec dependency pin is a range, not a pin | **High** | Dependency |
| WAL replay confusion after atomic rename (old -wal file) | **High** | Atomic swap |
| Shallow clone breaks repo_root_commit identity | **High** | Identity |
| FTS5 storage overhead wildly underestimated (50MB → 150-300MB) | **Medium** | Storage sizing |
| Migration functions not wrapped in savepoints | **Medium** | Migrations |
| MCP cache doesn't re-check identity after TTL | **Medium** | Identity |
| Chunk ID separator-collision bug (colon-concat of fields) | **Medium** | Schema |
| 6-char path hash is collision-prone at scale | **Low** | Path hashing |
| No fsync before os.rename | **Low** | Atomic swap |
| Working tree changes invisible to stale_check cache | **Low** | Concurrency |
| No `CHECK` constraint on `parse_mode` | **Low** | Schema |
| Missing index on symbol_lookup.file_path for disambiguation | **Low** | Performance |
| No registry for orphaned index cleanup | **Low** | Lifecycle |
