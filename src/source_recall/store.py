"""IndexStore: SQLite connection lifecycle, schema DDL, CRUD, migrations,
PID-file locking, and atomic swap.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from source_recall.models import (
    ChunkData,
    FileRecord,
    IndexLockError,
    ParseMode,
    RefData,
    SchemaVersionError,
)

# ---------------------------------------------------------------------------
# sqlite-vec availability (requires apsw for extension loading on macOS)
# ---------------------------------------------------------------------------

_HAS_SQLITE_VEC = False
try:
    import apsw  # noqa: F401
    import sqlite_vec  # noqa: F401

    _HAS_SQLITE_VEC = True
except ImportError:
    pass

_store_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 5

_DDL = """\
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
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
        CHECK (search_quality IN ('ast', 'regex', 'text_fallback')),
    branches        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_file
    ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_parent
    ON chunks(parent_chunk_id) WHERE parent_chunk_id IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, file_path, symbol_name,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
    VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF
    content, file_path, symbol_name ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
    INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
    VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
END;

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    parse_mode   TEXT NOT NULL DEFAULT 'ast'
        CHECK (parse_mode IN ('ast', 'text_fallback', 'regex')),
    mtime_ns     INTEGER,
    branch       TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_branch
    ON file_hashes(branch, file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_branches
    ON chunks(branches);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refs (
    source_chunk_id TEXT NOT NULL,
    target_symbol   TEXT NOT NULL,
    ref_type        TEXT NOT NULL
        CHECK (ref_type IN ('import', 'type_ref', 'call', 'inherits', 'decorator')),
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(target_symbol);
CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(source_chunk_id);

CREATE TABLE IF NOT EXISTS symbol_lookup (
    symbol_name TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbol_lookup(symbol_name);
"""

# ---------------------------------------------------------------------------
# Migrations — each carries its own version number.
# Wrapped in savepoints so partial failures roll back cleanly.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, str]] = [
    # (version, description, sql)
    # Version 1 is the initial schema — created by DDL above.
    (
        2,
        "add vec_chunks for vector search",
        # vec_chunks creation is handled in _ensure_vec_table() since
        # sqlite-vec virtual tables require the extension to be loaded.
        # This migration is a no-op placeholder to bump schema_version.
        "SELECT 1;",
    ),
    (
        3,
        "add refs and symbol_lookup tables",
        """
        CREATE TABLE IF NOT EXISTS refs (
            source_chunk_id TEXT NOT NULL,
            target_symbol   TEXT NOT NULL,
            ref_type        TEXT NOT NULL
                CHECK (ref_type IN ('import', 'type_ref', 'call', 'inherits', 'decorator')),
            FOREIGN KEY (source_chunk_id) REFERENCES chunks(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(target_symbol);
        CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(source_chunk_id);

        CREATE TABLE IF NOT EXISTS symbol_lookup (
            symbol_name TEXT NOT NULL,
            chunk_id    TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbol_lookup(symbol_name)
        """,
    ),
    # Migration 4 uses "alter_columns" key to signal conditional ALTER TABLE.
    # Handled specially in run_migrations() to check column existence first.
    (
        4,
        "add branch-aware columns for git-object chunking",
        """
        CREATE INDEX IF NOT EXISTS idx_file_hashes_branch
            ON file_hashes(branch, file_path)
        ;
        CREATE INDEX IF NOT EXISTS idx_chunks_branches
            ON chunks(branches)
        """,
    ),
    # Migration 5 is handled specially in run_migrations() because
    # CREATE TRIGGER contains semicolons inside BEGIN...END that
    # break the naive split-on-semicolon execution.
    (
        5,
        "narrow chunks_au trigger to FTS-indexed columns only",
        "SELECT 1",
    ),
]


# ---------------------------------------------------------------------------
# PID-file lock
# ---------------------------------------------------------------------------


def _acquire_lock(lock_path: Path, timeout: float = 0) -> None:
    """Acquire a PID-file advisory lock using atomic O_CREAT|O_EXCL.

    @param lock_path: Path to the lock file.
    @param timeout: Seconds to wait before giving up (0 = fail immediately).
    @raises IndexLockError: If the lock is held by a live process.
    """
    deadline = time.monotonic() + timeout
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        # Attempt atomic creation — fails if file already exists.
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                payload = json.dumps({"pid": os.getpid(), "started": _now_iso()})
                os.write(fd, payload.encode())
            finally:
                os.close(fd)
            return  # Lock acquired.
        except FileExistsError:
            pass

        # Lock file exists — check if holder is still alive.
        try:
            data = json.loads(lock_path.read_text())
            pid = data["pid"]
            os.kill(pid, 0)  # Signal 0: check if alive.
        except (json.JSONDecodeError, KeyError, ProcessLookupError, OSError):
            # Stale lock — remove and retry.
            lock_path.unlink(missing_ok=True)
            continue
        else:
            # Process is alive — wait or fail.
            if time.monotonic() >= deadline:
                raise IndexLockError(str(lock_path), pid)
            time.sleep(0.2)


def _release_lock(lock_path: Path) -> None:
    """Release the PID-file lock.

    @param lock_path: Path to the lock file.
    """
    try:
        data = json.loads(lock_path.read_text())
        if data.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)
    except Exception:
        lock_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------


class IndexStore:
    """Manages SQLite connections, schema, and CRUD for a single index.

    @param db_path: Path to the index.db file.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._vec_conn: Any = None  # apsw.Connection, lazily opened

    # -- Connection lifecycle -----------------------------------------------

    def open(self) -> sqlite3.Connection:
        """Open a connection with correct PRAGMAs.

        @returns: A configured SQLite connection.
        """
        if self._conn is not None:
            return self._conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close all connections if open."""
        if self._vec_conn is not None:
            self._vec_conn.close()
            self._vec_conn = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> IndexStore:
        """Context manager entry — opens the connection.

        @returns: Self with connection opened.
        """
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Context manager exit — always closes connections."""
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or open the connection.

        @returns: Active SQLite connection.
        """
        return self.open()

    # -- Schema -------------------------------------------------------------

    def create_schema(self) -> None:
        """Apply DDL and record initial schema version."""
        self.conn.executescript(_DDL)
        self._set_meta("schema_version", str(_SCHEMA_VERSION))

    def run_migrations(self) -> None:
        """Run any pending migrations, each in a savepoint.

        @raises SchemaVersionError: If the on-disk version is newer than
            this code supports.
        """
        current = self._get_schema_version()
        if current > _SCHEMA_VERSION:
            raise SchemaVersionError(on_disk=current, expected=_SCHEMA_VERSION)

        for version, description, sql in _MIGRATIONS:
            if version <= current:
                continue
            self.conn.execute(f"SAVEPOINT migration_{version}")
            try:
                # Migration 4 needs conditional ALTER TABLE — SQLite has
                # no ADD COLUMN IF NOT EXISTS. Run ALTER only if the
                # column is missing (handles fresh v4 DDL gracefully).
                if version == 4:
                    self._migrate_004_add_columns()

                # Migration 5 recreates the chunks_au trigger with a
                # column list. CREATE TRIGGER contains semicolons inside
                # BEGIN...END, so we can't split on ";".
                if version == 5:
                    self._migrate_005_narrow_trigger()

                # Use execute (not executescript) to stay within the
                # savepoint — executescript auto-commits.
                for stmt in sql.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        self.conn.execute(stmt)
                self.conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) "
                    "VALUES (?, ?, ?)",
                    (version, _now_iso(), description),
                )
                # Use direct execute (not _set_meta) to avoid commit()
                # which would release the savepoint.
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(version)),
                )
                self.conn.execute(f"RELEASE migration_{version}")
            except Exception:
                self.conn.execute(f"ROLLBACK TO migration_{version}")
                raise

    def _migrate_004_add_columns(self) -> None:
        """Conditionally add branch-awareness columns for migration v4.

        SQLite lacks ADD COLUMN IF NOT EXISTS, so we check PRAGMA
        table_info first. Safe to call on DBs that already have the
        columns (fresh v4 DDL).
        """
        chunks_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(chunks)")}
        if "branches" not in chunks_cols:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN branches TEXT NOT NULL DEFAULT ''"
            )

        fh_cols = {
            row[1] for row in self.conn.execute("PRAGMA table_info(file_hashes)")
        }
        if "branch" not in fh_cols:
            self.conn.execute(
                "ALTER TABLE file_hashes ADD COLUMN branch TEXT NOT NULL DEFAULT ''"
            )

    def _migrate_005_narrow_trigger(self) -> None:
        """Replace chunks_au trigger to only fire on FTS-indexed columns.

        The old trigger fired on ANY UPDATE to chunks, causing unnecessary
        FTS churn when only the branches column was updated.
        """
        self.conn.execute("DROP TRIGGER IF EXISTS chunks_au")
        # executescript auto-commits, so we use execute with the full
        # CREATE TRIGGER as a single statement (SQLite allows this).
        self.conn.execute(
            """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF
                content, file_path, symbol_name ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
                VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
                INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
                VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
            END"""
        )

    def _get_schema_version(self) -> int:
        """Read current schema version from meta.

        @returns: Integer schema version (0 if not set).
        """
        val = self._get_meta("schema_version")
        return int(val) if val else 0

    # -- Meta CRUD ----------------------------------------------------------

    def _set_meta(self, key: str, value: str) -> None:
        """Upsert a meta key.

        @param key: Meta key name.
        @param value: Meta value.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def _get_meta(self, key: str) -> str | None:
        """Read a meta value.

        @param key: Meta key name.
        @returns: Value string or None if not set.
        """
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta_batch(self, pairs: dict[str, str]) -> None:
        """Upsert multiple meta keys in one transaction.

        @param pairs: Key-value pairs to write.
        """
        with self._transaction():
            for key, value in pairs.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (key, value),
                )

    def get_meta(self, key: str) -> str | None:
        """Public accessor for meta values.

        @param key: Meta key name.
        @returns: Value string or None.
        """
        return self._get_meta(key)

    # -- Chunk CRUD ---------------------------------------------------------

    def insert_chunks(self, chunks: Sequence[ChunkData], branch: str = "") -> None:
        """Insert chunks in bulk, with optional branch tracking.

        When branch is non-empty, existing chunks get the branch appended
        to their comma-separated branches column (dedup-safe). New chunks
        are inserted with the given branch. FTS updated via triggers.

        @param chunks: Sequence of ChunkData to insert.
        @param branch: Branch name to associate (empty = legacy behavior).
        """
        if not chunks:
            return
        with self._transaction():
            # Batch-fetch existing chunk IDs + branches for dedup.
            existing_branches: dict[str, str] = {}
            if branch:
                all_ids = [c.chunk_id for c in chunks]
                # SQLite max variables is 999 — batch in groups.
                for i in range(0, len(all_ids), 500):
                    batch = all_ids[i : i + 500]
                    placeholders = ",".join("?" * len(batch))
                    rows = self.conn.execute(
                        f"SELECT id, branches FROM chunks WHERE id IN ({placeholders})",
                        batch,
                    ).fetchall()
                    for row in rows:
                        existing_branches[row[0]] = row[1]

            for c in chunks:
                if branch and c.chunk_id in existing_branches:
                    # Chunk exists — append branch if not already present.
                    # UPDATE branches only (won't trigger FTS re-index).
                    current = existing_branches[c.chunk_id]
                    current_set = set(current.split(",")) if current else set()
                    if branch not in current_set:
                        current_set.add(branch)
                        new_branches = ",".join(sorted(current_set))
                        self.conn.execute(
                            "UPDATE chunks SET branches = ? WHERE id = ?",
                            (new_branches, c.chunk_id),
                        )
                    continue  # Skip re-insert — chunk content is identical.

                self.conn.execute(
                    """INSERT OR REPLACE INTO chunks
                       (id, file_path, symbol_name, symbol_type, content,
                        start_line, end_line, parent_chunk_id, sub_chunk_index,
                        search_quality, branches)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.chunk_id,
                        c.file_path,
                        c.symbol_name,
                        c.symbol_type.value,
                        c.content,
                        c.start_line,
                        c.end_line,
                        c.parent_chunk_id,
                        c.sub_chunk_index,
                        c.search_quality.value,
                        branch,
                    ),
                )

    def delete_chunks_for_file(self, file_path: str) -> None:
        """Delete all chunks belonging to a file. Cascades to FTS via trigger.

        @param file_path: Repo-relative file path.
        """
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.commit()

    def get_chunk_count(self) -> int:
        """Count total chunks in the index.

        @returns: Number of chunks.
        """
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    # -- File hash CRUD -----------------------------------------------------

    def upsert_file_hash(self, record: FileRecord, branch: str = "") -> None:
        """Insert or update a file hash record.

        @param record: FileRecord with hash and parse mode.
        @param branch: Branch name to associate.
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO file_hashes
               (file_path, content_hash, parse_mode, mtime_ns, branch)
               VALUES (?, ?, ?, ?, ?)""",
            (
                record.file_path,
                record.content_hash,
                record.parse_mode.value,
                record.mtime_ns,
                branch,
            ),
        )
        self.conn.commit()

    def get_file_hash(self, file_path: str) -> FileRecord | None:
        """Look up a file hash record.

        @param file_path: Repo-relative file path.
        @returns: FileRecord or None if not tracked.
        """
        row = self.conn.execute(
            "SELECT file_path, content_hash, parse_mode, mtime_ns "
            "FROM file_hashes WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(
            file_path=row[0],
            content_hash=row[1],
            parse_mode=ParseMode(row[2]),
            mtime_ns=row[3],
        )

    def get_all_file_hashes(self, branch: str | None = None) -> dict[str, FileRecord]:
        """Load file hash records, optionally filtered by branch.

        @param branch: Filter to this branch. None = all records.
        @returns: Dict mapping file_path to FileRecord.
        """
        if branch is not None:
            rows = self.conn.execute(
                "SELECT file_path, content_hash, parse_mode, mtime_ns "
                "FROM file_hashes WHERE branch = ?",
                (branch,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT file_path, content_hash, parse_mode, mtime_ns FROM file_hashes"
            ).fetchall()
        return {
            row[0]: FileRecord(
                file_path=row[0],
                content_hash=row[1],
                parse_mode=ParseMode(row[2]),
                mtime_ns=row[3],
            )
            for row in rows
        }

    def delete_file_hash(self, file_path: str) -> None:
        """Remove a file hash record.

        @param file_path: Repo-relative file path.
        """
        self.conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
        self.conn.commit()

    def get_file_count_by_mode(self) -> dict[str, int]:
        """Count files grouped by parse mode.

        @returns: Dict mapping parse_mode to count.
        """
        rows = self.conn.execute(
            "SELECT parse_mode, COUNT(*) FROM file_hashes GROUP BY parse_mode"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    # -- FTS query ----------------------------------------------------------

    def fts_search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        """Run a BM25 full-text search.

        @param query: FTS5 query string.
        @param limit: Max results.
        @returns: List of dicts with chunk data and BM25 score.
        """
        # Escape special FTS5 characters for safety.
        safe_query = _fts_escape(query)
        if not safe_query.strip():
            return []

        rows = self.conn.execute(
            """SELECT
                 c.id, c.file_path, c.symbol_name, c.symbol_type,
                 c.content, c.start_line, c.end_line, c.search_quality,
                 -rank AS score, c.branches
               FROM chunks_fts
               JOIN chunks c ON c.rowid = chunks_fts.rowid
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit),
        ).fetchall()

        return [
            {
                "chunk_id": row[0],
                "file_path": row[1],
                "symbol_name": row[2],
                "symbol_type": row[3],
                "content": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "search_quality": row[7],
                "score": row[8],
                "branches": row[9],
            }
            for row in rows
        ]

    def symbol_search(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Exact-match search on symbol_name.

        @param symbol: Symbol name to match (case-insensitive).
        @param limit: Max results.
        @returns: List of dicts with chunk data.
        """
        rows = self.conn.execute(
            """SELECT id, file_path, symbol_name, symbol_type,
                      content, start_line, end_line, search_quality, branches
               FROM chunks
               WHERE symbol_name = ? COLLATE NOCASE
               LIMIT ?""",
            (symbol, limit),
        ).fetchall()

        return [
            {
                "chunk_id": row[0],
                "file_path": row[1],
                "symbol_name": row[2],
                "symbol_type": row[3],
                "content": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "search_quality": row[7],
                "score": 100.0,  # Exact matches get max score.
                "branches": row[8],
            }
            for row in rows
        ]

    # -- Refs / symbol_lookup operations ---------------------------------------

    def insert_refs(self, refs: Sequence[RefData]) -> None:
        """Batch-insert cross-references.

        @param refs: List of RefData to insert.
        """
        if not refs:
            return
        self.conn.executemany(
            "INSERT INTO refs (source_chunk_id, target_symbol, ref_type) "
            "VALUES (?, ?, ?)",
            [(r.source_chunk_id, r.target_symbol, r.ref_type) for r in refs],
        )
        self.conn.commit()

    def get_refs_for_chunk(self, chunk_id: str) -> list[RefData]:
        """Get all outgoing refs from a chunk.

        @param chunk_id: Source chunk ID.
        @returns: List of RefData.
        """
        from source_recall.models import RefType as _RefType

        rows = self.conn.execute(
            "SELECT source_chunk_id, target_symbol, ref_type FROM refs "
            "WHERE source_chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        return [
            RefData(
                source_chunk_id=row["source_chunk_id"],
                target_symbol=row["target_symbol"],
                ref_type=_RefType(row["ref_type"]),
            )
            for row in rows
        ]

    def get_chunks_referencing(self, symbol: str) -> list[dict[str, Any]]:
        """Find chunks that reference a given symbol (reverse lookup).

        @param symbol: Target symbol name.
        @returns: List of chunk dicts with ref_type.
        """
        rows = self.conn.execute(
            "SELECT r.ref_type, c.* FROM refs r "
            "JOIN chunks c ON c.id = r.source_chunk_id "
            "WHERE r.target_symbol = ?",
            (symbol,),
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_symbol_lookup(
        self, chunk_id: str, symbol_name: str, file_path: str
    ) -> None:
        """Register a symbol definition for graph resolution.

        @param chunk_id: Chunk that defines this symbol.
        @param symbol_name: Symbol name.
        @param file_path: File containing the definition.
        """
        self.conn.execute(
            "INSERT INTO symbol_lookup (symbol_name, chunk_id, file_path) "
            "VALUES (?, ?, ?)",
            (symbol_name, chunk_id, file_path),
        )
        self.conn.commit()

    def insert_symbol_lookups(self, entries: Sequence[tuple[str, str, str]]) -> None:
        """Batch-insert symbol lookup entries.

        @param entries: List of (chunk_id, symbol_name, file_path) tuples.
        """
        if not entries:
            return
        self.conn.executemany(
            "INSERT INTO symbol_lookup (symbol_name, chunk_id, file_path) "
            "VALUES (?, ?, ?)",
            [(sym, cid, fp) for cid, sym, fp in entries],
        )
        self.conn.commit()

    def lookup_symbol(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find chunks that define a symbol.

        @param symbol_name: Symbol to look up.
        @returns: List of dicts with chunk_id, file_path, etc.
        """
        rows = self.conn.execute(
            "SELECT sl.chunk_id, sl.file_path, c.symbol_name, c.symbol_type, "
            "c.content, c.start_line, c.end_line, c.search_quality, c.branches "
            "FROM symbol_lookup sl "
            "JOIN chunks c ON c.id = sl.chunk_id "
            "WHERE sl.symbol_name = ?",
            (symbol_name,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_refs_for_file(self, file_path: str) -> None:
        """Delete all refs originating from chunks in a file.

        @param file_path: Repo-relative file path.
        """
        self.conn.execute(
            "DELETE FROM refs WHERE source_chunk_id IN "
            "(SELECT id FROM chunks WHERE file_path = ?)",
            (file_path,),
        )
        self.conn.commit()

    def delete_symbol_lookups_for_file(self, file_path: str) -> None:
        """Delete all symbol_lookup entries for a file.

        @param file_path: Repo-relative file path.
        """
        self.conn.execute(
            "DELETE FROM symbol_lookup WHERE file_path = ?",
            (file_path,),
        )
        self.conn.commit()

    # -- Vector CRUD (sqlite-vec via apsw) ---------------------------------
    #
    # macOS Python doesn't compile sqlite3 with SQLITE_ENABLE_LOAD_EXTENSION.
    # We use apsw (which bundles its own sqlite3 build with extension support)
    # for all vec_chunks operations. The apsw connection opens the same
    # database file in WAL mode, so reads/writes interleave safely with the
    # sqlite3 connection used for FTS/chunks/meta.

    def _get_vec_conn(self) -> Any:
        """Get or open the apsw connection with sqlite-vec loaded.

        @returns: An apsw.Connection with the vec extension, or None.
        """
        if not _HAS_SQLITE_VEC:
            return None

        if self._vec_conn is not None:
            return self._vec_conn

        try:
            vec_conn = apsw.Connection(str(self.db_path))  # type: ignore[name-defined]
            vec_conn.execute("PRAGMA journal_mode = WAL")
            vec_conn.execute("PRAGMA busy_timeout = 5000")
            vec_conn.enable_load_extension(True)
            vec_conn.load_extension(sqlite_vec.loadable_path())  # type: ignore[name-defined]
            vec_conn.enable_load_extension(False)
            self._vec_conn = vec_conn
            return vec_conn
        except Exception:
            _store_logger.debug("Failed to open apsw vec connection", exc_info=True)
            return None

    def ensure_vec_table(self, dimensions: int) -> bool:
        """Create the vec_chunks virtual table if sqlite-vec is available.

        @param dimensions: Embedding vector dimensionality.
        @returns: True if table exists (created or already present).
        """
        vec_conn = self._get_vec_conn()
        if vec_conn is None:
            return False
        vec_conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{dimensions}]
            )"""
        )
        return True

    def has_vec_table(self) -> bool:
        """Check if the vec_chunks table exists.

        @returns: True if the table exists.
        """
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        return row is not None

    def insert_vectors(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Batch insert vectors into vec_chunks via apsw.

        @param chunk_ids: Chunk IDs matching rows in the chunks table.
        @param embeddings: Embedding vectors (must match table dimensions).
        """
        if not chunk_ids:
            return

        vec_conn = self._get_vec_conn()
        if vec_conn is None:
            return

        import struct

        for cid, emb in zip(chunk_ids, embeddings, strict=True):
            blob = struct.pack(f"{len(emb)}f", *emb)
            # vec0 doesn't support INSERT OR REPLACE — delete first.
            with contextlib.suppress(Exception):
                vec_conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
            vec_conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (cid, blob),
            )

    def search_vectors(
        self,
        query_embedding: list[float],
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """Search vec_chunks by distance, join with chunks via apsw.

        @param query_embedding: Query vector.
        @param top_k: Max results.
        @returns: List of dicts with chunk data and distance score.
        """
        vec_conn = self._get_vec_conn()
        if vec_conn is None:
            return []
        if not self.has_vec_table():
            return []

        import struct

        blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)

        rows = list(
            vec_conn.execute(
                """SELECT
                     c.id, c.file_path, c.symbol_name, c.symbol_type,
                     c.content, c.start_line, c.end_line, c.search_quality,
                     v.distance, c.branches
                   FROM vec_chunks v
                   JOIN chunks c ON c.id = v.chunk_id
                   WHERE v.embedding MATCH ?
                     AND k = ?
                   ORDER BY v.distance""",
                (blob, top_k),
            )
        )

        return [
            {
                "chunk_id": row[0],
                "file_path": row[1],
                "symbol_name": row[2],
                "symbol_type": row[3],
                "content": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "search_quality": row[7],
                "distance": row[8],
                "branches": row[9],
            }
            for row in rows
        ]

    def delete_vectors_by_file(self, file_path: str) -> None:
        """Delete vec_chunks rows for chunks belonging to a file.

        Uses the apsw connection since vec_chunks is a vec0 virtual table.

        @param file_path: Repo-relative file path.
        """
        if not self.has_vec_table():
            return
        vec_conn = self._get_vec_conn()
        if vec_conn is None:
            return
        vec_conn.execute(
            """DELETE FROM vec_chunks
               WHERE chunk_id IN (
                   SELECT id FROM chunks WHERE file_path = ?
               )""",
            (file_path,),
        )

    def get_vector_count(self) -> int:
        """Count rows in vec_chunks.

        @returns: Number of vectors stored (0 if table doesn't exist).
        """
        if not self.has_vec_table():
            return 0
        vec_conn = self._get_vec_conn()
        if vec_conn is None:
            return 0
        row = list(vec_conn.execute("SELECT COUNT(*) FROM vec_chunks"))
        return row[0][0] if row else 0

    # -- Atomic swap --------------------------------------------------------

    @staticmethod
    def atomic_swap(tmp_path: Path, target_path: Path) -> None:
        """Checkpoint WAL, fsync, clean stale sidecars, and rename.

        @param tmp_path: Temporary database that was just built.
        @param target_path: Final destination path.
        """
        # Checkpoint + truncate WAL on the temp file.
        conn = sqlite3.connect(str(tmp_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        # fsync the temp file.
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        # Remove stale WAL/SHM sidecars from the target path
        # to prevent replay confusion after rename.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target_path) + suffix)
            sidecar.unlink(missing_ok=True)

        # Atomic rename.
        os.rename(tmp_path, target_path)

    # -- Locking (delegates to module-level functions) ----------------------

    @staticmethod
    def acquire_lock(db_path: Path, timeout: float = 0) -> Path:
        """Acquire the PID-file lock for an index.

        @param db_path: Path to the index.db.
        @param timeout: Seconds to wait.
        @returns: Path to the lock file (for release).
        @raises IndexLockError: If lock is held.
        """
        lock_path = db_path.with_suffix(".lock")
        _acquire_lock(lock_path, timeout)
        return lock_path

    @staticmethod
    def release_lock(db_path: Path) -> None:
        """Release the PID-file lock for an index.

        @param db_path: Path to the index.db.
        """
        lock_path = db_path.with_suffix(".lock")
        _release_lock(lock_path)

    # -- Helpers ------------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Context manager for a SQLite transaction.

        @returns: Generator that commits on success, rolls back on error.
        """
        self.conn.execute("BEGIN")
        try:
            yield
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @staticmethod
    def clean_tmp_files(db_path: Path) -> list[Path]:
        """Remove stale .tmp.* files from a previous crashed build.

        @param db_path: Path to the index.db.
        @returns: List of removed paths.
        """
        removed: list[Path] = []
        parent = db_path.parent
        if not parent.exists():
            return removed
        current_tmp = f"{db_path.name}.tmp.{os.getpid()}"
        for p in parent.iterdir():
            if p.name.startswith(f"{db_path.name}.tmp.") and p.name != current_tmp:
                p.unlink(missing_ok=True)
                removed.append(p)
        return removed


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


_FTS5_STRIP = str.maketrans("", "", "\"'*^-")


def _fts_escape(query: str) -> str:
    """Escape an FTS5 query for safe matching.

    Wraps each token in double quotes to prevent FTS5 syntax
    interpretation (AND/OR/NOT/NEAR operators, column filters).
    Strips characters that have special meaning in FTS5 even
    inside quotes: ``"`` ``'`` ``*`` ``^`` ``-``.

    @param query: Raw user query.
    @returns: Escaped FTS5 query string.
    """
    tokens = query.split()
    escaped = []
    for token in tokens:
        clean = token.translate(_FTS5_STRIP)
        if clean:
            escaped.append(f'"{clean}"')
    return " ".join(escaped)


def _now_iso() -> str:
    """Current UTC time as ISO string.

    @returns: ISO-formatted timestamp.
    """
    return datetime.now(UTC).isoformat()


def get_index_dir(repo_path: Path) -> Path:
    """Compute the index directory for a repository.

    Uses a 12-char hash (48-bit) of the realpath for collision resistance.

    @param repo_path: Path to the repository root.
    @returns: Path to the index directory under ~/.local/share/source-recall/.
    """
    import hashlib

    real = str(repo_path.resolve())
    path_hash = hashlib.sha256(real.encode()).hexdigest()[:12]
    repo_name = repo_path.resolve().name
    base = Path.home() / ".local" / "share" / "source-recall"
    return base / f"{repo_name}-{path_hash}"


def get_db_path(repo_path: Path) -> Path:
    """Get the index.db path for a repository.

    @param repo_path: Path to the repository root.
    @returns: Path to index.db.
    """
    return get_index_dir(repo_path) / "index.db"
