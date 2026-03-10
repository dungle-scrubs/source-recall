"""IndexStore: SQLite connection lifecycle, schema DDL, CRUD, migrations,
PID-file locking, and atomic swap.
"""

from __future__ import annotations

import json
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
    SchemaVersionError,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

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
        CHECK (search_quality IN ('ast', 'regex', 'text_fallback'))
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

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
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
    mtime_ns     INTEGER
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    description TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Migrations — each carries its own version number.
# Wrapped in savepoints so partial failures roll back cleanly.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, str]] = [
    # (version, description, sql)
    # Version 1 is the initial schema — created by DDL above.
]


# ---------------------------------------------------------------------------
# PID-file lock
# ---------------------------------------------------------------------------


def _acquire_lock(lock_path: Path, timeout: float = 0) -> None:
    """Acquire a PID-file advisory lock.

    @param lock_path: Path to the lock file.
    @param timeout: Seconds to wait before giving up (0 = fail immediately).
    @raises IndexLockError: If the lock is held by a live process.
    """
    deadline = time.monotonic() + timeout

    while True:
        if lock_path.exists():
            try:
                data = json.loads(lock_path.read_text())
                pid = data["pid"]
                # Check if process is still alive.
                os.kill(pid, 0)
            except (json.JSONDecodeError, KeyError, ProcessLookupError, OSError):
                # Stale lock — steal it.
                lock_path.unlink(missing_ok=True)
            else:
                if time.monotonic() >= deadline:
                    raise IndexLockError(str(lock_path), pid)
                time.sleep(0.2)
                continue

        # Write our PID.
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"pid": os.getpid(), "started": _now_iso()}))

        # Re-read to confirm we won the race.
        try:
            data = json.loads(lock_path.read_text())
            if data["pid"] == os.getpid():
                return
        except Exception:
            pass

        if time.monotonic() >= deadline:
            raise IndexLockError(str(lock_path), 0)
        time.sleep(0.1)


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
        """Close the connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

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
                self.conn.executescript(sql)
                self.conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) "
                    "VALUES (?, ?, ?)",
                    (version, _now_iso(), description),
                )
                self._set_meta("schema_version", str(version))
                self.conn.execute(f"RELEASE migration_{version}")
            except Exception:
                self.conn.execute(f"ROLLBACK TO migration_{version}")
                raise

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

    def insert_chunks(self, chunks: Sequence[ChunkData]) -> None:
        """Insert chunks in bulk. FTS updated automatically via triggers.

        @param chunks: Sequence of ChunkData to insert.
        """
        if not chunks:
            return
        with self._transaction():
            self.conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (id, file_path, symbol_name, symbol_type, content,
                    start_line, end_line, parent_chunk_id, sub_chunk_index,
                    search_quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
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
                    )
                    for c in chunks
                ],
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

    def upsert_file_hash(self, record: FileRecord) -> None:
        """Insert or update a file hash record.

        @param record: FileRecord with hash and parse mode.
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO file_hashes
               (file_path, content_hash, parse_mode, mtime_ns)
               VALUES (?, ?, ?, ?)""",
            (
                record.file_path,
                record.content_hash,
                record.parse_mode.value,
                record.mtime_ns,
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

    def get_all_file_hashes(self) -> dict[str, FileRecord]:
        """Load all file hash records.

        @returns: Dict mapping file_path to FileRecord.
        """
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
                 -rank AS score
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
                      content, start_line, end_line, search_quality
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
            }
            for row in rows
        ]

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


def _fts_escape(query: str) -> str:
    """Escape an FTS5 query for safe matching.

    Wraps each token in double quotes to prevent FTS5 syntax
    interpretation (AND/OR/NOT/NEAR operators, column filters).

    @param query: Raw user query.
    @returns: Escaped FTS5 query string.
    """
    tokens = query.split()
    escaped = []
    for token in tokens:
        # Strip characters that break FTS5 even inside quotes.
        clean = token.replace('"', "").replace("'", "")
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
