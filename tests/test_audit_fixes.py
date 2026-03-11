"""Tests for issues found during audit."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from source_recall.models import ChunkData, ConfigError, SymbolType
from source_recall.store import IndexStore, _fts_escape

# ---------------------------------------------------------------------------
# Issue #1: PID-file lock uses O_CREAT|O_EXCL (no TOCTOU)
# ---------------------------------------------------------------------------


class TestAtomicLock:
    def test_concurrent_lock_attempts(self, tmp_path: Path) -> None:
        """Only one process wins the lock; the other gets IndexLockError."""
        import json
        import os

        from source_recall.models import IndexLockError

        db_path = tmp_path / "index.db"
        lock_path = db_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Acquire the lock first.
        IndexStore.acquire_lock(db_path, timeout=0)
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

        # Second attempt should fail immediately (our PID is alive).
        with pytest.raises(IndexLockError):
            IndexStore.acquire_lock(db_path, timeout=0)

        IndexStore.release_lock(db_path)

    def test_lock_file_created_atomically(self, tmp_path: Path) -> None:
        """Lock file is created with O_CREAT|O_EXCL (atomic)."""
        import json
        import os

        db_path = tmp_path / "index.db"
        IndexStore.acquire_lock(db_path, timeout=0)

        lock_path = db_path.with_suffix(".lock")
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()
        assert "started" in data

        IndexStore.release_lock(db_path)
        assert not lock_path.exists()


# ---------------------------------------------------------------------------
# Issue #2 & #3: IndexStore context manager (no connection leak)
# ---------------------------------------------------------------------------


class TestIndexStoreContextManager:
    def test_context_manager_opens_and_closes(self, tmp_path: Path) -> None:
        """IndexStore as context manager opens on entry, closes on exit."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()
            assert store._conn is not None
        # After exit, connection is closed.
        assert store._conn is None

    def test_context_manager_closes_on_error(self, tmp_path: Path) -> None:
        """IndexStore closes even if an exception occurs inside the block."""
        db_path = tmp_path / "test.db"
        with pytest.raises(RuntimeError), IndexStore(db_path) as store:
            store.create_schema()
            raise RuntimeError("boom")
        assert store._conn is None


# ---------------------------------------------------------------------------
# Issue #4: FTS trigger only fires on content/file_path/symbol_name changes
# ---------------------------------------------------------------------------


class TestFTSTriggerNarrow:
    def test_branches_update_does_not_churn_fts(self, tmp_path: Path) -> None:
        """Updating only branches column does NOT trigger FTS re-index."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()

            chunk = ChunkData(
                file_path="a.py",
                symbol_name="foo",
                symbol_type=SymbolType.FUNCTION,
                content="def foo(): pass",
                start_line=1,
                end_line=1,
            )
            store.insert_chunks([chunk], branch="main")

            # Get FTS rowid count before.
            fts_before = store.conn.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0]
            assert fts_before == 1

            # Append branch — should NOT fire FTS trigger.
            store.insert_chunks([chunk], branch="feature")

            # FTS should still have exactly 1 entry.
            fts_after = store.conn.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0]
            assert fts_after == 1

            # FTS search still works.
            results = store.fts_search("foo")
            assert len(results) == 1


# ---------------------------------------------------------------------------
# Issue #6: Bad TOML raises ConfigError
# ---------------------------------------------------------------------------


class TestBadTomlRaises:
    def test_malformed_toml_raises_config_error(self, tmp_path: Path) -> None:
        """Invalid TOML in .source-recall.toml raises ConfigError."""
        from source_recall.config import resolve_config

        toml_path = tmp_path / ".source-recall.toml"
        toml_path.write_text("[source-recall\n")  # Missing closing bracket.

        with pytest.raises(ConfigError):
            resolve_config(tmp_path)


# ---------------------------------------------------------------------------
# Issue #7: FTS escape strips *, ^, -
# ---------------------------------------------------------------------------


class TestFTSEscapeExtended:
    def test_wildcard_stripped(self) -> None:
        """Asterisk (prefix search) is stripped."""
        assert _fts_escape("test*") == '"test"'

    def test_caret_stripped(self) -> None:
        """Caret (boost) is stripped."""
        assert _fts_escape("^foo") == '"foo"'

    def test_hyphen_stripped(self) -> None:
        """Hyphen (NOT shorthand) is stripped."""
        assert _fts_escape("-excluded") == '"excluded"'

    def test_mixed_special_chars(self) -> None:
        """Multiple special chars stripped in one pass."""
        assert _fts_escape('"test*" -foo ^bar') == '"test" "foo" "bar"'


# ---------------------------------------------------------------------------
# Issue #8: insert_chunks batches existing-ID check
# ---------------------------------------------------------------------------


class TestInsertChunksBatchLookup:
    def test_many_chunks_with_branch(self, tmp_path: Path) -> None:
        """Inserting >500 chunks with branch tracking works (batched)."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()

            # Create 600 chunks to exercise the batch-500 code path.
            chunks = [
                ChunkData(
                    file_path=f"file_{i}.py",
                    symbol_name=f"func_{i}",
                    symbol_type=SymbolType.FUNCTION,
                    content=f"def func_{i}(): return {i}",
                    start_line=1,
                    end_line=1,
                )
                for i in range(600)
            ]
            store.insert_chunks(chunks, branch="main")
            assert store.get_chunk_count() == 600

            # Re-insert all with a different branch — tests batch lookup.
            store.insert_chunks(chunks, branch="feature")
            # Count should still be 600 (no duplicates).
            assert store.get_chunk_count() == 600


# ---------------------------------------------------------------------------
# Issue #9: stale_files removed from IndexStatus
# ---------------------------------------------------------------------------


class TestIndexStatusNoStaleFiles:
    def test_no_stale_files_field(self) -> None:
        """IndexStatus no longer has stale_files attribute."""
        from source_recall.models import IndexStatus

        fields = {f.name for f in IndexStatus.__dataclass_fields__.values()}
        assert "stale_files" not in fields


# ---------------------------------------------------------------------------
# Schema migration v5 (narrow trigger)
# ---------------------------------------------------------------------------


class TestMigrationV5:
    def test_migration_from_v4_to_v5(self, tmp_path: Path) -> None:
        """Migration v5 recreates chunks_au trigger with column list."""
        # Create a v4 database with the OLD trigger.
        db_path = tmp_path / "v4.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL DEFAULT '',
                symbol_type TEXT NOT NULL DEFAULT 'block',
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                parent_chunk_id TEXT, sub_chunk_index INTEGER,
                search_quality TEXT NOT NULL DEFAULT 'ast',
                branches TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE file_hashes (
                file_path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                parse_mode TEXT NOT NULL DEFAULT 'ast', mtime_ns INTEGER,
                branch TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content, file_path, symbol_name,
                content='chunks', content_rowid='rowid'
            );
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
                VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
            END;
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
                VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
            END;
            -- OLD trigger (fires on ANY update):
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
                VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
                INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
                VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
            END;
            CREATE TABLE refs (
                source_chunk_id TEXT NOT NULL, target_symbol TEXT NOT NULL,
                ref_type TEXT NOT NULL
            );
            CREATE TABLE symbol_lookup (
                symbol_name TEXT NOT NULL, chunk_id TEXT NOT NULL,
                file_path TEXT NOT NULL
            );
            INSERT INTO meta (key, value) VALUES ('schema_version', '4');
        """)
        conn.close()

        # Run migrations.
        store = IndexStore(db_path)
        store.open()
        store.run_migrations()

        # Schema version should be 5.
        assert store.get_meta("schema_version") == "5"

        # Verify trigger was replaced — check it references specific columns.
        triggers = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='chunks_au'"
        ).fetchone()
        assert triggers is not None
        assert "UPDATE OF" in triggers[0]

        store.close()


# ---------------------------------------------------------------------------
# Issue #11: Dead code removed
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    def test_no_read_content_via_git(self) -> None:
        """_read_content_via_git was removed from IndexBuilder."""
        from source_recall.builder import IndexBuilder

        assert not hasattr(IndexBuilder, "_read_content_via_git")

    def test_no_is_shallow_clone(self) -> None:
        """_is_shallow_clone was removed from IndexBuilder."""
        from source_recall.builder import IndexBuilder

        assert not hasattr(IndexBuilder, "_is_shallow_clone")


# ---------------------------------------------------------------------------
# Issue #14: _now_iso deduplicated
# ---------------------------------------------------------------------------


class TestNowIsoDedup:
    def test_builder_uses_store_now_iso(self) -> None:
        """builder.py imports _now_iso from store.py (no local definition)."""
        import inspect

        import source_recall.builder as builder_mod

        # _now_iso should not be defined in builder module's source.
        source = inspect.getsource(builder_mod)
        assert "def _now_iso" not in source


# ---------------------------------------------------------------------------
# Issue #16: clean command uses IndexStore
# ---------------------------------------------------------------------------


class TestCleanUsesIndexStore:
    def test_clean_no_raw_sqlite3(self) -> None:
        """cli.clean doesn't use raw sqlite3 — imports IndexStore instead."""
        import inspect

        from source_recall.cli import clean

        source = inspect.getsource(clean)
        assert "sqlite3" not in source
        assert "IndexStore" in source
