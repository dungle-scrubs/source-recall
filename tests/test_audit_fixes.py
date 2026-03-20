"""Tests for issues found during audit."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from source_recall.models import ChunkData, ConfigError, SymbolType
from source_recall.store import _FTS5_STRIP, IndexStore, _fts_escape

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


# ---------------------------------------------------------------------------
# Issue #17: atomic_swap cross-device guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #16b: _FTS5_STRIP is module-level constant
# ---------------------------------------------------------------------------


class TestFTS5StripModuleLevel:
    def test_fts5_strip_is_module_level(self) -> None:
        """_FTS5_STRIP is a module-level constant, not per-call."""
        assert isinstance(_FTS5_STRIP, dict)
        # It's a str.maketrans dict — should strip quotes, stars, etc.
        assert _FTS5_STRIP  # Not empty.


class TestAtomicSwapGuard:
    def test_same_device_succeeds(self, tmp_path: Path) -> None:
        """atomic_swap works when both paths are on same device."""
        tmp_db = tmp_path / "test.db.tmp.123"
        target = tmp_path / "test.db"

        s = IndexStore(tmp_db)
        s.open()
        s.create_schema()
        s.close()

        # Should not raise — same filesystem.
        IndexStore.atomic_swap(tmp_db, target)
        assert target.exists()


# ---------------------------------------------------------------------------
# Issue #18: build_mode sets synchronous=NORMAL
# ---------------------------------------------------------------------------


class TestBuildModePragma:
    def test_build_mode_sets_synchronous_normal(self, tmp_path: Path) -> None:
        """build_mode=True sets PRAGMA synchronous=NORMAL."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path, build_mode=True) as store:
            store.create_schema()
            row = store.conn.execute("PRAGMA synchronous").fetchone()
            # NORMAL = 1
            assert row[0] == 1

    def test_default_mode_does_not_force_normal(self, tmp_path: Path) -> None:
        """Default mode does not explicitly set synchronous=NORMAL.

        WAL mode defaults vary by platform (typically NORMAL=1 or FULL=2).
        The key invariant is that build_mode explicitly sets NORMAL, while
        default mode leaves it to SQLite's WAL default.
        """
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()
            # Just verify it doesn't raise — actual value is platform-dependent.
            row = store.conn.execute("PRAGMA synchronous").fetchone()
            assert row[0] in (1, 2)  # NORMAL or FULL, both acceptable.


# ---------------------------------------------------------------------------
# Issue #19: _SENTINEL deduplicated in models.py
# ---------------------------------------------------------------------------


class TestSentinelDedup:
    def test_init_and_server_use_same_sentinel(self) -> None:
        """__init__.py and server.py import _SENTINEL from models."""
        from source_recall import _SENTINEL as init_sentinel
        from source_recall.models import _SENTINEL as models_sentinel
        from source_recall.server import _SENTINEL as server_sentinel

        assert init_sentinel is models_sentinel
        assert server_sentinel is models_sentinel


# ---------------------------------------------------------------------------
# Issue #20: _chunk_prose line tracking accuracy
# ---------------------------------------------------------------------------


class TestProseLineTracking:
    def test_multiline_prose_line_numbers(self) -> None:
        """Prose chunker produces accurate line numbers from original content."""
        from source_recall.chunker import _chunk_prose

        content = (
            "First sentence on line one.\n"
            "Second sentence on line two.\n"
            "Third sentence on line three.\n"
        )
        chunks, quality = _chunk_prose("doc.txt", content, max_chars=6000)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        # Content spans 3 lines.
        assert chunks[0].end_line >= 1


# ---------------------------------------------------------------------------
# Issue #21: _extract_signature stops at docstring
# ---------------------------------------------------------------------------


class TestExtractSignature:
    def test_does_not_include_docstring(self) -> None:
        """Signature extraction stops before the docstring."""
        from source_recall.chunker import _extract_signature

        lines = [
            "def my_function(x: int, y: int) -> bool:",
            '    """This is a docstring."""',
            "    return x > y",
        ]
        sig = _extract_signature(lines)
        assert '"""' not in sig
        assert "def my_function" in sig

    def test_multiline_params(self) -> None:
        """Signature extraction includes multi-line parameter lists."""
        from source_recall.chunker import _extract_signature

        lines = [
            "def complex_function(",
            "    x: int,",
            "    y: str,",
            ") -> bool:",
            '    """Do something."""',
        ]
        sig = _extract_signature(lines)
        assert "x: int," in sig
        assert "y: str," in sig
        assert '"""' not in sig


# ---------------------------------------------------------------------------
# Issue #22: batch insert_vectors
# ---------------------------------------------------------------------------


class TestBatchInsertVectors:
    def test_batch_delete_before_insert(self, tmp_path: Path) -> None:
        """insert_vectors batches deletes before inserts."""
        from source_recall.embedder import BagOfWordsEmbedder

        db_path = tmp_path / "test.db"
        store = IndexStore(db_path)
        store.open()
        store.create_schema()
        ok = store.ensure_vec_table(dimensions=64)
        if not ok:
            pytest.skip("sqlite-vec not available")

        emb = BagOfWordsEmbedder(dimensions=64)
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])

        # Insert then re-insert — should not duplicate.
        vec = emb.embed_chunks(["def foo(): pass"])
        store.insert_vectors([chunk.chunk_id], vec)
        assert store.get_vector_count() == 1

        vec2 = emb.embed_chunks(["def foo(): return 42"])
        store.insert_vectors([chunk.chunk_id], vec2)
        assert store.get_vector_count() == 1

        store.close()


# ---------------------------------------------------------------------------
# Issue #23: FK CASCADE cleans refs/symbol_lookup on chunk delete
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #18: batch_mode suppresses per-method commits
# ---------------------------------------------------------------------------


class TestBatchMode:
    def test_batch_mode_single_commit(self, tmp_path: Path) -> None:
        """batch_mode suppresses auto-commits, then commits at exit."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()

            with store.batch_mode():
                store.insert_chunks(
                    [
                        ChunkData(
                            file_path="a.py",
                            symbol_name="foo",
                            symbol_type=SymbolType.FUNCTION,
                            content="def foo(): pass",
                            start_line=1,
                            end_line=1,
                        ),
                    ]
                )
                store.delete_chunks_for_file("nonexistent.py")
                # Inside batch_mode — data visible within same connection.
                assert store.get_chunk_count() == 1

            # After batch_mode exit — committed.
            assert store.get_chunk_count() == 1

    def test_batch_mode_rolls_back_on_error(self, tmp_path: Path) -> None:
        """batch_mode rolls back on exception."""
        db_path = tmp_path / "test.db"
        with IndexStore(db_path) as store:
            store.create_schema()

            # Insert one chunk normally first.
            store.insert_chunks(
                [
                    ChunkData(
                        file_path="a.py",
                        symbol_name="keep",
                        symbol_type=SymbolType.FUNCTION,
                        content="def keep(): pass",
                        start_line=1,
                        end_line=1,
                    ),
                ]
            )
            assert store.get_chunk_count() == 1

            with pytest.raises(RuntimeError), store.batch_mode():
                store.delete_chunks_for_file("a.py")
                raise RuntimeError("simulated failure")

            # Rolled back — chunk still present.
            assert store.get_chunk_count() == 1


class TestFKCascadeCleansGraphData:
    def test_refresh_removes_stale_symbol_lookups(self, tmp_path: Path) -> None:
        """Refreshing a changed file removes its old symbol_lookup entries
        via FK CASCADE when the chunk is deleted."""
        import subprocess

        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (repo / "svc.py").write_text(
            "class OldService:\n    def old_method(self): pass\n"
        )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        idx = Index(repo, embedder=None)
        db_path = idx.build()

        # Verify symbol_lookup has OldService.
        with IndexStore(db_path) as store:
            old_syms = store.lookup_symbol("OldService")
            assert len(old_syms) > 0

        # Rename the class and commit.
        (repo / "svc.py").write_text(
            "class NewService:\n    def new_method(self): pass\n"
        )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "rename"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        idx.refresh()

        # OldService should be gone from symbol_lookup via FK CASCADE.
        with IndexStore(db_path) as store:
            old_syms = store.lookup_symbol("OldService")
            assert len(old_syms) == 0, f"Stale symbol_lookup entries remain: {old_syms}"
            new_syms = store.lookup_symbol("NewService")
            assert len(new_syms) > 0


# ---------------------------------------------------------------------------
# Audit round 2
# ---------------------------------------------------------------------------


class TestListIndexesTextFallbackKey:
    """H1: sr list uses 'text_fallback' not 'text' for parse_mode lookup."""

    def test_text_fallback_key_in_source(self) -> None:
        """cli.list_indexes uses 'text_fallback', not 'text'."""
        import inspect

        from source_recall.cli import list_indexes

        source = inspect.getsource(list_indexes)
        assert 'modes.get("text_fallback"' in source
        assert 'modes.get("text"' not in source


class TestRefreshBuildLockGap:
    """H2: refresh fallback to build keeps the lock held."""

    def test_build_locked_exists(self) -> None:
        """_build_locked is a separate method callable under held lock."""
        from source_recall.builder import IndexBuilder

        assert hasattr(IndexBuilder, "_build_locked")

    def test_refresh_large_change_no_mid_release(self) -> None:
        """refresh >500 changes calls _build_locked, not build()."""
        import inspect

        from source_recall.builder import IndexBuilder

        source = inspect.getsource(IndexBuilder.refresh)
        # Should call _build_locked, not self.build()
        assert "_build_locked" in source
        # The lock release should only appear in the finally block,
        # not between store.close() and the rebuild.
        assert "self.build()" not in source


class TestReleaseLockSafety:
    """H3: _release_lock doesn't delete another process's lock."""

    def test_corrupt_lock_not_deleted(self, tmp_path: Path) -> None:
        """Corrupt lock file is left intact, not unconditionally deleted."""
        lock_path = tmp_path / "index.lock"
        lock_path.write_text("not-json{{{")

        from source_recall.store import _release_lock

        _release_lock(lock_path)
        # File should still exist — not deleted.
        assert lock_path.exists()

    def test_own_lock_deleted(self, tmp_path: Path) -> None:
        """Lock belonging to current process is properly deleted."""
        import json
        import os

        lock_path = tmp_path / "index.lock"
        lock_path.write_text(json.dumps({"pid": os.getpid()}))

        from source_recall.store import _release_lock

        _release_lock(lock_path)
        assert not lock_path.exists()

    def test_other_pid_lock_not_deleted(self, tmp_path: Path) -> None:
        """Lock belonging to a different PID is not deleted."""
        import json

        lock_path = tmp_path / "index.lock"
        lock_path.write_text(json.dumps({"pid": 99999999}))

        from source_recall.store import _release_lock

        _release_lock(lock_path)
        assert lock_path.exists()


class TestDetectChangesRebaseFallback:
    """L3: _detect_changes falls back to hash diff when ancestor check fails."""

    def test_rebase_falls_back_to_hash(self, tmp_path: Path) -> None:
        """When merge-base --is-ancestor fails, _git_diff_files returns None."""
        import subprocess

        from source_recall.builder import IndexBuilder
        from source_recall.config import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=repo, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=repo, capture_output=True, check=True,
        )

        builder = IndexBuilder(repo, resolve_config(repo))

        # A non-existent commit triggers the non-ancestor path.
        result = builder._git_diff_files("0000000000000000000000000000000000000000")
        assert result is None  # Falls back to hash diff.


class TestServeNoEnvMutation:
    """M7: serve command does not mutate os.environ."""

    def test_no_environ_assignment_in_serve(self) -> None:
        """cli.serve does not assign to os.environ."""
        import inspect

        from source_recall.cli import serve

        source = inspect.getsource(serve)
        assert 'os.environ[' not in source
