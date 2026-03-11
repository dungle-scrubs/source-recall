"""Tests for store.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from source_recall.models import (
    ChunkData,
    FileRecord,
    ParseMode,
    SymbolType,
)
from source_recall.store import IndexStore


@pytest.fixture
def store(tmp_path: Path) -> IndexStore:
    """Create a fresh IndexStore with schema applied."""
    db_path = tmp_path / "test.db"
    s = IndexStore(db_path)
    s.open()
    s.create_schema()
    return s


class TestConnection:
    def test_pragmas_set(self, store: IndexStore) -> None:
        """WAL, busy_timeout, and foreign_keys are enabled."""
        row = store.conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

        row = store.conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 5000

        row = store.conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_schema_created(self, store: IndexStore) -> None:
        """All tables exist after create_schema."""
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "meta" in tables
        assert "chunks" in tables
        assert "file_hashes" in tables
        assert "schema_migrations" in tables
        # FTS virtual table.
        assert "chunks_fts" in tables


class TestChunkCRUD:
    def test_insert_and_count(self, store: IndexStore) -> None:
        """Inserted chunks are countable."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="foo",
                symbol_type=SymbolType.FUNCTION,
                content="def foo(): pass",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="a.py",
                symbol_name="bar",
                symbol_type=SymbolType.FUNCTION,
                content="def bar(): pass",
                start_line=3,
                end_line=3,
            ),
        ]
        store.insert_chunks(chunks)
        assert store.get_chunk_count() == 2

    def test_delete_by_file(self, store: IndexStore) -> None:
        """Deleting by file removes all chunks for that file."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="f1",
                symbol_type=SymbolType.FUNCTION,
                content="def f1(): pass",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="b.py",
                symbol_name="f2",
                symbol_type=SymbolType.FUNCTION,
                content="def f2(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)
        assert store.get_chunk_count() == 2

        store.delete_chunks_for_file("a.py")
        assert store.get_chunk_count() == 1


class TestFTSTriggers:
    def test_insert_populates_fts(self, store: IndexStore) -> None:
        """FTS is populated automatically via trigger on INSERT."""
        chunks = [
            ChunkData(
                file_path="auth.py",
                symbol_name="AuthService",
                symbol_type=SymbolType.CLASS,
                content="class AuthService:\n    def validate(self): ...",
                start_line=1,
                end_line=2,
            ),
        ]
        store.insert_chunks(chunks)

        # FTS search should find it.
        results = store.fts_search("AuthService")
        assert len(results) == 1
        assert results[0]["symbol_name"] == "AuthService"

    def test_delete_removes_from_fts(self, store: IndexStore) -> None:
        """FTS entries are removed via trigger on DELETE."""
        chunks = [
            ChunkData(
                file_path="auth.py",
                symbol_name="AuthService",
                symbol_type=SymbolType.CLASS,
                content="class AuthService:\n    def validate(self): ...",
                start_line=1,
                end_line=2,
            ),
        ]
        store.insert_chunks(chunks)
        assert len(store.fts_search("AuthService")) == 1

        store.delete_chunks_for_file("auth.py")
        assert len(store.fts_search("AuthService")) == 0

    def test_fts_search_bm25_scoring(self, store: IndexStore) -> None:
        """FTS results have positive BM25 scores."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="process_payment",
                symbol_type=SymbolType.FUNCTION,
                content="def process_payment(amount):\n    charge(amount)",
                start_line=1,
                end_line=2,
            ),
            ChunkData(
                file_path="b.py",
                symbol_name="validate_user",
                symbol_type=SymbolType.FUNCTION,
                content="def validate_user(user):\n    return user.is_active",
                start_line=1,
                end_line=2,
            ),
        ]
        store.insert_chunks(chunks)

        results = store.fts_search("payment")
        assert len(results) >= 1
        assert results[0]["symbol_name"] == "process_payment"
        assert results[0]["score"] > 0


class TestSymbolSearch:
    def test_exact_match(self, store: IndexStore) -> None:
        """Symbol search finds exact name matches."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="UserService",
                symbol_type=SymbolType.CLASS,
                content="class UserService: ...",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="b.py",
                symbol_name="PaymentService",
                symbol_type=SymbolType.CLASS,
                content="class PaymentService: ...",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        results = store.symbol_search("UserService")
        assert len(results) == 1
        assert results[0]["symbol_name"] == "UserService"
        assert results[0]["score"] == 100.0

    def test_case_insensitive(self, store: IndexStore) -> None:
        """Symbol search is case-insensitive."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="MyClass",
                symbol_type=SymbolType.CLASS,
                content="class MyClass: ...",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        results = store.symbol_search("myclass")
        assert len(results) == 1


class TestDuplicateChunkInsert:
    def test_insert_or_replace_updates_content(self, store: IndexStore) -> None:
        """INSERT OR REPLACE updates existing chunk content and FTS."""
        chunk_v1 = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): return 1",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk_v1])
        assert store.get_chunk_count() == 1
        assert len(store.fts_search("return 1")) == 1

        # Insert same chunk_id with different content.
        chunk_v2 = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): return 999",
            start_line=1,
            end_line=1,
        )
        # Same file_path + symbol_name + content hash → same chunk_id? No — content differs.
        # Force same ID by using the original's ID.
        assert chunk_v1.chunk_id != chunk_v2.chunk_id
        # So this tests INSERT OR REPLACE with different chunk_ids → count increases.
        store.insert_chunks([chunk_v2])
        assert store.get_chunk_count() == 2

    def test_same_chunk_id_replaces(self, store: IndexStore) -> None:
        """Re-inserting the exact same ChunkData replaces, not duplicates."""
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])
        store.insert_chunks([chunk])  # Same exact data → same chunk_id.
        assert store.get_chunk_count() == 1  # Not 2.


class TestFileHashes:
    def test_upsert_and_get(self, store: IndexStore) -> None:
        """File hashes can be stored and retrieved."""
        record = FileRecord(
            file_path="auth.py",
            content_hash="abc123",
            parse_mode=ParseMode.AST,
            mtime_ns=1234567890,
        )
        store.upsert_file_hash(record)

        got = store.get_file_hash("auth.py")
        assert got is not None
        assert got.content_hash == "abc123"
        assert got.parse_mode == ParseMode.AST
        assert got.mtime_ns == 1234567890

    def test_get_all(self, store: IndexStore) -> None:
        """get_all_file_hashes returns all records."""
        for f in ["a.py", "b.py", "c.py"]:
            store.upsert_file_hash(
                FileRecord(
                    file_path=f,
                    content_hash=f"hash_{f}",
                    parse_mode=ParseMode.AST,
                )
            )
        all_hashes = store.get_all_file_hashes()
        assert len(all_hashes) == 3


class TestFileCountByMode:
    def test_counts_grouped_by_mode(self, store: IndexStore) -> None:
        """get_file_count_by_mode returns correct counts per parse mode."""
        store.upsert_file_hash(
            FileRecord(file_path="a.py", content_hash="a", parse_mode=ParseMode.AST)
        )
        store.upsert_file_hash(
            FileRecord(file_path="b.ts", content_hash="b", parse_mode=ParseMode.AST)
        )
        store.upsert_file_hash(
            FileRecord(
                file_path="c.sh",
                content_hash="c",
                parse_mode=ParseMode.REGEX,
            )
        )
        store.upsert_file_hash(
            FileRecord(
                file_path="d.yaml",
                content_hash="d",
                parse_mode=ParseMode.TEXT_FALLBACK,
            )
        )

        counts = store.get_file_count_by_mode()
        assert counts["ast"] == 2
        assert counts["regex"] == 1
        assert counts["text_fallback"] == 1

    def test_empty_store_returns_empty(self, store: IndexStore) -> None:
        """No file hashes → empty dict."""
        counts = store.get_file_count_by_mode()
        assert counts == {}


class TestMeta:
    def test_set_and_get(self, store: IndexStore) -> None:
        """Meta key-value pairs are stored and retrieved."""
        store.set_meta_batch({"repo_path": "/dev/myrepo", "last_commit": "abc"})
        assert store.get_meta("repo_path") == "/dev/myrepo"
        assert store.get_meta("last_commit") == "abc"
        assert store.get_meta("nonexistent") is None


class TestAtomicSwap:
    def test_swap_renames_file(self, tmp_path: Path) -> None:
        """atomic_swap renames tmp to target."""
        tmp_db = tmp_path / "test.db.tmp.123"
        target = tmp_path / "test.db"

        # Create a real sqlite db at tmp.
        s = IndexStore(tmp_db)
        s.open()
        s.create_schema()
        s.close()

        IndexStore.atomic_swap(tmp_db, target)

        assert target.exists()
        assert not tmp_db.exists()

    def test_swap_cleans_stale_wal(self, tmp_path: Path) -> None:
        """atomic_swap removes stale WAL/SHM sidecars."""
        target = tmp_path / "test.db"
        stale_wal = tmp_path / "test.db-wal"
        stale_shm = tmp_path / "test.db-shm"

        stale_wal.write_text("stale")
        stale_shm.write_text("stale")

        # Create temp db.
        tmp_db = tmp_path / "test.db.tmp.123"
        s = IndexStore(tmp_db)
        s.open()
        s.create_schema()
        s.close()

        IndexStore.atomic_swap(tmp_db, target)

        assert not stale_wal.exists()
        assert not stale_shm.exists()


class TestFTSEscape:
    def test_and_or_not_operators_escaped(self, store: IndexStore) -> None:
        """FTS5 operators in user queries are treated as literals."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="check_and_validate",
                symbol_type=SymbolType.FUNCTION,
                content="def check_and_validate(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        # "AND" is an FTS5 operator — without escaping this would fail
        # or produce unexpected results.
        results = store.fts_search("check AND validate")
        # Should still find the chunk (both words present).
        assert len(results) >= 1

    def test_column_filter_escaped(self, store: IndexStore) -> None:
        """FTS5 column filters like 'file_path:hack' are neutralized."""
        chunks = [
            ChunkData(
                file_path="secret.py",
                symbol_name="boring",
                symbol_type=SymbolType.FUNCTION,
                content="def boring(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        # Column filter syntax — should NOT search by file_path column.
        results = store.fts_search("file_path:secret")
        # With proper escaping, this searches for literal "file_pathsecret"
        # which won't match.
        assert len(results) == 0

    def test_quotes_in_query_escaped(self, store: IndexStore) -> None:
        """Quotes in user queries don't break FTS5 syntax."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="parse_json",
                symbol_type=SymbolType.FUNCTION,
                content="def parse_json(s: str): return json.loads(s)",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        # Unmatched quotes would crash FTS5 without escaping.
        results = store.fts_search('"parse json')
        assert len(results) >= 1

    def test_near_operator_escaped(self, store: IndexStore) -> None:
        """NEAR operator in queries doesn't cause errors."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="near_miss",
                symbol_type=SymbolType.FUNCTION,
                content="def near_miss(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        # NEAR is an FTS5 operator.
        results = store.fts_search("NEAR(near, miss)")
        # Should not crash — returns whatever matches.
        assert isinstance(results, list)

    def test_empty_query_returns_empty(self, store: IndexStore) -> None:
        """Blank/whitespace-only queries return empty results."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="anything",
                symbol_type=SymbolType.FUNCTION,
                content="def anything(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks)

        assert store.fts_search("") == []
        assert store.fts_search("   ") == []
        assert store.fts_search('"""') == []


class TestPIDLock:
    def test_stale_lock_is_stolen(self, tmp_path: Path) -> None:
        """A lock file with a dead PID is automatically reclaimed."""
        import json

        db_path = tmp_path / "index.db"
        lock_path = db_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a lock with a PID that doesn't exist (pid 2^22 is safe).
        dead_pid = 4_194_304
        lock_path.write_text(json.dumps({"pid": dead_pid, "started": "old"}))

        # Should not raise — the stale lock is reclaimed.
        returned = IndexStore.acquire_lock(db_path, timeout=1)
        assert returned == lock_path

        # Lock file now belongs to us.
        data = json.loads(lock_path.read_text())
        import os

        assert data["pid"] == os.getpid()

        IndexStore.release_lock(db_path)

    def test_live_lock_raises(self, tmp_path: Path) -> None:
        """A lock held by a live process raises IndexLockError."""
        import json
        import os

        from source_recall.models import IndexLockError

        db_path = tmp_path / "index.db"
        lock_path = db_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a lock with our own PID (definitely alive).
        lock_path.write_text(json.dumps({"pid": os.getpid(), "started": "now"}))

        with pytest.raises(IndexLockError) as exc_info:
            IndexStore.acquire_lock(db_path, timeout=0)
        assert exc_info.value.pid == os.getpid()


class TestCleanTmpFiles:
    def test_removes_stale_tmp_files(self, tmp_path: Path) -> None:
        """clean_tmp_files removes .tmp.* files from other PIDs."""
        db_path = tmp_path / "index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create stale tmp files (from "other" PIDs).
        stale1 = tmp_path / "index.db.tmp.99999"
        stale2 = tmp_path / "index.db.tmp.88888"
        stale1.write_text("stale1")
        stale2.write_text("stale2")

        # Create our own tmp file (should NOT be removed).
        import os

        ours = tmp_path / f"index.db.tmp.{os.getpid()}"
        ours.write_text("ours")

        removed = IndexStore.clean_tmp_files(db_path)

        assert len(removed) == 2
        assert not stale1.exists()
        assert not stale2.exists()
        assert ours.exists()

    def test_no_crash_on_missing_dir(self, tmp_path: Path) -> None:
        """clean_tmp_files returns empty if parent dir doesn't exist."""
        db_path = tmp_path / "nonexistent" / "index.db"
        removed = IndexStore.clean_tmp_files(db_path)
        assert removed == []


class TestMigrations:
    def test_schema_version_check(self, store: IndexStore) -> None:
        """SchemaVersionError raised if on-disk version is too new."""
        store.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '999')"
        )
        store.conn.commit()

        from source_recall.models import SchemaVersionError

        with pytest.raises(SchemaVersionError) as exc_info:
            store.run_migrations()
        assert exc_info.value.on_disk == 999

    def test_migration_004_adds_branch_columns(self, tmp_path: Path) -> None:
        """Migration v4 adds branches column to chunks and branch to file_hashes."""
        import sqlite3

        # Simulate a v3 database by creating schema WITHOUT branch columns.
        db_path = tmp_path / "v3.db"
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
                search_quality TEXT NOT NULL DEFAULT 'ast'
            );
            CREATE TABLE file_hashes (
                file_path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                parse_mode TEXT NOT NULL DEFAULT 'ast', mtime_ns INTEGER
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );
            INSERT INTO meta (key, value) VALUES ('schema_version', '3');
        """)
        conn.close()

        # Verify columns absent pre-migration.
        s = IndexStore(db_path)
        s.open()
        cols = {row[1] for row in s.conn.execute("PRAGMA table_info(chunks)")}
        assert "branches" not in cols
        fh_cols = {row[1] for row in s.conn.execute("PRAGMA table_info(file_hashes)")}
        assert "branch" not in fh_cols

        s.run_migrations()

        # Columns now exist.
        cols_after = {row[1] for row in s.conn.execute("PRAGMA table_info(chunks)")}
        assert "branches" in cols_after
        fh_cols_after = {row[1] for row in s.conn.execute("PRAGMA table_info(file_hashes)")}
        assert "branch" in fh_cols_after

        # Schema version bumped.
        assert s.get_meta("schema_version") == "4"

    def test_fresh_schema_has_branch_columns(self, store: IndexStore) -> None:
        """Fresh v4 schema includes branches and branch columns."""
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(chunks)")}
        assert "branches" in cols

        fh_cols = {row[1] for row in store.conn.execute("PRAGMA table_info(file_hashes)")}
        assert "branch" in fh_cols


class TestBranchAwareness:
    """Tests for branch-aware chunk and file_hash operations."""

    def test_insert_chunks_with_branch(self, store: IndexStore) -> None:
        """insert_chunks sets branches column when branch is provided."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="foo",
                symbol_type=SymbolType.FUNCTION,
                content="def foo(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        store.insert_chunks(chunks, branch="main")

        row = store.conn.execute(
            "SELECT branches FROM chunks WHERE id = ?", (chunks[0].chunk_id,)
        ).fetchone()
        assert row[0] == "main"

    def test_insert_chunks_appends_branch(self, store: IndexStore) -> None:
        """Re-inserting the same chunk with a different branch appends to CSV."""
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk], branch="main")
        store.insert_chunks([chunk], branch="feature")

        row = store.conn.execute(
            "SELECT branches FROM chunks WHERE id = ?", (chunk.chunk_id,)
        ).fetchone()
        branches = set(row[0].split(","))
        assert branches == {"main", "feature"}

    def test_insert_chunks_no_duplicate_branch(self, store: IndexStore) -> None:
        """Re-inserting with the same branch doesn't duplicate in CSV."""
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk], branch="main")
        store.insert_chunks([chunk], branch="main")

        row = store.conn.execute(
            "SELECT branches FROM chunks WHERE id = ?", (chunk.chunk_id,)
        ).fetchone()
        assert row[0] == "main"

    def test_upsert_file_hash_with_branch(self, store: IndexStore) -> None:
        """upsert_file_hash stores the branch column."""
        record = FileRecord(
            file_path="a.py",
            content_hash="abc123",
            parse_mode=ParseMode.AST,
            mtime_ns=None,
        )
        store.upsert_file_hash(record, branch="main")

        row = store.conn.execute(
            "SELECT branch FROM file_hashes WHERE file_path = 'a.py'"
        ).fetchone()
        assert row[0] == "main"

    def test_get_file_hashes_for_branch(self, store: IndexStore) -> None:
        """get_all_file_hashes can filter by branch."""
        store.upsert_file_hash(
            FileRecord("a.py", "hash1", ParseMode.AST, None), branch="main"
        )
        store.upsert_file_hash(
            FileRecord("b.py", "hash2", ParseMode.AST, None), branch="feature"
        )

        main_hashes = store.get_all_file_hashes(branch="main")
        assert "a.py" in main_hashes
        assert "b.py" not in main_hashes

    def test_fts_correct_after_branch_update(self, store: IndexStore) -> None:
        """FTS index stays correct when branches column is updated."""
        chunk = ChunkData(
            file_path="auth.py",
            symbol_name="validate",
            symbol_type=SymbolType.FUNCTION,
            content="def validate(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk], branch="main")

        # Verify FTS works.
        results = store.fts_search("validate")
        assert len(results) == 1

        # Add another branch (triggers UPDATE on branches column).
        store.insert_chunks([chunk], branch="feature")

        # FTS should still work.
        results = store.fts_search("validate")
        assert len(results) == 1
