"""Tests for store.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from source_recall.models import (
    ChunkData,
    FileRecord,
    ParseMode,
    SearchQuality,
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
        assert results[0].symbol_name == "AuthService"

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
        assert results[0].symbol_name == "process_payment"
        assert results[0].score > 0


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
        assert results[0].symbol_name == "UserService"
        assert results[0].score == 100.0

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


class TestTypedSearchRows:
    def test_search_methods_return_search_rows(self, store: IndexStore) -> None:
        """The four search methods return typed SearchRow instances.

        The store->querier boundary is a single frozen dataclass so the
        column->field mapping lives in one place and consumers are
        ty-guarded (finding B).
        """
        from source_recall.models import SearchRow

        chunk = ChunkData(
            file_path="a.py",
            symbol_name="TypedThing",
            symbol_type=SymbolType.CLASS,
            content="class TypedThing:\n    def go(self): ...",
            start_line=1,
            end_line=2,
        )
        store.insert_chunks([chunk])
        store.insert_symbol_lookups([(chunk.chunk_id, "TypedThing", "a.py")])

        fts = store.fts_search("TypedThing")
        sym = store.symbol_search("TypedThing")
        lookup = store.lookup_symbol("TypedThing")

        for rows in (fts, sym, lookup):
            assert len(rows) == 1
            row = rows[0]
            assert isinstance(row, SearchRow)
            assert row.chunk_id == chunk.chunk_id
            assert row.symbol_name == "TypedThing"
            assert row.file_path == "a.py"

        # FTS carries a BM25 score; symbol_search marks exact matches 100.0.
        assert fts[0].score != 0.0
        assert sym[0].score == 100.0


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

    def test_reinsert_without_branch_skips_fts_churn(self, store: IndexStore) -> None:
        """Re-inserting identical chunk without branch skips INSERT OR REPLACE.

        INSERT OR REPLACE fires DELETE + INSERT triggers, generating FTS
        tombstones even when content is unchanged.  The fix: detect existing
        identical chunks and skip the re-insert entirely (H1 audit fix).
        """
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])

        # FTS shadow table row count before re-insert — tombstones grow this.
        fts_data_before = store.conn.execute(
            "SELECT COUNT(*) FROM chunks_fts_data"
        ).fetchone()[0]

        # Re-insert same chunk without branch — should be a no-op.
        store.insert_chunks([chunk])

        # FTS shadow table should not grow (no DELETE+INSERT tombstone pair).
        fts_data_after = store.conn.execute(
            "SELECT COUNT(*) FROM chunks_fts_data"
        ).fetchone()[0]
        assert fts_data_after == fts_data_before
        assert store.get_chunk_count() == 1


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


class TestSymbolLookup:
    @staticmethod
    def _seed_definitions(store: IndexStore, symbol: str, n: int) -> None:
        """Insert ``n`` chunks each registered as a definition of ``symbol``."""
        chunks = [
            ChunkData(
                file_path=f"def{i}.py",
                symbol_name=symbol,
                symbol_type=SymbolType.FUNCTION,
                content=f"def {symbol}(): pass  # {i}",
                start_line=1,
                end_line=1,
            )
            for i in range(n)
        ]
        store.insert_chunks(chunks)
        store.insert_symbol_lookups([(c.chunk_id, symbol, c.file_path) for c in chunks])

    def test_lookup_symbol_returns_all_by_default(self, store: IndexStore) -> None:
        """Without a limit, lookup_symbol returns every matching definition."""
        self._seed_definitions(store, "widget", 5)
        rows = store.lookup_symbol("widget")
        assert len(rows) == 5

    def test_lookup_symbol_respects_limit(self, store: IndexStore) -> None:
        """A limit bounds the number of definitions returned."""
        self._seed_definitions(store, "widget", 5)
        rows = store.lookup_symbol("widget", limit=2)
        assert len(rows) == 2

    def test_lookup_symbols_batches_names(self, store: IndexStore) -> None:
        """lookup_symbols resolves multiple names in one query, deduping names."""
        self._seed_definitions(store, "alpha", 2)
        self._seed_definitions(store, "beta", 3)
        # Duplicate name in the request must not double-count rows.
        rows = store.lookup_symbols(["alpha", "beta", "alpha"])
        assert {r.symbol_name for r in rows} == {"alpha", "beta"}
        assert len(rows) == 5

    def test_lookup_symbols_limit_is_per_name(self, store: IndexStore) -> None:
        """lookup_symbols applies the limit per name, not globally.

        A global LIMIT would let a hot symbol starve later names; the
        per-name cap keeps every requested name represented.
        """
        self._seed_definitions(store, "alpha", 5)
        self._seed_definitions(store, "beta", 5)
        rows = store.lookup_symbols(["alpha", "beta"], limit=2)
        # 2 per name, and results are grouped in request order.
        assert [r.symbol_name for r in rows] == ["alpha", "alpha", "beta", "beta"]

    def test_lookup_symbols_empty(self, store: IndexStore) -> None:
        """An empty name list returns no rows and issues no query."""
        assert store.lookup_symbols([]) == []

    def test_lookup_symbols_beyond_sqlite_variable_limit(
        self, store: IndexStore
    ) -> None:
        """A name set larger than SQLITE_MAX_VARIABLE_NUMBER must not raise.

        A single ``IN (...)`` with >999 placeholders trips SQLite's bound
        parameter cap on some builds; lookup_symbols must batch instead of
        failing the whole graph expansion.
        """
        names = [f"sym_{i:05d}" for i in range(1500)]
        # Register a definition for every other name so grouping has real rows.
        present = names[::2]
        chunks = [
            ChunkData(
                file_path=f"{name}.py",
                symbol_name=name,
                symbol_type=SymbolType.FUNCTION,
                content=f"def {name}(): pass",
                start_line=1,
                end_line=1,
            )
            for name in present
        ]
        store.insert_chunks(chunks)
        store.insert_symbol_lookups(
            [(c.chunk_id, c.symbol_name, c.file_path) for c in chunks]
        )

        rows = store.lookup_symbols(names)

        resolved = {r.symbol_name for r in rows}
        assert resolved == set(present)
        assert len(rows) == len(present)

    def test_lookup_symbols_limit_keeps_highest_quality(
        self, store: IndexStore
    ) -> None:
        """The per-name cap keeps the strongest-quality definitions.

        With more definitions than ``limit`` and differing search_quality,
        the retained rows must be the highest quality, not an arbitrary
        first-N of the unordered result set.
        """
        qualities = [
            SearchQuality.TEXT_FALLBACK,
            SearchQuality.TEXT_FALLBACK,
            SearchQuality.AST,
            SearchQuality.REGEX,
            SearchQuality.TEXT_FALLBACK,
        ]
        chunks = [
            ChunkData(
                file_path=f"def{i}.py",
                symbol_name="widget",
                symbol_type=SymbolType.FUNCTION,
                content=f"def widget(): pass  # {i}",
                start_line=1,
                end_line=1,
                search_quality=q,
            )
            for i, q in enumerate(qualities)
        ]
        store.insert_chunks(chunks)
        store.insert_symbol_lookups(
            [(c.chunk_id, "widget", c.file_path) for c in chunks]
        )

        rows = store.lookup_symbols(["widget"], limit=2)

        assert len(rows) == 2
        # The two strongest qualities (ast, regex) survive the cap.
        assert [r.search_quality for r in rows] == ["ast", "regex"]


class TestHasVecTableMemo:
    def test_has_vec_table_memoized(self, store: IndexStore) -> None:
        """has_vec_table caches its result: once resolved to False it does
        not re-observe a table created outside ensure_vec_table."""
        assert store.has_vec_table() is False
        # Create a table named vec_chunks directly, bypassing the only
        # sanctioned creation path (ensure_vec_table).  The memoized
        # value must not change.
        store.conn.execute("CREATE TABLE vec_chunks (x)")
        assert store.has_vec_table() is False

    def test_ensure_vec_table_invalidates_cache(self, store: IndexStore) -> None:
        """ensure_vec_table drops the memoized negative so the table becomes
        visible on the next has_vec_table() call."""
        pytest.importorskip("sqlite_vec")
        assert store.has_vec_table() is False
        assert store.ensure_vec_table(8) is True
        assert store.has_vec_table() is True


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

    def test_swap_cleans_temp_sidecars(self, tmp_path: Path) -> None:
        """atomic_swap removes the temp file's own WAL/SHM sidecars.

        M-1 fix: the build opens the temp file in WAL mode, so it has
        its own ``<tmp>-wal`` / ``<tmp>-shm`` sidecars.  These must be
        cleaned up after the rename so they don't linger as junk in the
        index directory.
        """
        target = tmp_path / "test.db"
        tmp_db = tmp_path / "test.db.tmp.123"

        # Open the temp in WAL mode (as a real build does) and insert a
        # row so the -wal/-shm sidecars are materialized.
        s = IndexStore(tmp_db, build_mode=True)
        s.open()
        s.create_schema()
        s.conn.execute("INSERT INTO meta (key, value) VALUES ('k', 'v')")
        s.conn.commit()
        s.close()

        tmp_wal = tmp_path / "test.db.tmp.123-wal"
        tmp_shm = tmp_path / "test.db.tmp.123-shm"

        IndexStore.atomic_swap(tmp_db, target)

        assert target.exists()
        assert not tmp_db.exists()
        assert not tmp_wal.exists(), f"temp WAL left orphaned: {tmp_wal}"
        assert not tmp_shm.exists(), f"temp SHM left orphaned: {tmp_shm}"

    def test_swap_tolerates_missing_target_sidecar(self, tmp_path: Path) -> None:
        """A missing target -wal/-shm sidecar does not abort the swap."""
        target = tmp_path / "test.db"
        tmp_db = tmp_path / "test.db.tmp.123"
        s = IndexStore(tmp_db)
        s.open()
        s.create_schema()
        s.close()

        # No target sidecars exist — swap must complete without raising.
        IndexStore.atomic_swap(tmp_db, target)
        assert target.exists()
        assert not tmp_db.exists()

    def test_swap_tolerates_locked_target_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A target sidecar held by an open handle (PermissionError on
        Windows) does not abort the swap; it is left best-effort."""
        target = tmp_path / "test.db"
        stale_wal = tmp_path / "test.db-wal"
        stale_wal.write_text("stale")
        tmp_db = tmp_path / "test.db.tmp.123"
        s = IndexStore(tmp_db)
        s.open()
        s.create_schema()
        s.close()

        real_unlink = os.unlink

        def fake_unlink(path: object, *args: object, **kwargs: object) -> None:
            if str(path).endswith("test.db-wal"):
                raise PermissionError("sidecar held by an open handle")
            real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "unlink", fake_unlink)

        # Must not propagate the PermissionError.
        IndexStore.atomic_swap(tmp_db, target)
        assert target.exists()
        assert not tmp_db.exists()


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

    def test_fresh_schema_records_migration_history(self, store: IndexStore) -> None:
        """M-3: create_schema records baseline rows in schema_migrations.

        A freshly-built DB sets meta.schema_version but must also populate
        schema_migrations so the table is a complete history, not empty.
        """
        rows = store.conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        versions = [r[0] for r in rows]
        # Every migration version (1.._SCHEMA_VERSION) is recorded.
        from source_recall.store import _SCHEMA_VERSION

        assert versions == list(range(1, _SCHEMA_VERSION + 1)), versions

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
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, symbol_name)
                VALUES ('delete', old.rowid, old.content, old.file_path, old.symbol_name);
                INSERT INTO chunks_fts(rowid, content, file_path, symbol_name)
                VALUES (new.rowid, new.content, new.file_path, new.symbol_name);
            END;
            CREATE TABLE file_hashes (
                file_path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                parse_mode TEXT NOT NULL DEFAULT 'ast', mtime_ns INTEGER
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE refs (
                source_chunk_id TEXT NOT NULL, target_symbol TEXT NOT NULL,
                ref_type TEXT NOT NULL
            );
            CREATE TABLE symbol_lookup (
                symbol_name TEXT NOT NULL, chunk_id TEXT NOT NULL,
                file_path TEXT NOT NULL
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
        fh_cols_after = {
            row[1] for row in s.conn.execute("PRAGMA table_info(file_hashes)")
        }
        assert "branch" in fh_cols_after

        # Schema version bumped to latest.
        assert s.get_meta("schema_version") == "5"

    def test_fresh_schema_has_branch_columns(self, store: IndexStore) -> None:
        """Fresh v4 schema includes branches and branch columns."""
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(chunks)")}
        assert "branches" in cols

        fh_cols = {
            row[1] for row in store.conn.execute("PRAGMA table_info(file_hashes)")
        }
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


class TestFTSUpdateTrigger:
    """Behavioral coverage for the chunks_au AFTER UPDATE trigger.

    The INSERT (chunks_ai) and DELETE (chunks_ad) triggers have direct
    behavioral tests above. The UPDATE trigger (chunks_au) was only
    covered by the negative case (branches-only update does not churn
    FTS) and by migration-v5 SQL-text assertions. These tests perform
    real UPDATEs that change FTS-indexed columns and verify the index
    reflects the new content, not stale pre-update content.

    AGENTS.md marks the FTS5 external-content triggers as load-bearing:
    if chunks_au were dropped or narrowed, these tests fail.
    """

    def test_update_content_refreshes_fts(self, store: IndexStore) -> None:
        """UPDATE chunks SET content=... is reflected in FTS search."""
        chunk = ChunkData(
            file_path="svc.py",
            symbol_name="handler",
            symbol_type=SymbolType.FUNCTION,
            content="def handler(): return OLD_TOKEN",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])

        # FTS sees the original content.
        assert len(store.fts_search("OLD_TOKEN")) == 1
        assert len(store.fts_search("NEW_TOKEN")) == 0

        # Raw UPDATE (not INSERT OR REPLACE) exercises chunks_au.
        store.conn.execute(
            "UPDATE chunks SET content = ? WHERE id = ?",
            ("def handler(): return NEW_TOKEN", chunk.chunk_id),
        )

        # FTS must now reflect the new content and forget the old.
        assert len(store.fts_search("NEW_TOKEN")) == 1
        assert len(store.fts_search("OLD_TOKEN")) == 0

    def test_update_symbol_name_refreshes_fts(self, store: IndexStore) -> None:
        """UPDATE chunks SET symbol_name=... is reflected in FTS search.

        Uses a search term that appears ONLY in the symbol_name column
        (never in content) so the test isolates the symbol_name FTS
        column rather than matching the content column.
        """
        chunk = ChunkData(
            file_path="svc.py",
            symbol_name="OldSymbolTag",
            symbol_type=SymbolType.FUNCTION,
            content="def f(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])
        assert len(store.fts_search("OldSymbolTag")) == 1
        assert len(store.fts_search("NewSymbolTag")) == 0

        store.conn.execute(
            "UPDATE chunks SET symbol_name = ? WHERE id = ?",
            ("NewSymbolTag", chunk.chunk_id),
        )

        assert len(store.fts_search("NewSymbolTag")) == 1
        assert len(store.fts_search("OldSymbolTag")) == 0

    def test_update_file_path_refreshes_fts(self, store: IndexStore) -> None:
        """UPDATE chunks SET file_path=... is reflected in FTS search."""
        chunk = ChunkData(
            file_path="old_path.py",
            symbol_name="func",
            symbol_type=SymbolType.FUNCTION,
            content="def func(): pass",
            start_line=1,
            end_line=1,
        )
        store.insert_chunks([chunk])
        assert len(store.fts_search("old_path")) == 1
        assert len(store.fts_search("relocated_file")) == 0

        store.conn.execute(
            "UPDATE chunks SET file_path = ? WHERE id = ?",
            ("relocated_file.py", chunk.chunk_id),
        )

        assert len(store.fts_search("relocated_file")) == 1
        assert len(store.fts_search("old_path")) == 0


class TestAtomicSwapOrdering:
    """Verify the load-bearing ordering of atomic_swap steps.

    AGENTS.md invariant #4 specifies: WAL checkpoint -> fsync ->
    sidecar cleanup -> os.rename -> dir fsync. The existing swap tests
    only assert end states (file renamed, sidecars gone). These tests
    intercept os.fsync, os.rename, and _fsync_dir to verify the
    *sequence*: the data file is fsync'd before it is renamed, and the
    parent directory is fsync'd only after the rename (so the rename is
    durable). A refactor that reordered or dropped these would fail
    here even if the end state looked correct.
    """

    def test_data_fsync_precedes_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.fsync on the temp file runs before os.rename."""
        from source_recall import store as store_mod

        call_log: list[str] = []

        real_fsync = os.fsync
        real_rename = os.rename

        def spy_fsync(fd: int) -> None:
            call_log.append("fsync")
            real_fsync(fd)

        def spy_rename(src: str | os.PathLike, dst: str | os.PathLike) -> None:
            call_log.append("rename")
            real_rename(src, dst)

        monkeypatch.setattr(store_mod.os, "fsync", spy_fsync)
        monkeypatch.setattr(store_mod.os, "rename", spy_rename)

        target = tmp_path / "test.db"
        tmp_db = tmp_path / "test.db.tmp.123"
        s = IndexStore(tmp_db, build_mode=True)
        s.open()
        s.create_schema()
        s.conn.execute("INSERT INTO meta (key, value) VALUES ('k', 'v')")
        s.conn.commit()
        s.close()

        IndexStore.atomic_swap(tmp_db, target)

        # The data fsync must appear before the rename.
        assert "fsync" in call_log, "data fsync never ran"
        assert "rename" in call_log, "rename never ran"
        assert call_log.index("fsync") < call_log.index("rename"), (
            f"data fsync must precede rename, got {call_log}"
        )

    def test_dir_fsync_follows_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_fsync_dir (parent dir durability) runs after os.rename."""
        from source_recall import store as store_mod

        call_log: list[str] = []

        real_rename = os.rename

        def spy_rename(src: str | os.PathLike, dst: str | os.PathLike) -> None:
            call_log.append("rename")
            real_rename(src, dst)

        def spy_fsync_dir(path: Path) -> None:
            call_log.append("fsync_dir")

        monkeypatch.setattr(store_mod.os, "rename", spy_rename)
        monkeypatch.setattr(store_mod, "_fsync_dir", spy_fsync_dir)

        target = tmp_path / "test.db"
        tmp_db = tmp_path / "test.db.tmp.123"
        s = IndexStore(tmp_db, build_mode=True)
        s.open()
        s.create_schema()
        s.close()

        IndexStore.atomic_swap(tmp_db, target)

        assert "rename" in call_log, "rename never ran"
        assert "fsync_dir" in call_log, "dir fsync never ran"
        assert call_log.index("rename") < call_log.index("fsync_dir"), (
            f"dir fsync must follow rename, got {call_log}"
        )
