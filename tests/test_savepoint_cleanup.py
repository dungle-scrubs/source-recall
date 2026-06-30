"""Verify savepoints leave the connection clean after a failure.

SQLite's ``ROLLBACK TO sp`` unwinds changes but **leaves the
savepoint open** in the transaction stack — only ``RELEASE sp``
removes it.  ``run_migrations``, ``_transaction``, and nested
``batch_mode`` all rely on this contract: every failure path must
issue an explicit ``RELEASE sp`` after the rollback so repeated
failures in the same transaction window don't accumulate savepoints.

These tests force migration and savepoint failures and assert the
savepoint stack stays clean (no leftover entries of the same name).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from source_recall.store import IndexStore


def _make_store(tmp_path: Path) -> IndexStore:
    """Build a fresh IndexStore pinned to tmp_path."""
    db = tmp_path / "test.db"
    store = IndexStore(db)
    store.open()
    store.create_schema()
    return store


def _rewind_schema_version(store: IndexStore, version: int) -> None:
    """Force the on-disk schema_version meta to ``version`` so subsequent
    run_migrations will attempt to apply versions in (version, 5]."""
    store.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", str(version)),
    )
    # Clear any migration rows at >= version to mirror a true v3 DB.
    store.conn.execute("DELETE FROM schema_migrations WHERE version >= ?", (version,))
    store.conn.commit()


class TestSavepointReleaseOnRollback:
    def test_failed_migration_releases_savepoint(self, tmp_path: Path) -> None:
        """A failing migration does not leave its savepoint open.

        Subsequent meta writes must still succeed.
        """
        store = _make_store(tmp_path)
        _rewind_schema_version(store, 3)  # triggers migration 4 on next run
        with patch.object(store, "_migrate_004_add_columns") as m:
            m.side_effect = RuntimeError("simulated migration failure")
            with pytest.raises(RuntimeError, match="simulated migration failure"):
                store.run_migrations()

        # The savepoint must have been released — connection still works.
        store.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            ("after_failed_migration", "ok"),
        )
        store.conn.commit()
        row = store.conn.execute(
            "SELECT value FROM meta WHERE key = 'after_failed_migration'"
        ).fetchone()
        assert row is not None
        assert row[0] == "ok"

    def test_failed_transaction_releases_nested_savepoint(self, tmp_path: Path) -> None:
        """A failing nested _transaction inside batch_mode releases its savepoint."""
        store = _make_store(tmp_path)

        with store.batch_mode():
            with pytest.raises(RuntimeError, match="boom"):
                with store._transaction():
                    store.conn.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?)",
                        ("inside_tx", "x"),
                    )
                    raise RuntimeError("boom")

            # Outer batch is still alive — must be able to keep working.
            store.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("after_outer", "ok"),
            )

        # The inner write was rolled back, the outer write committed.
        keys = {row[0] for row in store.conn.execute("SELECT key FROM meta").fetchall()}
        assert "inside_tx" not in keys
        assert "after_outer" in keys

    def test_failed_nested_batch_releases_savepoint(self, tmp_path: Path) -> None:
        """A failing nested batch_mode releases its savepoint."""
        store = _make_store(tmp_path)

        with store.batch_mode():
            with pytest.raises(RuntimeError, match="boom"):
                with store.batch_mode():
                    store.conn.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?)",
                        ("inside_nested", "x"),
                    )
                    raise RuntimeError("boom")

            # Outer batch is still alive — must be able to keep working.
            store.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("after_outer", "ok"),
            )

        keys = {row[0] for row in store.conn.execute("SELECT key FROM meta").fetchall()}
        assert "inside_nested" not in keys
        assert "after_outer" in keys

    def test_run_migrations_succeeds_after_recovered_failure(
        self, tmp_path: Path
    ) -> None:
        """After a simulated failure, a second run_migrations succeeds cleanly.

        If the savepoint had leaked, retrying would attempt
        ``SAVEPOINT migration_4`` again and SQLite would reject the
        nested savepoint of the same name.
        """
        store = _make_store(tmp_path)
        _rewind_schema_version(store, 3)  # triggers migration 4 on next run
        original = store._migrate_004_add_columns
        calls = {"n": 0}

        def failing_then_ok(*args: object, **kwargs: object) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first call fails")
            return original(*args, **kwargs)

        with (
            patch.object(store, "_migrate_004_add_columns", failing_then_ok),
            pytest.raises(RuntimeError),
        ):
            store.run_migrations()

        # Second run should succeed.
        store.run_migrations()
        version = store.get_meta("schema_version")
        assert version is not None

    def test_repeated_failed_migrations_do_not_stack_savepoints(
        self, tmp_path: Path
    ) -> None:
        """Each failed migration must not leave a savepoint on the stack.

        SQLite's ``ROLLBACK TO sp`` retains ``sp`` in the transaction
        stack — it does NOT destroy the savepoint.  Code must explicitly
        ``RELEASE sp`` to clean up.  Without that RELEASE, repeated
        failures inside the same transaction window accumulate
        savepoints.  This test runs 3 failing migrations without any
        intervening commit and verifies that ``RELEASE migration_4``
        succeeds exactly zero times — proving the stack is clean.
        """
        store = _make_store(tmp_path)
        _rewind_schema_version(store, 3)

        for _ in range(3):
            # Re-arm the migration trigger without committing so the
            # savepoint stack from the previous failure is preserved.
            store.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", "3"),
            )
            with patch.object(store, "_migrate_004_add_columns") as m:
                m.side_effect = RuntimeError("boom")
                with pytest.raises(RuntimeError):
                    store.run_migrations()

        # Count how many times RELEASE migration_4 succeeds without an
        # intervening SAVEPOINT.  Without the cleanup fix, three
        # releases succeed (one per failed migration).  With the fix,
        # zero succeed — the savepoint stack is clean.
        successes = 0
        for _ in range(5):
            try:
                store.conn.execute("RELEASE migration_4")
                successes += 1
            except sqlite3.OperationalError:
                break
        assert successes == 0, (
            f"Expected savepoint stack to be empty after 3 failures, "
            f"but {successes} RELEASE migration_4 succeeded "
            f"(leaked {successes} savepoint(s))"
        )
        store.conn.commit()

    def test_repeated_failed_nested_transactions_do_not_stack_savepoints(
        self, tmp_path: Path
    ) -> None:
        """Repeated failing _transaction calls leave no savepoint stack growth.

        Mirrors the migration case but exercises the ``_transaction``
        savepoint path.  Without an explicit RELEASE after ROLLBACK TO,
        each failure leaks one ``sp_N`` onto the stack.
        """
        store = _make_store(tmp_path)
        # Wrap in batch_mode so the inner _transaction savepoints stay
        # alive long enough to observe the leak.
        with store.batch_mode():
            for _ in range(3):
                with pytest.raises(RuntimeError), store._transaction():
                    raise RuntimeError("boom")

            # Locate the highest ``sp_N`` savepoint that survived. Each
            # failure leaks one in the buggy code; the fix prevents any.
            n_released = 0
            for i in range(1, 10):
                try:
                    store.conn.execute(f"RELEASE sp_{i}")
                    n_released += 1
                except sqlite3.OperationalError:
                    break
            assert n_released == 0, (
                f"Leftover _transaction savepoints on stack: {n_released}"
            )
