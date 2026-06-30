"""Verify savepoints are released even on rollback.

SQLite's ``ROLLBACK TO sp`` unwinds changes but **leaves the savepoint
open**. The original code called ``ROLLBACK TO`` and then re-raised
without ``RELEASE``, leaking a savepoint on every failure.

These tests force a migration / savepoint failure and assert that
subsequent operations work normally.
"""

from __future__ import annotations

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
    store.conn.execute(
        "DELETE FROM schema_migrations WHERE version >= ?", (version,)
    )
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

    def test_failed_transaction_releases_nested_savepoint(
        self, tmp_path: Path
    ) -> None:
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
        keys = {
            row[0]
            for row in store.conn.execute("SELECT key FROM meta").fetchall()
        }
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

        keys = {
            row[0]
            for row in store.conn.execute("SELECT key FROM meta").fetchall()
        }
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

        with patch.object(
            store, "_migrate_004_add_columns", failing_then_ok
        ):
            with pytest.raises(RuntimeError):
                store.run_migrations()

        # Second run should succeed.
        store.run_migrations()
        version = store.get_meta("schema_version")
        assert version is not None