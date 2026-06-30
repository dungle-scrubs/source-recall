"""``.tmp.<pid>`` files must be cleaned up when a build fails.

The original code created a tmp file in ``_build_locked`` and only
called ``atomic_swap`` on the happy path. If the build crashed mid-way
(e.g. embedder OOM, chunker exception), the tmp file was left behind.
A ``finally: tmp_path.unlink(missing_ok=True)`` guarantees cleanup
without disturbing atomic_swap's success path.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config


def test_failed_build_removes_tmp_file(tmp_path: Path) -> None:
    """A failing build leaves no ``.tmp.<pid>`` file behind."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("x = 1\n", encoding="utf-8")

    builder = IndexBuilder(repo, resolve_config())

    # Force an error during the build's per-file indexing.
    with patch.object(builder, "_index_file") as m:
        m.side_effect = RuntimeError("simulated index failure")

        with pytest.raises(RuntimeError, match="simulated index failure"):
            builder.build()

    # The tmp file MUST NOT exist.
    pid = os.getpid()
    leftover = list((repo / ".source-recall").glob(f"index.db.tmp.{pid}*"))
    # The repo's index dir is monkeypatched into tmp_path via conftest.
    # The IndexBuilder writes to <tmp_path>/.source-recall/index.db.tmp.<pid>
    from source_recall.store import get_db_path

    db_path = get_db_path(repo.resolve())
    tmp_path_expected = db_path.parent / f"{db_path.name}.tmp.{pid}"
    assert not tmp_path_expected.exists(), (
        f"Build left tmp file behind: {tmp_path_expected}"
    )
    assert leftover == []


def test_successful_build_consumes_tmp(tmp_path: Path) -> None:
    """On success, atomic_swap renames the tmp into place — no leftover."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("x = 1\n", encoding="utf-8")

    builder = IndexBuilder(repo, resolve_config())
    builder.build()

    pid = os.getpid()
    from source_recall.store import get_db_path

    db_path = get_db_path(repo.resolve())
    leftover = list(db_path.parent.glob(f"index.db.tmp.{pid}*"))
    assert leftover == [], f"Success left tmp file behind: {leftover}"
    assert db_path.exists(), "Final index.db must exist after successful build"


def test_failed_build_removes_tmp_sidecars(tmp_path: Path) -> None:
    """A failing build also removes the tmp file's WAL/SHM sidecars.

    When ``_build_core`` raises after opening the temp index in WAL
    mode, the temp database has ``.tmp.<pid>-wal`` and
    ``.tmp.<pid>-shm`` sidecars that ``atomic_swap`` would normally
    clean.  Since ``atomic_swap`` only runs on success, the failure
    path must also unlink those sidecars or they leak across builds.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("x = 1\n", encoding="utf-8")

    builder = IndexBuilder(repo, resolve_config())

    # Patch deep enough that the temp index has already been opened
    # (and is therefore in WAL mode) before the exception is raised.
    # ``_build_core`` runs with the IndexStore as a context manager —
    # the context manager exit fires as we leave the with-block, so we
    # instead patch the final wrap that happens just before atomic_swap.
    from source_recall import store as store_mod
    from source_recall.store import get_db_path as _get_db_path

    real_atomic_swap = store_mod.IndexStore.atomic_swap

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Simulate the sidecar files existing right before swap —
        # represents the real failure where sidecars are present after
        # the sqlite3 batch commits but the rename didn't happen.
        db_path = _get_db_path(repo.resolve())
        pid = os.getpid()
        wal = db_path.parent / f"{db_path.name}.tmp.{pid}-wal"
        shm = db_path.parent / f"{db_path.name}.tmp.{pid}-shm"
        wal.write_bytes(b"")
        shm.write_bytes(b"")
        raise RuntimeError("simulated post-batch failure")

    with patch.object(store_mod.IndexStore, "atomic_swap", side_effect=boom):
        with pytest.raises(RuntimeError, match="simulated post-batch failure"):
            builder.build()

    pid = os.getpid()
    db_path = _get_db_path(repo.resolve())
    sidecars = [
        db_path.parent / f"{db_path.name}.tmp.{pid}-wal",
        db_path.parent / f"{db_path.name}.tmp.{pid}-shm",
    ]
    leftovers = [s for s in sidecars if s.exists()]
    assert leftovers == [], (
        f"Failed build left tmp sidecars behind: {leftovers}"
    )
    # Silence unused-name lint from the import-only pattern above.
    assert real_atomic_swap is not None