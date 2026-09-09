"""Regression tests for the two follow-up audit findings.

Finding 1 (HIGH): incremental refresh must never destroy a file's existing
index entries when re-chunking that file raises. A file that previously
indexed cleanly but now fails to chunk must keep its prior chunks/hashes.

Finding 2 (MEDIUM): ``atomic_swap`` must reject a cross-filesystem swap
BEFORE touching the live target's ``-wal``/``-shm`` sidecars, so a rejected
swap leaves the target and its sidecars untouched.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

import source_recall.builder as builder_mod
from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.store import IndexStore


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


def _git_commit(repo: Path, msg: str = "update") -> None:
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", msg, "--allow-empty"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# Finding 1: refresh must not lose data when re-chunking raises
# ---------------------------------------------------------------------------


class TestRefreshChunkFailurePreservesData:
    def test_reindex_failure_keeps_prior_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that fails to re-chunk on refresh keeps its old chunks,
        while the rest of the repo still refreshes."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        (repo / "b.py").write_text("def beta():\n    return 2\n")
        _git_commit(repo, "init")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=None)
        db_path = builder.build()

        # Capture a.py's original chunks — these must survive the refresh.
        with IndexStore(db_path) as store:
            old_a_ids = {
                row[0]
                for row in store.conn.execute(
                    "SELECT id FROM chunks WHERE file_path = 'a.py'"
                ).fetchall()
            }
        assert old_a_ids, "a.py should have chunks after the initial build"

        # Now make BOTH files change, but re-chunking a.py raises.
        (repo / "a.py").write_text("def alpha():\n    return 100\n")
        (repo / "b.py").write_text("def beta():\n    return 200\n")
        _git_commit(repo, "change both")

        real_chunk = builder_mod.chunk_file_with_refs

        def flaky_chunk(rel_path: str, content: str, *, max_chars: int = 6000):
            if rel_path == "a.py":
                raise ValueError("simulated chunker failure on a.py")
            return real_chunk(rel_path, content, max_chars=max_chars)

        monkeypatch.setattr(builder_mod, "chunk_file_with_refs", flaky_chunk)

        # Refresh must not raise, even though a.py fails to chunk.
        builder.refresh()

        with IndexStore(db_path) as store:
            a_ids = {
                row[0]
                for row in store.conn.execute(
                    "SELECT id FROM chunks WHERE file_path = 'a.py'"
                ).fetchall()
            }
            b_ids = {
                row[0]
                for row in store.conn.execute(
                    "SELECT id FROM chunks WHERE file_path = 'b.py'"
                ).fetchall()
            }

        # a.py's prior chunks must be intact (not silently deleted).
        assert a_ids == old_a_ids, (
            "a.py's prior chunks must be preserved when re-chunking fails"
        )
        # b.py must still refresh normally.
        assert b_ids, "b.py should still have chunks after refresh"

    def test_storage_failure_during_refresh_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine storage failure (not a chunk failure) must abort the
        refresh loudly, not be silently swallowed per-file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        _git_commit(repo, "init")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=None)
        builder.build()

        (repo / "a.py").write_text("def alpha():\n    return 100\n")
        _git_commit(repo, "change")

        # Simulate a storage-layer failure while re-indexing.
        def boom(self: IndexStore, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated storage failure")

        monkeypatch.setattr(IndexStore, "insert_chunks", boom)

        with pytest.raises(RuntimeError, match="simulated storage failure"):
            builder.refresh()


# ---------------------------------------------------------------------------
# Finding 2: cross-device atomic_swap must not touch the live target
# ---------------------------------------------------------------------------


class TestAtomicSwapCrossDeviceGuard:
    def test_rejected_swap_leaves_target_sidecars_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cross-filesystem swap is rejected before any target cleanup,
        so the target db AND its -wal/-shm sidecars are left untouched."""
        # A real (non-WAL) sqlite db as the temp build output.
        tmp_db = tmp_path / "index.db.tmp"
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("CREATE TABLE t (x)")
        conn.commit()
        conn.close()

        # The live target plus its sidecars, with known content.
        target = tmp_path / "index.db"
        target.write_bytes(b"LIVE_TARGET_DB")
        target_wal = Path(str(target) + "-wal")
        target_shm = Path(str(target) + "-shm")
        target_wal.write_bytes(b"LIVE_WAL")
        target_shm.write_bytes(b"LIVE_SHM")

        # Force the st_dev check to see two different filesystems.
        real_stat = Path.stat

        class _FakeStat:
            def __init__(self, dev: int) -> None:
                self.st_dev = dev

        def fake_stat(
            self: Path, *, follow_symlinks: bool = True
        ) -> os.stat_result | _FakeStat:
            if self == tmp_db:
                return _FakeStat(1000)
            if self == target.parent:
                return _FakeStat(2000)
            return real_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(Path, "stat", fake_stat)

        with pytest.raises(OSError, match="same filesystem"):
            IndexStore.atomic_swap(tmp_db, target)

        # The rejected swap must have touched nothing on the target side.
        assert target.read_bytes() == b"LIVE_TARGET_DB"
        assert target_wal.exists(), "target -wal must not be deleted on rejection"
        assert target_shm.exists(), "target -shm must not be deleted on rejection"
        assert target_wal.read_bytes() == b"LIVE_WAL"
        assert target_shm.read_bytes() == b"LIVE_SHM"
