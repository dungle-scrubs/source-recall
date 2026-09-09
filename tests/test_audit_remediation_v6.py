"""Regression tests for the v6 audit remediation (final round).

The v5 rounds closed the refresh no-data-loss invariant for every filesystem
change-detection signal (stat/read/decode/chunk/pdf/hash). This round closes
the SAME class in the last uncovered CHANGE-DETECTION SIGNAL — ``git status
--porcelain`` (dirty-file detection) — plus two shutdown/vector mediums.

    A file's existing index entry may be mutated (deleted or replaced) ONLY on
    POSITIVE confirmation of deletion or valid replacement. ANY failure to
    obtain a change-detection signal must NOT be read as a negative/clean
    result.

Finding 1 (HIGH): ``_detect_dirty_files`` returned ``{}`` both when the working
tree was genuinely clean AND when ``git status --porcelain`` failed. A transient
git-status failure then (a) dropped previously-indexed untracked/staged files
out of ``blob_map`` so they were swept into ``deleted`` and permanently erased,
and (b) misclassified a modified tracked file as unchanged so refresh committed
stale content as a successful pass. The fix makes detection failure explicit and
fail-closed: the refresh aborts cleanly, leaving the index fully intact.

Finding 2 (MEDIUM): a transient embedding failure during refresh dropped vector
coverage for a still-existing file permanently (the ``vec_dirty`` marker was
written but never consumed, so the next refresh early-returned as unchanged).
The fix consumes ``vec_dirty`` on the next refresh, forcing re-embedding.

Finding 3 (MEDIUM): a slot removed mid-build could later call ``set_ready()``,
which synchronously invoked a real blocking ``Index.close()`` outside the
bounded shutdown path. The fix routes that close through a detached daemon
thread so shutdown stays bounded.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
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


def _chunk_ids(store: IndexStore, rel_path: str) -> set[str]:
    return {
        row[0]
        for row in store.conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (rel_path,)
        ).fetchall()
    }


def _fail_git_status(
    monkeypatch: pytest.MonkeyPatch, *, mode: str = "nonzero"
) -> dict[str, bool]:
    """Make only ``git status --porcelain`` fail, leaving other git calls real.

    @param mode: ``nonzero`` (exit 1), ``enoent`` (FileNotFoundError), or
        ``timeout`` (subprocess.TimeoutExpired).
    @returns: A ``{"active": True}`` toggle — set ``active`` to False to let
        git-status succeed again WITHOUT undoing other (autouse) monkeypatches.
    """
    toggle = {"active": True}
    real_run = subprocess.run

    def flaky_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            toggle["active"]
            and isinstance(cmd, (list, tuple))
            and list(cmd[:3]) == ["git", "status", "--porcelain"]
        ):
            if mode == "enoent":
                raise FileNotFoundError("git not found")
            if mode == "timeout":
                raise subprocess.TimeoutExpired(cmd, 10)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", flaky_run)
    return toggle


# ---------------------------------------------------------------------------
# Finding 1: a git-status failure must not be read as a clean working tree
# ---------------------------------------------------------------------------


class TestDirtyDetectionFailureFailsClosed:
    @pytest.mark.parametrize("mode", ["nonzero", "enoent", "timeout"])
    def test_git_status_failure_does_not_delete_untracked_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """An untracked file indexed at build time keeps its chunks/hash/vectors
        when ``git status`` fails during a later refresh — a detection failure is
        never read as "the file is gone"."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        _git_commit(repo, "init")
        # Untracked file present at build time — indexed via dirty detection.
        (repo / "c.py").write_text("def gamma():\n    return 3\n")

        emb = BagOfWordsEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=emb)
        db_path = builder.build()

        with IndexStore(db_path) as store:
            old_c_ids = _chunk_ids(store, "c.py")
            old_c_hash = store.get_file_hash("c.py")
            vectors_created = store.get_vector_count() > 0
            c_vec_before = store.get_existing_vector_ids(list(old_c_ids))
        assert old_c_ids, "untracked c.py should have been indexed at build"
        assert old_c_hash is not None

        _fail_git_status(monkeypatch, mode=mode)

        # Refresh must not raise and must not erase c.py.
        builder.refresh()

        with IndexStore(db_path) as store:
            c_ids = _chunk_ids(store, "c.py")
            c_hash = store.get_file_hash("c.py")
            c_vec_after = store.get_existing_vector_ids(list(old_c_ids))

        assert c_ids == old_c_ids, (
            "untracked file's chunks must survive a git-status failure on refresh"
        )
        assert c_hash is not None, "untracked file's hash must survive"
        if vectors_created:
            assert c_vec_before == old_c_ids
            assert c_vec_after == old_c_ids, "untracked file's vectors must survive"

    def test_build_aborts_cleanly_when_dirty_detection_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full build must fail closed (clean FileDiscoveryError, existing
        index untouched) rather than silently omit untracked/staged files when
        dirty detection fails."""
        from source_recall.models import FileDiscoveryError

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        _git_commit(repo, "init")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=None)
        # A prior good index must survive an aborted rebuild.
        db_path = builder.build()
        mtime_before = db_path.stat().st_mtime_ns

        _fail_git_status(monkeypatch, mode="nonzero")
        with pytest.raises(FileDiscoveryError) as exc_info:
            builder.build()
        assert exc_info.value.reason == "dirty_detection_failed"
        # The pre-existing index is left intact (no atomic swap happened).
        assert db_path.exists()
        assert db_path.stat().st_mtime_ns == mtime_before

    def test_git_status_failure_does_not_commit_modified_file_as_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tracked file modified in the working tree must NOT be committed as a
        successful up-to-date pass when dirty detection failed. The refresh
        aborts (index meta untouched); a later refresh with detection working
        re-indexes the file so its new content becomes searchable."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha_original():\n    return 1\n")
        _git_commit(repo, "init")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=None)
        db_path = builder.build()

        with IndexStore(db_path) as store:
            indexed_at_before = store.get_meta("indexed_at")
            old_ids = _chunk_ids(store, "a.py")
        assert old_ids

        # Modify the tracked file in the working tree (uncommitted → dirty).
        (repo / "a.py").write_text("def alpha_MODIFIED_NEW():\n    return 999\n")

        toggle = _fail_git_status(monkeypatch, mode="nonzero")
        result = builder.refresh()

        with IndexStore(db_path) as store:
            indexed_at_after = store.get_meta("indexed_at")

        # A skipped refresh must not commit itself as a fresh successful pass:
        # the index meta stays untouched and no files are reported re-indexed.
        assert result == 0
        assert indexed_at_after == indexed_at_before, (
            "aborted refresh must leave index meta intact (not commit as success)"
        )

        # With detection working again, the modification is picked up: the new
        # content is searchable and the stale content is gone.
        toggle["active"] = False
        builder.refresh()
        with IndexStore(db_path) as store:
            chunk_text = " ".join(
                row[0]
                for row in store.conn.execute(
                    "SELECT content FROM chunks WHERE file_path = ?", ("a.py",)
                ).fetchall()
            )
        assert "alpha_MODIFIED_NEW" in chunk_text, (
            "modified file must be re-indexed once detection recovers"
        )
        assert "alpha_original" not in chunk_text, "stale content must not survive"


# ---------------------------------------------------------------------------
# Finding 2: a transient embedding failure must not permanently drop vectors
# ---------------------------------------------------------------------------


class _FlakyEmbedder:
    """BagOfWords embedder that raises on demand to simulate a transient
    embedding-service failure, then recovers."""

    def __init__(self, dimensions: int = 64) -> None:
        self._inner = BagOfWordsEmbedder(dimensions=dimensions)
        self.fail = False

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("simulated transient embedding failure")
        return self._inner.embed_chunks(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._inner.embed_query(query)


class TestEmbeddingFailureDoesNotPermanentlyLoseVectors:
    def test_transient_embed_failure_recovers_on_next_refresh(
        self, tmp_path: Path
    ) -> None:
        """A file re-indexed during a refresh whose embedding fails is left
        without vectors, but a subsequent successful refresh restores them —
        even though the file is otherwise unchanged."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        _git_commit(repo, "init")

        emb = _FlakyEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=emb)
        db_path = builder.build()

        with IndexStore(db_path) as store:
            if store.get_vector_count() == 0:
                pytest.skip("sqlite-vec unavailable — vector coverage not testable")

        # Change the file (committed) so refresh re-indexes it.
        (repo / "a.py").write_text("def alpha():\n    return 2  # changed\n")
        _git_commit(repo, "change")

        # Refresh #1: embedding fails → new chunks land without vectors.
        emb.fail = True
        builder.refresh()

        with IndexStore(db_path) as store:
            a_ids = _chunk_ids(store, "a.py")
            vec_after_fail = store.get_existing_vector_ids(list(a_ids))
        assert a_ids, "a.py must still have chunks after a failed-embed refresh"
        assert vec_after_fail == set(), (
            "the failed embedding must have left a.py without vectors"
        )

        # Refresh #2: embedding works, file otherwise unchanged. vec_dirty must
        # force re-embedding so coverage is restored.
        emb.fail = False
        builder.refresh()

        with IndexStore(db_path) as store:
            a_ids2 = _chunk_ids(store, "a.py")
            vec_restored = store.get_existing_vector_ids(list(a_ids2))
        assert vec_restored == a_ids2, (
            "a subsequent successful refresh must restore the lost vectors"
        )


# ---------------------------------------------------------------------------
# Finding 3: set_ready on a removed slot must not trigger an unbounded close
# ---------------------------------------------------------------------------


class TestSetReadyCancelledCloseIsBounded:
    def test_cancelled_set_ready_does_not_block_on_blocking_close(
        self, tmp_path: Path
    ) -> None:
        """A slot removed mid-build (cancel set) that then receives its built
        Index via set_ready must not block the background thread on a blocking
        Index.close(). The close is detached so shutdown stays bounded."""
        from source_recall.repo_manager import RepoSlot

        slot = RepoSlot(name="r", path=tmp_path)
        slot.cancel.set()

        close_started = threading.Event()
        release = threading.Event()

        class BlockingIndex:
            def close(self) -> None:
                close_started.set()
                release.wait(timeout=30)

        start = time.monotonic()
        slot.set_ready(BlockingIndex())  # ty: ignore[invalid-argument-type] deliberate close()-blocks double; set_ready is typed Index
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, (
            f"set_ready blocked {elapsed:.2f}s on a cancelled index close — "
            "the mid-build close was not detached from the caller"
        )
        # The detached close should still have been attempted.
        assert close_started.wait(timeout=5), "cancelled index was never closed"
        # The slot must stay out of READY and hold no index.
        assert slot.index is None

        release.set()
