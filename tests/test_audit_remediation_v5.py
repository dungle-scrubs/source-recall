"""Regression tests for the v5 audit remediation.

The single root problem across four review rounds: the incremental-refresh
path treats a file-ACCESS failure as either "file deleted" or "successfully
re-indexed", so a transient I/O error permanently destroys a still-existing
file's index entry. These tests pin the structural invariant:

    A file's existing index entry may be mutated (deleted or replaced) ONLY
    when the code has POSITIVELY confirmed either (a) the file is genuinely
    gone, or (b) valid replacement data was produced. Any uncertain/transient
    failure (stat, read, decode, chunk, pdf parse, pdf hash) MUST leave the
    prior index entry untouched.

Finding 1 (HIGH): a transient stat() failure during refresh discovery must
NOT classify a still-existing file as deleted.

Finding 2 (HIGH): a PDF whose post-extraction hashing (stat/read of the first
64 KB) raises OSError on refresh must keep its prior index entry, not commit a
placeholder hash over a destructive pre-delete.

Finding 4 (MEDIUM): shutdown must be bounded even when the real Index.close()
blocks (its internal writer lock is stuck behind a hung reader), not only when
slot-lock acquisition is contended.

Finding 5 (MEDIUM): token publication must be atomic — a concurrent reader
during a slow write must observe either the complete token or none, never a
partial prefix — while preserving the single-winner guarantee.
"""

from __future__ import annotations

import builtins
import errno
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Finding 1: a transient stat() failure must not delete a still-existing file
# ---------------------------------------------------------------------------


class TestTransientStatDoesNotDelete:
    def test_stat_failure_on_refresh_keeps_chunks_hash_and_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file whose stat() raises a transient (non-ENOENT) OSError during
        refresh discovery keeps its chunks, hash, and vectors — it is NOT
        reclassified as deleted just because a fallible stat dropped it from
        the discovered set."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        (repo / "b.py").write_text("def beta():\n    return 2\n")
        _git_commit(repo, "init")

        emb = BagOfWordsEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=emb)
        db_path = builder.build()

        with IndexStore(db_path) as store:
            old_a_ids = _chunk_ids(store, "a.py")
            old_a_hash = store.get_file_hash("a.py")
            vectors_created = store.get_vector_count() > 0
            a_vec_before = store.get_existing_vector_ids(list(old_a_ids))
        assert old_a_ids, "a.py should have chunks after the initial build"
        assert old_a_hash is not None

        # a.py's stat() fails transiently during refresh discovery; b.py is
        # fine. The failure must NOT be read as "a.py is gone".
        real_stat = Path.stat

        def flaky_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
            if self.name == "a.py":
                raise OSError(errno.EIO, "simulated transient stat failure")
            return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "stat", flaky_stat)

        # Refresh must not raise and must not erase a.py.
        builder.refresh()

        with IndexStore(db_path) as store:
            a_ids = _chunk_ids(store, "a.py")
            a_hash = store.get_file_hash("a.py")
            a_vec_after = store.get_existing_vector_ids(list(old_a_ids))

        assert a_ids == old_a_ids, (
            "a.py's chunks must survive a transient stat failure on refresh"
        )
        assert a_hash is not None, "a.py's file hash must survive"
        assert a_hash.content_hash == old_a_hash.content_hash
        if vectors_created:
            assert a_vec_before == old_a_ids
            assert a_vec_after == old_a_ids, "a.py's vectors must survive"


# ---------------------------------------------------------------------------
# Finding 2: a PDF-hash OSError on refresh must preserve prior index entry
# ---------------------------------------------------------------------------


class TestPdfHashFailurePreservesData:
    def test_pdf_hash_oserror_on_refresh_keeps_prior_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chunk_pdf() succeeds but the post-extraction 64 KB hash read raises
        OSError. The old bug caught it, substituted a placeholder hash, and
        committed the destructive pre-delete as a success. The fix routes it
        through _ChunkFailedError so the savepoint rolls back."""
        pytest.importorskip("fitz")
        import fitz

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "alpha beta gamma indexable pdf content here")
        doc.save(str(repo / "doc.pdf"))
        doc.close()
        _git_commit(repo, "init")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=None)
        db_path = builder.build()

        placeholder = hashlib.sha256(b"pdf").hexdigest()
        with IndexStore(db_path) as store:
            old_pdf_ids = _chunk_ids(store, "doc.pdf")
            old_hash = store.get_file_hash("doc.pdf")
        assert old_pdf_ids, "PDF should have chunks after the initial build"
        assert old_hash is not None
        assert old_hash.content_hash != placeholder, (
            "sanity: initial build produced a real content hash"
        )

        # chunk_pdf runs for real (succeeds), but the binary read of the first
        # 64 KB for the content hash raises OSError.
        real_open = builtins.open

        def flaky_open(
            file: object, mode: str = "r", *args: object, **kwargs: object
        ) -> object:
            if str(file).endswith("doc.pdf") and "b" in mode:
                raise OSError(errno.EIO, "simulated pdf hash read failure")
            return real_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "open", flaky_open)

        # Refresh must not raise; the PDF keeps its prior chunks AND its prior
        # (real) content hash — no placeholder is committed.
        builder.refresh()

        with IndexStore(db_path) as store:
            pdf_ids = _chunk_ids(store, "doc.pdf")
            new_hash = store.get_file_hash("doc.pdf")

        assert pdf_ids == old_pdf_ids, (
            "PDF's prior chunks must survive a hash-read OSError on refresh"
        )
        assert new_hash is not None
        assert new_hash.content_hash == old_hash.content_hash, (
            "PDF's prior content hash must survive — no placeholder committed"
        )
        assert new_hash.content_hash != placeholder


# ---------------------------------------------------------------------------
# Finding 4: shutdown bounded even when the real Index.close() blocks
# ---------------------------------------------------------------------------


class TestBoundedShutdownRealBlockingClose:
    def test_close_all_bounded_when_close_itself_blocks(self, tmp_path: Path) -> None:
        """close_all must return within the budget even when the slot lock is
        freely acquired but Index.close() itself blocks indefinitely (its
        internal writer lock stuck behind a hung reader)."""
        from source_recall.repo_manager import RepoManager

        manager = RepoManager()
        slot = manager.add(tmp_path)

        close_started = threading.Event()
        release = threading.Event()

        class BlockingIndex:
            def close(self) -> None:
                close_started.set()
                # Simulates Index.close() blocking on acquire_write() behind a
                # stuck reader — never returns within the budget.
                release.wait(timeout=30)

        slot.index = BlockingIndex()  # type: ignore[assignment]

        start = time.monotonic()
        manager.close_all(lock_timeout_s=0.5)
        elapsed = time.monotonic() - start

        assert close_started.is_set(), "close_all should have attempted the close"
        assert elapsed < 3.0, (
            f"close_all took {elapsed:.2f}s — a blocking close was not bounded"
        )

        # Let the abandoned close finish so the daemon thread can exit.
        release.set()


# ---------------------------------------------------------------------------
# Finding 5: token publication is atomic — no partial token ever observed
# ---------------------------------------------------------------------------


class TestAtomicTokenPublication:
    def test_reader_never_observes_partial_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """During a slow token write, a concurrent reader must see either a
        complete token or nothing — never a partial prefix."""
        from source_recall.daemon_config import load_or_create_token, load_token

        first_byte_written = threading.Event()
        reader_finished = threading.Event()
        observations: list[str] = []
        state = {"triggered": False}

        real_write = os.write

        def slow_write(fd: int, data: bytes) -> int:
            # One-shot: on the first multi-byte write (the token), write a
            # single byte, then pause so a concurrent reader can try to observe
            # the in-progress state before the rest lands.
            if not state["triggered"] and len(data) > 1:
                state["triggered"] = True
                n = real_write(fd, data[:1])
                first_byte_written.set()
                reader_finished.wait(timeout=5)
                return n
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", slow_write)

        def reader() -> None:
            first_byte_written.wait(timeout=5)
            for _ in range(200):
                tok = load_token()
                if tok is not None:
                    observations.append(tok)
            reader_finished.set()

        rt = threading.Thread(target=reader)
        rt.start()
        token = load_or_create_token()
        rt.join(timeout=10)

        assert state["triggered"], "the slow-write path must have been exercised"
        assert token, "a token must be produced"
        for obs in observations:
            assert obs == token, f"reader observed a partial/other token: {obs!r}"

    def test_concurrent_first_start_still_agrees_on_one_token(
        self, tmp_path: Path
    ) -> None:
        """The atomic-publication change must preserve the single-winner
        guarantee: concurrent first-starts converge on one token."""
        from source_recall.daemon_config import load_or_create_token

        results: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            tok = load_or_create_token()
            with lock:
                results.append(tok)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1
