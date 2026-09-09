"""Regression tests for the four follow-up audit findings (v4 review).

Finding 1 (HIGH): incremental refresh must not destroy a file's index
entries when the file READ fails (not just when chunking fails). A file that
indexed cleanly but whose read raises OSError on refresh must keep its prior
chunks/hashes/vectors — the savepoint rolls back the pre-delete.

Finding 2 (MEDIUM): with a non-loopback / --insecure bind, TrustedHost must
accept the LAN Host clients actually send (wildcard); the default loopback
bind must still reject a spoofed Host.

Finding 3 (MEDIUM): shutdown's close_all must complete within a bounded time
even when a slot lock is held by a stuck refresh.

Finding 4 (LOW): token creation must write the full token even under short
os.write returns.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import source_recall.builder as builder_mod
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
# Finding 1: a READ error on refresh must preserve prior chunks + vectors
# ---------------------------------------------------------------------------


class TestRefreshReadFailurePreservesData:
    def test_read_error_keeps_prior_chunks_and_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file whose READ raises OSError on refresh keeps its prior chunks
        AND vectors, while other files still refresh."""
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
            old_b_ids = _chunk_ids(store, "b.py")
            vectors_created = store.get_vector_count() > 0
            a_vec_before = store.get_existing_vector_ids(list(old_a_ids))
        assert old_a_ids, "a.py should have chunks after the initial build"
        if vectors_created:
            assert a_vec_before == old_a_ids, "a.py chunks should have vectors"

        # Modify both files WITHOUT committing → both are dirty and re-read
        # from the working tree via read_text on refresh.
        (repo / "a.py").write_text("def alpha():\n    return 100\n")
        (repo / "b.py").write_text("def gamma():\n    return 200\n")

        # The working-tree READ of a.py raises OSError; b.py reads normally.
        # Patch the OS-level read (not the chunker) so this exercises the read
        # path specifically — the exact path that returned [] and committed a
        # destructive delete before the fix.
        real_read_text = Path.read_text

        def flaky_read_text(
            self: Path, encoding: str | None = None, errors: str | None = None
        ) -> str:
            if self.name == "a.py":
                raise OSError("simulated read failure on a.py")
            return real_read_text(self, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        # Refresh must not raise even though a.py's read fails.
        builder.refresh()

        with IndexStore(db_path) as store:
            a_ids = _chunk_ids(store, "a.py")
            b_ids = _chunk_ids(store, "b.py")
            a_vec_after = store.get_existing_vector_ids(list(old_a_ids))

        # a.py keeps every prior chunk — no destructive commit.
        assert a_ids == old_a_ids, (
            "a.py's prior chunks must survive a refresh read failure"
        )
        if vectors_created:
            assert a_vec_after == old_a_ids, (
                "a.py's prior vectors must survive a refresh read failure"
            )
        # b.py still refreshed to its new content.
        assert b_ids, "b.py should still have chunks after refresh"
        assert b_ids != old_b_ids, "b.py should have re-chunked to new content"

    def test_pdf_parse_failure_on_refresh_keeps_prior_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PDF that fails to parse on refresh keeps its prior chunks."""
        fitz = pytest.importorskip("fitz")

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

        with IndexStore(db_path) as store:
            old_pdf_ids = _chunk_ids(store, "doc.pdf")
        assert old_pdf_ids, "PDF should have chunks after the initial build"

        # On refresh, PDF parsing raises.
        def boom_pdf(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated PDF parse failure")

        monkeypatch.setattr(builder_mod, "chunk_pdf", boom_pdf)

        # Refresh must not raise; the PDF keeps its prior chunks.
        builder.refresh()

        with IndexStore(db_path) as store:
            pdf_ids = _chunk_ids(store, "doc.pdf")
        assert pdf_ids == old_pdf_ids, (
            "PDF's prior chunks must survive a refresh parse failure"
        )


# ---------------------------------------------------------------------------
# Finding 2: TrustedHost policy for insecure / non-loopback binds
# ---------------------------------------------------------------------------


class TestTrustedHostPolicy:
    def test_helper_wildcards_non_loopback_and_strict_for_loopback(self) -> None:
        from source_recall.server import trusted_allowed_hosts

        assert trusted_allowed_hosts("0.0.0.0") == ["*"]
        assert trusted_allowed_hosts("192.168.1.10") == ["*"]
        loopback = trusted_allowed_hosts("127.0.0.1")
        assert "*" not in loopback
        assert "127.0.0.1" in loopback and "localhost" in loopback

    def test_server_secure_bind_rejects_spoofed_host(self, tmp_path: Path) -> None:
        from source_recall.server import create_app

        app = create_app(tmp_path, embedder=None, host="127.0.0.1")
        client = TestClient(app)
        resp = client.get("/health", headers={"host": "evil.example.com"})
        assert resp.status_code == 400

    def test_server_insecure_bind_accepts_arbitrary_host(self, tmp_path: Path) -> None:
        from source_recall.server import create_app

        app = create_app(tmp_path, embedder=None, host="0.0.0.0")
        client = TestClient(app)
        resp = client.get("/health", headers={"host": "192.168.1.42:7249"})
        assert resp.status_code == 200

    def test_daemon_insecure_bind_accepts_arbitrary_host(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        from source_recall import Index
        from source_recall.daemon import create_daemon_app
        from source_recall.daemon_config import DaemonConfig

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()
        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
            host="0.0.0.0",
        )
        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.get("/health", headers={"host": "192.168.1.42:7757"})
            assert resp.status_code == 200

    def test_daemon_secure_bind_rejects_spoofed_host(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        from source_recall import Index
        from source_recall.daemon import create_daemon_app
        from source_recall.daemon_config import DaemonConfig

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()
        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.get("/health", headers={"host": "evil.example.com"})
            assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Finding 3: shutdown must be bounded even when a slot lock is stuck
# ---------------------------------------------------------------------------


class TestBoundedShutdown:
    def test_close_all_bounded_when_slot_lock_held(self, tmp_path: Path) -> None:
        """close_all completes within a bounded time even if a slot's lock is
        held by a stuck refresh — the slot is force-closed."""
        from source_recall.repo_manager import RepoManager

        manager = RepoManager()
        slot = manager.add(tmp_path)

        closed = threading.Event()

        class FakeIndex:
            def close(self) -> None:
                closed.set()

        slot.index = FakeIndex()  # ty: ignore[invalid-assignment] deliberate close()-only double; RepoSlot.index is typed Index

        # A "stuck refresh" holds the slot lock and never releases it.
        release = threading.Event()

        def hold_lock() -> None:
            with slot.lock:
                release.wait(timeout=10)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        # Give the holder time to actually acquire the lock.
        for _ in range(100):
            if slot.lock.locked():
                break
            time.sleep(0.01)
        assert slot.lock.locked(), "holder thread should hold the slot lock"

        start = time.monotonic()
        manager.close_all(lock_timeout_s=0.5)
        elapsed = time.monotonic() - start

        # Shutdown must not block on the stuck lock indefinitely.
        assert elapsed < 3.0, f"close_all took {elapsed:.2f}s — not bounded"
        # The stuck slot is force-closed despite the held lock.
        assert closed.is_set(), "stuck slot should be force-closed"

        release.set()
        holder.join(timeout=5)


# ---------------------------------------------------------------------------
# Finding 4: token creation must survive short os.write returns
# ---------------------------------------------------------------------------


class TestTokenShortWrite:
    def test_token_fully_written_under_short_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short os.write must not truncate the persisted token file."""
        from source_recall.daemon_config import (
            load_or_create_token,
            token_file_path,
        )

        real_write = os.write
        state = {"shortened": False}

        def short_write(fd: int, data: bytes) -> int:
            # Force exactly one short write (the first multi-byte write, which
            # is the token write) to expose a non-looping writer.
            if not state["shortened"] and len(data) > 1:
                state["shortened"] = True
                return real_write(fd, data[:1])
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", short_write)

        token = load_or_create_token()
        assert state["shortened"], "test must have exercised a short write"

        on_disk = token_file_path().read_bytes().decode()
        assert on_disk == token, "token file must contain the full token"
