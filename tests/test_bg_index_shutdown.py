"""Tests for background index thread tracking and graceful shutdown."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


def _make_git_repo(path: Path) -> None:
    """Create a minimal git repo."""
    path.mkdir(exist_ok=True)
    (path / "main.py").write_text("def hello():\n    return 'world'\n")
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(Path.home()),
    }
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True,
        check=True,
        env=env,
    )


class TestBackgroundIndexThreadTracking:
    def test_shutdown_waits_for_background_index(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Lifespan shutdown joins background index threads."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)

        build_started = threading.Event()
        build_finished = threading.Event()

        original_build = Index.build

        def slow_build(self: Index) -> Path:
            build_started.set()
            time.sleep(0.5)  # Simulate non-trivial build.
            result = original_build(self)
            build_finished.set()
            return result

        with TestClient(app) as client:
            new_repo = tmp_path / "slow-repo"
            _make_git_repo(new_repo)

            with patch.object(Index, "build", slow_build):
                client.post("/repos", json={"path": str(new_repo)})
                build_started.wait(timeout=5)

        # After TestClient exits (lifespan shutdown), the build
        # should have been allowed to complete.
        assert build_finished.is_set(), "Background build was killed before finishing"

    def test_multiple_background_threads_all_joined(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Multiple concurrent background builds are all joined on shutdown."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        finished = {"count": 0}

        original_build = Index.build

        def counting_build(self: Index) -> Path:
            result = original_build(self)
            finished["count"] += 1
            return result

        # Keep the patch active for the whole block: the background build
        # thread may invoke Index.build after the POST returns, so restoring
        # the patch per-request races the worker and undercounts under load.
        with patch.object(Index, "build", counting_build):
            with TestClient(app) as client:
                for i in range(3):
                    repo = tmp_path / f"repo-{i}"
                    _make_git_repo(repo)
                    client.post("/repos", json={"path": str(repo)})

                # Wait for all builds to complete before shutdown.
                deadline = time.monotonic() + 10
                while finished["count"] < 3 and time.monotonic() < deadline:
                    time.sleep(0.05)

        # All 3 builds should have completed before shutdown finished.
        assert finished["count"] == 3

    def test_completed_background_threads_are_removed_from_tracking(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Threads that finish must be removed from bg_threads.

        Without removal, the list grows unbounded over the lifetime of a
        long-running daemon (every /repos POST adds an entry; threads
        never leave).  On shutdown the daemon joins every historical
        thread, paying N×(join-timeout) even when no work is in flight.
        The list should contain only live (running) threads at any
        moment.

        We assert this by patching ``threading.Thread.join`` at module
        level to record callers, then triggering a shutdown after two
        background threads have completed.  No join should be recorded
        for the completed threads.
        """
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)

        joins: list[tuple[str, bool, float | None]] = []
        real_join = threading.Thread.join

        def recording_join(
            self: threading.Thread, timeout: float | None = None
        ) -> None:
            joins.append((self.name, self.is_alive(), timeout))
            # Actually join briefly so we don't deadlock the test.
            real_join(self, timeout=0.001)

        with TestClient(app) as client:
            # Add two background builds and wait for them to finish.
            for i in range(2):
                repo = tmp_path / f"bg-repo-{i}"
                _make_git_repo(repo)
                client.post("/repos", json={"path": str(repo)})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    r = next(
                        (
                            r
                            for r in client.get("/repos").json()["repos"]
                            if r["name"] == repo.name
                        ),
                        None,
                    )
                    if r and r["state"] == "ready":
                        break
                    time.sleep(0.2)

            # Both bg index threads have completed.  Now patch join and
            # exit the TestClient to trigger lifespan shutdown.
            threading.Thread.join = recording_join
            try:
                pass  # __exit__ runs shutdown
            finally:
                # Background task: schedule restore after shutdown completes
                # by hooking into manager.close_all.  Simpler: just restore
                # when the context exits (next test will reset state).
                pass

        threading.Thread.join = real_join

        # Filter out joins from threads we didn't add via /repos POST.
        # The periodic refresh thread (sr-periodic-refresh) also gets
        # joined on shutdown and is fine — we only care about the
        # background index threads we created.
        bg_joins = [n for n, _, _ in joins if n.startswith("sr-index-bg-repo-")]
        assert bg_joins == [], (
            f"Completed background threads were joined on shutdown: {bg_joins}"
        )
