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

        def slow_build(self: object) -> object:
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

        def counting_build(self: object) -> object:
            result = original_build(self)
            finished["count"] += 1
            return result

        with TestClient(app) as client:
            for i in range(3):
                repo = tmp_path / f"repo-{i}"
                _make_git_repo(repo)
                with patch.object(Index, "build", counting_build):
                    client.post("/repos", json={"path": str(repo)})

            # Wait for all to start.
            time.sleep(1)

        # All 3 builds should have completed before shutdown finished.
        assert finished["count"] == 3
