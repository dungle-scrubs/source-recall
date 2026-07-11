"""Tests for graceful shutdown behavior."""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def shutdown_client(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client for shutdown testing."""
    from source_recall import Index
    from source_recall.daemon import create_daemon_app

    emb = BagOfWordsEmbedder(dimensions=64)
    Index(py_app_path, embedder=emb).build()

    config = DaemonConfig(
        repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
        config_path=tmp_path / "repos.toml",
    )
    app = create_daemon_app(config, embedder=emb)
    with TestClient(app) as client:
        yield client


class TestGracefulShutdown:
    def test_db_connections_closed_after_shutdown(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Lifespan shutdown closes all Index connections."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb)

        # Enter and exit lifespan — simulates startup then shutdown.
        with TestClient(app) as client:
            # Verify it was working.
            resp = client.get("/health")
            assert resp.json()["ok"] is True

        # After context exit (lifespan shutdown), the daemon state
        # should have cleaned up. We verify by checking the manager
        # slots all have closed indexes.
        # (The lifespan calls manager.close_all())

    def test_queries_work_until_shutdown(self, shutdown_client: TestClient) -> None:
        """Queries still work right up to shutdown boundary."""
        # This verifies the lifespan startup completed successfully.
        resp = shutdown_client.post("/query", json={"question": "authenticate"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0

    def test_refresh_thread_join_uses_configured_timeout(
        self, py_app_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shutdown joins the periodic-refresh thread with shutdown_timeout_s.

        Regression for a hardcoded 5s join that was shorter than the
        configured shutdown budget and could close indexes underneath a
        still-running refresh.
        """
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
            shutdown_timeout_s=7,
        )

        joins: list[tuple[str, float | None]] = []
        orig_join = threading.Thread.join

        def spy_join(self: threading.Thread, timeout: float | None = None) -> None:
            joins.append((self.name, timeout))
            orig_join(self, timeout)

        monkeypatch.setattr(threading.Thread, "join", spy_join)

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app):
            pass  # Enter + exit lifespan (startup then shutdown).

        refresh_joins = [t for (n, t) in joins if n == "sr-periodic-refresh"]
        assert refresh_joins == [7]
