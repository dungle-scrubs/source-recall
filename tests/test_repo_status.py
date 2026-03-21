"""Tests for per-repo status endpoint."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def status_client(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client for status testing."""
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


class TestPerRepoStatus:
    def test_returns_ready_for_indexed_repo(
        self, status_client: TestClient, py_app_path: Path
    ) -> None:
        """GET /repos/{name}/status returns ready state with details."""
        name = py_app_path.name
        resp = status_client.get(f"/repos/{name}/status")
        assert resp.status_code == 200

        data = resp.json()
        assert data["state"] == "ready"
        assert data["name"] == name
        assert data["file_count"] > 0
        assert data["chunk_count"] > 0
        assert "indexed_at" in data

    def test_returns_404_for_unknown_repo(self, status_client: TestClient) -> None:
        """GET /repos/{name}/status returns 404 for unknown repo."""
        resp = status_client.get("/repos/nonexistent/status")
        assert resp.status_code == 404

    def test_returns_error_state_for_bad_repo(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """GET /repos/{name}/status returns error state."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        bad = tmp_path / "bad-repo"
        bad.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=py_app_path, name="good"),
                DaemonConfig.RepoEntry(path=bad, name="bad"),
            ],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.get("/repos/bad/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["state"] == "error"
            assert data["error"] is not None
