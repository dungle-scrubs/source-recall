"""Tests for SSE progress stream endpoint."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def sse_client(py_app_path: Path, tmp_path: Path) -> Generator[TestClient, None, None]:
    """Daemon client for SSE testing."""
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


class TestSSEProgress:
    def test_endpoint_exists(self, sse_client: TestClient, py_app_path: Path) -> None:
        """GET /repos/{name}/progress returns a streaming response."""
        name = py_app_path.name
        resp = sse_client.get(f"/repos/{name}/progress")
        # When repo is ready (not indexing), should complete immediately
        # with a completion event.
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_emits_complete_for_ready_repo(
        self, sse_client: TestClient, py_app_path: Path
    ) -> None:
        """SSE stream emits a complete event for already-ready repos."""
        name = py_app_path.name
        resp = sse_client.get(f"/repos/{name}/progress")
        assert resp.status_code == 200

        # Parse SSE events from body.
        body = resp.text
        assert "event: complete" in body

    def test_returns_404_for_unknown_repo(self, sse_client: TestClient) -> None:
        """SSE endpoint returns 404 for unknown repo."""
        resp = sse_client.get("/repos/nonexistent/progress")
        assert resp.status_code == 404
