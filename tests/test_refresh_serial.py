"""Tests for refresh serialization and targeted refresh."""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def refresh_client(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client for refresh testing (no rate limit)."""
    from source_recall import Index
    from source_recall.daemon import create_daemon_app

    emb = BagOfWordsEmbedder(dimensions=64)
    Index(py_app_path, embedder=emb).build()

    config = DaemonConfig(
        repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
        config_path=tmp_path / "repos.toml",
    )
    # Create app with rate limit disabled for testing.
    app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
    with TestClient(app) as client:
        yield client


class TestRefreshSerialization:
    def test_concurrent_refreshes_serialize(
        self, refresh_client: TestClient, py_app_path: Path
    ) -> None:
        """Two concurrent /refresh calls serialize — second waits, not 429."""
        results: list[int] = []

        def do_refresh() -> None:
            resp = refresh_client.post(f"/refresh?repo={py_app_path.name}")
            results.append(resp.status_code)

        t1 = threading.Thread(target=do_refresh)
        t2 = threading.Thread(target=do_refresh)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # Both should succeed (serialized, not rejected).
        assert results == [200, 200]


class TestTargetedRefresh:
    def test_refresh_with_files_parameter(
        self, refresh_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /refresh with files param re-indexes specified files."""
        resp = refresh_client.post(
            f"/refresh?repo={py_app_path.name}",
            json={"files": ["myapp/auth.py"]},
        )
        assert resp.status_code == 200
        assert "files_updated" in resp.json()

    def test_refresh_without_files_does_full(
        self, refresh_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /refresh without files does incremental refresh."""
        resp = refresh_client.post(f"/refresh?repo={py_app_path.name}")
        assert resp.status_code == 200
        assert resp.json()["files_updated"] >= 0
