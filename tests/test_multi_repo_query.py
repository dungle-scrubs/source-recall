"""Tests for multi-repo query routing."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


class _CountingEmbedder(BagOfWordsEmbedder):
    """BagOfWordsEmbedder that records every embed_query call.

    Test double aligned with the real method (same name and signature)
    instead of an instance-attribute spy over the bound method.
    """

    def __init__(self, dimensions: int, calls: list[str]) -> None:
        super().__init__(dimensions=dimensions)
        self.calls = calls

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return super().embed_query(query)


@pytest.fixture
def multi_repo_client(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client with two indexed repos."""
    from source_recall import Index
    from source_recall.daemon import create_daemon_app

    emb = BagOfWordsEmbedder(dimensions=64)

    # First repo — py-app fixture.
    Index(py_app_path, embedder=emb).build()

    # Second repo — copy of py-app with different name.
    repo2 = tmp_path / "repo2"
    shutil.copytree(py_app_path, repo2)
    Index(repo2, embedder=emb).build()

    config = DaemonConfig(
        repos=[
            DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name),
            DaemonConfig.RepoEntry(path=repo2, name="repo2"),
        ],
        config_path=tmp_path / "repos.toml",
    )
    app = create_daemon_app(config, embedder=emb)
    with TestClient(app) as client:
        yield client


class TestMultiRepoQuery:
    def test_query_without_repo_searches_all(
        self, multi_repo_client: TestClient
    ) -> None:
        """POST /query without repo returns results from all repos."""
        resp = multi_repo_client.post("/query", json={"question": "authenticate"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0

        # Results should have repo_name field.
        assert all("repo_name" in r for r in results)

    def test_results_tagged_with_repo_name(self, multi_repo_client: TestClient) -> None:
        """Each result includes the repo it came from."""
        resp = multi_repo_client.post("/query", json={"question": "authenticate"})
        results = resp.json()["results"]
        repo_names = {r["repo_name"] for r in results}
        # Should have results from both repos.
        assert len(repo_names) == 2

    def test_query_with_repo_filter(
        self, multi_repo_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /query with repo filters to that repo."""
        resp = multi_repo_client.post(
            "/query",
            json={"question": "authenticate", "repo": py_app_path.name},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        # All results from the specified repo.
        assert all(r.get("repo_name") == py_app_path.name for r in results)

    def test_fanout_embeds_query_once(self, py_app_path: Path, tmp_path: Path) -> None:
        """A multi-repo fan-out embeds the query exactly once, not per repo.

        The daemon computes the query vector once and reuses it across all
        repos (they share ``state['embedder']``).
        """
        import shutil

        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        calls: list[str] = []
        emb = _CountingEmbedder(dimensions=64, calls=calls)
        Index(py_app_path, embedder=emb).build()
        repo2 = tmp_path / "repo2"
        shutil.copytree(py_app_path, repo2)
        Index(repo2, embedder=emb).build()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name),
                DaemonConfig.RepoEntry(path=repo2, name="repo2"),
            ],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb)

        with TestClient(app) as client:
            # Drain the startup warmup (which also embeds) before counting.
            warmup = getattr(app.state, "warmup_thread", None)
            if warmup is not None:
                warmup.join(timeout=5)
            before = len(calls)
            resp = client.post("/query", json={"question": "authenticate"})
            assert resp.status_code == 200
            # Two ready repos → fan-out, but only one embed call.
            assert len(resp.json()["results"]) > 0
            assert len(calls) - before == 1
