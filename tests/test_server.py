"""Tests for the query server."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def indexed_app(py_app_path: Path) -> Generator[TestClient, None, None]:
    """Build an index with vectors, then create a test client.

    Uses context manager so the lifespan (model loading) runs.

    @returns: FastAPI TestClient bound to the indexed repo.
    """
    from source_recall import Index
    from source_recall.server import create_app

    emb = BagOfWordsEmbedder(dimensions=64)
    idx = Index(py_app_path, embedder=emb)
    idx.build()

    app = create_app(py_app_path, embedder=emb)
    with TestClient(app) as client:
        yield client


class TestWarmup:
    def test_default_embedder_is_shared_across_repositories(
        self, py_app_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto mode constructs one model for every loaded repository."""
        import shutil

        from source_recall import Index
        from source_recall.server import create_app

        repo2 = py_app_path.parent / "repo2"
        shutil.copytree(py_app_path, repo2)
        Index(py_app_path, embedder=None).build()
        Index(repo2, embedder=None).build()

        created: list[BagOfWordsEmbedder] = []

        def create_default(_encode_batch_size: int = 32) -> BagOfWordsEmbedder:
            instance = BagOfWordsEmbedder(dimensions=64)
            created.append(instance)
            return instance

        monkeypatch.setattr(
            Index, "_create_default_embedder", staticmethod(create_default)
        )

        app = create_app([py_app_path, repo2])
        with TestClient(app):
            app.state.warmup_thread.join(timeout=5)

        assert len(created) == 1

    def test_startup_warms_embedder(
        self, py_app_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server startup spawns a thread that warms the embedder.

        The warmup must run (embed_query called) but must not block the
        lifespan startup — it happens on a background daemon thread.
        """
        from source_recall import Index
        from source_recall.server import create_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        calls: list[str] = []
        orig = emb.embed_query

        def spy(q: str) -> list[float]:
            calls.append(q)
            return orig(q)

        monkeypatch.setattr(emb, "embed_query", spy)

        app = create_app(py_app_path, embedder=emb)
        with TestClient(app):
            warmup = app.state.warmup_thread
            assert warmup is not None
            warmup.join(timeout=5)

        assert "warmup" in calls


class TestQueryEndpoint:
    def test_returns_ranked_results(self, indexed_app: TestClient) -> None:
        """POST /query returns results array with query_ms."""
        resp = indexed_app.post("/query", json={"question": "authenticate"})
        assert resp.status_code == 200

        data = resp.json()
        assert "results" in data
        assert "query_ms" in data
        assert len(data["results"]) > 0
        assert data["query_ms"] >= 0

        # Each result has expected fields.
        r = data["results"][0]
        assert "chunk_id" in r
        assert "file_path" in r
        assert "content" in r
        assert "score" in r
        assert "match_reason" in r


class TestStatusEndpoint:
    def test_returns_index_metrics(self, indexed_app: TestClient) -> None:
        """GET /status returns file/chunk/vector counts."""
        resp = indexed_app.get("/status")
        assert resp.status_code == 200

        data = resp.json()
        assert data["file_count"] > 0
        assert data["chunk_count"] > 0
        assert data["vector_count"] > 0
        assert data["embed_model"] == "BagOfWordsEmbedder"
        assert data["embed_dimensions"] == 64
        assert data["db_size_bytes"] > 0
        assert data["indexed_at"] != ""


class TestRefreshEndpoint:
    def test_returns_update_count(self, indexed_app: TestClient) -> None:
        """POST /refresh returns files_updated and refresh_ms."""
        resp = indexed_app.post("/refresh")
        assert resp.status_code == 200

        data = resp.json()
        assert "files_updated" in data
        assert "refresh_ms" in data
        assert data["files_updated"] == 0  # No changes since build.
        assert data["refresh_ms"] >= 0


class TestQueryTopK:
    def test_top_k_limits_results(self, indexed_app: TestClient) -> None:
        """POST /query with top_k=2 returns exactly 2 results."""
        resp = indexed_app.post("/query", json={"question": "authenticate", "top_k": 2})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_empty_question_returns_empty(self, indexed_app: TestClient) -> None:
        """POST /query with empty string returns empty results, not 500."""
        resp = indexed_app.post("/query", json={"question": ""})
        assert resp.status_code == 200
        assert resp.json()["results"] == []


class TestHealthEndpoint:
    def test_returns_ok(self, indexed_app: TestClient) -> None:
        """GET /health returns ok=true with repos and uptime."""
        resp = indexed_app.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["ok"] is True
        assert len(data["repos"]) == 1
        assert data["uptime_s"] >= 0


class TestReposEndpoint:
    def test_lists_loaded_repos(self, indexed_app: TestClient) -> None:
        """GET /repos returns repo info with stats."""
        resp = indexed_app.get("/repos")
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["repos"]) == 1
        repo = data["repos"][0]
        assert repo["file_count"] > 0
        assert repo["chunk_count"] > 0


class TestNonexistentRepo:
    def test_query_unknown_repo_returns_404(self, indexed_app: TestClient) -> None:
        """POST /query with unknown repo name returns 404."""
        resp = indexed_app.post(
            "/query", json={"question": "test", "repo": "nonexistent"}
        )
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]

    def test_status_unknown_repo_returns_404(self, indexed_app: TestClient) -> None:
        """GET /status with unknown repo name returns 404."""
        resp = indexed_app.get("/status?repo=nonexistent")
        assert resp.status_code == 404


class TestCORSRestriction:
    def test_cors_rejects_foreign_origin(self, indexed_app: TestClient) -> None:
        """Preflight from a non-localhost origin should be denied."""
        resp = indexed_app.options(
            "/query",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # A restricted CORS config will NOT echo back the foreign origin.
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin != "*"
        assert "evil.example.com" not in allow_origin

    def test_cors_allows_localhost_origin(self, indexed_app: TestClient) -> None:
        """Preflight from localhost should be allowed."""
        resp = indexed_app.options(
            "/query",
            headers={
                "Origin": "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert "127.0.0.1" in allow_origin or allow_origin == "*"


class TestMultiRepo:
    def test_query_requires_repo_when_multiple(self, py_app_path: Path) -> None:
        """POST /query without repo returns 400 when multiple repos loaded."""
        import shutil

        from source_recall import Index
        from source_recall.server import create_app

        # Create a second repo.
        repo2 = py_app_path.parent / "repo2"
        shutil.copytree(py_app_path, repo2)

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()
        Index(repo2, embedder=emb).build()

        app = create_app([py_app_path, repo2], embedder=emb)
        with TestClient(app) as client:
            # Without repo → 400.
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 400

            # With repo → 200.
            repo_name = py_app_path.name
            resp = client.post("/query", json={"question": "test", "repo": repo_name})
            assert resp.status_code == 200
            assert len(resp.json()["results"]) > 0

        shutil.rmtree(repo2)

    def test_status_requires_repo_when_multiple(self, py_app_path: Path) -> None:
        """GET /status without repo returns 400 when multiple repos loaded."""
        import shutil

        from source_recall import Index
        from source_recall.server import create_app

        repo2 = py_app_path.parent / "repo2"
        shutil.copytree(py_app_path, repo2)

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()
        Index(repo2, embedder=emb).build()

        app = create_app([py_app_path, repo2], embedder=emb)
        with TestClient(app) as client:
            # Without repo → 400.
            resp = client.get("/status")
            assert resp.status_code == 400

            # With repo → 200.
            resp = client.get(f"/status?repo={py_app_path.name}")
            assert resp.status_code == 200
            assert resp.json()["file_count"] > 0

        shutil.rmtree(repo2)

    def test_refresh_requires_repo_when_multiple(self, py_app_path: Path) -> None:
        """POST /refresh without repo returns 400 when multiple repos loaded."""
        import shutil

        from source_recall import Index
        from source_recall.server import create_app

        repo2 = py_app_path.parent / "repo2"
        shutil.copytree(py_app_path, repo2)

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()
        Index(repo2, embedder=emb).build()

        app = create_app([py_app_path, repo2], embedder=emb)
        with TestClient(app) as client:
            # Without repo → 400.
            resp = client.post("/refresh")
            assert resp.status_code == 400

            # With repo → 200.
            resp = client.post(f"/refresh?repo={py_app_path.name}")
            assert resp.status_code == 200

        shutil.rmtree(repo2)
