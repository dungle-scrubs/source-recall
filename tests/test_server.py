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
