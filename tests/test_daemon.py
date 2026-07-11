"""Tests for the daemon-mode server (multi-repo registry)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def daemon_config(py_app_path: Path, tmp_path: Path) -> DaemonConfig:
    """DaemonConfig with one pre-indexed repo."""
    from source_recall import Index

    emb = BagOfWordsEmbedder(dimensions=64)
    Index(py_app_path, embedder=emb).build()

    toml_path = tmp_path / "repos.toml"
    return DaemonConfig(
        repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
        config_path=toml_path,
    )


@pytest.fixture
def daemon_client(
    daemon_config: DaemonConfig,
) -> Generator[TestClient, None, None]:
    """TestClient for the daemon-mode app."""
    from source_recall.daemon import create_daemon_app

    emb = BagOfWordsEmbedder(dimensions=64)
    app = create_daemon_app(daemon_config, embedder=emb)
    with TestClient(app) as client:
        yield client


class TestGetRepos:
    def test_returns_registered_repos_with_state(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """GET /repos returns repos with state field."""
        resp = daemon_client.get("/repos")
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["repos"]) == 1
        repo = data["repos"][0]
        assert repo["name"] == py_app_path.name
        assert repo["state"] == "ready"
        assert "path" in repo


class TestPostRepos:
    def test_add_repo_returns_201(
        self, daemon_client: TestClient, tmp_path: Path
    ) -> None:
        """POST /repos adds a new repo and returns 201."""
        new_repo = tmp_path / "new-repo"
        new_repo.mkdir()
        # Create minimal git structure so index can work.
        (new_repo / "hello.py").write_text("x = 1\n")

        resp = daemon_client.post("/repos", json={"path": str(new_repo)})
        assert resp.status_code == 201
        assert resp.json()["name"] == "new-repo"
        # State is queued or indexing (background thread may start immediately).
        assert resp.json()["state"] in ("queued", "indexing")

    def test_add_repo_appears_in_list(
        self, daemon_client: TestClient, tmp_path: Path
    ) -> None:
        """Added repo shows up in GET /repos."""
        new_repo = tmp_path / "another-repo"
        new_repo.mkdir()
        (new_repo / "main.py").write_text("y = 2\n")

        daemon_client.post("/repos", json={"path": str(new_repo)})

        resp = daemon_client.get("/repos")
        names = [r["name"] for r in resp.json()["repos"]]
        assert "another-repo" in names

    def test_add_nonexistent_path_returns_400(self, daemon_client: TestClient) -> None:
        """POST /repos with bad path returns 400."""
        resp = daemon_client.post("/repos", json={"path": "/nonexistent/fake/repo"})
        assert resp.status_code == 400

    def test_add_duplicate_returns_409(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /repos for already-registered repo returns 409."""
        resp = daemon_client.post("/repos", json={"path": str(py_app_path)})
        assert resp.status_code == 409

    def test_add_writes_config_atomically(
        self,
        daemon_client: TestClient,
        daemon_config: DaemonConfig,
        tmp_path: Path,
    ) -> None:
        """POST /repos persists the new repo to repos.toml."""
        new_repo = tmp_path / "persisted-repo"
        new_repo.mkdir()
        (new_repo / "f.py").write_text("z = 3\n")

        daemon_client.post("/repos", json={"path": str(new_repo)})

        # Config file should exist and contain the new repo.
        assert daemon_config.config_path is not None
        content = daemon_config.config_path.read_text()
        assert "persisted-repo" in content


class TestDeleteRepos:
    def test_delete_returns_200(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """DELETE /repos/{name} removes and returns 200."""
        name = py_app_path.name
        resp = daemon_client.delete(f"/repos/{name}")
        assert resp.status_code == 200

        # Verify it's gone.
        repos = daemon_client.get("/repos").json()["repos"]
        assert all(r["name"] != name for r in repos)

    def test_delete_nonexistent_returns_404(self, daemon_client: TestClient) -> None:
        """DELETE /repos/{name} for unknown repo returns 404."""
        resp = daemon_client.delete("/repos/nonexistent")
        assert resp.status_code == 404

    def test_delete_writes_config(
        self,
        daemon_client: TestClient,
        daemon_config: DaemonConfig,
        py_app_path: Path,
    ) -> None:
        """DELETE /repos/{name} updates repos.toml."""
        name = py_app_path.name
        daemon_client.delete(f"/repos/{name}")

        assert daemon_config.config_path is not None
        content = daemon_config.config_path.read_text()
        assert str(py_app_path) not in content


class TestDaemonHealth:
    def test_health_shows_repos(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """GET /health includes repo names."""
        resp = daemon_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert py_app_path.name in data["repos"]


class TestDaemonQuery:
    def test_query_against_registered_repo(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /query works against daemon-managed repos."""
        resp = daemon_client.post(
            "/query",
            json={"question": "authenticate", "repo": py_app_path.name},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0

    def test_query_single_repo_without_name(self, daemon_client: TestClient) -> None:
        """POST /query without repo works when only one repo loaded."""
        resp = daemon_client.post("/query", json={"question": "authenticate"})
        assert resp.status_code == 200


class TestDaemonRefresh:
    def test_refresh_against_registered_repo(
        self, daemon_client: TestClient, py_app_path: Path
    ) -> None:
        """POST /refresh works against daemon-managed repos."""
        resp = daemon_client.post(f"/refresh?repo={py_app_path.name}")
        assert resp.status_code == 200
        assert "files_updated" in resp.json()


class TestDaemonAuth:
    def test_request_without_token_returns_401(
        self, daemon_config: DaemonConfig
    ) -> None:
        """Every route requires the local auth token."""
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        app = create_daemon_app(daemon_config, embedder=emb)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            # Strip the token the autouse fixture injects.
            client.headers.pop("x-sr-token", None)
            resp = client.get("/health")
        assert resp.status_code == 401

    def test_request_with_wrong_token_returns_401(
        self, daemon_config: DaemonConfig
    ) -> None:
        """A bad token is rejected."""
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        app = create_daemon_app(daemon_config, embedder=emb)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            resp = client.get("/health", headers={"X-SR-Token": "nope"})
        assert resp.status_code == 401

    def test_request_with_token_returns_200(self, daemon_client: TestClient) -> None:
        """The correct token (injected by the fixture) authorizes."""
        resp = daemon_client.get("/health")
        assert resp.status_code == 200

    def test_bearer_token_authorizes(self, daemon_config: DaemonConfig) -> None:
        """The Authorization: Bearer form is also accepted."""
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        app = create_daemon_app(daemon_config, embedder=emb)
        token = app.state.sr_token
        with TestClient(app, base_url="http://127.0.0.1") as client:
            client.headers.pop("x-sr-token", None)
            resp = client.get("/health", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


class TestDaemonHostValidation:
    def test_foreign_host_header_rejected(self, daemon_client: TestClient) -> None:
        """A non-loopback Host header is rejected (DNS-rebinding defense)."""
        resp = daemon_client.get("/health", headers={"host": "evil.example.com"})
        assert resp.status_code == 400

    def test_loopback_host_accepted(self, daemon_client: TestClient) -> None:
        """localhost Host header passes."""
        resp = daemon_client.get("/health", headers={"host": "localhost"})
        assert resp.status_code == 200


class TestAddRepoContainment:
    def test_add_repo_rejects_disallowed_path(self, daemon_client: TestClient) -> None:
        """A real directory outside the allowed roots is rejected with 403."""
        # mkdtemp lands in the system temp dir — outside the user's home and
        # outside the pytest tmp tree that holds the registered repo.
        outside = tempfile.mkdtemp()
        try:
            resp = daemon_client.post("/repos", json={"path": outside})
            assert resp.status_code == 403
        finally:
            os.rmdir(outside)
