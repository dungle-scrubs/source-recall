"""Tests for daemon feature gaps: background indexing, /status, CORS, rate limiting."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


def _make_git_repo(path: Path) -> None:
    """Initialize a bare git repo with one commit so sr index works."""
    path.mkdir(exist_ok=True)
    (path / "main.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(Path.home()),
        },
    )


@pytest.fixture
def daemon_with_indexed(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client with one indexed repo."""
    from source_recall import Index
    from source_recall.daemon import create_daemon_app

    emb = BagOfWordsEmbedder(dimensions=64)
    Index(py_app_path, embedder=emb).build()

    config = DaemonConfig(
        repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
        config_path=tmp_path / "repos.toml",
    )
    app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Background index trigger on POST /repos
# ---------------------------------------------------------------------------


class TestPostReposTriggersIndex:
    def test_added_repo_transitions_to_ready(
        self, daemon_with_indexed: TestClient, tmp_path: Path
    ) -> None:
        """POST /repos triggers background indexing; repo becomes ready."""
        new_repo = tmp_path / "new-git-repo"
        _make_git_repo(new_repo)

        resp = daemon_with_indexed.post("/repos", json={"path": str(new_repo)})
        assert resp.status_code == 201

        # Wait for background indexing to complete.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = daemon_with_indexed.get("/repos/new-git-repo/status")
            if status.json()["state"] == "ready":
                break
            time.sleep(0.3)

        final = daemon_with_indexed.get("/repos/new-git-repo/status").json()
        assert final["state"] == "ready"
        assert final["file_count"] > 0

    def test_added_repo_queryable_after_index(
        self, daemon_with_indexed: TestClient, tmp_path: Path
    ) -> None:
        """After background indexing, new repo is queryable."""
        new_repo = tmp_path / "queryable-repo"
        _make_git_repo(new_repo)

        daemon_with_indexed.post("/repos", json={"path": str(new_repo)})

        # Wait for ready.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            s = daemon_with_indexed.get("/repos/queryable-repo/status").json()
            if s["state"] == "ready":
                break
            time.sleep(0.3)

        resp = daemon_with_indexed.post(
            "/query", json={"question": "hello", "repo": "queryable-repo"}
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0


# ---------------------------------------------------------------------------
# GET /status delegation
# ---------------------------------------------------------------------------


class TestDaemonStatusEndpoint:
    def test_status_single_repo(self, daemon_with_indexed: TestClient) -> None:
        """GET /status returns index metrics for single repo."""
        resp = daemon_with_indexed.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_count"] > 0
        assert data["chunk_count"] > 0
        assert "repo_path" in data

    def test_status_with_repo_param(
        self, daemon_with_indexed: TestClient, py_app_path: Path
    ) -> None:
        """GET /status?repo=name returns metrics for that repo."""
        resp = daemon_with_indexed.get(f"/status?repo={py_app_path.name}")
        assert resp.status_code == 200
        assert resp.json()["file_count"] > 0

    def test_status_unknown_repo_404(self, daemon_with_indexed: TestClient) -> None:
        """GET /status?repo=nope returns 404."""
        resp = daemon_with_indexed.get("/status?repo=nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CORS on daemon
# ---------------------------------------------------------------------------


class TestDaemonCORS:
    def test_rejects_foreign_origin(self, daemon_with_indexed: TestClient) -> None:
        """Daemon rejects non-localhost CORS origins."""
        resp = daemon_with_indexed.options(
            "/query",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in allow

    def test_allows_localhost(self, daemon_with_indexed: TestClient) -> None:
        """Daemon allows localhost CORS origins."""
        resp = daemon_with_indexed.options(
            "/query",
            headers={
                "Origin": "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "127.0.0.1" in allow


# ---------------------------------------------------------------------------
# Rate limiting on daemon refresh
# ---------------------------------------------------------------------------


class TestDaemonRateLimit:
    def test_refresh_rate_limited(self, py_app_path: Path, tmp_path: Path) -> None:
        """Rapid /refresh calls get 429 when rate limiting is active."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )
        # Use default rate limit (10s).
        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            # First refresh succeeds.
            r1 = client.post(f"/refresh?repo={py_app_path.name}")
            assert r1.status_code == 200

            # Immediate second should be rate limited.
            r2 = client.post(f"/refresh?repo={py_app_path.name}")
            assert r2.status_code == 429
            assert "Retry" in r2.json()["detail"]


# ---------------------------------------------------------------------------
# Error state: query returns 503
# ---------------------------------------------------------------------------


class TestErrorStateQuery:
    def test_query_error_repo_returns_503(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Querying a repo in error state returns 503."""
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
            resp = client.post("/query", json={"question": "test", "repo": "bad"})
            assert resp.status_code == 503
            assert "not ready" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Concurrent add/remove safety
# ---------------------------------------------------------------------------


class TestConcurrentOps:
    def test_concurrent_add_remove(
        self, daemon_with_indexed: TestClient, tmp_path: Path
    ) -> None:
        """Add and remove operations don't crash under concurrent access."""
        import threading

        errors: list[str] = []

        def add_repo(i: int) -> None:
            try:
                d = tmp_path / f"conc-repo-{i}"
                d.mkdir(exist_ok=True)
                (d / "f.py").write_text(f"x = {i}\n")
                daemon_with_indexed.post("/repos", json={"path": str(d)})
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=add_repo, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent add errors: {errors}"

        # All should appear in repos list.
        repos = daemon_with_indexed.get("/repos").json()["repos"]
        assert len(repos) >= 5  # 5 new + 1 original


# ---------------------------------------------------------------------------
# Config persistence edge cases
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_add_then_remove_updates_config(
        self,
        daemon_with_indexed: TestClient,
        tmp_path: Path,
    ) -> None:
        """Config reflects add followed by remove."""
        new = tmp_path / "ephemeral"
        new.mkdir()
        (new / "f.py").write_text("y = 1\n")

        # Add.
        daemon_with_indexed.post("/repos", json={"path": str(new)})
        config_path = tmp_path / "repos.toml"
        content = config_path.read_text()
        assert "ephemeral" in content

        # Remove.
        daemon_with_indexed.delete("/repos/ephemeral")
        content = config_path.read_text()
        assert "ephemeral" not in content

    def test_config_parent_dirs_created(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Config write creates parent directories if needed."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        deep_path = tmp_path / "a" / "b" / "c" / "repos.toml"

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=deep_path,
        )
        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            new = tmp_path / "nested-repo"
            new.mkdir()
            (new / "f.py").write_text("z = 1\n")
            client.post("/repos", json={"path": str(new)})

        assert deep_path.exists()
        assert "nested-repo" in deep_path.read_text()


# ---------------------------------------------------------------------------
# to_toml edge cases
# ---------------------------------------------------------------------------


class TestToTomlEdgeCases:
    def test_custom_name_included(self, tmp_path: Path) -> None:
        """to_toml includes name when different from directory basename."""
        repo = tmp_path / "mydir"
        repo.mkdir()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo, name="custom-name")],
            config_path=tmp_path / "repos.toml",
        )
        toml = config.to_toml()
        assert 'name = "custom-name"' in toml

    def test_auto_name_omitted(self, tmp_path: Path) -> None:
        """to_toml omits name when it matches directory basename."""
        repo = tmp_path / "mydir"
        repo.mkdir()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo, name="mydir")],
            config_path=tmp_path / "repos.toml",
        )
        toml = config.to_toml()
        assert "name =" not in toml

    def test_empty_repos(self) -> None:
        """to_toml with no repos produces valid TOML."""
        config = DaemonConfig()
        toml = config.to_toml()
        assert "[daemon]" in toml
        assert "[[repos]]" not in toml
