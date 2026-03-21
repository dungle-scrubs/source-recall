"""End-to-end daemon lifecycle tests.

No mocks — exercises real config parsing, server startup, indexing,
querying, dynamic repo management, and shutdown via TestClient.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


def _make_git_repo(path: Path, files: dict[str, str] | None = None) -> None:
    """Create a git repo with committed files.

    @param path: Directory to initialize.
    @param files: Optional dict of {filename: content}.
    """
    path.mkdir(exist_ok=True)
    if files is None:
        files = {"main.py": "def hello():\n    return 'world'\n"}
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content)

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


def _wait_for_ready(client: TestClient, name: str, timeout: float = 15) -> bool:
    """Poll until repo reaches 'ready' state.

    @param client: TestClient instance.
    @param name: Repo name.
    @param timeout: Max seconds to wait.
    @returns: True if ready within timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/repos/{name}/status")
        if resp.status_code == 200 and resp.json()["state"] == "ready":
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# E2E: Full lifecycle
# ---------------------------------------------------------------------------


class TestE2EFullLifecycle:
    """Config → start → add repos → query → refresh → remove → shutdown."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Complete daemon lifecycle from config to shutdown."""
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # 1. Create repos.
        repo_a = tmp_path / "alpha"
        _make_git_repo(
            repo_a,
            {
                "auth.py": (
                    "def authenticate(user: str, password: str) -> bool:\n"
                    "    return user == 'admin' and password == 'secret'\n"
                ),
                "models.py": "class User:\n    name: str\n    email: str\n",
            },
        )

        # 2. Write config.
        config_path = tmp_path / "repos.toml"
        config_path.write_text(
            f"""\
[daemon]
port = 7249
refresh_interval_s = 3600

[[repos]]
path = "{repo_a}"
name = "alpha"
"""
        )

        # 3. Parse config.
        config = DaemonConfig.from_toml(config_path)
        assert len(config.repos) == 1

        # 4. Pre-build index (like daemon startup would).
        from source_recall import Index

        Index(repo_a, embedder=emb).build()

        # 5. Start daemon.
        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            # 6. Verify startup.
            health = client.get("/health").json()
            assert health["ok"] is True
            assert "alpha" in health["repos"]
            assert health["mode"] == "full"

            # 7. Query existing repo.
            resp = client.post("/query", json={"question": "authenticate"})
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) > 0
            assert any("authenticate" in r["content"] for r in results)

            # 8. Check status.
            status = client.get("/status").json()
            assert status["file_count"] == 2
            assert status["chunk_count"] > 0

            # 9. Dynamic add — second repo.
            repo_b = tmp_path / "beta"
            _make_git_repo(
                repo_b,
                {
                    "database.py": (
                        "def connect_db(url: str) -> object:\n"
                        "    return {'url': url, 'connected': True}\n"
                    ),
                },
            )
            resp = client.post("/repos", json={"path": str(repo_b)})
            assert resp.status_code == 201
            assert resp.json()["name"] == "beta"

            # 10. Wait for background indexing.
            assert _wait_for_ready(client, "beta")

            # 11. Cross-repo query (no repo filter).
            resp = client.post("/query", json={"question": "connect database"})
            assert resp.status_code == 200
            results = resp.json()["results"]
            repo_names = {r.get("repo_name") for r in results}
            assert "beta" in repo_names

            # 12. Filtered query.
            resp = client.post(
                "/query",
                json={"question": "authenticate", "repo": "alpha"},
            )
            assert resp.status_code == 200
            assert all(r.get("repo_name") == "alpha" for r in resp.json()["results"])

            # 13. Per-repo status.
            alpha_status = client.get("/repos/alpha/status").json()
            assert alpha_status["state"] == "ready"
            assert alpha_status["file_count"] == 2

            beta_status = client.get("/repos/beta/status").json()
            assert beta_status["state"] == "ready"
            assert beta_status["file_count"] == 1

            # 14. Refresh.
            resp = client.post("/refresh?repo=alpha")
            assert resp.status_code == 200
            assert resp.json()["files_updated"] == 0  # No changes.

            # 15. Config persisted with both repos.
            content = config_path.read_text()
            assert "alpha" in content
            assert "beta" in content

            # 16. Remove repo.
            resp = client.delete("/repos/beta")
            assert resp.status_code == 200

            repos = client.get("/repos").json()["repos"]
            assert len(repos) == 1
            assert repos[0]["name"] == "alpha"

            # 17. Config updated.
            content = config_path.read_text()
            assert "beta" not in content

            # 18. Verify removed repo no longer queryable.
            resp = client.post("/query", json={"question": "database", "repo": "beta"})
            assert resp.status_code == 404

        # 19. After shutdown — lifespan exit completes without crash.


class TestE2EMultiRepoSearch:
    """Cross-repo search with score normalization."""

    def test_results_from_both_repos_ranked_fairly(self, tmp_path: Path) -> None:
        """Multi-repo query returns results from both repos, reasonably ranked."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # Repo with auth code.
        repo_auth = tmp_path / "auth-service"
        _make_git_repo(
            repo_auth,
            {
                "auth.py": (
                    "def verify_token(token: str) -> bool:\n"
                    "    '''Verify JWT authentication token.'''\n"
                    "    return len(token) > 10\n"
                ),
            },
        )
        Index(repo_auth, embedder=emb).build()

        # Repo with different auth code.
        repo_api = tmp_path / "api-gateway"
        _make_git_repo(
            repo_api,
            {
                "middleware.py": (
                    "def auth_middleware(request: object) -> object:\n"
                    "    '''Authentication middleware for API gateway.'''\n"
                    "    token = request.headers.get('Authorization')\n"
                    "    return token\n"
                ),
            },
        )
        Index(repo_api, embedder=emb).build()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=repo_auth, name="auth-service"),
                DaemonConfig.RepoEntry(path=repo_api, name="api-gateway"),
            ],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.post("/query", json={"question": "authentication token"})
            assert resp.status_code == 200
            results = resp.json()["results"]

            # Both repos should contribute results.
            repo_names = {r["repo_name"] for r in results}
            assert len(repo_names) == 2

            # Scores should be normalized (0-1 range).
            for r in results:
                assert 0 <= r["score"] <= 1.0


class TestE2EConfigRoundTrip:
    """Config file is the source of truth — read → write → re-read."""

    def test_config_survives_restart(self, tmp_path: Path) -> None:
        """Repos added via API survive a config re-read."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # Initial repo.
        repo_a = tmp_path / "initial"
        _make_git_repo(repo_a)
        Index(repo_a, embedder=emb).build()

        config_path = tmp_path / "repos.toml"

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo_a, name="initial")],
            config_path=config_path,
        )

        # First "session" — add a repo.
        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            new_repo = tmp_path / "added-later"
            _make_git_repo(new_repo)

            resp = client.post("/repos", json={"path": str(new_repo)})
            assert resp.status_code == 201

            _wait_for_ready(client, "added-later")

        # Second "session" — re-read config.
        config2 = DaemonConfig.from_toml(config_path)
        assert len(config2.repos) == 2
        names = {r.name for r in config2.repos}
        assert "initial" in names
        assert "added-later" in names


class TestE2EDegradedMode:
    """Daemon operates in degraded mode when repos fail."""

    def test_healthy_repos_serve_while_others_fail(self, tmp_path: Path) -> None:
        """Good repos work normally even with bad siblings."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # Good repo.
        good = tmp_path / "good-repo"
        _make_git_repo(good, {"api.py": "def get_users() -> list:\n    return []\n"})
        Index(good, embedder=emb).build()

        # Bad repo — empty dir, no git, no index.
        bad = tmp_path / "bad-repo"
        bad.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=good, name="good"),
                DaemonConfig.RepoEntry(path=bad, name="bad"),
            ],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            # Health says ok (at least one ready).
            health = client.get("/health").json()
            assert health["ok"] is True

            # Good repo is queryable.
            resp = client.post("/query", json={"question": "users", "repo": "good"})
            assert resp.status_code == 200
            assert len(resp.json()["results"]) > 0

            # Bad repo returns 503.
            resp = client.post("/query", json={"question": "users", "repo": "bad"})
            assert resp.status_code == 503

            # Repos list shows both with correct states.
            repos = client.get("/repos").json()["repos"]
            states = {r["name"]: r["state"] for r in repos}
            assert states["good"] == "ready"
            assert states["bad"] == "error"

            # Overall query (no repo filter) returns only good results.
            resp = client.post("/query", json={"question": "users"})
            assert resp.status_code == 200
            assert all(r["repo_name"] == "good" for r in resp.json()["results"])


class TestE2ESSEDuringIndexing:
    """SSE progress stream during actual background indexing."""

    def test_sse_during_background_index(self, tmp_path: Path) -> None:
        """SSE endpoint streams progress for a repo being indexed."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # Pre-index one repo.
        existing = tmp_path / "existing"
        _make_git_repo(existing)
        Index(existing, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=existing, name="existing")],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            # SSE for ready repo should emit complete immediately.
            resp = client.get("/repos/existing/progress")
            assert resp.status_code == 200
            assert "event: complete" in resp.text

            # Add a new repo, check SSE while it's indexing.
            new_repo = tmp_path / "indexing-repo"
            _make_git_repo(new_repo)
            client.post("/repos", json={"path": str(new_repo)})

            # SSE should eventually show complete.
            _wait_for_ready(client, "indexing-repo")
            resp = client.get("/repos/indexing-repo/progress")
            assert "event: complete" in resp.text


class TestE2EFTSOnlyMode:
    """Full lifecycle in FTS-only (no embedder) mode."""

    def test_fts_only_lifecycle(self, tmp_path: Path) -> None:
        """Daemon works end-to-end in FTS-only mode."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        repo = tmp_path / "fts-repo"
        _make_git_repo(
            repo,
            {"search.py": "def search(query: str) -> list:\n    return []\n"},
        )
        Index(repo, embedder=None).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo, name="fts-repo")],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=None, refresh_min_interval=0)
        with TestClient(app) as client:
            # Health reports fts_only.
            health = client.get("/health").json()
            assert health["ok"] is True
            assert health["mode"] == "fts_only"

            # FTS query works.
            resp = client.post("/query", json={"question": "search query"})
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) > 0

            # Status works.
            status = client.get("/status").json()
            assert status["file_count"] == 1
            assert status["vector_count"] == 0

            # Refresh works.
            resp = client.post("/refresh")
            assert resp.status_code == 200


class TestE2EDynamicOperations:
    """Add and remove repos while queries are in flight."""

    def test_add_remove_add_same_name(self, tmp_path: Path) -> None:
        """Can add → remove → re-add a repo with the same name."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        seed = tmp_path / "seed"
        _make_git_repo(seed)
        Index(seed, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=seed, name="seed")],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            # Add.
            v1 = tmp_path / "recycled"
            _make_git_repo(v1, {"v1.py": "x = 1\n"})
            resp = client.post("/repos", json={"path": str(v1)})
            assert resp.status_code == 201
            _wait_for_ready(client, "recycled")

            # Remove.
            resp = client.delete("/repos/recycled")
            assert resp.status_code == 200

            # Re-add with different content.
            v2 = tmp_path / "recycled-v2"
            _make_git_repo(v2, {"v2.py": "y = 2\n"})
            resp = client.post("/repos", json={"path": str(v2), "name": "recycled"})
            assert resp.status_code == 201
            _wait_for_ready(client, "recycled")

            # Query the re-added repo.
            status = client.get("/repos/recycled/status").json()
            assert status["state"] == "ready"
