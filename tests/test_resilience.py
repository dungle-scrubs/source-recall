"""Tests for daemon startup resilience."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


class TestBadRepoSurvival:
    def test_daemon_starts_with_corrupt_repo(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Daemon starts even if one repo has no index (marked error)."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        # Build index for good repo.
        Index(py_app_path, embedder=emb).build()

        # Bad repo — exists as dir but has no index.
        bad_repo = tmp_path / "bad-repo"
        bad_repo.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=py_app_path, name="good"),
                DaemonConfig.RepoEntry(path=bad_repo, name="bad"),
            ],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            # Daemon is alive.
            resp = client.get("/health")
            assert resp.status_code == 200

            # Good repo is ready, bad repo is error.
            repos = client.get("/repos").json()["repos"]
            states = {r["name"]: r["state"] for r in repos}
            assert states["good"] == "ready"
            assert states["bad"] == "error"

    def test_good_repo_queryable_despite_bad_sibling(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Good repos remain queryable even when siblings fail."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        bad_repo = tmp_path / "bad-repo"
        bad_repo.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=py_app_path, name="good"),
                DaemonConfig.RepoEntry(path=bad_repo, name="bad"),
            ],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.post(
                "/query",
                json={"question": "authenticate", "repo": "good"},
            )
            assert resp.status_code == 200
            assert len(resp.json()["results"]) > 0


class TestFtsOnlyMode:
    def test_health_reports_mode(self, py_app_path: Path, tmp_path: Path) -> None:
        """Health reports mode=fts_only when embedder is None."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        # Build FTS-only index.
        Index(py_app_path, embedder=None).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name="fts-repo")],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=None)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            assert data["mode"] == "fts_only"

    def test_health_reports_full_mode(self, py_app_path: Path, tmp_path: Path) -> None:
        """Health reports mode=full when embedder is loaded."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name="vec-repo")],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb)
        with TestClient(app) as client:
            resp = client.get("/health")
            data = resp.json()
            assert data["mode"] == "full"
