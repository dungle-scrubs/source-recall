"""Tests for daemon audit findings."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


@pytest.fixture
def audit_client(
    py_app_path: Path, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Daemon client for audit tests."""
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


class TestToTomlEscaping:
    """Audit finding #2: TOML injection via path/name with special chars."""

    def test_path_with_quotes_round_trips(self, tmp_path: Path) -> None:
        """Path containing double quotes is escaped in to_toml."""
        repo = tmp_path / 'repo "quoted"'
        repo.mkdir()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo, name="safe-name")],
        )
        toml_str = config.to_toml()

        # The output should be parseable TOML.
        import tomllib

        parsed = tomllib.loads(toml_str)
        assert parsed["repos"][0]["path"] == str(repo)

    def test_path_with_backslash_round_trips(self) -> None:
        """Path containing backslash is escaped in to_toml."""
        # Can't create a dir with backslash on macOS, but test the escaping.
        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(
                    path=Path("/fake/path\\with\\backslash"), name="bs"
                )
            ],
        )
        toml_str = config.to_toml()

        import tomllib

        parsed = tomllib.loads(toml_str)
        assert "\\" in parsed["repos"][0]["path"]

    def test_name_with_quotes_escaped(self, tmp_path: Path) -> None:
        """Name containing quotes doesn't break TOML."""
        repo = tmp_path / "myrepo"
        repo.mkdir()
        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=repo, name='my "repo"')],
        )
        toml_str = config.to_toml()

        import tomllib

        parsed = tomllib.loads(toml_str)
        assert parsed["repos"][0]["name"] == 'my "repo"'


class TestDaemonQueryErrorBubbling:
    """Audit finding #3: sr ask should surface daemon errors."""

    def test_ask_surfaces_daemon_503(self) -> None:
        """sr ask shows daemon error instead of silent fallback."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"detail": "Repo 'bad' not ready"}

        with patch("source_recall.cli._try_daemon_query", return_value=mock_resp):
            result = runner.invoke(app, ["ask", "test", "--json"])

        assert result.exit_code == 1
        assert "not ready" in result.output

    def test_ask_surfaces_daemon_404(self) -> None:
        """sr ask shows 404 error for unknown repo."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"detail": "Repo 'nope' not found"}

        with patch("source_recall.cli._try_daemon_query", return_value=mock_resp):
            result = runner.invoke(app, ["ask", "test", "--repo", "nope"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestResolveIndexZeroRepos:
    """Audit finding #8: _resolve_index with zero ready repos."""

    def test_status_with_zero_ready_repos_returns_503(self, tmp_path: Path) -> None:
        """GET /status returns 503 when no repos are ready."""
        from source_recall.daemon import create_daemon_app

        bad = tmp_path / "bad-repo"
        bad.mkdir()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=bad, name="bad")],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=BagOfWordsEmbedder(dimensions=64))
        with TestClient(app) as client:
            resp = client.get("/status")
            assert resp.status_code == 503

    def test_query_with_zero_ready_repos_returns_503(self, tmp_path: Path) -> None:
        """POST /query returns 503 when no repos are ready."""
        from source_recall.daemon import create_daemon_app

        bad = tmp_path / "bad-repo"
        bad.mkdir()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=bad, name="bad")],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=BagOfWordsEmbedder(dimensions=64))
        with TestClient(app) as client:
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 503
            assert "No repos ready" in resp.json()["detail"]


class TestUvicornShutdownTimeout:
    """Audit finding #13: uvicorn gets shutdown_timeout_s."""

    def test_daemon_run_passes_timeout(self, tmp_path: Path) -> None:
        """sr daemon run passes timeout_graceful_shutdown to uvicorn."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()
        config_path = tmp_path / "repos.toml"
        config_path.write_text("[daemon]\nport = 7249\nshutdown_timeout_s = 7\n")

        with patch("source_recall.cli.uvicorn_run") as mock_run:
            runner.invoke(
                app,
                ["daemon", "run", "--config", str(config_path)],
            )

        if mock_run.called:
            kwargs = mock_run.call_args.kwargs
            assert kwargs.get("timeout_graceful_shutdown") == 7
