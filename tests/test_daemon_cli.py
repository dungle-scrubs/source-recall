"""Tests for daemon CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from source_recall.cli import app

runner = CliRunner()


class TestDaemonRun:
    def test_daemon_run_loads_config_and_starts(self, tmp_path: Path) -> None:
        """sr daemon run loads repos.toml and starts uvicorn."""
        config_path = tmp_path / "repos.toml"
        config_path.write_text("[daemon]\nport = 7249\n")

        with patch("source_recall.cli.uvicorn_run") as mock_run:
            result = runner.invoke(app, ["daemon", "run", "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        # Verify it was called with host/port from config.
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["port"] == 7249
        assert call_kwargs.kwargs["host"] == "127.0.0.1"

    def test_daemon_run_missing_config_exits(self) -> None:
        """sr daemon run with missing config exits with error."""
        result = runner.invoke(
            app, ["daemon", "run", "--config", "/nonexistent/repos.toml"]
        )
        assert result.exit_code != 0


class TestAddCommand:
    def test_add_posts_to_daemon(self, tmp_path: Path) -> None:
        """sr add sends POST /repos to daemon."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "name": "myrepo",
            "path": str(repo),
            "state": "queued",
            "error": None,
        }

        with patch("source_recall.cli._daemon_post", return_value=mock_resp):
            result = runner.invoke(app, ["add", str(repo)])

        assert result.exit_code == 0
        assert "myrepo" in result.output

    def test_add_shows_error_on_failure(self) -> None:
        """sr add shows error when daemon returns error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"detail": "bad path"}

        with patch("source_recall.cli._daemon_post", return_value=mock_resp):
            result = runner.invoke(app, ["add", "/fake/path"])

        assert result.exit_code == 1


class TestRemoveCommand:
    def test_remove_deletes_from_daemon(self) -> None:
        """sr remove sends DELETE /repos/{name} to daemon."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "removed", "name": "myrepo"}

        with patch("source_recall.cli._daemon_delete", return_value=mock_resp):
            result = runner.invoke(app, ["remove", "myrepo"])

        assert result.exit_code == 0
        assert "myrepo" in result.output


class TestReposCommand:
    def test_repos_lists_from_daemon(self) -> None:
        """sr repos fetches GET /repos from daemon."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "repos": [
                {
                    "name": "alpha",
                    "path": "/dev/alpha",
                    "state": "ready",
                    "error": None,
                },
                {
                    "name": "beta",
                    "path": "/dev/beta",
                    "state": "indexing",
                    "error": None,
                },
            ]
        }

        with patch("source_recall.cli._daemon_get", return_value=mock_resp):
            result = runner.invoke(app, ["repos"])

        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_repos_daemon_down_shows_error(self) -> None:
        """sr repos shows error when daemon unreachable."""
        with patch(
            "source_recall.cli._daemon_get",
            side_effect=ConnectionError("refused"),
        ):
            result = runner.invoke(app, ["repos"])

        assert result.exit_code == 1
        assert (
            "not running" in result.output.lower() or "refused" in result.output.lower()
        )
