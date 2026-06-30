"""Tests for sr ask daemon-first behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from source_recall.cli import app

runner = CliRunner()


class TestAskDaemonFirst:
    def test_ask_queries_daemon_when_running(self) -> None:
        """sr ask tries daemon HTTP first when running."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "chunk_id": "abc123",
                    "file_path": "src/auth.py",
                    "symbol_name": "authenticate",
                    "symbol_type": "function",
                    "content": "def authenticate(): pass",
                    "score": 0.95,
                    "start_line": 1,
                    "end_line": 1,
                    "search_quality": "ast",
                    "match_reason": "fts",
                    "repo_name": "myrepo",
                }
            ],
            "query_ms": 12.3,
        }

        with patch("source_recall.cli._try_daemon_query", return_value=mock_resp):
            result = runner.invoke(app, ["ask", "authenticate", "--json"])

        assert result.exit_code == 0
        assert "authenticate" in result.output

    def test_ask_falls_back_to_in_process(self, py_app_path: Path) -> None:
        """sr ask falls back to in-process when daemon is down."""
        from source_recall import Index
        from source_recall.embedder import BagOfWordsEmbedder

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        # Daemon not running — ConnectionError.
        with patch(
            "source_recall.cli._try_daemon_query",
            return_value=None,
        ):
            result = runner.invoke(
                app, ["ask", "authenticate", str(py_app_path), "--json"]
            )

        assert result.exit_code == 0
        assert "authenticate" in result.output or "auth" in result.output

    def test_ask_with_repo_filter(self) -> None:
        """sr ask --repo passes filter to daemon."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [],
            "query_ms": 5.0,
        }

        with patch(
            "source_recall.cli._try_daemon_query", return_value=mock_resp
        ) as mock_query:
            result = runner.invoke(
                app, ["ask", "test query", "--repo", "myrepo", "--json"]
            )

        assert result.exit_code == 0
        # Verify repo was passed through.
        call_kwargs = mock_query.call_args
        assert call_kwargs.kwargs.get("repo") == "myrepo" or "myrepo" in str(
            call_kwargs
        )
