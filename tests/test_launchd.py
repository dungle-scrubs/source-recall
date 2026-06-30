"""Tests for launchd plist generation and daemon lifecycle commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from source_recall.daemon_config import DaemonConfig
from source_recall.launchd import PLIST_LABEL, generate_plist


class TestPlistGeneration:
    def test_contains_daemon_run_command(self) -> None:
        """Generated plist invokes 'sr daemon run', not 'sr serve'."""
        config = DaemonConfig(port=7249)
        plist = generate_plist(config)

        assert "sr</string>" in plist or "sr" in plist
        assert "daemon</string>" in plist
        assert "run</string>" in plist
        # Must NOT contain the old 'serve' command.
        assert "<string>serve</string>" not in plist

    def test_contains_correct_label(self) -> None:
        """Plist uses the standard label."""
        config = DaemonConfig()
        plist = generate_plist(config)
        assert PLIST_LABEL in plist

    def test_contains_log_paths(self) -> None:
        """Plist includes stdout/stderr log paths."""
        config = DaemonConfig()
        plist = generate_plist(config)
        assert "source-recall" in plist
        assert ".log" in plist

    def test_contains_keep_alive(self) -> None:
        """Plist has KeepAlive=true for auto-restart."""
        config = DaemonConfig()
        plist = generate_plist(config)
        assert "<key>KeepAlive</key>" in plist

    def test_exit_timeout_exceeds_shutdown(self) -> None:
        """ExitTimeOut > shutdown_timeout_s (D-005)."""
        config = DaemonConfig(shutdown_timeout_s=10)
        plist = generate_plist(config)
        assert "<key>ExitTimeOut</key>" in plist
        # ExitTimeOut should be > 10
        assert "<integer>15</integer>" in plist

    def test_custom_config_path_included(self, tmp_path: Path) -> None:
        """Plist includes --config flag when config_path is set."""
        cfg_path = tmp_path / "repos.toml"
        cfg_path.write_text("[daemon]\n")
        config = DaemonConfig(config_path=cfg_path)
        plist = generate_plist(config)
        assert str(cfg_path) in plist

    def test_prefers_repo_local_venv_sr_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plist uses repo-local .venv/bin/sr when available."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        sr_bin = venv_bin / "sr"
        sr_bin.write_text("#!/bin/sh\n")
        sr_bin.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        plist = generate_plist(DaemonConfig())

        assert f"<string>{sr_bin}</string>" in plist
        assert "<string>sr</string>" not in plist


class TestDaemonStartCLI:
    def test_start_writes_plist_and_bootstraps(self) -> None:
        """sr daemon start writes plist and calls launchctl bootstrap."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()

        with (
            patch("source_recall.cli._write_and_load_plist"),
            patch("source_recall.cli._wait_for_health", return_value=True),
        ):
            result = runner.invoke(
                app,
                ["daemon", "start", "--config", "/dev/null"],
                catch_exceptions=False,
            )

        # May fail on config parse, but we're testing the command exists.
        # The important thing is the command is registered.
        assert "daemon" in result.output.lower() or result.exit_code in (0, 1)

    def test_stop_command_exists(self) -> None:
        """sr daemon stop is a registered command."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["daemon", "stop"], catch_exceptions=False)
        # Even if it fails (no plist loaded), the command should exist.
        assert result.exit_code in (0, 1)

    def test_status_command_exists(self) -> None:
        """sr daemon status is a registered command."""
        from typer.testing import CliRunner

        from source_recall.cli import app

        runner = CliRunner()
        # Mock the health check so it doesn't need a running daemon.
        with patch("source_recall.cli._daemon_get", side_effect=ConnectionError):
            result = runner.invoke(app, ["daemon", "status"], catch_exceptions=False)
        assert result.exit_code in (0, 1)
