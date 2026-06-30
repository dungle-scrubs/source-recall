"""Tests for the sr --version command."""

from __future__ import annotations

from typer.testing import CliRunner

from source_recall.__init__ import __version__ as pkg_version
from source_recall.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag_prints_version_and_exits_zero(self) -> None:
        """sr --version prints the package version and exits 0."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0, result.output
        assert pkg_version in result.output

    def test_version_short_flag_works(self) -> None:
        """sr -v prints the package version and exits 0."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0, result.output
        assert pkg_version in result.output

    def test_version_does_not_require_subcommand(self) -> None:
        """--version is eager: works without a subcommand."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Should not print the help text.
        assert "Quick start" not in result.output
