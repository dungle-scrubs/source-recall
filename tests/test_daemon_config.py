"""Tests for daemon configuration parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from source_recall.daemon_config import DaemonConfig


class TestFromToml:
    def test_parses_multi_repo_config(self, tmp_path: Path) -> None:
        """DaemonConfig.from_toml() parses [daemon] + [[repos]]."""
        repo_a = tmp_path / "repo-a"
        repo_b = tmp_path / "repo-b"
        repo_a.mkdir()
        repo_b.mkdir()

        toml_path = tmp_path / "repos.toml"
        toml_path.write_text(
            f"""\
[daemon]
host = "127.0.0.1"
port = 8000
refresh_interval_s = 600

[[repos]]
path = "{repo_a}"

[[repos]]
path = "{repo_b}"
name = "custom-name"
"""
        )

        config = DaemonConfig.from_toml(toml_path)

        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.refresh_interval_s == 600
        assert len(config.repos) == 2
        assert config.repos[0].path == repo_a
        assert config.repos[0].name == "repo-a"  # auto-derived
        assert config.repos[1].path == repo_b
        assert config.repos[1].name == "custom-name"

    def test_defaults_when_daemon_section_missing(self, tmp_path: Path) -> None:
        """Defaults apply when [daemon] section is omitted."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        toml_path = tmp_path / "repos.toml"
        toml_path.write_text(
            f"""\
[[repos]]
path = "{repo}"
"""
        )

        config = DaemonConfig.from_toml(toml_path)

        assert config.host == "127.0.0.1"
        assert config.port == 7249
        assert config.refresh_interval_s == 300
        assert config.shutdown_timeout_s == 10
        assert len(config.repos) == 1

    def test_expands_tilde_and_resolves_symlinks(self, tmp_path: Path) -> None:
        """Paths with ~ are expanded and symlinks are resolved."""
        real_dir = tmp_path / "real-repo"
        real_dir.mkdir()
        link = tmp_path / "link-repo"
        link.symlink_to(real_dir)

        toml_path = tmp_path / "repos.toml"
        toml_path.write_text(
            f"""\
[[repos]]
path = "{link}"
"""
        )

        config = DaemonConfig.from_toml(toml_path)
        assert config.repos[0].path == real_dir.resolve()

    def test_invalid_toml_raises_config_error(self, tmp_path: Path) -> None:
        """Malformed TOML raises ConfigError."""
        from source_recall.models import ConfigError

        toml_path = tmp_path / "repos.toml"
        toml_path.write_text("[[[[invalid toml")

        with pytest.raises(ConfigError, match="invalid TOML"):
            DaemonConfig.from_toml(toml_path)

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        """Non-existent file raises ConfigError."""
        from source_recall.models import ConfigError

        with pytest.raises(ConfigError, match="not found"):
            DaemonConfig.from_toml(tmp_path / "nope.toml")

    def test_nonexistent_repo_path_raises_config_error(self, tmp_path: Path) -> None:
        """Repo path that doesn't exist raises ConfigError."""
        from source_recall.models import ConfigError

        toml_path = tmp_path / "repos.toml"
        toml_path.write_text(
            """\
[[repos]]
path = "/nonexistent/path/to/repo"
"""
        )

        with pytest.raises(ConfigError, match="does not exist"):
            DaemonConfig.from_toml(toml_path)

    def test_empty_repos_is_valid(self, tmp_path: Path) -> None:
        """Config with no repos is valid (daemon starts empty)."""
        toml_path = tmp_path / "repos.toml"
        toml_path.write_text(
            """\
[daemon]
port = 7249
"""
        )

        config = DaemonConfig.from_toml(toml_path)
        assert config.repos == []

    def test_config_path_stored(self, tmp_path: Path) -> None:
        """The config file path is stored for atomic writes."""
        toml_path = tmp_path / "repos.toml"
        toml_path.write_text("[daemon]\n")

        config = DaemonConfig.from_toml(toml_path)
        assert config.config_path == toml_path


class TestDefaultConfigPath:
    def test_default_path(self) -> None:
        """Default config path is ~/.config/source-recall/repos.toml."""
        assert DaemonConfig.default_config_path() == (
            Path.home() / ".config" / "source-recall" / "repos.toml"
        )


class TestToToml:
    def test_round_trips(self, tmp_path: Path) -> None:
        """to_toml() produces valid TOML that re-parses identically."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        config = DaemonConfig(
            host="127.0.0.1",
            port=8080,
            refresh_interval_s=120,
            shutdown_timeout_s=5,
            repos=[DaemonConfig.RepoEntry(path=repo, name="myrepo")],
            config_path=tmp_path / "repos.toml",
        )

        toml_str = config.to_toml()
        out_path = tmp_path / "repos.toml"
        out_path.write_text(toml_str)

        reloaded = DaemonConfig.from_toml(out_path)
        assert reloaded.port == 8080
        assert reloaded.refresh_interval_s == 120
        assert len(reloaded.repos) == 1
        assert reloaded.repos[0].name == "myrepo"
