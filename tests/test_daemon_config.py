"""Tests for daemon configuration parser."""

from __future__ import annotations

import os
import stat
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
    def test_default_path_composes_from_dir(self) -> None:
        """default_config_path() is default_config_dir()/repos.toml."""
        assert (
            DaemonConfig.default_config_path()
            == DaemonConfig.default_config_dir() / "repos.toml"
        )

    def test_canonical_default_dir_is_config_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real (unpatched) default dir is ~/.config/source-recall."""
        # Undo the autouse config-dir isolation so we can assert the real
        # canonical location this test exists to regression-guard.
        monkeypatch.undo()
        assert DaemonConfig.default_config_dir() == (
            Path.home() / ".config" / "source-recall"
        )


class TestTokenLifecycle:
    """load_or_create_token: atomic O_EXCL create + fail-closed permissions."""

    def test_creates_token_with_0600(self, tmp_path: Path) -> None:
        """A freshly created token file is owner-read/write only."""
        from source_recall.daemon_config import (
            load_or_create_token,
            token_file_path,
        )

        token = load_or_create_token()
        assert token
        path = token_file_path()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600

    def test_existing_token_not_overwritten(self, tmp_path: Path) -> None:
        """When the token file already exists, its value is read, not clobbered.

        This is the O_EXCL path: create fails with FileExistsError, so we must
        fall back to reading the existing secret rather than generating a new
        one (which would diverge from an already-running daemon).
        """
        from source_recall.daemon_config import (
            load_or_create_token,
            token_file_path,
        )

        path = token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, b"preexisting-token")
        os.close(fd)

        assert load_or_create_token() == "preexisting-token"

    def test_concurrent_first_start_agree_on_one_token(self, tmp_path: Path) -> None:
        """Concurrent first-starts converge on a single token, never diverge."""
        import threading

        from source_recall.daemon_config import load_or_create_token

        results: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            tok = load_or_create_token()
            with lock:
                results.append(tok)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1

    def test_fail_closed_when_chmod_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the token file cannot be made 0600, creation fails closed."""
        from source_recall import daemon_config as dc

        def boom(*_a: object, **_k: object) -> None:
            raise OSError("chmod refused")

        monkeypatch.setattr(dc.os, "fchmod", boom)
        with pytest.raises(dc.TokenSecurityError):
            dc.load_or_create_token()

    def test_fail_closed_when_perms_stay_permissive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If perms remain permissive after chmod, creation fails closed."""
        from source_recall import daemon_config as dc

        # fchmod is a no-op, so the file keeps whatever mode os.open gave it,
        # which the fail-closed check must reject if it is not 0600.
        monkeypatch.setattr(dc.os, "fchmod", lambda *_a, **_k: None)

        real_fstat = os.fstat

        def fake_fstat(fd: int, *a: object, **k: object) -> object:
            st = real_fstat(fd, *a, **k)

            class _S:
                st_mode = (st.st_mode & ~0o777) | 0o644

            return _S()

        monkeypatch.setattr(dc.os, "fstat", fake_fstat)
        with pytest.raises(dc.TokenSecurityError):
            dc.load_or_create_token()

    def test_symlink_token_rejected(self, tmp_path: Path) -> None:
        """A token path that is a symlink is refused (no symlink following)."""
        from source_recall.daemon_config import (
            TokenSecurityError,
            load_or_create_token,
            token_file_path,
        )

        target = tmp_path / "attacker-token"
        target.write_text("attacker-known-token")
        path = token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)

        with pytest.raises(TokenSecurityError):
            load_or_create_token()


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
