"""Daemon configuration: repos.toml parser and writer."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from source_recall.models import ConfigError


class DaemonConfig(BaseModel):
    """Daemon-mode configuration parsed from repos.toml.

    @param host: Bind address (localhost only by default).
    @param port: HTTP port.
    @param refresh_interval_s: Seconds between periodic refreshes.
    @param shutdown_timeout_s: Max seconds for graceful shutdown drain.
    @param repos: List of repository entries to manage.
    @param config_path: Path to the source repos.toml (for atomic writes).
    """

    class RepoEntry(BaseModel):
        """A single repository in the daemon config.

        @param path: Absolute, resolved path to the repo root.
        @param name: Short display name (defaults to directory basename).
        """

        path: Path
        name: str = ""

    host: str = "127.0.0.1"
    port: int = Field(default=7249, gt=0, le=65535)
    refresh_interval_s: int = Field(default=300, gt=0)
    shutdown_timeout_s: int = Field(default=10, gt=0)
    repos: list[RepoEntry] = Field(default_factory=list)
    config_path: Path | None = None

    @classmethod
    def default_config_path(cls) -> Path:
        """Return the default config file location.

        @returns: ~/.config/source-recall/repos.toml
        """
        return Path.home() / ".config" / "source-recall" / "repos.toml"

    @classmethod
    def from_toml(cls, path: Path) -> DaemonConfig:
        """Parse a repos.toml file into a DaemonConfig.

        Expands ~ in paths, resolves symlinks, and validates that
        repo directories exist.

        @param path: Path to the TOML config file.
        @returns: Parsed DaemonConfig.
        @raises ConfigError: If file missing, invalid TOML, or bad paths.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError("config_path", str(path), "config file not found")

        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("config_path", str(path), f"invalid TOML: {e}") from e

        daemon_section = data.get("daemon", {})
        raw_repos = data.get("repos", [])

        repos: list[DaemonConfig.RepoEntry] = []
        for entry in raw_repos:
            raw_path = entry.get("path", "")
            resolved = Path(raw_path).expanduser().resolve()
            if not resolved.is_dir():
                raise ConfigError(
                    "repos.path",
                    str(raw_path),
                    "path does not exist or is not a directory",
                )
            name = entry.get("name", "") or resolved.name
            repos.append(cls.RepoEntry(path=resolved, name=name))

        return cls(
            host=daemon_section.get("host", "127.0.0.1"),
            port=daemon_section.get("port", 7249),
            refresh_interval_s=daemon_section.get("refresh_interval_s", 300),
            shutdown_timeout_s=daemon_section.get("shutdown_timeout_s", 10),
            repos=repos,
            config_path=path.resolve(),
        )

    @staticmethod
    def _toml_escape(s: str) -> str:
        """Escape a string for TOML basic string value.

        Handles backslashes, double quotes, and control characters
        that are forbidden as literals inside TOML basic strings.

        @param s: Raw string.
        @returns: Escaped string safe for use inside TOML double quotes.
        """
        s = s.replace("\\", "\\\\")
        s = s.replace('"', '\\"')
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        s = s.replace("\t", "\\t")
        return s

    def to_toml(self) -> str:
        """Serialize this config to TOML format.

        @returns: TOML string suitable for writing to repos.toml.
        """
        esc = self._toml_escape
        lines = ["[daemon]"]
        lines.append(f'host = "{esc(self.host)}"')
        lines.append(f"port = {self.port}")
        lines.append(f"refresh_interval_s = {self.refresh_interval_s}")
        lines.append(f"shutdown_timeout_s = {self.shutdown_timeout_s}")

        for repo in self.repos:
            lines.append("")
            lines.append("[[repos]]")
            lines.append(f'path = "{esc(str(repo.path))}"')
            if repo.name != repo.path.name:
                lines.append(f'name = "{esc(repo.name)}"')

        lines.append("")
        return "\n".join(lines)
