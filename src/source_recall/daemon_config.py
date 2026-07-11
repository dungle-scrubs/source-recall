"""Daemon configuration: repos.toml parser and writer."""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from source_recall.models import ConfigError

# Name of the local auth-token file. The token gates every daemon route so a
# browser or other local process cannot drive the unauthenticated API
# (DNS-rebinding exfiltration). It lives at ONE canonical location — the default
# config dir — independent of any ``--config`` path, so the daemon and the CLI
# client always agree on the same secret. See ``load_or_create_token``.
_TOKEN_FILENAME = "token"


class TokenSecurityError(RuntimeError):
    """Raised when the auth-token file cannot be secured to 0600.

    Accepting a world- or group-readable token file would let any local process
    read the secret and drive the daemon API, so token creation fails closed if
    the file cannot be confined to owner-only permissions.
    """


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
    def default_config_dir(cls) -> Path:
        """Return the default config directory.

        @returns: ~/.config/source-recall
        """
        return Path.home() / ".config" / "source-recall"

    @classmethod
    def default_config_path(cls) -> Path:
        """Return the default config file location.

        @returns: ~/.config/source-recall/repos.toml
        """
        return cls.default_config_dir() / "repos.toml"

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


# ---------------------------------------------------------------------------
# Local auth token
# ---------------------------------------------------------------------------
#
# The daemon binds to loopback but is otherwise unauthenticated. A local
# auth token defeats two related attacks: (1) a malicious web page can hit
# 127.0.0.1 from the victim's browser (DNS-rebinding / CSRF) and drive the
# API; (2) any local process could add an arbitrary path and exfiltrate its
# contents. Every daemon route requires the token via an ``Authorization:
# Bearer <token>`` or ``X-SR-Token`` header. The token lives at the canonical
# default config dir with 0600 permissions — NOT beside a custom ``--config``
# repos.toml — so the CLI client (which always reads the default location) and a
# custom-config daemon agree on the same secret.


def token_file_path() -> Path:
    """Return the canonical path to the daemon auth-token file.

    The token always lives in the default config dir, independent of any
    ``--config`` path, so the daemon and the CLI client never disagree.

    @returns: Absolute path to the token file.
    """
    return DaemonConfig.default_config_dir() / _TOKEN_FILENAME


def _enforce_owner_only_fd(fd: int, path: Path) -> None:
    """Confine an OPEN token file to 0600, failing closed if it cannot be.

    Operates on the file descriptor, not the pathname, so it cannot be tricked
    into chmod-ing a different file the path was swapped to (TOCTOU) or the
    target of a symlink.

    @param fd: Open descriptor for the token file.
    @param path: Path (for error messages only).
    @raises TokenSecurityError: If fchmod fails or the mode is still not 0600.
    """
    try:
        os.fchmod(fd, 0o600)
    except OSError as e:
        raise TokenSecurityError(
            f"Could not set 0600 on auth-token file {path}: {e}. "
            "Refusing to use a token file that may be readable by others."
        ) from e
    mode = stat.S_IMODE(os.fstat(fd).st_mode)
    if mode != 0o600:
        raise TokenSecurityError(
            f"Auth-token file {path} has mode {oct(mode)} after chmod; "
            "expected 0600. Refusing to use a permissive token file."
        )


def _read_all_fd(fd: int) -> bytes:
    """Read a small file fully from an open descriptor."""
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def load_or_create_token() -> str:
    """Load the daemon auth token, generating one on first use.

    The token is a URL-safe random string persisted with 0600 permissions.

    Publication is ATOMIC: the token is written in full to a private,
    randomly-named temp file (``O_CREAT | O_EXCL | O_NOFOLLOW``, 0600) and then
    linked into place with ``os.link``. Because the token file appears at its
    canonical path only via that link — and only after the temp file is fully
    written — a concurrent reader ever sees either the complete token or no
    file at all, never a partial or empty one.

    ``os.link`` also elects a single winner across concurrent first-starts: it
    fails with ``FileExistsError`` if the canonical path already exists, so the
    losers read the winner's token instead of diverging. ``os.rename`` would
    give atomicity but silently overwrite, destroying that single-winner
    guarantee, so link is used deliberately.

    All permission and read operations go through the file descriptor
    (``O_NOFOLLOW`` + ``fstat`` / ``fchmod``) so a symlink or a pathname swapped
    underneath us cannot redirect them. Permission tightening is fail-closed —
    a token that cannot be confined to a regular, owner-only 0600 file is
    rejected.

    @returns: The auth token string.
    @raises TokenSecurityError: If the token file cannot be secured.
    """
    path = token_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: an already-published token is, by construction, complete —
    # so if the canonical file exists we simply read it (no rewrite).
    if path.exists():
        existing = _read_existing_token_secure(path)
        if existing is not None:
            return existing
        raise TokenSecurityError(
            f"Auth-token file {path} exists but is empty; refusing to "
            "overwrite a possibly in-progress or corrupt token."
        )

    token = secrets.token_urlsafe(32)
    data = token.encode()
    # Random temp name so a hostile pre-created file cannot win the O_EXCL
    # create and so concurrent writers never collide on the temp.
    tmp = path.parent / f".{_TOKEN_FILENAME}.{os.getpid()}.{secrets.token_hex(8)}.tmp"

    try:
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            # os.write may perform a short write; loop until every byte lands
            # so the published file is never truncated.
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
            _enforce_owner_only_fd(fd, tmp)
            os.fsync(fd)
        finally:
            os.close(fd)

        try:
            # Atomic publication + single-winner election in one step: link
            # fails if the canonical path already exists.
            os.link(tmp, path)
        except FileExistsError:
            existing = _read_existing_token_secure(path)
            if existing is not None:
                return existing
            # Canonical path exists but is empty — a create raced ahead of its
            # write, or the file is corrupt. Fail closed rather than diverge.
            raise TokenSecurityError(
                f"Auth-token file {path} exists but is empty; refusing to "
                "overwrite a possibly in-progress or corrupt token."
            ) from None
    finally:
        # The published inode survives via ``path``; drop the temp name.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)

    return token


def _open_token_no_follow(path: Path) -> int:
    """Open an existing token file for reading without following symlinks.

    Verifies via ``fstat`` on the descriptor that the target is a regular file
    owned by the current user, so a planted symlink or a foreign-owned file
    cannot supply an attacker-known token.

    @param path: Token file path.
    @returns: Open read-only descriptor.
    @raises TokenSecurityError: If the path is a symlink, not a regular file, or
        not owned by the current user.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        raise TokenSecurityError(
            f"Refusing to read auth-token file {path}: {e} "
            "(is it a symlink pointing outside the config dir?)."
        ) from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise TokenSecurityError(
                f"Auth-token file {path} is not a regular file; refusing to use it."
            )
        if st.st_uid != os.getuid():
            raise TokenSecurityError(
                f"Auth-token file {path} is not owned by the current user; "
                "refusing to trust a foreign-owned token."
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_existing_token_secure(path: Path, attempts: int = 20) -> str | None:
    """Securely read a token another process may still be writing.

    Re-opens with ``O_NOFOLLOW`` each attempt, tightens perms fail-closed on the
    descriptor, and retries briefly to cover the window between a concurrent
    ``O_EXCL`` create and its write. Returns ``None`` if the file stays empty.

    @param path: Token file path.
    @param attempts: Max read attempts before giving up.
    @returns: The token string, or ``None`` if it never became non-empty.
    """
    import time

    for _ in range(attempts):
        try:
            fd = _open_token_no_follow(path)
        except OSError:
            time.sleep(0.01)
            continue
        try:
            _enforce_owner_only_fd(fd, path)
            value = _read_all_fd(fd).decode("utf-8", "strict").strip()
        finally:
            os.close(fd)
        if value:
            return value
        time.sleep(0.01)
    return None


def load_token() -> str | None:
    """Read the daemon auth token without creating one.

    Used by the CLI client so ``sr ask`` / ``sr refresh`` can forward the
    token. Reads the canonical default location without following symlinks.
    Returns ``None`` when no token exists yet (daemon never started).

    @returns: The token string, or ``None`` if absent.
    """
    path = token_file_path()
    if not path.exists():
        return None
    try:
        fd = _open_token_no_follow(path)
    except TokenSecurityError:
        raise
    except OSError:
        return None
    try:
        value = _read_all_fd(fd).decode("utf-8", "strict").strip()
    finally:
        os.close(fd)
    return value or None
