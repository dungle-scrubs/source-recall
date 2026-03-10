"""3-level config resolution: env vars > repo-local TOML > defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "node_modules/",
    "__pycache__/",
    ".git/",
    "dist/",
    "build/",
    "*.min.js",
    "*.map",
    "*.lock",
    "package-lock.json",
    "*.pyc",
    ".env",
    "venv/",
    ".venv/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    "*.egg-info/",
)


class SRConfig(BaseSettings):
    """Source-recall configuration.

    Resolution order (highest wins):
    1. Environment variables prefixed with ``SR_``
    2. Repo-local ``.source-recall.toml``
    3. Hard-coded defaults

    @param max_file_size: Skip files larger than this (bytes).
    @param top_k: Number of results returned by queries.
    @param chunk_max_chars: Character threshold for sub-chunking.
    @param auto_refresh: Refresh stale indexes before queries.
    @param exclude_patterns: Glob patterns to exclude from indexing.
    """

    model_config = SettingsConfigDict(
        env_prefix="SR_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    max_file_size: int = Field(default=100_000, gt=0)
    top_k: int = Field(default=8, gt=0, le=100)
    chunk_max_chars: int = Field(default=6000, gt=500)
    auto_refresh: bool = True
    exclude_patterns: tuple[str, ...] = _DEFAULT_EXCLUDES
    embed_enabled: bool = True
    embed_batch_size: int = Field(default=32, gt=0, le=512)

    @field_validator("exclude_patterns", mode="before")
    @classmethod
    def _coerce_excludes(cls, v: Any) -> tuple[str, ...]:
        """Accept lists from TOML and coerce to tuple.

        @param v: Raw value from config source.
        @returns: Tuple of exclude patterns.
        """
        if isinstance(v, list):
            return tuple(v)
        return v


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return the [source-recall] table.

    @param path: Absolute path to the TOML file.
    @returns: Dict of config values (empty if file missing or no table).
    """
    if not path.is_file():
        return {}
    try:
        import tomllib

        data = tomllib.loads(path.read_text())
        return data.get("source-recall", data)
    except Exception:
        return {}


def resolve_config(
    repo_path: str | Path | None = None,
    **overrides: Any,
) -> SRConfig:
    """Build the effective config from all sources.

    Priority: overrides (caller intent) > env vars > TOML > defaults.

    @param repo_path: Repository root — used to locate .source-recall.toml.
    @param overrides: Explicit keyword overrides (equivalent to env vars).
    @returns: Resolved SRConfig instance.
    """
    toml_values: dict[str, Any] = {}
    if repo_path is not None:
        toml_path = Path(repo_path) / ".source-recall.toml"
        toml_values = _load_toml(toml_path)

    # TOML provides defaults that env vars and overrides can beat.
    # pydantic-settings reads env vars automatically.
    merged = {**toml_values, **overrides}

    # Filter out None values so pydantic defaults aren't overridden by None.
    merged = {k: v for k, v in merged.items() if v is not None}

    return SRConfig(**merged)


def format_config(config: SRConfig) -> str:
    """Render the resolved config as TOML for ``sr config show``.

    @param config: Resolved config instance.
    @returns: TOML-formatted string.
    """
    lines = ["[source-recall]"]
    lines.append(f"max_file_size = {config.max_file_size}")
    lines.append(f"top_k = {config.top_k}")
    lines.append(f"chunk_max_chars = {config.chunk_max_chars}")
    lines.append(f"auto_refresh = {'true' if config.auto_refresh else 'false'}")
    lines.append(f"embed_enabled = {'true' if config.embed_enabled else 'false'}")
    lines.append(f"embed_batch_size = {config.embed_batch_size}")

    excludes = ", ".join(f'"{p}"' for p in config.exclude_patterns)
    lines.append(f"exclude_patterns = [{excludes}]")

    return "\n".join(lines) + "\n"
