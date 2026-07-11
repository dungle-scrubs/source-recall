"""3-level config resolution: env vars > repo-local TOML > defaults."""

from __future__ import annotations

import os
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
    @param graph_expand_enabled: Expand top results along the ref graph.
        Set ``SR_GRAPH_EXPAND_ENABLED=false`` (or in TOML) to skip
        expansion entirely and avoid its ref-graph round-trips.
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
    rerank_enabled: bool = False
    graph_expand_enabled: bool = True

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
    @raises ConfigError: If the file exists but contains invalid TOML.
    """
    if not path.is_file():
        return {}
    import tomllib

    from source_recall.models import ConfigError

    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError("toml_file", str(path), f"invalid TOML: {e}") from e
    return data.get("source-recall", data)


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

    # Determine which fields are explicitly set via SR_ env vars.
    # pydantic-settings init kwargs beat env vars, so we must exclude
    # TOML values for fields that have an env var set — otherwise TOML
    # would incorrectly override the env var.
    env_prefix = "SR_"
    env_keys = {
        name
        for name in SRConfig.model_fields
        if f"{env_prefix}{name.upper()}" in os.environ
    }

    # Layer: start with TOML (lowest), exclude fields set by env vars,
    # then apply explicit overrides (highest).
    merged = {k: v for k, v in toml_values.items() if k not in env_keys}
    merged.update({k: v for k, v in overrides.items() if v is not None})

    # SRConfig reads SR_ env vars automatically via pydantic-settings.
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
    lines.append(f"rerank_enabled = {'true' if config.rerank_enabled else 'false'}")
    lines.append(
        f"graph_expand_enabled = {'true' if config.graph_expand_enabled else 'false'}"
    )

    excludes = ", ".join(f'"{p}"' for p in config.exclude_patterns)
    lines.append(f"exclude_patterns = [{excludes}]")

    return "\n".join(lines) + "\n"
