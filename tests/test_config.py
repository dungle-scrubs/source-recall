"""Tests for config.py."""

from __future__ import annotations

from pathlib import Path

from source_recall.config import SRConfig, format_config, resolve_config


class TestResolveConfig:
    def test_defaults(self) -> None:
        """Default config values are sensible."""
        config = resolve_config()
        assert config.max_file_size == 100_000
        assert config.top_k == 8
        assert config.chunk_max_chars == 6000
        assert config.auto_refresh is True
        assert "node_modules/" in config.exclude_patterns

    def test_overrides_beat_defaults(self) -> None:
        """Keyword overrides take priority."""
        config = resolve_config(top_k=15, max_file_size=50_000)
        assert config.top_k == 15
        assert config.max_file_size == 50_000

    def test_env_vars(self, monkeypatch: object) -> None:
        """SR_ env vars override defaults."""
        import os

        monkeypatch.setattr(os, "environ", {**os.environ, "SR_TOP_K": "20"})  # type: ignore[attr-defined]
        config = resolve_config()
        assert config.top_k == 20

    def test_toml_config(self, tmp_path: Path) -> None:
        """Repo-local .source-recall.toml is read."""
        toml = tmp_path / ".source-recall.toml"
        toml.write_text("[source-recall]\ntop_k = 12\nmax_file_size = 200000\n")
        config = resolve_config(tmp_path)
        assert config.top_k == 12
        assert config.max_file_size == 200_000

    def test_overrides_beat_toml(self, tmp_path: Path) -> None:
        """Explicit overrides beat TOML values."""
        toml = tmp_path / ".source-recall.toml"
        toml.write_text("[source-recall]\ntop_k = 12\n")
        config = resolve_config(tmp_path, top_k=5)
        assert config.top_k == 5


class TestConfigValidation:
    def test_negative_max_file_size_rejected(self) -> None:
        """max_file_size must be > 0."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SRConfig(max_file_size=-1)

    def test_zero_top_k_rejected(self) -> None:
        """top_k must be > 0."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SRConfig(top_k=0)

    def test_top_k_over_100_rejected(self) -> None:
        """top_k must be <= 100."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SRConfig(top_k=101)

    def test_chunk_max_chars_below_minimum_rejected(self) -> None:
        """chunk_max_chars must be > 500."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SRConfig(chunk_max_chars=100)


class TestFormatConfig:
    def test_round_trip(self) -> None:
        """format_config produces valid TOML-like output."""
        config = SRConfig()
        text = format_config(config)
        assert "[source-recall]" in text
        assert "top_k = 8" in text
        assert "max_file_size = 100000" in text
        assert "auto_refresh = true" in text
