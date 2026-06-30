from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mise_config() -> Path:
    """Return the project mise config path (legacy dotfile or modern form)."""
    modern = ROOT / "mise.toml"
    if modern.exists():
        return modern
    return ROOT / ".mise.toml"


def test_mise_declares_project_tools() -> None:
    mise_config = _mise_config()
    assert mise_config.exists(), (
        "expected mise.toml (or legacy .mise.toml) at project root"
    )

    config = tomllib.loads(mise_config.read_text())

    assert config["tools"]["python"] == (ROOT / ".python-version").read_text().strip()
    assert config["tools"]["uv"] == "0.11.25"
    assert config["tools"]["just"] == "1.51.0"
