from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mise_declares_project_tools() -> None:
    mise_config = ROOT / ".mise.toml"
    assert mise_config.exists()

    config = tomllib.loads(mise_config.read_text())

    assert config["tools"]["python"] == (ROOT / ".python-version").read_text().strip()
    assert config["tools"]["uv"] == "0.11.15"
    assert config["tools"]["just"] == "1.51.0"
