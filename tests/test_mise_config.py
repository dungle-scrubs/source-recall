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


def test_ci_workflows_pin_the_mise_uv_version() -> None:
    """Every workflow setup-uv step pins mise.toml's uv version.

    The uv_build backend requires >=0.8,<0.12 and mise.toml is the local
    source of truth; drift between CI and local dev breaks the lock
    resolution silently otherwise.
    """
    config = tomllib.loads(_mise_config().read_text())
    uv_version = config["tools"]["uv"]
    expected = f'version: "{uv_version}"'

    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows, "expected workflow files under .github/workflows"

    stale: list[str] = []
    for workflow in workflows:
        text = workflow.read_text()
        # Every setup-uv usage carries a version pin; count them so a
        # workflow that adds setup-uv without a pin is caught too.
        setup_uv_usages = text.count("astral-sh/setup-uv")
        pins = text.count(expected)
        if pins < setup_uv_usages:
            stale.append(
                f"{workflow.name}: {setup_uv_usages} setup-uv step(s), "
                f"{pins} pin(s) for uv {uv_version}"
            )
    assert stale == [], f"workflows not pinning mise's uv {uv_version}: {stale}"
