"""Guard exact version pins mandated by AGENTS.md invariant #1.

tree-sitter and tree-sitter-language-pack must stay pinned exactly
(``==``) because the C ABI breaks across minor versions. See
AGENTS.md "tree-sitter pins are exact".
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dependencies() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return pyproject["project"]["dependencies"]


def test_tree_sitter_pins_are_exact() -> None:
    deps = _dependencies()
    assert "tree-sitter==0.25.2" in deps
    assert "tree-sitter-language-pack==0.13.0" in deps
