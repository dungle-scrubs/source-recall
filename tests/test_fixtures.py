"""Tests for shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path


def test_py_app_path_is_isolated_copy(py_app_path: Path) -> None:
    """py_app_path mutations do not modify the checked-in fixture tree."""
    fixture_auth = Path(__file__).parent / "fixtures" / "py-app" / "myapp" / "auth.py"
    before = fixture_auth.read_text()

    auth_file = py_app_path / "myapp" / "auth.py"
    auth_file.write_text(auth_file.read_text() + "\n# fixture isolation marker\n")

    assert fixture_auth.read_text() == before
