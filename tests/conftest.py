"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def ts_app_path() -> Path:
    """Path to the TypeScript fixture app."""
    return FIXTURES_DIR / "ts-app"


@pytest.fixture
def py_app_path(tmp_path: Path) -> Path:
    """Path to an isolated copy of the Python fixture app."""
    dst = tmp_path / "py-app"
    shutil.copytree(FIXTURES_DIR / "py-app", dst)
    return dst


@pytest.fixture
def mixed_path() -> Path:
    """Path to the mixed-language fixture."""
    return FIXTURES_DIR / "mixed"


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repo with some files.

    @returns: Path to the temp repo root.
    """
    # Copy py-app fixture into tmp_path.
    src = FIXTURES_DIR / "py-app"
    dst = tmp_path / "repo"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(autouse=True)
def clean_index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect index storage to tmp_path to avoid polluting user data."""
    index_base = tmp_path / "source-recall-indexes"
    index_base.mkdir()

    # Monkeypatch the get_index_dir function.
    import source_recall.store as store_mod

    def patched_get_index_dir(repo_path: Path) -> Path:
        import hashlib

        real = str(repo_path.resolve())
        path_hash = hashlib.sha256(real.encode()).hexdigest()[:16]
        repo_name = repo_path.resolve().name
        return index_base / f"{repo_name}-{path_hash}"

    def patched_get_db_path(repo_path: Path) -> Path:
        return patched_get_index_dir(repo_path) / "index.db"

    monkeypatch.setattr(store_mod, "get_index_dir", patched_get_index_dir)
    monkeypatch.setattr(store_mod, "get_db_path", patched_get_db_path)

    # Some modules import get_db_path directly. Patch those aliases too,
    # otherwise tests can leak paths across test cases after module import.
    import source_recall.builder as builder_mod
    import source_recall.querier as querier_mod

    monkeypatch.setattr(builder_mod, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(querier_mod, "get_db_path", patched_get_db_path)
