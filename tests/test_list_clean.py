"""Tests for sr list and sr clean commands.

These commands enumerate the shared index base directory.  They must
route through ``store.get_index_base`` (which the autouse
``clean_index_dir`` fixture monkeypatches to ``tmp_path``) so they never
touch ``~/.local/share/source-recall/``.

Covers the H-1 audit finding: the commands previously hardcoded the
home path and had no tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import source_recall.store as store_mod
from source_recall.cli import app

runner = CliRunner()


def _index_base() -> Path:
    """Read the (monkeypatched) index base at call time, not import time."""
    return store_mod.get_index_base()


def _index_a_repo(repo: Path) -> None:
    """Build a minimal index for ``repo`` so it shows up in sr list."""
    from source_recall import Index

    repo.mkdir(parents=True, exist_ok=True)
    (repo / "app.py").write_text("def hello():\n    return 'world'\n")
    Index(repo, embedder=None).build()


class TestListIndexes:
    def test_list_finds_built_index(self, tmp_path: Path) -> None:
        """sr list enumerates the monkeypatched index base and reports repos."""
        repo = tmp_path / "myrepo"
        _index_a_repo(repo)

        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0, result.output

        entries = json.loads(result.output)
        assert len(entries) == 1
        assert entries[0]["repo_path"] == str(repo.resolve())
        assert entries[0]["chunk_count"] >= 1

    def test_list_empty_when_no_indexes(self, tmp_path: Path) -> None:
        """sr list with an empty index base reports nothing."""
        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == []

    def test_list_does_not_touch_home(self, tmp_path: Path) -> None:
        """The index base is tmp_path, not ~/.local/share/source-recall.

        If list hardcoded the home path (the H-1 bug), this test would
        either read real user data or fail because the directory is
        absent.  The fixture guarantees the base is under tmp_path.
        """
        base = _index_base()
        assert str(base).startswith(str(tmp_path))

        repo = tmp_path / "tracked"
        _index_a_repo(repo)

        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        # Only the repo we built appears — no leakage from home.
        entries = json.loads(result.output)
        assert len(entries) == 1


class TestCleanIndexes:
    def test_clean_removes_orphaned_index(self, tmp_path: Path) -> None:
        """sr clean removes indexes whose repo_path no longer exists."""
        repo = tmp_path / "gonerepo"
        _index_a_repo(repo)

        # Sanity: the index dir exists under the patched base.
        base = _index_base()
        index_dirs = [p for p in base.iterdir() if p.is_dir()]
        assert len(index_dirs) == 1

        # "Delete" the repo by renaming it.
        repo.rename(tmp_path / "moved-elsewhere")

        result = runner.invoke(app, ["clean", "--json"])
        assert result.exit_code == 0, result.output

        removed = json.loads(result.output)
        assert len(removed) == 1
        assert removed[0]["action"] == "removed"
        # Index directory is gone.
        assert not index_dirs[0].exists()

    def test_clean_dry_run_keeps_index(self, tmp_path: Path) -> None:
        """sr clean --dry-run reports but does not remove."""
        repo = tmp_path / "mayberepo"
        _index_a_repo(repo)
        repo.rename(tmp_path / "moved-again")

        base = _index_base()
        index_dirs = [p for p in base.iterdir() if p.is_dir()]
        assert len(index_dirs) == 1

        result = runner.invoke(app, ["clean", "--dry-run", "--json"])
        assert result.exit_code == 0, result.output

        removed = json.loads(result.output)
        assert len(removed) == 1
        assert removed[0]["action"] == "would_remove"
        # Still present.
        assert index_dirs[0].exists()

    def test_clean_keeps_live_index(self, tmp_path: Path) -> None:
        """sr clean does not remove an index whose repo still exists."""
        repo = tmp_path / "aliverepo"
        _index_a_repo(repo)

        result = runner.invoke(app, ["clean", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == []
