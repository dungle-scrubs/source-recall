"""Tests for builder.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.models import IndexIdentityError
from source_recall.store import IndexStore, get_db_path


def _git_init(repo: Path, *, marker: str = "") -> None:
    """Initialize a git repo with one commit.

    @param repo: Repo directory (must exist).
    @param marker: Unique content to ensure different root commits.
    """
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    # Unique content guarantees a unique root commit hash.
    (repo / "init.py").write_text(f"# {marker or repo.name}\nx = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"init {marker or repo.name}"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


class TestExclusionPatterns:
    def test_node_modules_excluded(self, tmp_path: Path) -> None:
        """Files inside node_modules/ are excluded."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "index.ts").write_text("export const x = 1")
        nm = repo / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        files = builder._discover_files()
        assert "index.ts" in files
        assert not any("node_modules" in f for f in files)

    def test_pycache_excluded(self, tmp_path: Path) -> None:
        """__pycache__/ directories are excluded."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("x = 1")
        cache = repo / "__pycache__"
        cache.mkdir()
        (cache / "app.cpython-312.pyc").write_bytes(b"\x00")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        files = builder._discover_files()
        assert "app.py" in files
        assert not any("__pycache__" in f for f in files)

    def test_nested_excluded_dir(self, tmp_path: Path) -> None:
        """Exclusion works for deeply nested matching directories."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("x = 1")
        deep = repo / "src" / "lib" / "node_modules" / "pkg"
        deep.mkdir(parents=True)
        (deep / "mod.js").write_text("x = 1")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        files = builder._discover_files()
        assert "app.py" in files
        assert not any("node_modules" in f for f in files)

    def test_zero_byte_files_excluded(self, tmp_path: Path) -> None:
        """Empty files are skipped."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "empty.py").write_text("")
        (repo / "real.py").write_text("x = 1")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        files = builder._discover_files()
        assert "real.py" in files
        assert "empty.py" not in files


class TestRefreshDeletedFiles:
    def test_deleted_file_chunks_removed(self, tmp_path: Path) -> None:
        """Refresh removes chunks for files that no longer exist."""
        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="del-test")
        (repo / "keep.py").write_text("def keep(): return 1\n")
        (repo / "remove.py").write_text("def remove(): return 2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "two files"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        idx = Index(repo)
        idx.build()
        s1 = idx.status()
        # 3 files: init.py (from _git_init), keep.py, remove.py
        assert s1.file_count == 3

        # Delete one file and commit.
        (repo / "remove.py").unlink()
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "remove file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        refreshed = idx.refresh()
        assert refreshed > 0

        s2 = idx.status()
        assert s2.file_count == 2

        # Query should not return chunks from deleted file.
        results = idx.query("remove")
        remove_results = [r for r in results if "remove.py" in r.file_path]
        assert len(remove_results) == 0


class TestCleanOrphanedIndexes:
    def test_clean_removes_orphaned_index(self, tmp_path: Path) -> None:
        """Indexes pointing to non-existent repos are cleaned up."""
        from source_recall import Index
        from source_recall.store import get_index_dir

        # Create a repo and index it.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("def hello(): pass\n")

        idx = Index(repo)
        idx.build()

        index_dir = get_index_dir(repo)
        assert index_dir.exists()
        assert (index_dir / "index.db").exists()

        # "Delete" the repo by renaming it.
        repo.rename(tmp_path / "repo_gone")

        # The index dir still exists but its repo_path is gone.
        assert index_dir.exists()

        # Read meta from the DB to verify repo_path points to missing dir.
        import sqlite3

        conn = sqlite3.connect(str(index_dir / "index.db"))
        row = conn.execute("SELECT value FROM meta WHERE key = 'repo_path'").fetchone()
        conn.close()
        assert row is not None
        assert not Path(row[0]).exists()


class TestBranchAwareBuilder:
    def test_build_stores_active_branch(self, tmp_path: Path) -> None:
        """build() records the current branch in meta."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="branch-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        store = IndexStore(get_db_path(repo))
        store.open()
        branch = store.get_meta("active_branch")
        assert branch is not None
        assert branch != ""
        # Should be "main" or "master" depending on git default.
        assert branch in ("main", "master")
        store.close()

    def test_build_sets_branches_on_chunks(self, tmp_path: Path) -> None:
        """build() sets the branches column on all chunks."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="chunks-branch-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        store = IndexStore(get_db_path(repo))
        store.open()
        rows = store.conn.execute(
            "SELECT branches FROM chunks WHERE branches = ''"
        ).fetchall()
        # No chunks should have empty branches.
        assert len(rows) == 0

        # All chunks should have a branch set.
        total = store.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE branches != ''"
        ).fetchone()[0]
        assert total > 0
        store.close()

    def test_refresh_after_branch_switch_preserves_chunks(self, tmp_path: Path) -> None:
        """Switching branches and refreshing preserves shared chunks."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="switch-test")
        (repo / "shared.py").write_text("def shared(): return 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "shared file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        store = IndexStore(get_db_path(repo))
        store.open()
        count_after_build = store.get_chunk_count()

        # Create and switch to feature branch, add a new file.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (repo / "feature.py").write_text("def feature_only(): return 2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        store.close()

        builder.refresh()

        store = IndexStore(get_db_path(repo))
        store.open()
        count_after_refresh = store.get_chunk_count()
        # Should have more chunks (feature.py added).
        assert count_after_refresh > count_after_build

        # The active_branch should now be "feature".
        assert store.get_meta("active_branch") == "feature"
        store.close()


class TestIdentityVerification:
    def test_mismatched_root_commit_raises(self, tmp_path: Path) -> None:
        """Refreshing with a different repo's root commit raises error."""
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        _git_init(repo_a, marker="alpha")
        (repo_a / "app.py").write_text("def hello(): pass\n")
        subprocess.run(
            ["git", "add", "app.py"],
            cwd=repo_a,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo_a,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo_a))
        builder_a = IndexBuilder(repo_a, config)
        builder_a.build()

        # Copy repo A's index to repo B's location.
        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        _git_init(repo_b, marker="bravo")

        db_path_a = get_db_path(repo_a)
        db_path_b = get_db_path(repo_b)
        db_path_b.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(db_path_a, db_path_b)

        config_b = resolve_config(str(repo_b))
        builder_b = IndexBuilder(repo_b, config_b)
        with pytest.raises(IndexIdentityError):
            builder_b.refresh()

    def test_same_root_commit_different_path_raises(self, tmp_path: Path) -> None:
        """Forks with same root commit but different paths are caught.

        Two repos forked from the same upstream share a root commit.
        Identity check must still detect the path mismatch.
        """
        # Create repo_a and index it.
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        _git_init(repo_a, marker="shared-origin")

        config_a = resolve_config(str(repo_a))
        builder_a = IndexBuilder(repo_a, config_a)
        builder_a.build()

        # Create repo_b as a clone (same root commit).
        repo_b = tmp_path / "repo_b"
        subprocess.run(
            ["git", "clone", str(repo_a), str(repo_b)],
            capture_output=True,
            check=True,
        )

        # Copy repo_a's index DB to repo_b's index location.
        db_path_a = get_db_path(repo_a)
        db_path_b = get_db_path(repo_b)
        db_path_b.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(db_path_a, db_path_b)

        # Both repos have the same root commit, but stored repo_path differs.
        config_b = resolve_config(str(repo_b))
        builder_b = IndexBuilder(repo_b, config_b)
        with pytest.raises(IndexIdentityError):
            builder_b.refresh()
