"""Tests for git-object indexing edge cases and integration gaps.

Covers:
  1. File deletion during git-object refresh
  2. Non-git repo build fallback
  3. FTS integrity after blob-SHA fast-path skip
  4. _detect_changes compatibility with blob-SHA content_hash
  5. _read_git_blob with binary/non-UTF8 content
  6. Dirty file on refresh
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.store import IndexStore, get_db_path


def _git_init(repo: Path, *, marker: str = "") -> None:
    """Initialize a git repo with one commit."""
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    (repo / "init.py").write_text(f"# {marker or repo.name}\nx = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"init {marker or repo.name}"],
        cwd=repo, capture_output=True, check=True,
    )


# =========================================================================
# Gap 1: File deletion during git-object refresh
# =========================================================================


class TestGitObjectRefreshDeletion:
    def test_deleted_file_removed_on_branch_switch(self, tmp_path: Path) -> None:
        """When a file exists on main but not feature, refresh removes it."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="del-branch-test")
        (repo / "keep.py").write_text("def keep(): return 1\n")
        (repo / "remove_me.py").write_text("def remove(): return 2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "two files"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            assert store.get_file_hash("remove_me.py") is not None

        # Create feature branch that deletes remove_me.py.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo, capture_output=True, check=True,
        )
        (repo / "remove_me.py").unlink()
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "delete file"],
            cwd=repo, capture_output=True, check=True,
        )

        changed = builder.refresh()
        assert changed > 0

        with IndexStore(db_path) as store:
            store.run_migrations()
            # File hash should be gone.
            assert store.get_file_hash("remove_me.py") is None
            # Chunks should be gone.
            rows = store.conn.execute(
                "SELECT id FROM chunks WHERE file_path = 'remove_me.py'"
            ).fetchall()
            assert len(rows) == 0


# =========================================================================
# Gap 2: Non-git repo build fallback
# =========================================================================


class TestNonGitRepoBuild:
    def test_build_works_without_git(self, tmp_path: Path) -> None:
        """Build on a plain directory (no .git) uses filesystem + sha256."""
        repo = tmp_path / "plain-repo"
        repo.mkdir()
        (repo / "app.py").write_text("def hello(): pass\n")
        (repo / "lib.py").write_text("x = 42\n")

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            # Both files should be indexed.
            assert store.get_file_hash("app.py") is not None
            assert store.get_file_hash("lib.py") is not None
            # content_hash should be sha256 (64 hex chars), not blob SHA (40).
            rec = store.get_file_hash("app.py")
            assert len(rec.content_hash) == 64
            # Chunks should exist.
            assert store.get_chunk_count() > 0


# =========================================================================
# Gap 3: FTS integrity after blob-SHA fast-path skip
# =========================================================================


class TestFtsIntegrityAfterBranchSwitch:
    def test_fts_search_returns_skipped_chunks(self, tmp_path: Path) -> None:
        """Chunks that took the fast path (skip) are still FTS-searchable."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="fts-test")
        (repo / "shared.py").write_text(
            "def calculate_fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\n"
        )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "shared file"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        # FTS should find the function before branch switch.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            results_before = store.fts_search("fibonacci")
            assert len(results_before) > 0

        # Switch to feature branch (shared.py identical).
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo, capture_output=True, check=True,
        )
        (repo / "feature.py").write_text("def feature_func(): return 99\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature file"],
            cwd=repo, capture_output=True, check=True,
        )

        builder.refresh()

        # FTS should still find fibonacci in shared.py (fast-path chunk).
        with IndexStore(db_path) as store:
            store.run_migrations()
            results_after = store.fts_search("fibonacci")
            assert len(results_after) > 0
            fib_results = [r for r in results_after if "shared.py" in r["file_path"]]
            assert len(fib_results) > 0

    def test_fts_finds_chunk_after_two_branch_switches(self, tmp_path: Path) -> None:
        """FTS works after build → feature → refresh → main → refresh."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="fts-cycle-test")
        (repo / "core.py").write_text(
            "class DatabaseConnection:\n"
            "    def connect(self, host, port):\n"
            "        self.host = host\n"
            "        self.port = port\n"
        )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "core"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        # Switch to feature.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "empty feature"],
            cwd=repo, capture_output=True, check=True,
        )
        builder.refresh()

        # Switch back to main/master.
        subprocess.run(["git", "checkout", "main"], cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", "master"], cwd=repo, capture_output=True)
        builder.refresh()

        # FTS should still find DatabaseConnection.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            results = store.fts_search("DatabaseConnection")
            assert len(results) > 0


# =========================================================================
# Gap 4: _detect_changes with blob-SHA content_hash
# =========================================================================


class TestLegacyDetectChangesWithBlobHash:
    def test_legacy_fallback_after_git_object_build(self, tmp_path: Path) -> None:
        """Legacy _detect_changes works even when content_hash is a blob SHA.

        After a git-object build, content_hash is a 40-char blob SHA.
        If _try_git_object_refresh returns None (e.g. shallow clone on
        refresh but not on build), _detect_changes compares sha256 of
        current content against the stored blob SHA. Every file looks
        'changed' — that's acceptable (conservative), but must not crash.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="legacy-compat-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        # Verify content_hash is blob SHA (40 chars).
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rec = store.get_file_hash("app.py")
            assert len(rec.content_hash) == 40

        # Force legacy path by monkeypatching _try_git_object_refresh.
        original = builder._try_git_object_refresh
        builder._try_git_object_refresh = lambda store, branch: None

        # Modify a file so there's something to detect.
        (repo / "app.py").write_text("def hello(): return 'changed'\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "change app"],
            cwd=repo, capture_output=True, check=True,
        )

        # Should not crash — may re-index everything (conservative).
        changed = builder.refresh()
        assert changed >= 1  # At least app.py detected as changed.

        # Restore.
        builder._try_git_object_refresh = original


# =========================================================================
# Gap 5: _read_git_blob with binary/non-UTF8 content
# =========================================================================


class TestReadGitBlobBinary:
    def test_binary_blob_does_not_crash(self, tmp_path: Path) -> None:
        """Binary file content read via git blob doesn't crash the build."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="binary-test")

        # Write a file with non-UTF8 bytes.
        binary_content = b"\x80\x81\x82\xff\xfe def hello(): pass\n"
        (repo / "data.py").write_bytes(binary_content)
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "binary file"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)

        # Build should not crash.
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rec = store.get_file_hash("data.py")
            assert rec is not None


# =========================================================================
# Gap 6: Dirty file on refresh
# =========================================================================


class TestDirtyFileOnRefresh:
    def test_uncommitted_modification_detected_on_refresh(
        self, tmp_path: Path
    ) -> None:
        """Refresh detects and re-indexes uncommitted file changes."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="dirty-refresh-test")
        (repo / "app.py").write_text("def hello(): return 'original'\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        # Modify without committing.
        (repo / "app.py").write_text("def hello(): return 'dirty_refresh'\n")

        changed = builder.refresh()
        assert changed >= 1

        # Verify the new content is in the index.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rows = store.conn.execute(
                "SELECT content FROM chunks WHERE file_path = 'app.py'"
            ).fetchall()
            assert any("dirty_refresh" in row[0] for row in rows)

    def test_new_untracked_file_indexed_on_refresh(self, tmp_path: Path) -> None:
        """Refresh picks up brand-new untracked files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="untracked-refresh-test")
        (repo / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo, capture_output=True, check=True,
        )

        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config)
        builder.build()

        # Add a new file without committing.
        (repo / "brand_new.py").write_text("def brand_new(): return 'surprise'\n")

        changed = builder.refresh()
        assert changed >= 1

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rec = store.get_file_hash("brand_new.py")
            assert rec is not None
