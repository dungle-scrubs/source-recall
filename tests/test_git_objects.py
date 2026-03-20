"""Tests for git-object-based indexing helpers (Phase 1)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config


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
    (repo / "init.py").write_text(f"# {marker or repo.name}\nx = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"init {marker or repo.name}"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


def _make_builder(repo: Path) -> IndexBuilder:
    """Create a builder for a repo with default config."""
    config = resolve_config(str(repo))
    return IndexBuilder(repo, config)


# =========================================================================
# M1.1: _discover_files_git_objects()
# =========================================================================


class TestDiscoverFilesGitObjects:
    def test_returns_tuples_for_committed_files(self, tmp_path: Path) -> None:
        """Parses git ls-tree output into (path, blob_sha) tuples."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="discover-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder = _make_builder(repo)
        result = builder._discover_files_git_objects()

        assert result is not None
        # Should have at least init.py and app.py.
        paths = {path for path, _sha in result}
        assert "app.py" in paths
        assert "init.py" in paths

        # Each entry is (str, str) — path and 40-char hex SHA.
        for path, sha in result:
            assert isinstance(path, str)
            assert isinstance(sha, str)
            assert len(sha) == 40

    def test_returns_none_for_non_git_directory(self, tmp_path: Path) -> None:
        """Returns None when the directory is not a git repo."""
        repo = tmp_path / "not-a-repo"
        repo.mkdir()
        (repo / "app.py").write_text("x = 1\n")

        builder = _make_builder(repo)
        result = builder._discover_files_git_objects()
        assert result is None

    def test_excludes_excluded_patterns(self, tmp_path: Path) -> None:
        """Respects _is_excluded() filtering on discovered files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="exclude-test")

        # Create a file that should be excluded (node_modules/).
        nm = repo / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        (repo / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "--force", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "with excluded"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder = _make_builder(repo)
        result = builder._discover_files_git_objects()

        assert result is not None
        paths = {path for path, _sha in result}
        assert "app.py" in paths
        assert not any("node_modules" in p for p in paths)

    def test_handles_empty_repo(self, tmp_path: Path) -> None:
        """Returns empty list for a repo with no files in HEAD tree."""
        repo = tmp_path / "repo"
        repo.mkdir()
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

        builder = _make_builder(repo)
        # Empty repo has no HEAD — should return None (not crash).
        result = builder._discover_files_git_objects()
        assert result is None


# =========================================================================
# M1.2: _read_git_blob()
# =========================================================================


class TestReadGitBlob:
    def test_reads_committed_file_content(self, tmp_path: Path) -> None:
        """Returns file content for a committed file's blob SHA."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="blob-read-test")
        content = "def hello():\n    return 'world'\n"
        (repo / "app.py").write_text(content)
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Get the blob SHA for app.py.
        result = subprocess.run(
            ["git", "rev-parse", "HEAD:app.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        blob_sha = result.stdout.strip()

        builder = _make_builder(repo)
        blob_content = builder._read_git_blob(blob_sha)

        assert blob_content is not None
        assert blob_content == content

    def test_returns_none_for_nonexistent_blob(self, tmp_path: Path) -> None:
        """Returns None for a SHA that doesn't exist in the repo."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="missing-blob-test")

        builder = _make_builder(repo)
        result = builder._read_git_blob("0" * 40)
        assert result is None


# =========================================================================
# M1.3: _detect_dirty_files()
# =========================================================================


class TestDetectDirtyFiles:
    def test_detects_modified_file(self, tmp_path: Path) -> None:
        """Finds modified-but-uncommitted files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="dirty-test")
        (repo / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Modify without committing.
        (repo / "app.py").write_text("x = 2\n")

        builder = _make_builder(repo)
        dirty = builder._detect_dirty_files()

        assert "app.py" in dirty
        # The SHA should be 40 hex chars.
        assert len(dirty["app.py"]) == 40

    def test_computes_correct_blob_sha(self, tmp_path: Path) -> None:
        """Synthetic blob SHA matches git hash-object output."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="hash-test")
        (repo / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Modify file.
        new_content = "x = 42\n"
        (repo / "app.py").write_text(new_content)

        # Compute expected blob SHA.
        expected = subprocess.run(
            ["git", "hash-object", "--stdin"],
            cwd=repo,
            input=new_content,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        builder = _make_builder(repo)
        dirty = builder._detect_dirty_files()

        assert dirty["app.py"] == expected

    def test_detects_untracked_files(self, tmp_path: Path) -> None:
        """Detects new untracked files as dirty."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="untracked-test")

        (repo / "new_file.py").write_text("y = 99\n")

        builder = _make_builder(repo)
        dirty = builder._detect_dirty_files()

        assert "new_file.py" in dirty

    def test_returns_empty_for_clean_tree(self, tmp_path: Path) -> None:
        """Returns empty dict when working tree is clean."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="clean-test")

        builder = _make_builder(repo)
        dirty = builder._detect_dirty_files()

        assert dirty == {}


# =========================================================================
# M1.4: _is_shallow_clone()
# =========================================================================


class TestIsShallowClone:
    def test_returns_false_for_normal_repo(self, tmp_path: Path) -> None:
        """Normal repo is not shallow."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="normal-test")

        builder = _make_builder(repo)
        assert builder._is_shallow_clone() is False

    def test_returns_true_for_shallow_clone(self, tmp_path: Path) -> None:
        """Shallow clone is detected as shallow."""
        # Create a source repo with some commits.
        source = tmp_path / "source"
        source.mkdir()
        _git_init(source, marker="shallow-source")
        (source / "a.py").write_text("a = 1\n")
        subprocess.run(["git", "add", "."], cwd=source, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "second"],
            cwd=source,
            capture_output=True,
            check=True,
        )

        # Shallow clone with depth=1 (file:// required — local clones
        # ignore --depth).
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "--depth=1", f"file://{source}", str(shallow)],
            capture_output=True,
            check=True,
        )

        builder = _make_builder(shallow)
        assert builder._is_shallow_clone() is True
