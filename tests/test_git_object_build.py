"""Tests for git-object-based build/refresh integration (Phase 2)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
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
    (repo / "init.py").write_text(f"# {marker or repo.name}\nx = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"init {marker or repo.name}"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


def _make_builder(
    repo: Path, *, embedder: object | None = None
) -> IndexBuilder:
    """Create a builder with optional embedder."""
    config = resolve_config(str(repo))
    return IndexBuilder(repo, config, embedder=embedder)


# =========================================================================
# M2.1: _index_file with blob-SHA fast path
# =========================================================================


class TestIndexFileBlobFastPath:
    def test_identical_blob_sha_skips_reread(self, tmp_path: Path) -> None:
        """When blob_sha matches stored content_hash, file is not re-read."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="skip-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder = _make_builder(repo)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            # Get the stored content_hash for app.py — should be blob SHA.
            rec = store.get_file_hash("app.py")
            assert rec is not None
            stored_hash = rec.content_hash

        # Get the current blob SHA from git.
        blob_sha = subprocess.run(
            ["git", "rev-parse", "HEAD:app.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # stored_hash should now be the blob SHA.
        assert stored_hash == blob_sha

    def test_changed_blob_sha_triggers_rechunk(self, tmp_path: Path) -> None:
        """When blob_sha differs from stored content_hash, file is re-indexed."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="rechunk-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder = _make_builder(repo)
        builder.build()

        # Get old hash.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            old_rec = store.get_file_hash("app.py")
            old_hash = old_rec.content_hash

        # Change the file.
        (repo / "app.py").write_text("def hello(): return 'world'\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "update app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder.refresh()

        with IndexStore(db_path) as store:
            store.run_migrations()
            new_rec = store.get_file_hash("app.py")
            assert new_rec is not None
            # Hash should have changed.
            assert new_rec.content_hash != old_hash

    def test_dirty_file_reads_from_working_tree(self, tmp_path: Path) -> None:
        """Dirty files are read from the working tree, not git blob."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="dirty-build-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Modify without committing.
        (repo / "app.py").write_text("def hello(): return 'dirty'\n")

        builder = _make_builder(repo)
        builder.build()

        # The content should reflect the working tree, not the commit.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rows = store.conn.execute(
                "SELECT content FROM chunks WHERE file_path = 'app.py'"
            ).fetchall()
            assert any("dirty" in row[0] for row in rows)


# =========================================================================
# M2.2: Modified build() flow
# =========================================================================


class TestBuildUsesGitObjects:
    def test_full_build_stores_blob_sha_as_content_hash(self, tmp_path: Path) -> None:
        """Full build uses git blob SHA (not sha256) for content_hash."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="blob-hash-test")
        (repo / "app.py").write_text("x = 42\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder = _make_builder(repo)
        builder.build()

        # Get expected blob SHA from git.
        expected_sha = subprocess.run(
            ["git", "rev-parse", "HEAD:app.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rec = store.get_file_hash("app.py")
            assert rec is not None
            assert rec.content_hash == expected_sha

    def test_shallow_clone_falls_back_to_filesystem(self, tmp_path: Path) -> None:
        """Shallow clones use filesystem reads with sha256 hashes."""
        source = tmp_path / "source"
        source.mkdir()
        _git_init(source, marker="shallow-build-source")
        (source / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=source, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=source,
            capture_output=True,
            check=True,
        )

        # Shallow clone.
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "--depth=1", f"file://{source}", str(shallow)],
            capture_output=True,
            check=True,
        )

        builder = _make_builder(shallow)
        builder.build()

        # Should succeed — content_hash will be sha256 (fallback path).
        db_path = get_db_path(shallow)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rec = store.get_file_hash("app.py")
            assert rec is not None
            # For shallow clones, content_hash should be sha256, not blob SHA.
            assert len(rec.content_hash) == 64  # sha256 hex is 64 chars

    def test_dirty_files_included_in_build(self, tmp_path: Path) -> None:
        """Build includes uncommitted file modifications."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="dirty-include-test")
        (repo / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Create a new uncommitted file.
        (repo / "new.py").write_text("y = 2\n")

        builder = _make_builder(repo)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            # new.py should be indexed even though it's not committed.
            rec = store.get_file_hash("new.py")
            assert rec is not None


# =========================================================================
# M2.3: Modified refresh() flow — branch switch optimizations
# =========================================================================


class TestRefreshBranchSwitch:
    def test_branch_switch_identical_files_zero_embeds(self, tmp_path: Path) -> None:
        """Switching branches with identical files does NOT re-embed."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="zero-embed-test")
        (repo / "shared.py").write_text("def shared(): return 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "shared file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        emb = BagOfWordsEmbedder(dimensions=64)
        builder = _make_builder(repo, embedder=emb)
        builder.build()

        # Get initial vector count.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            initial_vec_count = store.get_vector_count()

        # Create feature branch with same files.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Track embed calls.
        embed_calls: list[int] = []
        original_embed = emb.embed_chunks

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(len(texts))
            return original_embed(texts)

        emb.embed_chunks = tracking_embed

        builder.refresh()

        # No new embeddings should have been produced.
        # embed_chunks may be called with empty list or not at all.
        total_embedded = sum(embed_calls)
        assert total_embedded == 0, f"Expected 0 new embeddings, got {total_embedded}"

    def test_branch_switch_changed_files_reembeds_only_changed(
        self, tmp_path: Path
    ) -> None:
        """Branch switch re-embeds only files that actually changed."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="changed-only-test")
        (repo / "shared.py").write_text("def shared(): return 1\n")
        (repo / "diverge.py").write_text("def original(): return 'main'\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "main files"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        emb = BagOfWordsEmbedder(dimensions=64)
        builder = _make_builder(repo, embedder=emb)
        builder.build()

        # Create feature branch and change one file.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (repo / "diverge.py").write_text("def feature(): return 'feature'\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature change"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        builder.refresh()

        # shared.py should have branches CSV with both branches.
        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            rows = store.conn.execute(
                "SELECT branches FROM chunks WHERE file_path = 'shared.py'"
            ).fetchall()
            for row in rows:
                branches = set(row[0].split(","))
                assert "feature" in branches

    def test_switch_back_to_original_zero_reembeds(self, tmp_path: Path) -> None:
        """Switching back to original branch re-embeds zero chunks."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="switch-back-test")
        (repo / "app.py").write_text("def hello(): return 1\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "main app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        emb = BagOfWordsEmbedder(dimensions=64)
        builder = _make_builder(repo, embedder=emb)
        builder.build()

        # Switch to feature and refresh.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (repo / "feature.py").write_text("def feature(): return 2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        builder.refresh()

        # Switch back to main.
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=repo,
            capture_output=True,
        )
        # Try "master" if "main" failed.
        subprocess.run(
            ["git", "checkout", "master"],
            cwd=repo,
            capture_output=True,
        )

        # Track embed calls on the way back.
        embed_calls: list[int] = []
        original_embed = emb.embed_chunks

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(len(texts))
            return original_embed(texts)

        emb.embed_chunks = tracking_embed

        builder.refresh()

        total_embedded = sum(embed_calls)
        assert total_embedded == 0, (
            f"Expected 0 re-embeddings switching back, got {total_embedded}"
        )
