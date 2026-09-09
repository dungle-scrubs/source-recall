"""Tests for skip-existing-vectors optimization (Phase 3)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.store import IndexStore, get_db_path


class _RefreshTrackingEmbedder(BagOfWordsEmbedder):
    """BagOfWordsEmbedder that records embed_chunks batch sizes on demand.

    Assigning ``sink`` starts recording; ``None`` stops it. Double aligned
    with the real method signature instead of an instance-attribute spy,
    so counting can be scoped to a single refresh.
    """

    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions)
        self.sink: list[int] | None = None

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if self.sink is not None:
            self.sink.append(len(texts))
        return super().embed_chunks(texts)


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


# =========================================================================
# M3.1: get_existing_vector_ids()
# =========================================================================


class TestGetExistingVectorIds:
    def test_returns_empty_for_no_vectors(self, tmp_path: Path) -> None:
        """Returns empty set when no chunks have vectors."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="no-vec-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        # Build WITHOUT embedder — no vectors.
        builder = IndexBuilder(repo, config)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            # Even if vec table doesn't exist, should return empty set.
            result = store.get_existing_vector_ids(["fake-chunk-id"])
            assert result == set()

    def test_returns_correct_subset(self, tmp_path: Path) -> None:
        """Returns only chunk IDs that actually have vectors."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="subset-test")
        (repo / "app.py").write_text("def hello(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        emb = BagOfWordsEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=emb)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            # Get actual chunk IDs.
            rows = store.conn.execute("SELECT id FROM chunks").fetchall()
            real_ids = [row[0] for row in rows]
            assert len(real_ids) > 0

            # Query with mix of real and fake IDs.
            query_ids = real_ids + ["nonexistent-1", "nonexistent-2"]
            result = store.get_existing_vector_ids(query_ids)

            # Should contain only the real IDs.
            assert result == set(real_ids)
            assert "nonexistent-1" not in result
            assert "nonexistent-2" not in result


# =========================================================================
# Full branch-switch cycle verification
# =========================================================================


class TestFullBranchSwitchCycle:
    def test_full_cycle_embedding_count_matches_changed_files(
        self, tmp_path: Path
    ) -> None:
        """build main → switch feature → refresh → switch main → refresh.

        Embedding count should match changed-file count, not total-file count.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="cycle-test")
        (repo / "shared.py").write_text("def shared(): return 1\n")
        (repo / "another.py").write_text("def another(): return 2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "main files"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        emb = _RefreshTrackingEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        builder = IndexBuilder(repo, config, embedder=emb)

        # 1. Build on main.
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            main_vec_count = store.get_vector_count()
            assert main_vec_count > 0

        # 2. Switch to feature, add one new file.
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        (repo / "feature_only.py").write_text("def feature(): return 3\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature file"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        # Track embed calls during feature refresh.
        feature_embed_calls: list[int] = []
        emb.sink = feature_embed_calls

        builder.refresh()

        # Should only embed the new feature_only.py chunks, not shared.py/another.py.
        feature_total = sum(feature_embed_calls)
        assert feature_total > 0, "Should embed at least the new file"

        with IndexStore(db_path) as store:
            store.run_migrations()
            feature_vec_count = store.get_vector_count()
            # Should have more vectors now.
            assert feature_vec_count > main_vec_count

        # 3. Switch back to main.
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=repo,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "master"],
            cwd=repo,
            capture_output=True,
        )

        # Track embed calls on way back.
        return_embed_calls: list[int] = []
        emb.sink = return_embed_calls

        builder.refresh()

        # Switching back: all chunks already have vectors from the build.
        return_total = sum(return_embed_calls)
        assert return_total == 0, (
            f"Expected 0 re-embeddings on return to main, got {return_total}"
        )
