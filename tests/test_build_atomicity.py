"""Tests for per-file transactional atomicity during full builds (M-1).

The build loop is wrapped in ``batch_mode()`` so all per-file writes
(chunks + refs + symbol_lookup + file_hash) commit as a single
transaction rather than N separate commits per file.  This test
verifies the batching by counting commits: a batched build issues far
fewer commits than the unbatched one-per-write-call path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from source_recall.builder import IndexBuilder
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.store import IndexStore, get_db_path


def _git_init(repo: Path, *, marker: str = "") -> None:
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
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
        check=True,
    )


class TestBuildBatchAtomicity:
    def test_build_batches_per_file_writes(self, tmp_path: Path) -> None:
        """A build with many files commits in one batch, not one-per-file.

        We count conn.commit() calls during a build.  With batch_mode
        wrapping the loop, the per-file writes (insert_chunks,
        insert_refs, insert_symbol_lookups, upsert_file_hash) do not
        each commit — they all join the single outer batch transaction.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="batch")
        # Several files, each producing chunks + refs + symbol lookups.
        for i in range(5):
            (repo / f"f{i}.py").write_text(
                f"import os\ndef func_{i}():\n    return {i}\n"
            )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "files"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        emb = BagOfWordsEmbedder(dimensions=64)
        builder = IndexBuilder(repo, config, embedder=emb)

        # Count REAL commits: _auto_commit only commits when not inside
        # a batch, so we count only those calls that actually reach
        # conn.commit().
        commit_count = {"n": 0}
        original_auto_commit = IndexStore._auto_commit

        def counting_auto_commit(self):
            if self._batch_depth == 0:
                commit_count["n"] += 1
            return original_auto_commit(self)

        with patch.object(IndexStore, "_auto_commit", counting_auto_commit):
            builder.build()

        # 5 files, each previously triggering ~4 real commits via
        # _auto_commit (insert_refs, insert_symbol_lookups,
        # upsert_file_hash) plus insert_chunks' own _transaction
        # commit = ~20 real commits.  With the batch wrapper, those
        # _auto_commit calls are no-ops (batch_depth > 0), so the real
        # commit count drops dramatically.
        assert commit_count["n"] < 8, (
            f"Expected few real commits (<8), got {commit_count['n']} — "
            "per-file writes are not being batched into one transaction"
        )

    def test_build_produces_consistent_index(self, tmp_path: Path) -> None:
        """A completed build has matching chunk/ref/symbol counts."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo, marker="consistency")
        (repo / "a.py").write_text("import os\ndef alpha():\n    return os.getcwd()\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "a"],
            cwd=repo,
            capture_output=True,
            check=True,
        )

        config = resolve_config(str(repo))
        emb = BagOfWordsEmbedder(dimensions=64)
        builder = IndexBuilder(repo, config, embedder=emb)
        builder.build()

        db_path = get_db_path(repo)
        with IndexStore(db_path) as store:
            store.run_migrations()
            chunk_count = store.get_chunk_count()
            ref_count = store.conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
            sym_count = store.conn.execute(
                "SELECT COUNT(*) FROM symbol_lookup"
            ).fetchone()[0]

        # alpha() references os → at least one ref.
        assert chunk_count > 0
        assert ref_count > 0, "Refs missing — per-file batch lost them"
        assert sym_count > 0, "Symbol lookups missing — per-file batch lost them"
