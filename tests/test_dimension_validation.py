"""Tests for embedding-dimension validation on query open (H-3).

Before the fix, switching embedders (different dimensionality) silently
degraded vector search to empty results because vec_chunks' schema is
fixed at creation and a mismatched query vector errors inside the
swallowed ``except Exception``.  The fix logs a clear warning on open.
"""

from __future__ import annotations

import logging
from pathlib import Path

from source_recall import Index
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.querier import IndexQuerier


def _build_repo(repo: Path) -> None:
    """Create and index a tiny repo with the default embedder."""
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "app.py").write_text("def authenticate(user):\n    return user\n")
    emb = BagOfWordsEmbedder(dimensions=64)
    Index(repo, embedder=emb).build()


class TestEmbedDimensionValidation:
    def test_mismatched_dimension_logs_warning(self, tmp_path: Path, caplog) -> None:
        """Querying with a different-dimension embedder logs a clear warning.

        The query still succeeds (FTS fallback) — the fix only adds a
        specific, actionable dimension-mismatch warning instead of the
        generic "Vector search failed" message.
        """
        repo = tmp_path / "repo"
        _build_repo(repo)

        # Query with an embedder of a DIFFERENT dimension.
        wrong_emb = BagOfWordsEmbedder(dimensions=32)
        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=wrong_emb)

        with caplog.at_level(logging.WARNING, logger="source_recall.querier"):
            results = querier.query("authenticate")

        # Query must still return FTS results.
        assert len(results) > 0

        joined = " ".join(r.message for r in caplog.records)
        assert "dimension" in joined.lower(), (
            f"Expected a dimension-mismatch warning, got: {caplog.records!r}"
        )
        querier.close()

    def test_matching_dimension_no_warning(self, tmp_path: Path, caplog) -> None:
        """Querying with the same-dimension embedder logs no warning."""
        repo = tmp_path / "repo"
        _build_repo(repo)

        right_emb = BagOfWordsEmbedder(dimensions=64)
        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=right_emb)

        with caplog.at_level(logging.WARNING, logger="source_recall.querier"):
            querier.query("authenticate")

        warnings = [r for r in caplog.records if "dimension" in r.message.lower()]
        assert warnings == [], f"Unexpected dimension warning: {warnings!r}"
        querier.close()

    def test_no_embedder_no_warning(self, tmp_path: Path, caplog) -> None:
        """FTS-only (no embedder) never warns about dimensions."""
        repo = tmp_path / "repo"
        _build_repo(repo)

        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=None)

        with caplog.at_level(logging.WARNING, logger="source_recall.querier"):
            querier.query("authenticate")

        warnings = [r for r in caplog.records if "dimension" in r.message.lower()]
        assert warnings == []
        querier.close()
