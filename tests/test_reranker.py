"""Tests for the reranker module."""

from __future__ import annotations

import pytest

from source_recall.reranker import DummyReranker, Reranker


class TestRerankerProtocol:
    def test_dummy_reranker_preserves_order(self) -> None:
        """DummyReranker returns items in original order with same scores."""
        reranker = DummyReranker()
        items = [
            {"chunk_id": "a", "content": "first"},
            {"chunk_id": "b", "content": "second"},
            {"chunk_id": "c", "content": "third"},
        ]
        scored = reranker.rerank("query", items)
        assert [s[0]["chunk_id"] for s in scored] == ["a", "b", "c"]

    def test_dummy_satisfies_protocol(self) -> None:
        """DummyReranker satisfies the Reranker protocol."""
        reranker = DummyReranker()
        assert isinstance(reranker, Reranker)

    @pytest.mark.slow
    def test_cross_encoder_reranker_reorders(self) -> None:
        """CrossEncoderReranker reorders by relevance score."""
        from source_recall.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        items = [
            {"chunk_id": "irrelevant", "content": "The weather is nice today."},
            {
                "chunk_id": "relevant",
                "content": "def authenticate(user, password): return check_credentials(user, password)",
            },
            {
                "chunk_id": "somewhat",
                "content": "import logging\nlogger = logging.getLogger(__name__)",
            },
        ]
        scored = reranker.rerank("authentication function", items)

        # The relevant chunk should rank first.
        assert scored[0][0]["chunk_id"] == "relevant"
        # Scores should be descending.
        scores = [s[1] for s in scored]
        assert scores == sorted(scores, reverse=True)
