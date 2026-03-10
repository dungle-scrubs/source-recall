"""Reranker: cross-encoder relevance scoring for result reordering."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking search results.

    A reranker scores (query, document) pairs and returns them
    sorted by relevance.
    """

    def rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank items by relevance to query.

        @param query: User query.
        @param items: Candidate results (must have 'content' key).
        @returns: List of (item, score) sorted by score descending.
        """
        ...


class DummyReranker:
    """No-op reranker that preserves original order.

    Used as fallback when cross-encoder is unavailable.
    """

    def rerank(
        self,
        _query: str,
        items: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Return items in original order with decreasing scores.

        @param _query: User query (ignored).
        @param items: Candidate results.
        @returns: (item, score) pairs in original order.
        """
        return [(item, 1.0 - i * 0.01) for i, item in enumerate(items)]


class CrossEncoderReranker:
    """Reranker using a cross-encoder model for relevance scoring.

    Loads the model lazily on first use.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        """Initialize with a cross-encoder model name.

        @param model_name: HuggingFace model identifier.
        """
        self._model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the cross-encoder model.

        @returns: CrossEncoder instance.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank items using cross-encoder relevance scoring.

        @param query: User query.
        @param items: Candidate results (must have 'content' key).
        @returns: (item, score) sorted by score descending.
        """
        if not items:
            return []

        model = self._ensure_model()
        pairs = [(query, item["content"]) for item in items]
        scores = model.predict(pairs).tolist()

        scored = list(zip(items, scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
