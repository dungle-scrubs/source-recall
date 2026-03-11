"""Embedding providers: Protocol, local CodeRankEmbed, and test BagOfWords."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Embedding provider contract.

    Implementations must handle the asymmetry between code chunks
    (embedded as-is) and queries (which may need task prefixes).
    """

    @property
    def dimensions(self) -> int:
        """Dimensionality of the output vectors.

        @returns: Number of dimensions.
        """
        ...

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed code chunks (no query prefix).

        @param texts: Raw code strings.
        @returns: List of embedding vectors, one per input.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (with task prefix if required).

        @param query: User's search query.
        @returns: Single embedding vector.
        """
        ...


# ---------------------------------------------------------------------------
# CodeRankEmbedder — local inference via sentence-transformers
# ---------------------------------------------------------------------------

_CODERANK_MODEL = "nomic-ai/CodeRankEmbed"
_CODERANK_DIMENSIONS = 768
_QUERY_PREFIX = "Represent this query for searching relevant code: "


class CodeRankEmbedder:
    """Local CodeRankEmbed via sentence-transformers + ONNX Runtime.

    Downloads the model (~522 MB) on first use and caches in
    ``~/.cache/huggingface/``.

    @param show_progress: Show download progress bar (default: True).
    """

    def __init__(self, *, show_progress: bool = True) -> None:
        self._show_progress = show_progress
        self._model: object | None = None

    def _load_model(self) -> object:
        """Lazy-load the SentenceTransformer model.

        @returns: Loaded SentenceTransformer instance.
        """
        if self._model is not None:
            return self._model

        import os

        from sentence_transformers import SentenceTransformer

        # Cap PyTorch threads to half the CPU cores to avoid starving
        # the OS during long builds.  Users can override via env vars.
        if "OMP_NUM_THREADS" not in os.environ:
            import multiprocessing

            cap = max(1, multiprocessing.cpu_count() // 2)
            import torch

            torch.set_num_threads(cap)

        logger.info("Loading %s (first run downloads ~522 MB)...", _CODERANK_MODEL)
        self._model = SentenceTransformer(
            _CODERANK_MODEL,
            trust_remote_code=True,
        )
        return self._model

    @property
    def dimensions(self) -> int:
        """Output dimensionality (768 for CodeRankEmbed).

        @returns: 768.
        """
        return _CODERANK_DIMENSIONS

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed code chunks without query prefix.

        @param texts: Raw code strings.
        @returns: List of 768-d vectors.
        """
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)  # type: ignore[union-attr]
        return embeddings.tolist()  # type: ignore[union-attr]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with CodeRankEmbed's required prefix.

        @param query: User's search query.
        @returns: 768-d vector.
        """
        model = self._load_model()
        prefixed = f"{_QUERY_PREFIX}{query}"
        return model.encode([prefixed], show_progress_bar=False)[0].tolist()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# BagOfWordsEmbedder — test-only, produces genuine similarity
# ---------------------------------------------------------------------------


class BagOfWordsEmbedder:
    """Test embedder with genuine cosine similarity from shared vocabulary.

    Shared words → high cosine, disjoint words → low cosine.
    This catches real retrieval bugs that hash-derived random vectors
    would miss.

    @param dimensions: Output vector dimensionality.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dim = dimensions

    @property
    def dimensions(self) -> int:
        """Output dimensionality.

        @returns: Configured dimensions.
        """
        return self._dim

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed code chunks as bag-of-words vectors.

        @param texts: Raw code strings.
        @returns: List of L2-normalized vectors.
        """
        return [self._bow(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query as a bag-of-words vector.

        @param query: User's search query.
        @returns: L2-normalized vector.
        """
        return self._bow(query)

    def _bow(self, text: str) -> list[float]:
        """Convert text to a bag-of-words vector.

        Each word hashes to a bucket; collisions accumulate.
        The result is L2-normalized.

        @param text: Input text.
        @returns: Normalized float vector.
        """
        vec = [0.0] * self._dim
        for word in text.lower().split():
            idx = hash(word) % self._dim
            vec[idx] += 1.0
        # L2 normalize.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
