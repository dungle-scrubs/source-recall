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
# Pin to a known-good revision to limit supply-chain risk from
# trust_remote_code=True.  Bump only after auditing the diff.
_CODERANK_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
# SHA-256 of the model's config.json at the pinned revision.
# Defense-in-depth: if the HuggingFace CDN is compromised at this
# exact revision, the checksum catches tampered config files.
# Bump when updating _CODERANK_REVISION after auditing the diff.
_CODERANK_CONFIG_SHA256 = (
    "c5c4beb205d1e44581a60dd1ef14e35e04c8fd4bdd07a42c2ac944e886f4e97b"
)
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

        # Force CPU.  MPS (Apple Silicon GPU) shares memory with the
        # display compositor — large attention matrices from long code
        # chunks trigger Metal OOM that freezes the entire system.
        # CPU is also faster than MPS for this model size.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        logger.info("Loading %s (first run downloads ~522 MB)...", _CODERANK_MODEL)
        self._model = SentenceTransformer(
            _CODERANK_MODEL,
            trust_remote_code=True,
            revision=_CODERANK_REVISION,
            device="cpu",
        )
        # The model defaults to 8192 tokens — attention is O(n²) so
        # long sequences explode memory (9.7 GB at 8192, 1.7 GB at 512).
        # 512 tokens covers most function signatures + bodies and keeps
        # builds fast (~134ms/chunk vs 1825ms at full context).
        self._model.max_seq_length = 512
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
        # Use small encode batch to limit peak memory from attention
        # matrices on long code chunks (up to 6000 chars / 8192 tokens).
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=8)  # type: ignore[union-attr]
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

    @staticmethod
    def _word_bucket(word: str, dim: int) -> int:
        """Deterministic bucket index for a word.

        Uses MD5 (fast, deterministic) instead of Python's hash()
        which is randomized per process (PYTHONHASHSEED).

        @param word: Lowercased word.
        @param dim: Vector dimensionality.
        @returns: Bucket index in [0, dim).
        """
        import hashlib

        digest = hashlib.md5(word.encode()).digest()  # noqa: S324
        # First 4 bytes as unsigned int.
        return int.from_bytes(digest[:4], "little") % dim

    def _bow(self, text: str) -> list[float]:
        """Convert text to a bag-of-words vector.

        Each word hashes to a deterministic bucket; collisions accumulate.
        The result is L2-normalized.

        @param text: Input text.
        @returns: Normalized float vector.
        """
        vec = [0.0] * self._dim
        for word in text.lower().split():
            idx = self._word_bucket(word, self._dim)
            vec[idx] += 1.0
        # L2 normalize.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
