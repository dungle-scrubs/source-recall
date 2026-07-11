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
# Defense-in-depth: config.json carries the ``auto_map`` that wires the
# ``trust_remote_code=True`` model + config classes, so a tampered config
# at this exact revision is a supply-chain vector.  ``_verify_config_checksum``
# recomputes this over the cached bytes before the model loads and refuses
# to proceed on mismatch.  Bump after auditing the diff when updating
# _CODERANK_REVISION.
_CODERANK_CONFIG_SHA256 = (
    "5ff856a41d0f53ef2d74520627d464bd75c2efd8f26f381bd528654895c29b6c"
)
_QUERY_PREFIX = "Represent this query for searching relevant code: "

# Upper bound on the sentence-transformers encode batch size.  Sequences
# are truncated to 512 tokens (see ``max_seq_length`` below), so attention
# matrices are bounded per sequence; the batch dimension still multiplies
# peak memory, so cap it to keep long-chunk builds from exhausting RAM.
_MAX_ENCODE_BATCH = 32


class EmbedderVerificationError(RuntimeError):
    """Raised when a downloaded model artifact fails integrity verification.

    Signals that the cached ``config.json`` for the pinned revision does not
    match ``_CODERANK_CONFIG_SHA256``, so the model is refused rather than
    loaded via its ``trust_remote_code`` path against unverified config.
    """


class CodeRankEmbedder:
    """Local CodeRankEmbed via sentence-transformers + ONNX Runtime.

    Downloads the model (~522 MB) on first use and caches in
    ``~/.cache/huggingface/``.

    @param show_progress: Show download progress bar (default: True).
    @param encode_batch_size: Batch size handed to ``model.encode``.
        Clamped to ``[1, _MAX_ENCODE_BATCH]`` so an over-large config value
        cannot trigger an attention-matrix OOM.
    """

    def __init__(
        self, *, show_progress: bool = True, encode_batch_size: int = 32
    ) -> None:
        self._show_progress = show_progress
        self._encode_batch_size = max(1, min(int(encode_batch_size), _MAX_ENCODE_BATCH))
        self._model: object | None = None

    def _load_model(self) -> object:
        """Lazy-load the SentenceTransformer model.

        @returns: Loaded SentenceTransformer instance.
        """
        if self._model is not None:
            return self._model

        import os

        # Verify config.json integrity BEFORE constructing the model: the
        # config drives the trust_remote_code auto_map, so it must match the
        # pinned checksum before any remote code is trusted.
        self._verify_config_checksum()

        from sentence_transformers import SentenceTransformer

        # Force CPU.  MPS (Apple Silicon GPU) shares memory with the
        # display compositor — large attention matrices from long code
        # chunks trigger Metal OOM that freezes the entire system.
        # CPU is also faster than MPS for this model size.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        logger.info("Loading %s (first run downloads ~522 MB)...", _CODERANK_MODEL)
        model = SentenceTransformer(
            _CODERANK_MODEL,
            trust_remote_code=True,
            revision=_CODERANK_REVISION,
            device="cpu",
        )
        # The model defaults to 8192 tokens — attention is O(n²) so
        # long sequences explode memory (9.7 GB at 8192, 1.7 GB at 512).
        # 512 tokens covers most function signatures + bodies and keeps
        # builds fast (~134ms/chunk vs 1825ms at full context).
        model.max_seq_length = 512
        self._model = model
        return self._model

    def _verify_config_checksum(self) -> None:
        """Verify the pinned revision's config.json matches its checksum.

        Resolves the cached ``config.json`` for ``_CODERANK_REVISION`` (via
        the HuggingFace cache, downloading it alone if not yet present),
        hashes its bytes, and compares against ``_CODERANK_CONFIG_SHA256``.

        @raises EmbedderVerificationError: If the config cannot be located or
            its SHA-256 does not match the pinned constant.
        """
        import hashlib

        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(
            _CODERANK_MODEL, "config.json", revision=_CODERANK_REVISION
        )
        # try_to_load_from_cache returns the file path (str) when cached, a
        # _CACHED_NO_EXIST sentinel when known-absent, or None when unknown.
        # Fall back to fetching just config.json for the pinned revision.
        if isinstance(cached, str):
            config_path = cached
        else:
            config_path = hf_hub_download(
                _CODERANK_MODEL, "config.json", revision=_CODERANK_REVISION
            )

        with open(config_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()

        if digest != _CODERANK_CONFIG_SHA256:
            raise EmbedderVerificationError(
                f"config.json checksum mismatch for {_CODERANK_MODEL}"
                f"@{_CODERANK_REVISION}: expected {_CODERANK_CONFIG_SHA256}, "
                f"got {digest}. Refusing to load a possibly tampered model."
            )

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
        # Batch size comes from config (embed_batch_size), clamped in
        # __init__ to a safe max.  Sequences are truncated to 512 tokens,
        # so the per-sequence attention matrix is bounded; the clamp keeps
        # the batch dimension from multiplying peak memory unboundedly.
        embeddings = model.encode(  # type: ignore[union-attr]
            texts, show_progress_bar=False, batch_size=self._encode_batch_size
        )
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
