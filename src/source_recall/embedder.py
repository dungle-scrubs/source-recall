"""Embedding providers: Protocol, local CodeRankEmbed, and test BagOfWords."""

from __future__ import annotations

import logging
import re
import threading
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
#
# PRIMARY integrity guarantee: a full 40-hex git commit SHA content-addresses
# the entire repo tree at that revision, so HuggingFace Hub can only resolve it
# to the exact file bytes that were audited.  A branch/tag ref would be mutable
# and defeat the pin, so ``_verify_model_integrity`` refuses to load unless this
# is a full commit SHA (see ``_FULL_COMMIT_SHA_RE``).
_CODERANK_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"

# Full 40-hex git commit SHA.  Used to assert the revision pin above is a
# content-addressing commit SHA rather than a mutable ref.
_FULL_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

# SHA-256 of EVERY file that is executed when the model loads under
# ``trust_remote_code=True`` at ``_CODERANK_REVISION``: config.json plus each
# remote ``.py`` module named by config.json's ``auto_map`` (the configuration
# and modeling classes that trust_remote_code imports and runs).  Checksumming
# config.json alone is insufficient — the auto_map modules are the code that
# actually executes.  ``_verify_model_integrity`` resolves each file for the
# pinned revision, recomputes its SHA-256, and refuses to load on ANY mismatch,
# missing file, or auto_map module absent from this set.  This is
# defense-in-depth layered on the commit-SHA revision pin.  Bump after auditing
# the diff when updating _CODERANK_REVISION.
_CODERANK_TRUSTED_FILE_SHA256 = {
    "config.json": ("5ff856a41d0f53ef2d74520627d464bd75c2efd8f26f381bd528654895c29b6c"),
    "configuration_hf_nomic_bert.py": (
        "8632792e922e62ab1a6feaab15baf406e223c0547a21de23ab4f520b2b36d674"
    ),
    "modeling_hf_nomic_bert.py": (
        "502ccfb9c2d5dac976109ac2f04dc3125d8540329e5d4af92bc587d6dc65edcd"
    ),
}
_QUERY_PREFIX = "Represent this query for searching relevant code: "


def _auto_map_module_files(auto_map: dict[str, object]) -> list[str]:
    """Discover the remote ``.py`` module files named by a config auto_map.

    ``auto_map`` maps entries like
    ``"AutoModel" -> "modeling_hf_nomic_bert.NomicBertModel"``; the part before
    the final dot is the module whose file is ``<module>.py``.  These are the
    exact files ``trust_remote_code=True`` imports and executes, so each must be
    integrity-checked before the model loads.

    @param auto_map: The ``auto_map`` mapping from config.json.
    @returns: Sorted unique ``.py`` filenames referenced by the auto_map.
    """
    files: set[str] = set()
    for ref in auto_map.values():
        if not isinstance(ref, str) or "." not in ref:
            continue
        module = ref.rsplit(".", 1)[0]
        # Cross-repo refs look like ``repo_id--module``; keep the local module.
        module = module.split("--")[-1]
        files.add(f"{module}.py")
    return sorted(files)


# Upper bound on the sentence-transformers encode batch size.  Sequences
# are truncated to 512 tokens (see ``max_seq_length`` below), so attention
# matrices are bounded per sequence; the batch dimension still multiplies
# peak memory, so cap it to keep long-chunk builds from exhausting RAM.
_MAX_ENCODE_BATCH = 32


class EmbedderVerificationError(RuntimeError):
    """Raised when a model artifact fails the trust_remote_code integrity gate.

    Signals one of: the revision pin is not a full commit SHA; a trusted file
    (config.json or an ``auto_map`` remote module) could not be resolved for the
    pinned revision; its bytes do not match the pinned SHA-256; or config.json's
    ``auto_map`` names a module absent from the pinned checksum set.  In every
    case the model is refused rather than loaded via its ``trust_remote_code``
    path against unverified code.
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
        # Guards _load_model so the startup warmup thread and the first real
        # query cannot both construct (and download) the model.
        self._model_lock = threading.Lock()

    def _load_model(self) -> object:
        """Lazy-load the SentenceTransformer model.

        Idempotent under concurrency via double-checked locking: the Stage-4
        startup warmup thread can race a real query, and without the lock both
        would construct — and download — the model, spiking memory.

        @returns: Loaded SentenceTransformer instance.
        """
        if self._model is not None:
            return self._model

        with self._model_lock:
            # Re-check under the lock: another thread may have finished the
            # load while we waited on the lock.
            if self._model is not None:
                return self._model

            import os

            # Verify integrity BEFORE constructing the model: the config drives
            # the trust_remote_code auto_map and the remote .py modules it
            # imports, so config.json AND every executed module must match their
            # pinned checksums before any remote code is trusted.
            self._verify_model_integrity()

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

    def _resolve_trusted_file(self, filename: str) -> str:
        """Resolve a trusted model file to a local path for the pinned revision.

        Prefers the HuggingFace cache; downloads the single file for
        ``_CODERANK_REVISION`` if not yet cached.

        @param filename: Repo-relative filename (e.g. ``config.json``).
        @returns: Absolute path to the resolved file.
        @raises EmbedderVerificationError: If the file cannot be resolved.
        """
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(
            _CODERANK_MODEL, filename, revision=_CODERANK_REVISION
        )
        # try_to_load_from_cache returns the file path (str) when cached, a
        # _CACHED_NO_EXIST sentinel when known-absent, or None when unknown.
        if isinstance(cached, str):
            return cached
        try:
            return hf_hub_download(
                _CODERANK_MODEL, filename, revision=_CODERANK_REVISION
            )
        except Exception as e:
            raise EmbedderVerificationError(
                f"Could not resolve trusted file '{filename}' for "
                f"{_CODERANK_MODEL}@{_CODERANK_REVISION}: {e}. "
                "Refusing to load an unverifiable model."
            ) from e

    def _check_file_sha(self, filename: str) -> str:
        """Verify a trusted file's SHA-256 against the pinned set.

        @param filename: Repo-relative filename to verify.
        @returns: The verified file's local path (for further parsing).
        @raises EmbedderVerificationError: If the file is not in the pinned set
            or its bytes do not match the pinned SHA-256.
        """
        import hashlib

        expected = _CODERANK_TRUSTED_FILE_SHA256.get(filename)
        if expected is None:
            raise EmbedderVerificationError(
                f"Trusted file '{filename}' (referenced by config.json auto_map) "
                f"is not in the pinned checksum set for {_CODERANK_MODEL}"
                f"@{_CODERANK_REVISION}. Refusing to execute unpinned remote code."
            )
        path = self._resolve_trusted_file(filename)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected:
            raise EmbedderVerificationError(
                f"'{filename}' checksum mismatch for {_CODERANK_MODEL}"
                f"@{_CODERANK_REVISION}: expected {expected}, got {digest}. "
                "Refusing to load a possibly tampered model."
            )
        return path

    def _verify_model_integrity(self) -> None:
        """Fail-closed integrity gate for the trust_remote_code load path.

        1. Asserts ``_CODERANK_REVISION`` is a full commit SHA — the primary
           guarantee that the Hub resolves the pin to exact audited bytes.
        2. Verifies config.json against its pinned SHA-256 (it drives the
           auto_map, so it is checked first).
        3. Parses the verified config's ``auto_map`` to discover the remote
           ``.py`` modules ``trust_remote_code`` will execute, and verifies each
           against the pinned set.
        4. Verifies any already-materialized copy of those modules in the
           transformers dynamic-module cache (the code that is actually
           imported and run), which the Hub-snapshot check alone would miss.

        @raises EmbedderVerificationError: On a non-SHA revision, an
            unresolvable trusted file, any checksum mismatch, or an auto_map
            module missing from the pinned set.
        """
        import json

        if not _FULL_COMMIT_SHA_RE.fullmatch(_CODERANK_REVISION):
            raise EmbedderVerificationError(
                f"_CODERANK_REVISION {_CODERANK_REVISION!r} is not a full 40-hex "
                "commit SHA. Only a commit SHA content-pins the model's files; "
                "refusing to load against a mutable ref."
            )

        # config.json first — it names the remote modules to execute.
        config_path = self._check_file_sha("config.json")
        with open(config_path, "rb") as handle:
            config = json.loads(handle.read())

        auto_map = config.get("auto_map", {})
        if not isinstance(auto_map, dict):
            auto_map = {}
        module_files = _auto_map_module_files(auto_map)
        for module_file in module_files:
            self._check_file_sha(module_file)

        # The Hub snapshot is verified, but trust_remote_code executes a COPY
        # of these modules under the transformers dynamic-module cache and, for
        # a pinned revision, will not overwrite an existing copy. Verify any
        # such materialized copy too so a stale or pre-planted file there cannot
        # execute behind a clean snapshot.
        self._verify_execution_cache(module_files)

    def _verify_execution_cache(self, module_files: list[str]) -> None:
        """Verify materialized copies of remote modules in the exec cache.

        ``trust_remote_code`` copies each remote ``.py`` into the transformers
        dynamic-module cache (``HF_MODULES_CACHE/transformers_modules/...``) and
        imports THAT copy. For a pinned revision transformers will not overwrite
        an existing copy, so a stale or pre-planted file there would execute even
        though the pristine Hub snapshot passes verification. Any copy found for
        the pinned revision must match the pinned checksum. Absence is safe:
        transformers then materializes the copy from the already-verified
        snapshot bytes.

        The cache path's repo-name component is sanitized in a
        transformers-version-dependent way, so copies are located by globbing on
        the exact commit-revision directory + module filename rather than a
        reconstructed path.

        @param module_files: Remote module filenames to verify.
        @raises EmbedderVerificationError: On any checksum mismatch.
        """
        import hashlib
        from pathlib import Path

        try:
            from transformers.utils import HF_MODULES_CACHE
        except Exception:
            # transformers layout unknown — the snapshot check + revision pin
            # remain in force.
            return

        base = Path(HF_MODULES_CACHE) / "transformers_modules"
        if not base.is_dir():
            return

        for filename in module_files:
            expected = _CODERANK_TRUSTED_FILE_SHA256.get(filename)
            for cached in base.glob(f"**/{_CODERANK_REVISION}/{filename}"):
                if not cached.is_file():
                    continue
                digest = hashlib.sha256(cached.read_bytes()).hexdigest()
                if expected is None or digest != expected:
                    raise EmbedderVerificationError(
                        f"Executed remote-module copy {cached} does not match "
                        f"the pinned checksum for {_CODERANK_MODEL}"
                        f"@{_CODERANK_REVISION}. Refusing to run a possibly "
                        "tampered dynamic-module cache."
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
