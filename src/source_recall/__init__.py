"""source-recall: Code search and retrieval for AI coding tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.config import SRConfig, resolve_config
from source_recall.models import (
    _SENTINEL,
    ConfigError,
    FileDiscoveryError,
    IndexIdentityError,
    IndexLockError,
    IndexNotFoundError,
    IndexStatus,
    QueryResult,
    SchemaVersionError,
    SourceRecallError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from source_recall.embedder import Embedder

logger = logging.getLogger(__name__)

__all__ = [
    "Index",
    "SRConfig",
    "QueryResult",
    "IndexStatus",
    "SourceRecallError",
    "IndexNotFoundError",
    "IndexLockError",
    "IndexIdentityError",
    "SchemaVersionError",
    "ConfigError",
    "FileDiscoveryError",
]


class Index:
    """Thin facade over IndexBuilder + IndexQuerier.

    @param repo_path: Path to the repository root (default: cwd).
    @param on_progress: Optional callback(file_path, current, total).
    @param embedder: Embedder instance, None to disable, or omit to auto-create.
        When omitted (default), a CodeRankEmbedder is created if embed_enabled
        is True in config. Pass None explicitly to force FTS-only mode.
    @param kwargs: Config overrides passed to resolve_config.
    """

    def __init__(
        self,
        repo_path: str | Path = ".",
        on_progress: Callable[[str, int, int], None] | None = None,
        embedder: Embedder | None | object = _SENTINEL,
        **kwargs: object,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = resolve_config(self.repo_path, **kwargs)
        self._on_progress = on_progress

        if embedder is _SENTINEL:
            # Auto-create embedder based on config.
            if self.config.embed_enabled:
                self._embedder = self._create_default_embedder()
            else:
                self._embedder = None
        else:
            self._embedder = embedder  # type: ignore[assignment]

        self._reranker: object = _SENTINEL  # Lazy-loaded.
        self._querier: object | None = None  # Cached IndexQuerier.

    @staticmethod
    def _create_default_embedder() -> Embedder | None:
        """Attempt to create a CodeRankEmbedder.

        @returns: CodeRankEmbedder instance or None on failure.
        """
        try:
            from source_recall.embedder import CodeRankEmbedder

            return CodeRankEmbedder()
        except Exception:
            logger.warning(
                "Could not create CodeRankEmbedder — falling back to FTS-only",
                exc_info=True,
            )
            return None

    def _get_reranker(self) -> object | None:
        """Lazily create a CrossEncoderReranker if config allows.

        Only loads the model when rerank_enabled is True in config.
        The model is ~90MB and takes ~2s to load, so we defer until
        the first query and only when explicitly enabled.

        @returns: Reranker instance or None.
        """
        if self._reranker is not _SENTINEL:
            return self._reranker

        if not getattr(self.config, "rerank_enabled", False):
            self._reranker = None
            return None

        try:
            from source_recall.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        except Exception:
            logger.warning("Could not create CrossEncoderReranker — skipping rerank")
            self._reranker = None
        return self._reranker

    def close(self) -> None:
        """Close underlying database connections.

        Safe to call multiple times. Should be called when the Index
        is no longer needed, or use the context manager instead.
        """
        self._close_querier()

    def __enter__(self) -> Index:
        """Context manager entry.

        @returns: Self.
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Context manager exit — closes connections."""
        self.close()

    def build(self) -> Path:
        """Build the full index with atomic swap.

        @returns: Path to the index database.
        """
        from source_recall.builder import IndexBuilder

        # Invalidate cached querier — the DB will be replaced.
        self._close_querier()
        builder = IndexBuilder(
            self.repo_path, self.config, self._on_progress, self._embedder
        )
        return builder.build()

    def refresh(self) -> int:
        """Incrementally refresh the index.

        @returns: Number of files re-indexed.
        """
        from source_recall.builder import IndexBuilder

        # Invalidate cached querier — the DB may change.
        self._close_querier()
        builder = IndexBuilder(
            self.repo_path, self.config, self._on_progress, self._embedder
        )
        return builder.refresh()

    def _close_querier(self) -> None:
        """Close and discard the cached querier."""
        if self._querier is not None:
            self._querier.close()  # type: ignore[union-attr]
            self._querier = None

    def _get_querier(self) -> object:
        """Get or create a cached IndexQuerier.

        Reuses the same querier (and DB connection) across queries
        to avoid per-query migration checks and connection overhead.

        @returns: IndexQuerier instance.
        """
        if self._querier is not None:
            return self._querier

        from source_recall.querier import IndexQuerier

        self._querier = IndexQuerier(
            self.repo_path, self.config, self._embedder, self._get_reranker()
        )
        return self._querier

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        branch: str | None = None,
    ) -> list[QueryResult]:
        """Search the index.

        @param question: Natural language or symbol query.
        @param top_k: Override number of results.
        @param branch: Filter to this branch. None = active branch.
        @returns: Ranked list of QueryResult.
        """
        querier = self._get_querier()
        return querier.query(question, top_k=top_k, branch=branch)  # type: ignore[union-attr]

    def status(self) -> IndexStatus:
        """Get index status information.

        @returns: IndexStatus with all metrics.
        """
        querier = self._get_querier()
        return querier.status()  # type: ignore[union-attr]
