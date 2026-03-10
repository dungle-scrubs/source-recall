"""source-recall: Code search and retrieval for AI coding tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.config import SRConfig, resolve_config
from source_recall.models import (
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

_SENTINEL = object()


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

    def build(self) -> Path:
        """Build the full index with atomic swap.

        @returns: Path to the index database.
        """
        from source_recall.builder import IndexBuilder

        builder = IndexBuilder(
            self.repo_path, self.config, self._on_progress, self._embedder
        )
        return builder.build()

    def refresh(self) -> int:
        """Incrementally refresh the index.

        @returns: Number of files re-indexed.
        """
        from source_recall.builder import IndexBuilder

        builder = IndexBuilder(
            self.repo_path, self.config, self._on_progress, self._embedder
        )
        return builder.refresh()

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Search the index.

        @param question: Natural language or symbol query.
        @param top_k: Override number of results.
        @returns: Ranked list of QueryResult.
        """
        from source_recall.querier import IndexQuerier

        querier = IndexQuerier(self.repo_path, self.config, self._embedder)
        try:
            return querier.query(question, top_k=top_k)
        finally:
            querier.close()

    def status(self) -> IndexStatus:
        """Get index status information.

        @returns: IndexStatus with all metrics.
        """
        from source_recall.querier import IndexQuerier

        querier = IndexQuerier(self.repo_path, self.config)
        try:
            return querier.status()
        finally:
            querier.close()
