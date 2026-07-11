"""source-recall: Code search and retrieval for AI coding tools."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.concurrency import ReaderWriterLock
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

try:
    __version__ = _pkg_version("source-recall")
except PackageNotFoundError:  # pragma: no cover - editable/source checkout
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from collections.abc import Callable

    from source_recall.embedder import Embedder

logger = logging.getLogger(__name__)

# Shared reader/writer lock (moved to source_recall.concurrency so the
# querier can reuse it for its own store-lifetime lock).  ``Index`` uses
# it so concurrent query/status run in parallel while refresh/build/close
# take an exclusive write lock to swap the underlying querier (M-2 fix).
_ReaderWriterLock = ReaderWriterLock


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
    "__version__",
]


class Index:
    """Thin facade over IndexBuilder + IndexQuerier.

    @param repo_path: Path to the repository root (default: cwd).
    @param on_progress: Optional callback(file_path, current, total).
    @param on_phase: Optional callback(phase) for build lifecycle progress.
    @param on_progress_detail: Optional callback(detail) for build timing detail.
    @param embedder: Embedder instance, None to disable, or omit to auto-create.
        When omitted (default), a CodeRankEmbedder is created if embed_enabled
        is True in config. Pass None explicitly to force FTS-only mode.
    @param kwargs: Config overrides passed to resolve_config.
    """

    def __init__(
        self,
        repo_path: str | Path = ".",
        on_progress: Callable[[str, int, int], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_progress_detail: Callable[[dict[str, object]], None] | None = None,
        embedder: Embedder | None | object = _SENTINEL,
        **kwargs: object,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = resolve_config(self.repo_path, **kwargs)
        self._on_progress = on_progress
        self._on_phase = on_phase
        self._on_progress_detail = on_progress_detail

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
        # M2: reader/writer lock.  query()/status() are readers (may run
        # concurrently); refresh()/build()/close() are writers (exclusive),
        # so a refresh cannot close the connection mid-query.
        self._querier_lock = _ReaderWriterLock()

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
            self.repo_path,
            self.config,
            self._on_progress,
            self._embedder,
            on_phase=self._on_phase,
            on_progress_detail=self._on_progress_detail,
        )
        return builder.build()

    def refresh(self, *, files: list[str] | None = None) -> int:
        """Incrementally refresh the index.

        @param files: Optional list of repo-relative paths to re-index.
            When provided, only those files are re-indexed (targeted refresh).
            When omitted, full change detection runs.
        @returns: Number of files re-indexed.
        """
        from source_recall.builder import IndexBuilder

        # Invalidate cached querier — the DB may change.
        self._close_querier()
        builder = IndexBuilder(
            self.repo_path,
            self.config,
            self._on_progress,
            self._embedder,
            on_phase=self._on_phase,
            on_progress_detail=self._on_progress_detail,
        )
        return builder.refresh(files=files)

    def _close_querier(self) -> None:
        """Close and discard the cached querier.

        Thread-safe: takes the *write* lock so concurrent readers (queries)
        drain before the connection is closed (M2).

        @raises nothing.
        """
        self._querier_lock.acquire_write()
        try:
            if self._querier is not None:
                self._querier.close()  # type: ignore[union-attr]
                self._querier = None
        finally:
            self._querier_lock.release_write()

    def _ensure_querier(self) -> object:
        """Create a querier if one doesn't exist.

        Caller MUST already hold the *write* lock.

        @returns: IndexQuerier instance.
        """
        if self._querier is not None:
            return self._querier

        from source_recall.querier import IndexQuerier

        self._querier = IndexQuerier(
            self.repo_path, self.config, self._embedder, self._get_reranker()
        )
        return self._querier

    def _with_querier(self) -> tuple[object, _ReaderWriterLock]:
        """Return the cached querier and the lock, holding the *read* lock.

        The caller MUST call ``release_read`` when done with the querier.
        Lazily opens the querier under the write lock on first access.
        Holding the read lock for the query's duration prevents a
        concurrent refresh from closing the connection mid-flight (M2).

        A concurrent ``refresh``/``build``/``close`` can set
        ``self._querier`` to None at any write-lock boundary, so after
        (re-)acquiring the read lock we re-check and loop until we hold
        the read lock over a non-None querier (C-1 fix).

        @returns: (IndexQuerier instance, lock).
        """
        while True:
            self._querier_lock.acquire_read()
            if self._querier is not None:
                return self._querier, self._querier_lock
            # Read lock held but no querier — upgrade to write lock.
            self._querier_lock.release_read()
            self._querier_lock.acquire_write()
            try:
                # Re-check: another writer may have created it first.
                if self._querier is None:
                    self._ensure_querier()
            finally:
                self._querier_lock.release_write()
            # Loop back: re-acquire the read lock and re-verify.  If a
            # concurrent close ran between release_write and the next
            # acquire_read, we'll see None again and reopen.

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        branch: str | None = None,
        query_vec: list[float] | None = None,
    ) -> list[QueryResult]:
        """Search the index.

        Thread-safe: concurrent ``query`` calls share the cached querier
        under the read lock and run in parallel.  A concurrent
        ``refresh()`` takes the write lock and waits for in-flight
        queries to drain, so it cannot close the connection mid-flight
        (M2 fix).

        @param question: Natural language or symbol query.
        @param top_k: Override number of results.
        @param branch: Filter to this branch. None = active branch.
        @param query_vec: Precomputed query embedding to reuse for vector
            search (skips re-embedding). Callers fanning one query across
            several indexes that share an embedder pass it to embed once.
        @returns: Ranked list of QueryResult.
        """
        querier, lock = self._with_querier()
        try:
            return querier.query(  # type: ignore[union-attr]
                question, top_k=top_k, branch=branch, query_vec=query_vec
            )
        finally:
            lock.release_read()

    def status(self) -> IndexStatus:
        """Get index status information.

        @returns: IndexStatus with all metrics.
        """
        querier, lock = self._with_querier()
        try:
            return querier.status()  # type: ignore[union-attr]
        finally:
            lock.release_read()
