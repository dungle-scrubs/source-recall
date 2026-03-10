"""Query server: loads model once, serves queries over HTTP."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

_SENTINEL = object()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Search query request.

    @param question: Natural language or symbol query.
    @param top_k: Max results (default: config value).
    """

    question: str
    top_k: int | None = None


class QueryResultResponse(BaseModel):
    """Single search result.

    @param chunk_id: Unique chunk identifier.
    @param file_path: Repo-relative path.
    @param symbol_name: Symbol name (may be empty).
    @param symbol_type: Structural classification.
    @param content: Chunk text.
    @param score: Retrieval score.
    @param start_line: First line in source file.
    @param end_line: Last line in source file.
    @param search_quality: Parse fidelity (ast/regex/text_fallback).
    @param match_reason: How the result was found.
    """

    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    content: str
    score: float
    start_line: int
    end_line: int
    search_quality: str
    match_reason: str


class QueryResponse(BaseModel):
    """Search response.

    @param results: Ranked list of results.
    @param query_ms: Query latency in milliseconds.
    """

    results: list[QueryResultResponse]
    query_ms: float


class StatusResponse(BaseModel):
    """Index status response.

    @param repo_path: Repository path.
    @param file_count: Total indexed files.
    @param chunk_count: Total chunks.
    @param vector_count: Chunks with vectors.
    @param embed_model: Embedding model name.
    @param embed_dimensions: Vector dimensions.
    @param db_size_bytes: Database file size.
    @param indexed_at: Last index timestamp.
    """

    repo_path: str
    file_count: int
    chunk_count: int
    vector_count: int
    embed_model: str
    embed_dimensions: int
    db_size_bytes: int
    indexed_at: str


class RefreshResponse(BaseModel):
    """Refresh response.

    @param files_updated: Number of files re-indexed.
    @param refresh_ms: Refresh latency in milliseconds.
    """

    files_updated: int
    refresh_ms: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    repo_path: Path,
    embedder: Any | None = _SENTINEL,
) -> FastAPI:
    """Create a FastAPI app bound to a specific repository index.

    Loads the embedder and opens the index on startup. All query
    requests share the pre-loaded model — no per-request load cost.

    @param repo_path: Absolute path to the repository root.
    @param embedder: Embedder instance (omit for auto-create, None for FTS-only).
    @returns: Configured FastAPI application.
    """
    # Shared state — populated during lifespan startup.
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Load embedder and open index on server start."""
        from source_recall import Index

        logger.info("Loading embedder for %s...", repo_path)
        t0 = time.monotonic()

        idx = Index(repo_path, embedder=embedder)
        idx.status()

        elapsed = time.monotonic() - t0
        logger.info("Server ready (model loaded in %.1fs)", elapsed)

        state["index"] = idx
        state["repo_path"] = repo_path
        yield

    app = FastAPI(
        title="source-recall",
        description="Code search and retrieval server.",
        lifespan=lifespan,
    )

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        """Search the index.

        @param req: Query request with question and optional top_k.
        @returns: Ranked results with query latency.
        """
        idx: Any = state["index"]

        t0 = time.monotonic()
        results = idx.query(req.question, top_k=req.top_k)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return QueryResponse(
            results=[
                QueryResultResponse(
                    chunk_id=r.chunk_id,
                    file_path=r.file_path,
                    symbol_name=r.symbol_name,
                    symbol_type=r.symbol_type,
                    content=r.content,
                    score=r.score,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    search_quality=r.search_quality,
                    match_reason=r.match_reason,
                )
                for r in results
            ],
            query_ms=round(elapsed_ms, 1),
        )

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        """Get index status.

        @returns: Index metrics.
        """
        idx: Any = state["index"]
        s = idx.status()
        return StatusResponse(
            repo_path=s.repo_path,
            file_count=s.file_count,
            chunk_count=s.chunk_count,
            vector_count=s.vector_count,
            embed_model=s.embed_model,
            embed_dimensions=s.embed_dimensions,
            db_size_bytes=s.db_size_bytes,
            indexed_at=s.indexed_at,
        )

    @app.post("/refresh", response_model=RefreshResponse)
    async def refresh() -> RefreshResponse:
        """Incrementally refresh the index.

        @returns: Number of files updated with latency.
        """
        idx: Any = state["index"]

        t0 = time.monotonic()
        count = idx.refresh()
        elapsed_ms = (time.monotonic() - t0) * 1000

        return RefreshResponse(
            files_updated=count,
            refresh_ms=round(elapsed_ms, 1),
        )

    return app
