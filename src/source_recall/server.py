"""Query server: loads model once, serves queries over HTTP."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
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
    @param repo: Repo name to query (required when multiple repos served).
    """

    question: str
    top_k: int | None = None
    repo: str | None = None


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


class HealthResponse(BaseModel):
    """Health check response.

    @param ok: Whether the server is ready to handle queries.
    @param repos: List of loaded repo names.
    @param uptime_s: Seconds since server started.
    """

    ok: bool
    repos: list[str]
    uptime_s: float


class RepoInfo(BaseModel):
    """Info about a single loaded repo.

    @param name: Short name (directory basename).
    @param path: Absolute path.
    @param file_count: Total indexed files.
    @param chunk_count: Total chunks.
    @param vector_count: Chunks with vectors.
    """

    name: str
    path: str
    file_count: int
    chunk_count: int
    vector_count: int


class ReposResponse(BaseModel):
    """List of all loaded repos.

    @param repos: Repo details.
    """

    repos: list[RepoInfo]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    repo_paths: list[Path] | Path,
    embedder: Any | None = _SENTINEL,
) -> FastAPI:
    """Create a FastAPI app serving one or more repository indexes.

    Loads the embedder and opens indexes on startup. All query
    requests share the pre-loaded model — no per-request load cost.

    @param repo_paths: One or more repo root paths to serve.
    @param embedder: Embedder instance (omit for auto-create, None for FTS-only).
    @returns: Configured FastAPI application.
    """
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]

    # Shared state — populated during lifespan startup.
    state: dict[str, Any] = {"indexes": {}, "started_at": 0.0}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Load embedder and open indexes on server start."""
        from source_recall import Index

        t0 = time.monotonic()
        state["started_at"] = t0

        for repo_path in repo_paths:
            name = repo_path.name
            logger.info("Loading index for %s (%s)...", name, repo_path)
            idx = Index(repo_path, embedder=embedder)
            idx.status()
            state["indexes"][name] = {"index": idx, "path": repo_path}

        elapsed = time.monotonic() - t0
        names = list(state["indexes"].keys())
        logger.info(
            "Server ready (%d repos loaded in %.1fs): %s", len(names), elapsed, names
        )
        yield

    app = FastAPI(
        title="source-recall",
        description="Code search and retrieval server.",
        lifespan=lifespan,
    )

    def _resolve_index(repo: str | None) -> Any:
        """Resolve a repo name to its Index, or return the only one.

        @param repo: Repo name (required if multiple repos loaded).
        @returns: Index instance.
        @raises HTTPException: If repo not found or ambiguous.
        """
        indexes = state["indexes"]
        if len(indexes) == 1 and repo is None:
            return next(iter(indexes.values()))["index"]
        if repo is None:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple repos loaded. Specify 'repo': one of {list(indexes.keys())}",
            )
        if repo not in indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Repo '{repo}' not found. Available: {list(indexes.keys())}",
            )
        return indexes[repo]["index"]

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness check — returns ok if indexes are loaded.

        @returns: Health status with repo list and uptime.
        """
        return HealthResponse(
            ok=len(state["indexes"]) > 0,
            repos=list(state["indexes"].keys()),
            uptime_s=round(time.monotonic() - state["started_at"], 1),
        )

    @app.get("/repos", response_model=ReposResponse)
    async def repos() -> ReposResponse:
        """List all loaded repos with stats.

        @returns: Repo details.
        """
        items = []
        for name, entry in state["indexes"].items():
            s = entry["index"].status()
            items.append(
                RepoInfo(
                    name=name,
                    path=str(entry["path"]),
                    file_count=s.file_count,
                    chunk_count=s.chunk_count,
                    vector_count=s.vector_count,
                )
            )
        return ReposResponse(repos=items)

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        """Search an index.

        @param req: Query request with question, optional top_k and repo.
        @returns: Ranked results with query latency.
        """
        idx = _resolve_index(req.repo)

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
    async def status(repo: str | None = None) -> StatusResponse:
        """Get index status.

        @param repo: Repo name (optional if single repo).
        @returns: Index metrics.
        """
        idx = _resolve_index(repo)
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
    async def refresh(repo: str | None = None) -> RefreshResponse:
        """Incrementally refresh an index.

        @param repo: Repo name (optional if single repo).
        @returns: Number of files updated with latency.
        """
        idx = _resolve_index(repo)

        t0 = time.monotonic()
        count = idx.refresh()
        elapsed_ms = (time.monotonic() - t0) * 1000

        return RefreshResponse(
            files_updated=count,
            refresh_ms=round(elapsed_ms, 1),
        )

    return app
