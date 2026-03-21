"""Daemon-mode server: multi-repo registry with dynamic add/remove."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from source_recall.daemon_config import DaemonConfig
from source_recall.models import _SENTINEL
from source_recall.repo_manager import RepoManager, SlotState
from source_recall.server import (
    QueryRequest,
    QueryResponse,
    QueryResultResponse,
    RefreshResponse,
    StatusResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models specific to daemon mode
# ---------------------------------------------------------------------------


class DaemonHealthResponse(BaseModel):
    """Daemon health check response with mode info.

    @param ok: Whether the daemon has at least one ready repo.
    @param repos: List of registered repo names.
    @param uptime_s: Seconds since daemon started.
    @param mode: 'full' if embedder loaded, 'fts_only' otherwise.
    """

    ok: bool
    repos: list[str]
    uptime_s: float
    mode: str


class AddRepoRequest(BaseModel):
    """Request to register a new repo.

    @param path: Absolute path to the repository root.
    @param name: Optional display name (defaults to dir basename).
    """

    path: str
    name: str | None = None


class RepoStateResponse(BaseModel):
    """Info about a registered repo including lifecycle state.

    @param name: Short display name.
    @param path: Absolute path.
    @param state: Lifecycle state (queued/indexing/ready/error).
    @param error: Error message if state is error.
    """

    name: str
    path: str
    state: str
    error: str | None = None


class RepoDetailStatus(BaseModel):
    """Detailed status for a single repo.

    @param name: Short display name.
    @param path: Absolute path.
    @param state: Lifecycle state.
    @param error: Error message if state is error.
    @param file_count: Total indexed files.
    @param chunk_count: Total chunks.
    @param vector_count: Chunks with vectors.
    @param indexed_at: Last index timestamp.
    """

    name: str
    path: str
    state: str
    error: str | None = None
    file_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    indexed_at: str = ""


class DaemonReposResponse(BaseModel):
    """List of all daemon-managed repos.

    @param repos: Repo details with state.
    """

    repos: list[RepoStateResponse]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


class RefreshRequest(BaseModel):
    """Optional body for targeted refresh.

    @param files: Specific files to re-index (repo-relative paths).
    """

    files: list[str] | None = None


def create_daemon_app(
    config: DaemonConfig,
    embedder: Any | None = _SENTINEL,
    *,
    refresh_min_interval: float = 10.0,
) -> FastAPI:
    """Create a daemon-mode FastAPI app with dynamic repo registry.

    Repos are initialized from config on startup. New repos can be
    added/removed via HTTP API. Config is persisted atomically.

    @param config: Parsed daemon configuration.
    @param embedder: Embedder instance (omit for auto-create, None for FTS-only).
    @returns: Configured FastAPI application.
    """
    state: dict[str, Any] = {
        "manager": None,
        "config": config,
        "embedder": None,
        "started_at": 0.0,
        "bg_threads": [],  # Tracked background index threads.
        "bg_threads_lock": threading.Lock(),
    }

    # Rate limiter for /refresh.
    _REFRESH_MIN_INTERVAL = refresh_min_interval
    _refresh_lock = threading.Lock()
    _last_refresh: dict[str, float] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Initialize repo manager and open indexes on startup."""
        from source_recall import Index

        t0 = time.monotonic()
        state["started_at"] = t0

        # Resolve embedder.
        if embedder is _SENTINEL:
            try:
                from source_recall.embedder import CodeRankEmbedder

                state["embedder"] = CodeRankEmbedder()
            except Exception:
                logger.warning(
                    "Could not load embedder — running in FTS-only mode",
                    exc_info=True,
                )
                state["embedder"] = None
        else:
            state["embedder"] = embedder

        # Build repo manager from config.
        manager = RepoManager.from_config(config)
        state["manager"] = manager

        # Open indexes for all registered repos.
        for name, slot in list(manager.slots.items()):
            try:
                idx = Index(slot.path, embedder=state["embedder"])
                idx.status()  # Opens DB, verifies index exists.
                slot.set_ready(idx)
                logger.info("Loaded repo %s (%s)", name, slot.path)
            except Exception:
                logger.warning(
                    "Failed to load index for %s — marked as error",
                    name,
                    exc_info=True,
                )
                slot.set_error("Failed to load index")

        elapsed = time.monotonic() - t0
        names = list(manager.slots.keys())
        logger.info("Daemon ready (%d repos in %.1fs): %s", len(names), elapsed, names)

        # Start periodic refresh thread.
        _stop_periodic = threading.Event()

        def _periodic_refresh_loop() -> None:
            """Background thread: refresh each ready repo periodically."""
            interval = config.refresh_interval_s
            while not _stop_periodic.wait(timeout=interval):
                for slot_name, slot in list(manager.slots.items()):
                    if slot.state != SlotState.READY or slot.index is None:
                        continue
                    # Skip if lock is held (another refresh in progress).
                    acquired = slot.lock.acquire(blocking=False)
                    if not acquired:
                        logger.debug(
                            "Periodic refresh skipped for %s (lock held)",
                            slot_name,
                        )
                        continue
                    try:
                        slot.index.refresh()
                        logger.debug("Periodic refresh completed for %s", slot_name)
                    except Exception:
                        logger.warning(
                            "Periodic refresh failed for %s",
                            slot_name,
                            exc_info=True,
                        )
                    finally:
                        slot.lock.release()

        refresh_thread = threading.Thread(
            target=_periodic_refresh_loop, daemon=True, name="sr-periodic-refresh"
        )
        refresh_thread.start()

        yield

        # Shutdown: stop periodic refresh, wait for background builds,
        # then close all indexes.
        _stop_periodic.set()
        refresh_thread.join(timeout=5)

        # Join any in-flight background index threads.
        with state["bg_threads_lock"]:
            threads = list(state["bg_threads"])
        for t in threads:
            t.join(timeout=config.shutdown_timeout_s)

        manager.close_all()

    app = FastAPI(
        title="source-recall daemon",
        description="Multi-repo code search daemon.",
        lifespan=lifespan,
    )

    # CORS — localhost only.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_manager() -> RepoManager:
        """Get the RepoManager instance.

        @returns: RepoManager.
        """
        return state["manager"]

    def _resolve_index(repo: str | None) -> Any:
        """Resolve a repo name to its Index.

        @param repo: Repo name (required if multiple repos).
        @returns: Index instance.
        @raises HTTPException: If repo not found or not ready.
        """
        manager = _get_manager()
        ready = {n: s for n, s in manager.slots.items() if s.state == SlotState.READY}

        if len(ready) == 1 and repo is None:
            slot = next(iter(ready.values()))
            return slot.index

        if repo is None and len(ready) == 0:
            raise HTTPException(
                status_code=503,
                detail="No repos ready",
            )

        if repo is None:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple repos loaded. Specify 'repo': one of {list(ready.keys())}",
            )

        if repo not in manager.slots:
            raise HTTPException(
                status_code=404,
                detail=f"Repo '{repo}' not found. Available: {list(manager.slots.keys())}",
            )

        slot = manager.slots[repo]
        if slot.state != SlotState.READY:
            raise HTTPException(
                status_code=503,
                detail=f"Repo '{repo}' is in state '{slot.state}' (not ready)",
            )

        return slot.index

    def _persist_config() -> None:
        """Atomically write current config to repos.toml.

        Uses tmp file + rename for crash safety.
        """
        cfg = state["config"]
        if cfg.config_path is None:
            return

        # Rebuild repos list from current manager state (under lock to
        # prevent concurrent add/remove from mutating during iteration).
        manager = _get_manager()
        with manager._lock:
            cfg.repos = [
                DaemonConfig.RepoEntry(path=slot.path, name=slot.name)
                for slot in manager.slots.values()
            ]

        toml_str = cfg.to_toml()
        config_path = cfg.config_path

        # Atomic write: write to tmp in same dir, then rename.
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=config_path.parent, suffix=".tmp", prefix=".repos-"
        )
        closed = False
        try:
            os.write(fd, toml_str.encode())
            os.fsync(fd)
            os.close(fd)
            closed = True
            os.rename(tmp_path, config_path)
        except Exception:
            if not closed:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=DaemonHealthResponse)
    def health() -> DaemonHealthResponse:
        """Liveness check with mode indicator.

        @returns: Health status with repo list, uptime, and mode.
        """
        manager = _get_manager()
        ready = [n for n, s in manager.slots.items() if s.state == SlotState.READY]
        mode = "full" if state["embedder"] is not None else "fts_only"
        return DaemonHealthResponse(
            ok=len(ready) > 0,
            repos=list(manager.slots.keys()),
            uptime_s=round(time.monotonic() - state["started_at"], 1),
            mode=mode,
        )

    @app.get("/repos", response_model=DaemonReposResponse)
    def list_repos() -> DaemonReposResponse:
        """List all registered repos with state.

        @returns: Repo details.
        """
        manager = _get_manager()
        items = []
        for slot in manager.slots.values():
            items.append(
                RepoStateResponse(
                    name=slot.name,
                    path=str(slot.path),
                    state=slot.state.value,
                    error=slot.error,
                )
            )
        return DaemonReposResponse(repos=items)

    @app.get("/repos/{name}/status", response_model=RepoDetailStatus)
    def repo_status(name: str) -> RepoDetailStatus:
        """Detailed status for a single repo.

        @param name: Repo name.
        @returns: State, file/chunk counts, timestamps.
        @raises HTTPException: 404 if repo not found.
        """
        manager = _get_manager()
        try:
            slot = manager.get(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        result = RepoDetailStatus(
            name=slot.name,
            path=str(slot.path),
            state=slot.state.value,
            error=slot.error,
        )

        # Enrich with index stats if available.
        if slot.index is not None and slot.state == SlotState.READY:
            try:
                s = slot.index.status()
                result.file_count = s.file_count
                result.chunk_count = s.chunk_count
                result.vector_count = s.vector_count
                result.indexed_at = s.indexed_at
            except Exception:
                pass  # Stats unavailable — return base info.

        return result

    @app.get("/repos/{name}/progress")
    def repo_progress(name: str) -> Any:
        """SSE stream of indexing progress for a repo.

        Streams events while repo is indexing. Emits a final
        'complete' event when done. Returns immediately with
        'complete' if repo is already ready.

        @param name: Repo name.
        @returns: StreamingResponse with text/event-stream.
        @raises HTTPException: 404 if repo not found.
        """
        from starlette.responses import StreamingResponse

        manager = _get_manager()
        try:
            slot = manager.get(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        import asyncio
        import json as _json

        async def event_generator() -> AsyncIterator[str]:
            """Yield SSE events for indexing progress.

            @returns: SSE-formatted event strings.
            """
            # If already ready or error, emit single event and close.
            if slot.state in (SlotState.READY, SlotState.ERROR):
                event_data = _json.dumps(
                    {
                        "state": slot.state.value,
                        "name": slot.name,
                    }
                )
                event_type = "complete" if slot.state == SlotState.READY else "error"
                yield f"event: {event_type}\ndata: {event_data}\n\n"
                return

            # Poll state while indexing.
            while slot.state == SlotState.INDEXING or slot.state == SlotState.QUEUED:
                event_data = _json.dumps(
                    {
                        "state": slot.state.value,
                        "name": slot.name,
                        "current": slot.progress_current,
                        "total": slot.progress_total,
                        "file": slot.progress_file,
                    }
                )
                yield f"event: progress\ndata: {event_data}\n\n"
                await asyncio.sleep(0.5)

            # Final event.
            event_data = _json.dumps(
                {
                    "state": slot.state.value,
                    "name": slot.name,
                }
            )
            event_type = "complete" if slot.state == SlotState.READY else "error"
            yield f"event: {event_type}\ndata: {event_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/repos", response_model=RepoStateResponse, status_code=201)
    def add_repo(req: AddRepoRequest) -> RepoStateResponse:
        """Register a new repo.

        @param req: Repo path and optional name.
        @returns: Created repo info.
        @raises HTTPException: 400 if path invalid, 409 if duplicate.
        """
        path = Path(req.path).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist or is not a directory: {path}",
            )

        manager = _get_manager()
        name = req.name or path.name

        try:
            slot = manager.add(path, name=name)
        except ValueError as e:
            if "already registered" in str(e):
                raise HTTPException(status_code=409, detail=str(e)) from e
            raise HTTPException(status_code=400, detail=str(e)) from e

        _persist_config()

        # Trigger background indexing (D-004).
        def _background_index(s: Any) -> None:
            """Build index for a newly added repo in background."""
            from source_recall import Index

            s.set_indexing()
            try:
                idx = Index(
                    s.path,
                    embedder=state["embedder"],
                    on_progress=s.update_progress,
                )
                idx.build()
                # Re-open for querying.
                idx = Index(s.path, embedder=state["embedder"])
                idx.status()
                s.set_ready(idx)
                logger.info("Background index complete for %s", s.name)
            except Exception:
                logger.warning(
                    "Background index failed for %s",
                    s.name,
                    exc_info=True,
                )
                s.set_error("Index build failed")

        t = threading.Thread(
            target=_background_index,
            args=(slot,),
            name=f"sr-index-{slot.name}",
        )
        with state["bg_threads_lock"]:
            state["bg_threads"].append(t)
        t.start()

        return RepoStateResponse(
            name=slot.name,
            path=str(slot.path),
            state=slot.state.value,
            error=slot.error,
        )

    @app.delete("/repos/{name}")
    def remove_repo(name: str) -> dict[str, str]:
        """Unregister a repo.

        @param name: Repo name to remove.
        @returns: Confirmation message.
        @raises HTTPException: 404 if not found.
        """
        manager = _get_manager()
        try:
            manager.remove(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        _persist_config()

        return {"status": "removed", "name": name}

    def _query_single_repo(
        name: str, idx: Any, question: str, top_k: int | None, branch: str | None
    ) -> list[QueryResultResponse]:
        """Query a single repo and tag results with repo name.

        @param name: Repo name.
        @param idx: Index instance.
        @param question: Query string.
        @param top_k: Max results.
        @param branch: Branch filter.
        @returns: List of tagged QueryResultResponse.
        """
        results = idx.query(question, top_k=top_k, branch=branch)
        return [
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
                repo_name=name,
            )
            for r in results
        ]

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        """Search indexes.

        When ``repo`` is specified, searches that repo only.
        When omitted, fans out to all ready repos and merges
        results by normalized score.

        @param req: Query request.
        @returns: Ranked results.
        """
        manager = _get_manager()
        ready = {
            n: s
            for n, s in manager.slots.items()
            if s.state == SlotState.READY and s.index is not None
        }

        t0 = time.monotonic()

        if req.repo is not None:
            # Single-repo query.
            if req.repo not in manager.slots:
                raise HTTPException(
                    status_code=404,
                    detail=f"Repo '{req.repo}' not found. Available: {list(manager.slots.keys())}",
                )
            slot = manager.slots[req.repo]
            if slot.state != SlotState.READY or slot.index is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Repo '{req.repo}' not ready (state: {slot.state})",
                )
            all_results = _query_single_repo(
                req.repo, slot.index, req.question, req.top_k, req.branch
            )
        elif len(ready) == 1:
            # Single ready repo — no fan-out needed.
            name, slot = next(iter(ready.items()))
            all_results = _query_single_repo(
                name, slot.index, req.question, req.top_k, req.branch
            )
        elif len(ready) == 0:
            raise HTTPException(status_code=503, detail="No repos ready")
        else:
            # Fan out to all ready repos, normalize scores per-repo.
            all_results = []
            for name, slot in ready.items():
                repo_results = _query_single_repo(
                    name, slot.index, req.question, req.top_k, req.branch
                )
                if repo_results:
                    max_score = max(r.score for r in repo_results)
                    if max_score > 0:
                        for r in repo_results:
                            r.score = r.score / max_score
                all_results.extend(repo_results)

            # Sort by normalized score descending.
            all_results.sort(key=lambda r: r.score, reverse=True)

            # Trim to top_k.
            if req.top_k:
                all_results = all_results[: req.top_k]

        elapsed_ms = (time.monotonic() - t0) * 1000

        return QueryResponse(
            results=all_results,
            query_ms=round(elapsed_ms, 1),
        )

    @app.post("/refresh", response_model=RefreshResponse)
    def refresh(
        repo: str | None = None,
        body: RefreshRequest | None = None,
    ) -> RefreshResponse:
        """Incrementally refresh an index.

        Serializes via per-repo lock (D-001). Concurrent calls wait
        instead of being rejected. Optional ``files`` body targets
        specific files for re-indexing.

        @param repo: Repo name (optional if single repo).
        @param body: Optional targeted refresh with file list.
        @returns: Number of files updated.
        @raises HTTPException: 429 if rate limited.
        """
        # Resolve the slot (not just the index) for per-repo locking.
        manager = _get_manager()
        ready = {n: s for n, s in manager.slots.items() if s.state == SlotState.READY}

        if len(ready) == 1 and repo is None:
            slot = next(iter(ready.values()))
        elif repo is None:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple repos loaded. Specify 'repo': one of {list(ready.keys())}",
            )
        elif repo not in manager.slots:
            raise HTTPException(
                status_code=404,
                detail=f"Repo '{repo}' not found.",
            )
        else:
            slot = manager.slots[repo]
            if slot.state != SlotState.READY:
                raise HTTPException(
                    status_code=503,
                    detail=f"Repo '{repo}' not ready (state: {slot.state})",
                )

        idx = slot.index

        # Rate limit per repo (check before acquiring per-repo lock).
        key = slot.name
        now = time.monotonic()
        if _REFRESH_MIN_INTERVAL > 0:
            with _refresh_lock:
                last = _last_refresh.get(key, 0.0)
                if now - last < _REFRESH_MIN_INTERVAL:
                    remaining = round(_REFRESH_MIN_INTERVAL - (now - last), 1)
                    raise HTTPException(
                        status_code=429,
                        detail=f"Refresh rate limited. Retry in {remaining}s.",
                    )
                _last_refresh[key] = now

        # Serialize per-repo — concurrent calls wait.
        target_files = body.files if body is not None else None
        with slot.lock:
            t0 = time.monotonic()
            count = idx.refresh(files=target_files)
            elapsed_ms = (time.monotonic() - t0) * 1000

        return RefreshResponse(
            files_updated=count,
            refresh_ms=round(elapsed_ms, 1),
        )

    @app.get("/status", response_model=StatusResponse)
    def status(repo: str | None = None) -> StatusResponse:
        """Get index status — delegates to the repo's Index.

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

    return app
