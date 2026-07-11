"""Daemon-mode server: multi-repo registry with dynamic add/remove."""

from __future__ import annotations

import logging
import os
import secrets
import signal
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from source_recall.daemon_config import DaemonConfig, load_or_create_token
from source_recall.models import _SENTINEL
from source_recall.repo_manager import RepoManager, SlotState
from source_recall.server import (
    QueryRequest,
    QueryResponse,
    QueryResultResponse,
    RefreshResponse,
    StatusResponse,
    _warm_embedder,
    _warm_reranker,
)

logger = logging.getLogger(__name__)

# Module-level stop event — set on SIGTERM/SIGINT so the periodic
# refresh loop and any in-flight background index threads can drain
# before uvicorn exits, even if uvicorn's own signal handler races
# ahead of us (H-2 audit fix).
_stop_event = threading.Event()


def _install_shutdown_handlers() -> Any:
    """Register SIGTERM/SIGINT handlers that set ``_stop_event``.

    Idempotent — calling more than once replaces the previous handler
    so the latest stop event is honoured.

    @returns: The installed handler (callable).
    """

    def _handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %d — initiating graceful shutdown", signum)
        _stop_event.set()

    prior_int = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGTERM, _handler)
        # Skip SIGINT modification in pytest (default handler is used
        # for Ctrl-C in test runners).
        if prior_int != signal.default_int_handler:
            signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        # signal() can fail on non-main threads or restricted contexts.
        # Lifespan shutdown will still fire on uvicorn's lifespan exit.
        logger.debug("Could not install signal handlers", exc_info=True)
    return _handler


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
    # Local auth token — generated (or loaded) up front so it is available
    # to the auth dependency on the very first request, independent of the
    # lifespan. Stored 0600 at the canonical default config dir (NOT beside a
    # custom --config), so a custom-config daemon and the CLI client — which
    # always reads the default location — agree on the same secret.
    token = load_or_create_token()

    state: dict[str, Any] = {
        "manager": None,
        "config": config,
        "embedder": None,
        "started_at": 0.0,
        "bg_threads": [],  # Tracked background index threads.
        "bg_threads_lock": threading.Lock(),
        "token": token,
    }

    def _require_token(
        authorization: str | None = Header(default=None),
        x_sr_token: str | None = Header(default=None),
    ) -> None:
        """Reject requests without the local auth token.

        Accepts the token via ``Authorization: Bearer <token>`` or the
        ``X-SR-Token`` header. Compared in constant time. This is the sole
        gate that stops a browser (DNS-rebinding/CSRF) or another local
        process from driving the API and exfiltrating indexed source.

        @raises HTTPException: 401 if the token is missing or wrong.
        """
        provided = x_sr_token
        if provided is None and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                provided = value.strip()
        expected = state["token"]
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid auth token")

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

        # Clear any leftover stop signal from a prior daemon instance in
        # the same process (e.g. a previous test's lifespan shutdown set
        # the module-level event). Without this, the periodic refresh
        # loop sees the event already set and exits before the first tick.
        _stop_event.clear()

        # Install signal handlers so SIGTERM/SIGINT outside uvicorn's
        # own handling still triggers a graceful drain (H-2 audit fix).
        _install_shutdown_handlers()

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
        # Note: _stop_event is module-level so a SIGTERM received
        # before this scope enters still triggers a clean drain (H-2
        # audit fix).  Local alias kept for readability.
        stop_event = _stop_event

        def _periodic_refresh_loop() -> None:
            """Background thread: refresh each ready repo periodically."""
            interval = config.refresh_interval_s
            while not stop_event.wait(timeout=interval):
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

        # Warm the shared embedder (and any per-repo reranker) on a
        # background daemon thread so the first query does not pay the
        # model-load cold spike. Does not block startup; failures are
        # logged and ignored inside the _warm_* helpers.
        def _warmup() -> None:
            _warm_embedder(state["embedder"])
            seen: set[int] = set()
            for slot in list(manager.slots.values()):
                idx = slot.index
                if idx is None:
                    continue
                try:
                    reranker = idx._get_reranker()
                except Exception:
                    reranker = None
                    logger.warning("Reranker warmup failed", exc_info=True)
                if reranker is not None and id(reranker) not in seen:
                    seen.add(id(reranker))
                    _warm_reranker(reranker)

        warmup_thread = threading.Thread(target=_warmup, daemon=True, name="sr-warmup")
        warmup_thread.start()
        _app.state.warmup_thread = warmup_thread

        yield

        # Shutdown: stop periodic refresh, wait for background builds,
        # then close all indexes.  Setting the module-level event
        # ensures the loop also responds to SIGTERM/SIGINT, not just
        # the lifespan exit (H-2 audit fix).
        _stop_event.set()
        # Join with the configured shutdown budget (same as the build
        # threads below), not a hardcoded 5s. close_all() then acquires
        # each slot lock so it can never close an index underneath a
        # still-running periodic refresh.
        refresh_thread.join(timeout=config.shutdown_timeout_s)

        # Cancel signal handlers on lifespan exit — their job is done.
        # Re-raise any signal sent during shutdown so the OS sees the
        # default disposition (exit code 130 for SIGINT, 143 for SIGTERM).
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

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
        # Every route requires the local auth token.
        dependencies=[Depends(_require_token)],
        # Disable FastAPI's built-in docs/schema routes: they sit OUTSIDE the
        # auth dependency and this is a machine API, so an unauthenticated
        # /docs, /redoc, or /openapi.json would leak the route surface.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Publish the token so in-process callers (e.g. the test client) can
    # authenticate without touching the filesystem.
    app.state.sr_token = token

    # Host-header validation — defeats DNS rebinding, where a page on an
    # attacker-controlled domain resolves to 127.0.0.1 and the browser
    # sends that domain as the Host. Only loopback names (plus whatever
    # host the operator explicitly bound) are accepted.
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    allowed_hosts = list(
        dict.fromkeys(["localhost", "127.0.0.1", "0.0.0.0", config.host])
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

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

    def _repo_path_allowed(path: Path) -> bool:
        """Containment policy for repos added over the network (POST /repos).

        Accepting *any* absolute path over HTTP turns the daemon into a
        file-exfiltration primitive (register ``/`` then query its
        contents). A path is accepted only when it is:

        1. already registered (idempotent re-add), or
        2. inside the user's home directory, or
        3. inside a directory that already contains a registered repo —
           the operator already exposed that tree by registering a repo
           there, so siblings are within the same trust boundary.

        Everything else (``/etc``, ``/var``, other users' homes, ...) is
        rejected with 403. This is an opt-in allowlist rather than a
        blanket accept. Repos loaded from the trusted on-disk config
        (``RepoManager.from_config``) bypass this check by design.

        @param path: Resolved candidate repo path.
        @returns: True if the path is within an allowed root.
        """
        manager = _get_manager()
        registered = list(manager.slots.values())
        if any(slot.path == path for slot in registered):
            return True
        roots = {Path.home().resolve()}
        roots.update(slot.path.parent for slot in registered)
        return any(path == root or root in path.parents for root in roots)

    def _persist_config() -> None:
        """Atomically write current config to repos.toml.

        Uses tmp file + rename for crash safety.
        """
        cfg = state["config"]
        if cfg.config_path is None:
            return

        # Snapshot the current repos under the manager lock, then
        # serialize outside the lock so we don't hold it during I/O.
        manager = _get_manager()
        cfg.repos = [
            DaemonConfig.RepoEntry(path=p, name=n) for n, p in manager.snapshot_repos()
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
                        "phase": slot.progress_phase,
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
                        "phase": slot.progress_phase,
                        "detail": slot.progress_detail,
                    }
                )
                yield f"event: progress\ndata: {event_data}\n\n"
                await asyncio.sleep(0.5)

            # Final event.
            event_data = _json.dumps(
                {
                    "state": slot.state.value,
                    "name": slot.name,
                    "phase": slot.progress_phase,
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

        Paths are constrained by ``_repo_path_allowed`` — arbitrary
        absolute paths are rejected with 403 so the network API cannot be
        used to index and exfiltrate files outside the operator's trust
        boundary.

        @param req: Repo path and optional name.
        @returns: Created repo info.
        @raises HTTPException: 400 if path invalid, 403 if path disallowed,
            409 if duplicate.
        """
        path = Path(req.path).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist or is not a directory: {path}",
            )
        if not _repo_path_allowed(path):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Path is outside the allowed roots (home directory or a "
                    f"directory already holding a registered repo): {path}"
                ),
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
                    on_phase=s.update_progress_phase,
                    on_progress_detail=s.update_progress_detail,
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
        slot.index_thread = t
        with state["bg_threads_lock"]:
            state["bg_threads"].append(t)
        t.start()

        # Wrap join so the thread is removed from bg_threads once it
        # finishes (whether via daemon shutdown or any other caller).
        # Without this the list grows unbounded over the daemon's
        # lifetime and shutdown joins every historical thread, paying
        # N×(join-timeout) even when no work is in flight.
        _orig_join = t.join

        def _join_then_cleanup(timeout: float | None = None) -> None:
            try:
                _orig_join(timeout=timeout)
            finally:
                with state["bg_threads_lock"]:
                    if t in state["bg_threads"]:
                        state["bg_threads"].remove(t)

        t.join = _join_then_cleanup  # type: ignore[method-assign]

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
        name: str,
        idx: Any,
        question: str,
        top_k: int | None,
        branch: str | None,
        query_vec: list[float] | None = None,
    ) -> list[QueryResultResponse]:
        """Query a single repo and tag results with repo name.

        @param name: Repo name.
        @param idx: Index instance.
        @param question: Query string.
        @param top_k: Max results.
        @param branch: Branch filter.
        @param query_vec: Precomputed query embedding shared across repos.
        @returns: List of tagged QueryResultResponse.
        """
        results = idx.query(question, top_k=top_k, branch=branch, query_vec=query_vec)
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

        # Embed the query once, at the daemon layer, and reuse the vector
        # across every repo. All ready repos are opened with the shared
        # ``state["embedder"]`` (see the lifespan startup loop), so a single
        # query vector is correct for all of them; a per-repo re-embed would
        # be pure duplicate work on the fan-out path. FTS-only mode
        # (embedder is None) and empty queries produce no vector.
        embedder = state["embedder"]
        query_vec: list[float] | None = None
        if embedder is not None and req.question.strip():
            try:
                query_vec = embedder.embed_query(req.question)
            except Exception:
                logger.warning("Query embedding failed", exc_info=True)

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
                req.repo, slot.index, req.question, req.top_k, req.branch, query_vec
            )
        elif len(ready) == 1:
            # Single ready repo — no fan-out needed.
            name, slot = next(iter(ready.items()))
            all_results = _query_single_repo(
                name, slot.index, req.question, req.top_k, req.branch, query_vec
            )
        elif len(ready) == 0:
            raise HTTPException(status_code=503, detail="No repos ready")
        else:
            # Fan out to all ready repos, normalize scores per-repo.
            all_results = []
            for name, slot in ready.items():
                repo_results = _query_single_repo(
                    name, slot.index, req.question, req.top_k, req.branch, query_vec
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
