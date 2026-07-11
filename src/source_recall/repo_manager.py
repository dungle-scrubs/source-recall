"""Repo registry: manages per-repo state, Index instances, and locks."""

from __future__ import annotations

import copy
import enum
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from source_recall import Index
    from source_recall.daemon_config import DaemonConfig

logger = logging.getLogger(__name__)


class SlotState(enum.StrEnum):
    """Lifecycle state for a managed repository.

    queued → indexing → ready
                     → error
    """

    QUEUED = "queued"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class RepoSlot:
    """Per-repo state: owns the Index instance and a serialization lock.

    @param name: Short display name.
    @param path: Absolute, resolved path to repo root.
    """

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = Path(path).resolve()
        self.state = SlotState.QUEUED
        self.error: str | None = None
        self.index: Index | None = None
        self.lock = threading.Lock()
        # Cancel flag for in-flight background indexing (H-2).  When set,
        # set_ready() discards the built Index instead of storing it, so a
        # removed repo never leaks an open connection.
        self.cancel = threading.Event()
        # The background indexing thread building this slot, if any.
        # Registered by the daemon so remove() can join it.
        self.index_thread: threading.Thread | None = None
        # Progress tracking for SSE.
        self.progress_current: int = 0
        self.progress_total: int = 0
        self.progress_file: str = ""
        self.progress_phase: str = "queued"
        self.progress_detail: dict[str, object] = {}

    def set_indexing(self) -> None:
        """Atomically transition to INDEXING state."""
        with self.lock:
            self.state = SlotState.INDEXING
            self.progress_current = 0
            self.progress_total = 0
            self.progress_file = ""
            self.progress_phase = "scanning"
            self.progress_detail = {}

    def update_progress(self, file_path: str, current: int, total: int) -> None:
        """Update indexing progress (called from builder callback).

        Writes total before current so readers never see current > total.
        Uses the slot lock for atomicity across all three fields (M3).

        @param file_path: Current file being indexed.
        @param current: Files processed so far.
        @param total: Total files to process.
        """
        with self.lock:
            self.progress_total = total
            self.progress_current = current
            self.progress_file = file_path

    def update_progress_phase(self, phase: str) -> None:
        """Update coarse indexing phase for progress observers.

        @param phase: Build lifecycle phase such as scanning or embedding.
        """
        with self.lock:
            self.progress_phase = phase

    def update_progress_detail(self, detail: dict[str, object]) -> None:
        """Update structured indexing detail for progress observers.

        Deep-copies the detail so later mutation of nested values in the
        caller's dict does not leak into the stored snapshot (L-2 fix).

        @param detail: Machine-readable timing/span attributes.
        """
        with self.lock:
            self.progress_detail = copy.deepcopy(detail)

    def set_ready(self, index: Index) -> None:
        """Atomically transition to READY with an open Index.

        If the slot has been cancelled (removed while indexing), the
        passed Index is closed immediately and the slot stays out of
        READY — preventing a leaked, never-closed connection (H-2).

        @param index: The loaded Index instance.
        """
        with self.lock:
            if self.cancel.is_set():
                # Repo was removed mid-build: discard the connection.
                try:
                    index.close()
                except Exception:
                    logger.warning(
                        "Error closing cancelled index for %s",
                        self.name,
                        exc_info=True,
                    )
                return
            self.index = index
            self.state = SlotState.READY
            self.error = None
            self.progress_phase = "ready"
            self.progress_detail = {}

    def set_error(self, message: str) -> None:
        """Atomically transition to ERROR state.

        @param message: Human-readable error description.
        """
        with self.lock:
            self.state = SlotState.ERROR
            self.error = message
            self.progress_phase = "error"
            self.progress_detail = {}

    def close(self) -> None:
        """Close the Index if open. Safe to call multiple times."""
        if self.index is not None:
            try:
                self.index.close()
            except Exception:
                logger.warning("Error closing index for %s", self.name, exc_info=True)
            self.index = None


class RepoManager:
    """Registry of RepoSlots — add, remove, list managed repos.

    Thread-safe for slot access via the internal lock.
    """

    def __init__(self) -> None:
        self.slots: dict[str, RepoSlot] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: DaemonConfig) -> RepoManager:
        """Create a RepoManager pre-populated from a DaemonConfig.

        Skips repos whose paths no longer exist (logs a warning).

        @param config: Parsed daemon configuration.
        @returns: Populated RepoManager.
        """
        mgr = cls()
        for entry in config.repos:
            if not entry.path.is_dir():
                logger.warning(
                    "Skipping repo %s: path %s does not exist",
                    entry.name,
                    entry.path,
                )
                continue
            try:
                mgr.add(entry.path, name=entry.name)
            except ValueError:
                logger.warning("Skipping duplicate repo %s", entry.name, exc_info=True)
        return mgr

    def add(self, path: Path, *, name: str | None = None) -> RepoSlot:
        """Register a new repo.

        @param path: Path to the repository root.
        @param name: Optional display name (defaults to dir basename).
        @returns: The created RepoSlot.
        @raises ValueError: If path doesn't exist or name is duplicate.
        """
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            msg = f"Path does not exist or is not a directory: {resolved}"
            raise ValueError(msg)

        slot_name = name or resolved.name

        with self._lock:
            if slot_name in self.slots:
                msg = f"Repo '{slot_name}' already registered"
                raise ValueError(msg)

            slot = RepoSlot(name=slot_name, path=resolved)
            self.slots[slot_name] = slot
            return slot

    def remove(self, name: str, *, join_timeout_s: float = 30.0) -> None:
        """Unregister and close a repo.

        If a background indexing thread is building this slot, it is
        signalled to cancel and joined (bounded by ``join_timeout_s``)
        so the orphaned thread cannot call ``set_ready`` on a removed
        slot and leak an open connection (H-2).

        @param name: Repo name to remove.
        @param join_timeout_s: Max seconds to wait for an in-flight build.
        @raises KeyError: If name not found.
        """
        with self._lock:
            if name not in self.slots:
                msg = f"Repo '{name}' not found"
                raise KeyError(msg)

            slot = self.slots.pop(name)

        # Signal any in-flight background build to discard its result.
        slot.cancel.set()
        # Join the indexing thread if one is registered.
        thread = slot.index_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout_s)

        # Close outside lock to avoid holding it during I/O.
        slot.close()

    def get(self, name: str) -> RepoSlot:
        """Look up a slot by name.

        @param name: Repo name.
        @returns: The RepoSlot.
        @raises KeyError: If not found.
        """
        with self._lock:
            if name not in self.slots:
                msg = f"Repo '{name}' not found"
                raise KeyError(msg)
            return self.slots[name]

    def list_repos(self) -> list[dict[str, Any]]:
        """Return summary info for all registered repos.

        @returns: List of dicts with name, path, state, error.
        """
        with self._lock:
            return [
                {
                    "name": slot.name,
                    "path": str(slot.path),
                    "state": slot.state.value,
                    "error": slot.error,
                }
                for slot in self.slots.values()
            ]

    def snapshot_repos(self) -> list[tuple[str, Path]]:
        """Snapshot (name, path) for all registered repos under the lock.

        Used by config persistence so the caller can serialize outside
        the lock without reaching into ``_lock`` directly (M-2).

        @returns: List of (name, path) tuples.
        """
        with self._lock:
            return [(slot.name, slot.path) for slot in self.slots.values()]

    def close_all(self) -> None:
        """Close all Index instances. Called during shutdown.

        Each slot is closed while holding its per-slot lock — the same
        lock the periodic-refresh loop takes around ``index.refresh()`` —
        so a refresh that is still draining after the shutdown join
        finishes before the connection is torn down. Without this,
        close_all could close the SQLite connection mid-refresh.
        """
        with self._lock:
            for slot in self.slots.values():
                with slot.lock:
                    slot.close()
