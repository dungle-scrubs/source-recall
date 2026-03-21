"""Repo registry: manages per-repo state, Index instances, and locks."""

from __future__ import annotations

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
        # Progress tracking for SSE.
        self.progress_current: int = 0
        self.progress_total: int = 0
        self.progress_file: str = ""

    def set_indexing(self) -> None:
        """Atomically transition to INDEXING state."""
        with self.lock:
            self.state = SlotState.INDEXING
            self.progress_current = 0
            self.progress_total = 0
            self.progress_file = ""

    def update_progress(self, file_path: str, current: int, total: int) -> None:
        """Update indexing progress (called from builder callback).

        @param file_path: Current file being indexed.
        @param current: Files processed so far.
        @param total: Total files to process.
        """
        self.progress_file = file_path
        self.progress_current = current
        self.progress_total = total

    def set_ready(self, index: Index) -> None:
        """Atomically transition to READY with an open Index.

        @param index: The loaded Index instance.
        """
        with self.lock:
            self.index = index
            self.state = SlotState.READY
            self.error = None

    def set_error(self, message: str) -> None:
        """Atomically transition to ERROR state.

        @param message: Human-readable error description.
        """
        with self.lock:
            self.state = SlotState.ERROR
            self.error = message

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

    def remove(self, name: str) -> None:
        """Unregister and close a repo.

        @param name: Repo name to remove.
        @raises KeyError: If name not found.
        """
        with self._lock:
            if name not in self.slots:
                msg = f"Repo '{name}' not found"
                raise KeyError(msg)

            slot = self.slots.pop(name)

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

    def close_all(self) -> None:
        """Close all Index instances. Called during shutdown."""
        with self._lock:
            for slot in self.slots.values():
                slot.close()
