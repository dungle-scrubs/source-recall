"""Tests for removing a repo while a background index is in flight (H-2).

Before the fix, ``RepoManager.remove`` popped the slot and called
``slot.close()`` (a no-op while ``index is None`` during a build).  The
background thread kept running and, on completion, called
``slot.set_ready(idx)`` on the now-orphaned slot — opening a database
connection that was never closed.

The fix: the slot carries a ``cancel`` Event and an ``index_thread``
reference.  ``remove`` sets cancel and joins the thread; ``set_ready``
checks cancel and discards the Index if cancelled.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from source_recall.repo_manager import RepoManager, RepoSlot, SlotState


class _FakeIndex:
    """Minimal Index double that records close() calls."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestRemoveRepoCancelsBackgroundIndex:
    def test_set_ready_after_cancel_closes_index(self, tmp_path: Path) -> None:
        """set_ready on a cancelled slot closes the Index, never stores it."""
        slot = RepoSlot(name="r", path=tmp_path)
        slot.cancel.set()

        idx = _FakeIndex()
        slot.set_ready(idx)  # type: ignore[arg-type]

        assert idx.closed, "Cancelled set_ready must close the passed Index"
        assert slot.index is None, "Cancelled slot must not store the Index"
        assert slot.state != SlotState.READY

    def test_set_ready_without_cancel_stores_index(self, tmp_path: Path) -> None:
        """Normal (non-cancelled) set_ready stores the Index as before."""
        slot = RepoSlot(name="r", path=tmp_path)

        idx = _FakeIndex()
        slot.set_ready(idx)  # type: ignore[arg-type]

        assert not idx.closed
        assert slot.index is idx
        assert slot.state == SlotState.READY

    def test_remove_sets_cancel_and_joins_thread(self, tmp_path: Path) -> None:
        """remove() signals cancel and joins the indexing thread."""
        mgr = RepoManager()
        slot = mgr.add(tmp_path, name="r")

        finished = threading.Event()

        def build() -> None:
            slot.set_indexing()
            # Simulate a build that respects cancel by finishing promptly.
            finished.wait(timeout=5)
            slot.set_ready(_FakeIndex())  # type: ignore[arg-type]

        t = threading.Thread(target=build, name="sr-index-r")
        slot.index_thread = t
        t.start()

        # Let the build thread reach its wait point.
        time.sleep(0.1)

        # Allow the build to proceed, then remove (which cancels + joins).
        finished.set()
        mgr.remove("r", join_timeout_s=5)

        t.join(timeout=5)
        assert not t.is_alive(), "remove must join the indexing thread"
        assert "r" not in mgr.slots

    def test_remove_with_blocking_build_unblocks_via_cancel(
        self, tmp_path: Path
    ) -> None:
        """A build blocked past remove does not leak a READY orphaned slot."""
        mgr = RepoManager()
        slot = mgr.add(tmp_path, name="r")

        proceed = threading.Event()
        captured: list[RepoSlot] = [slot]

        def build() -> None:
            slot.set_indexing()
            proceed.wait(timeout=5)
            slot.set_ready(_FakeIndex())  # type: ignore[arg-type]

        t = threading.Thread(target=build, name="sr-index-r")
        slot.index_thread = t
        t.start()
        time.sleep(0.1)

        # Remove while build is still blocked.  The join timeout lets
        # remove return even if the build hasn't finished yet.
        mgr.remove("r", join_timeout_s=0.2)
        assert "r" not in mgr.slots

        # Now let the build finish; set_ready must discard due to cancel.
        proceed.set()
        t.join(timeout=5)

        orphan = captured[0]
        assert orphan.state != SlotState.READY, (
            "Orphaned slot must not reach READY after removal"
        )
        assert orphan.index is None, "Orphaned slot must not hold a leaked Index"

    def test_remove_unknown_raises_keyerror(self) -> None:
        mgr = RepoManager()
        try:
            mgr.remove("nope")
        except KeyError:
            return
        raise AssertionError("remove of unknown name must raise KeyError")
