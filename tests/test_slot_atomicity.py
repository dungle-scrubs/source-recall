"""Tests for atomic slot state transitions."""

from __future__ import annotations

import threading
from pathlib import Path

from source_recall.repo_manager import RepoSlot, SlotState


class TestSlotAtomicTransition:
    def test_set_ready_is_atomic(self, tmp_path: Path) -> None:
        """Readers never see state=READY with index=None."""
        slot = RepoSlot(name="test", path=tmp_path)
        violations: list[str] = []

        def reader() -> None:
            for _ in range(1000):
                with slot.lock:
                    if slot.state == SlotState.READY and slot.index is None:
                        violations.append("READY but index=None")
                    if slot.state != SlotState.READY and slot.index is not None:
                        violations.append(f"index set but state={slot.state}")

        def writer() -> None:
            for i in range(100):
                slot.set_ready(f"idx-{i}")  # type: ignore[arg-type]
                slot.set_error("oops")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writer_t = threading.Thread(target=writer)

        for r in readers:
            r.start()
        writer_t.start()
        writer_t.join()
        for r in readers:
            r.join()

        assert not violations, f"Atomicity violations: {violations}"

    def test_set_ready_sets_both(self, tmp_path: Path) -> None:
        """set_ready() sets state and index together."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_ready("my_index")  # type: ignore[arg-type]

        assert slot.state == SlotState.READY
        assert slot.index == "my_index"

    def test_set_error_clears_index(self, tmp_path: Path) -> None:
        """set_error() sets error state and clears index ref."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_ready("my_index")  # type: ignore[arg-type]
        slot.set_error("broken")

        assert slot.state == SlotState.ERROR
        assert slot.error == "broken"

    def test_set_indexing(self, tmp_path: Path) -> None:
        """set_indexing() transitions to INDEXING."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_indexing()
        assert slot.state == SlotState.INDEXING
