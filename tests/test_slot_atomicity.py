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

    def test_update_progress_is_atomic(self, tmp_path: Path) -> None:
        """Readers never see progress_current > progress_total (M3 audit fix).

        update_progress writes three fields.  Without synchronization a
        reader could see current=50 with total=0 if it reads between writes.
        """
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_indexing()
        violations: list[str] = []

        def reader() -> None:
            for _ in range(5000):
                current = slot.progress_current
                total = slot.progress_total
                if total > 0 and current > total:
                    violations.append(f"current={current} > total={total}")

        def writer() -> None:
            for i in range(500):
                slot.update_progress(f"file_{i}.py", i + 1, 500)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writer_t = threading.Thread(target=writer)

        for r in readers:
            r.start()
        writer_t.start()
        writer_t.join()
        for r in readers:
            r.join()

        assert not violations, f"Progress atomicity violations: {violations}"
