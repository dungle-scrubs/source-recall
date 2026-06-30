"""Tests for RepoSlot.update_progress_detail deep copy (L-2).

Before the fix, ``update_progress_detail`` did a shallow ``dict(detail)``
copy.  If the builder mutated nested values in its detail dict between
callbacks, the slot's snapshot reflected live mutation.
"""

from __future__ import annotations

from pathlib import Path

from source_recall.repo_manager import RepoSlot


class TestProgressDetailDeepCopy:
    def test_nested_mutation_does_not_leak(self, tmp_path: Path) -> None:
        """Mutating the original detail's nested values after update
        does not change the slot's stored copy."""
        slot = RepoSlot(name="r", path=tmp_path)
        slot.set_indexing()

        original = {"spans": {"embed": 1.0}, "total": 10}
        slot.update_progress_detail(original)

        # Mutate the original's nested dict.
        original["spans"]["embed"] = 999.0
        original["total"] = 999

        stored = slot.progress_detail
        assert stored["spans"]["embed"] == 1.0, (
            "Nested mutation leaked into the slot's stored detail (shallow copy bug)"
        )
        assert stored["total"] == 10
