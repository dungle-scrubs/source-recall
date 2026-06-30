"""Tests for RepoManager.snapshot_repos and config persistence (M-2).

The daemon's ``_persist_config`` previously reached into
``RepoManager._lock`` (a private attribute) to snapshot the repos list
for serialization.  The fix adds a public ``snapshot_repos()`` method
and serializes outside the lock.
"""

from __future__ import annotations

from pathlib import Path

from source_recall.repo_manager import RepoManager


class TestSnapshotRepos:
    def test_snapshot_returns_name_path_tuples(self, tmp_path: Path) -> None:
        """snapshot_repos returns (name, path) for each registered repo."""
        mgr = RepoManager()
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        mgr.add(a, name="alpha")
        mgr.add(b, name="beta")

        snap = mgr.snapshot_repos()
        assert ("alpha", a) in snap
        assert ("beta", b) in snap
        assert len(snap) == 2

    def test_snapshot_is_empty_when_no_repos(self) -> None:
        """snapshot_repos returns [] for an empty manager."""
        mgr = RepoManager()
        assert mgr.snapshot_repos() == []

    def test_snapshot_reflects_removal(self, tmp_path: Path) -> None:
        """Removing a repo is reflected in the next snapshot."""
        mgr = RepoManager()
        a = tmp_path / "a"
        a.mkdir()
        mgr.add(a, name="alpha")
        mgr.remove("alpha")

        assert mgr.snapshot_repos() == []
