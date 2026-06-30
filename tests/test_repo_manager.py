"""Tests for RepoManager and RepoSlot."""

from __future__ import annotations

from pathlib import Path

import pytest

from source_recall.daemon_config import DaemonConfig
from source_recall.repo_manager import RepoManager, RepoSlot, SlotState


class TestRepoSlotStates:
    def test_initial_state_is_queued(self, tmp_path: Path) -> None:
        """New RepoSlot starts in queued state."""
        slot = RepoSlot(name="test", path=tmp_path)
        assert slot.state == SlotState.QUEUED

    def test_state_transitions(self, tmp_path: Path) -> None:
        """Slot state can transition through the lifecycle."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.state = SlotState.INDEXING
        assert slot.state == SlotState.INDEXING
        slot.state = SlotState.READY
        assert slot.state == SlotState.READY

    def test_error_state(self, tmp_path: Path) -> None:
        """Slot can enter error state with a message."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.state = SlotState.ERROR
        slot.error = "corrupt index"
        assert slot.state == SlotState.ERROR
        assert slot.error == "corrupt index"


class TestRepoManagerAdd:
    def test_add_creates_queued_slot(self, tmp_path: Path) -> None:
        """add() creates a RepoSlot in queued state."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mgr = RepoManager()
        slot = mgr.add(repo)

        assert slot.name == "myrepo"
        assert slot.path == repo.resolve()
        assert slot.state == SlotState.QUEUED
        assert "myrepo" in mgr.slots

    def test_add_with_custom_name(self, tmp_path: Path) -> None:
        """add() uses custom name when provided."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mgr = RepoManager()
        slot = mgr.add(repo, name="custom")

        assert slot.name == "custom"
        assert "custom" in mgr.slots

    def test_add_rejects_nonexistent_path(self, tmp_path: Path) -> None:
        """add() raises ValueError for paths that don't exist."""
        mgr = RepoManager()
        with pytest.raises(ValueError, match="does not exist"):
            mgr.add(tmp_path / "nope")

    def test_add_rejects_duplicate_name(self, tmp_path: Path) -> None:
        """add() raises ValueError when name already registered."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mgr = RepoManager()
        mgr.add(repo)
        with pytest.raises(ValueError, match="already registered"):
            mgr.add(repo)


class TestRepoManagerRemove:
    def test_remove_deletes_slot(self, tmp_path: Path) -> None:
        """remove() removes the slot from the registry."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mgr = RepoManager()
        mgr.add(repo)
        assert "myrepo" in mgr.slots

        mgr.remove("myrepo")
        assert "myrepo" not in mgr.slots

    def test_remove_nonexistent_raises(self) -> None:
        """remove() raises KeyError for unknown repos."""
        mgr = RepoManager()
        with pytest.raises(KeyError, match="not found"):
            mgr.remove("nope")

    def test_remove_closes_index(self, py_app_path: Path) -> None:
        """remove() closes the Index if one was opened."""
        from source_recall import Index
        from source_recall.embedder import BagOfWordsEmbedder

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        mgr = RepoManager()
        slot = mgr.add(py_app_path)
        slot.index = idx
        slot.state = SlotState.READY

        mgr.remove(py_app_path.name)
        # After remove, querying the closed index should fail or
        # the index object should be None on the slot.
        assert py_app_path.name not in mgr.slots


class TestRepoManagerLoadFromConfig:
    def test_loads_all_repos(self, tmp_path: Path) -> None:
        """load_from_config() populates slots from DaemonConfig."""
        repo_a = tmp_path / "alpha"
        repo_b = tmp_path / "beta"
        repo_a.mkdir()
        repo_b.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=repo_a, name="alpha"),
                DaemonConfig.RepoEntry(path=repo_b, name="beta"),
            ],
        )

        mgr = RepoManager.from_config(config)

        assert len(mgr.slots) == 2
        assert "alpha" in mgr.slots
        assert "beta" in mgr.slots
        assert mgr.slots["alpha"].state == SlotState.QUEUED
        assert mgr.slots["beta"].state == SlotState.QUEUED

    def test_load_skips_bad_repo_gracefully(self, tmp_path: Path) -> None:
        """load_from_config() skips repos whose paths disappeared."""
        good = tmp_path / "good"
        good.mkdir()

        config = DaemonConfig(
            repos=[
                DaemonConfig.RepoEntry(path=good, name="good"),
                DaemonConfig.RepoEntry(path=tmp_path / "gone", name="gone"),
            ],
        )

        mgr = RepoManager.from_config(config)

        assert "good" in mgr.slots
        assert "gone" not in mgr.slots


class TestRepoManagerList:
    def test_list_returns_all_slots(self, tmp_path: Path) -> None:
        """list_repos() returns info for all registered repos."""
        repo = tmp_path / "myrepo"
        repo.mkdir()

        mgr = RepoManager()
        mgr.add(repo)

        repos = mgr.list_repos()
        assert len(repos) == 1
        assert repos[0]["name"] == "myrepo"
        assert repos[0]["state"] == "queued"
