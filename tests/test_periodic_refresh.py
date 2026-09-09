"""Tests for periodic background refresh."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.repo_manager import RepoManager

_POLL_INTERVAL_S = 0.05
_POLL_TIMEOUT_S = 10.0


def _poll_until(
    predicate: Callable[[], object], timeout: float = _POLL_TIMEOUT_S
) -> bool:
    """Poll ``predicate()`` until it is truthy or the deadline passes.

    Bounded, load-tolerant replacement for a fixed ``time.sleep``: returns
    as soon as the condition is met instead of always burning the full
    wait, and still gives slow CI runners the full timeout to catch up.

    @param predicate: Zero-arg callable checked each poll tick.
    @param timeout: Max seconds to wait.
    @returns: True if the predicate became truthy before the deadline.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(_POLL_INTERVAL_S)
    return bool(predicate())


class TestPeriodicRefresh:
    def test_periodic_triggers_after_interval(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Periodic refresh fires after the configured interval."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            refresh_interval_s=1,  # 1 second for fast test.
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        refresh_count = {"n": 0}

        def counting_refresh(_self: object) -> int:
            refresh_count["n"] += 1
            return 0

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            # Verify daemon is up.
            assert client.get("/health").json()["ok"] is True

            # Patch refresh after startup, then poll for the first tick
            # instead of sleeping a fixed duration.
            with patch.object(Index, "refresh", counting_refresh):
                triggered = _poll_until(lambda: refresh_count["n"] >= 1)

        # Should have been called at least once.
        assert triggered, "periodic refresh did not fire within the deadline"
        assert refresh_count["n"] >= 1

    def test_periodic_refresh_works_after_prior_daemon_shutdown(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Periodic refresh fires even if a prior daemon already shut down.

        Regression: ``_stop_event`` is module-level and was set on
        lifespan shutdown but never cleared. A second daemon in the same
        process (e.g. the next test) saw the event already set, so its
        periodic refresh loop exited before the first tick.
        """
        import source_recall.daemon as daemon_mod
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            refresh_interval_s=1,
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        # Simulate a prior daemon shutdown that set the module-level event.
        daemon_mod._stop_event.set()
        assert daemon_mod._stop_event.is_set(), "precondition: event is set"

        refresh_count = {"n": 0}

        def counting_refresh(_self: object) -> int:
            refresh_count["n"] += 1
            return 0

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            assert client.get("/health").json()["ok"] is True

            # After startup, the event must be clear so the loop runs.
            assert not daemon_mod._stop_event.is_set(), (
                "lifespan startup must clear _stop_event so the periodic "
                "loop runs even after a prior daemon shut down in the "
                "same process"
            )

            with patch.object(Index, "refresh", counting_refresh):
                triggered = _poll_until(lambda: refresh_count["n"] >= 1)

        assert triggered, "periodic refresh did not fire within the deadline"
        assert refresh_count["n"] >= 1

    def test_periodic_skips_when_already_refreshing(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Periodic refresh skips a repo whose slot lock is already held.

        Holds the repo's slot lock (as a real in-progress refresh would)
        and asserts the periodic loop's own refresh is never invoked for
        that repo while the lock is held, then confirms it fires once the
        lock is released.
        """
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            refresh_interval_s=1,
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        refresh_count = {"n": 0}

        def counting_refresh(_self: object) -> int:
            refresh_count["n"] += 1
            return 0

        # Capture the RepoManager instance the daemon builds internally so
        # the test can reach the real per-repo lock the periodic loop
        # checks with ``lock.acquire(blocking=False)``.
        captured: dict[str, RepoManager] = {}
        original_from_config = RepoManager.from_config

        def capturing_from_config(config_arg: DaemonConfig) -> RepoManager:
            manager = original_from_config(config_arg)
            captured["manager"] = manager
            return manager

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with (
            patch.object(
                RepoManager, "from_config", staticmethod(capturing_from_config)
            ),
            TestClient(app) as client,
        ):
            assert client.get("/health").json()["ok"] is True

            manager = captured["manager"]
            slot = manager.slots[py_app_path.name]

            with patch.object(Index, "refresh", counting_refresh):
                # Hold the slot lock as an in-progress refresh would.
                assert slot.lock.acquire(blocking=False)
                try:
                    # Give the periodic loop several ticks to try (and
                    # skip) while the lock is held.
                    time.sleep(0.3)
                    assert refresh_count["n"] == 0, (
                        "periodic refresh must not run while the slot lock is held"
                    )
                finally:
                    slot.lock.release()

                # Once released, the next tick should refresh normally.
                triggered = _poll_until(lambda: refresh_count["n"] >= 1)

        assert triggered, "periodic refresh did not resume after lock release"
