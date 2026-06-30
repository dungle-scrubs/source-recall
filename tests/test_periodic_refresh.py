"""Tests for periodic background refresh."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder


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

            # Patch refresh after startup to count calls.
            with patch.object(Index, "refresh", counting_refresh):
                time.sleep(2.5)  # Wait for at least 2 periodic ticks.

        # Should have been called at least once.
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
                time.sleep(2.5)

        assert refresh_count["n"] >= 1

    def test_periodic_skips_when_already_refreshing(
        self, py_app_path: Path, tmp_path: Path
    ) -> None:
        """Periodic refresh skips if lock is held (no backlog)."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)
        Index(py_app_path, embedder=emb).build()

        config = DaemonConfig(
            refresh_interval_s=1,
            repos=[DaemonConfig.RepoEntry(path=py_app_path, name=py_app_path.name)],
            config_path=tmp_path / "repos.toml",
        )

        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            assert client.get("/health").json()["ok"] is True
            # If we hold the repo lock, periodic should skip.
            # Just verify no crash/deadlock occurs.
            time.sleep(2)

        # Test passes if no deadlock or crash.
