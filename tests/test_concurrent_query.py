"""Tests for Index concurrent-query parallelism (M-2 fix).

Before M-2, ``Index.query`` held an exclusive ``_querier_lock`` for the
entire call, serializing all concurrent queries on one Index.  After
M-2, queries share a read lock and run in parallel; refresh/build/close
take the exclusive write lock.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from source_recall import Index
from source_recall.embedder import BagOfWordsEmbedder


def _build_index(repo: Path) -> Index:
    """Build a tiny index so queries resolve without error."""
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "app.py").write_text("def authenticate(user):\n    return user\n")
    emb = BagOfWordsEmbedder(dimensions=64)
    Index(repo, embedder=emb).build()
    return Index(repo, embedder=emb)


class TestConcurrentQueries:
    def test_concurrent_queries_overlap(self, tmp_path: Path) -> None:
        """Two concurrent queries on one Index execute in parallel.

        Each query sleeps inside a patched ``fts_search``.  Under the old
        exclusive lock the wall time would be ~2x the sleep; under the
        read-shared lock it should be ~1x.
        """
        idx = _build_index(tmp_path / "repo")

        from source_recall import querier as querier_mod

        sleep_s = 0.4
        original = querier_mod.IndexQuerier.query

        # Wrap query to add a measurable delay; both threads must be able
        # to be inside it at once.
        in_flight = {"n": 0, "max": 0}
        lock = threading.Lock()

        def slow_query(self, question, *, top_k=None, branch=None, query_vec=None):  # type: ignore[no-untyped-def]
            with lock:
                in_flight["n"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["n"])
            time.sleep(sleep_s)
            try:
                return original(
                    self, question, top_k=top_k, branch=branch, query_vec=query_vec
                )
            finally:
                with lock:
                    in_flight["n"] -= 1

        querier_mod.IndexQuerier.query = slow_query  # type: ignore[method-assign]
        try:
            t0 = time.monotonic()
            threads = [
                threading.Thread(target=idx.query, args=("authenticate",))
                for _ in range(2)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            elapsed = time.monotonic() - t0
        finally:
            querier_mod.IndexQuerier.query = original  # type: ignore[method-assign]

        # If queries serialized (old behavior), max in-flight would be 1
        # and elapsed would be ~2*sleep_s.  With the read lock, both are
        # in flight simultaneously.
        assert in_flight["max"] == 2, (
            f"queries did not run concurrently (max in-flight={in_flight['max']})"
        )
        assert elapsed < 2 * sleep_s, (
            f"queries serialized: elapsed={elapsed:.2f}s, expected <{2 * sleep_s}s"
        )

    def test_refresh_waits_for_in_flight_query(self, tmp_path: Path) -> None:
        """A refresh (write lock) waits for in-flight queries (read lock) to drain."""
        idx = _build_index(tmp_path / "repo")

        from source_recall import querier as querier_mod

        original = querier_mod.IndexQuerier.query
        query_started = threading.Event()
        query_can_finish = threading.Event()

        def blocking_query(self, question, *, top_k=None, branch=None, query_vec=None):  # type: ignore[no-untyped-def]
            query_started.set()
            query_can_finish.wait(timeout=5)
            return original(
                self, question, top_k=top_k, branch=branch, query_vec=query_vec
            )

        querier_mod.IndexQuerier.query = blocking_query  # type: ignore[method-assign]
        try:
            qt = threading.Thread(target=idx.query, args=("authenticate",))
            qt.start()
            assert query_started.wait(timeout=2)

            # Refresh should block (write lock) until the query finishes.
            refreshed = {"done": False}

            def do_refresh() -> None:
                idx.refresh()
                refreshed["done"] = True

            rt = threading.Thread(target=do_refresh)
            rt.start()
            time.sleep(0.1)
            assert not refreshed["done"], "refresh proceeded while query in flight"

            query_can_finish.set()
            rt.join(timeout=10)
            qt.join(timeout=10)
            assert refreshed["done"], "refresh never completed after query drained"
        finally:
            querier_mod.IndexQuerier.query = original  # type: ignore[method-assign]
