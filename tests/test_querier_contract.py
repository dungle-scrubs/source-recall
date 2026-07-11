"""Tests for the querier result-length contract and graph expansion (M-5).

Before the fix:
- ``query`` could return more than ``top_k`` results because the final
  slice was ``final_ids[: k + expansion_slots]`` where ``expansion_slots``
  could be ``min(3, k)`` even when ``len(final_ids) == k``.
- ``_graph_expand`` mutated the caller's ``all_chunks`` dict in place,
  making the data flow hard to reason about.

The fix caps results strictly at ``top_k`` and has ``_graph_expand`` write
expansion data into a local dict that the caller merges explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from source_recall import Index
from source_recall.config import resolve_config
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.models import RefData, RefType
from source_recall.querier import IndexQuerier
from source_recall.store import IndexStore, get_db_path


def _build_with_refs(repo: Path) -> None:
    """Build an index where a unique caller references 5 helpers.

    Refs are inserted manually (bypassing the qualified-name resolution
    gap in the chunker) so graph expansion actually fires during query.
    """
    repo.mkdir(parents=True, exist_ok=True)
    # A unique caller name so it's the sole top FTS/vector hit.
    (repo / "a.py").write_text(
        "def the_unique_caller_xyz():\n    return helper_one()\n"
    )
    (repo / "b.py").write_text(
        "def helper_one():\n    return 1\n"
        "def helper_two():\n    return 2\n"
        "def helper_three():\n    return 3\n"
        "def helper_four():\n    return 4\n"
        "def helper_five():\n    return 5\n"
    )
    emb = BagOfWordsEmbedder(dimensions=64)
    Index(repo, embedder=emb).build()

    # Insert call refs from the caller to each helper so expansion fires.
    with IndexStore(get_db_path(repo)) as store:
        store.run_migrations()
        caller = store.conn.execute(
            "SELECT id FROM chunks WHERE symbol_name = 'the_unique_caller_xyz'"
        ).fetchone()[0]
        for name in (
            "helper_one",
            "helper_two",
            "helper_three",
            "helper_four",
            "helper_five",
        ):
            store.insert_refs(
                [
                    RefData(
                        source_chunk_id=caller,
                        target_symbol=name,
                        ref_type=RefType.CALL,
                    )
                ]
            )
        store.conn.commit()


class TestResultLengthContract:
    def test_query_returns_at_most_top_k_with_expansion(self, tmp_path: Path) -> None:
        """len(results) must never exceed top_k, even with graph expansion."""
        repo = tmp_path / "repo"
        _build_with_refs(repo)
        idx = Index(repo, embedder=BagOfWordsEmbedder(dimensions=64))
        # Query the unique caller name so it ranks #1, then expansion
        # pulls in helpers.  Without the fix, top_k=3 could return 6.
        for k in (1, 2, 3, 5):
            results = idx.query("the_unique_caller_xyz", top_k=k)
            assert len(results) <= k, (
                f"top_k={k} but got {len(results)} results "
                f"(reasons: {[r.match_reason for r in results]})"
            )
        idx.close()


class TestGraphExpansionToggle:
    def test_expansion_fires_by_default(self, tmp_path: Path) -> None:
        """With the default config, graph expansion pulls in referenced helpers.

        FTS-only (embedder=None) so the helpers are *not* already surfaced
        by vector search — they can only enter results via ref expansion.
        """
        repo = tmp_path / "repo"
        _build_with_refs(repo)
        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=None)
        results = querier.query("the_unique_caller_xyz", top_k=8)
        assert any(r.match_reason == "graph_expansion" for r in results), (
            "expected graph expansion to contribute results by default"
        )
        querier.close()

    def test_expansion_disabled_by_config(self, tmp_path: Path) -> None:
        """graph_expand_enabled=False short-circuits expansion entirely."""
        repo = tmp_path / "repo"
        _build_with_refs(repo)
        config = resolve_config(str(repo), graph_expand_enabled=False)
        querier = IndexQuerier(repo, config, embedder=None)
        results = querier.query("the_unique_caller_xyz", top_k=8)
        assert all(r.match_reason != "graph_expansion" for r in results), (
            "expansion ran despite graph_expand_enabled=False"
        )
        querier.close()


class TestGraphExpansionNoMutation:
    def test_graph_expand_does_not_mutate_input(self, tmp_path: Path) -> None:
        """_graph_expand writes to a local dict, not the caller's all_chunks."""
        repo = tmp_path / "repo"
        _build_with_refs(repo)

        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=BagOfWordsEmbedder(dimensions=64))
        store = querier._get_store()

        caller_id = store.conn.execute(
            "SELECT id FROM chunks WHERE symbol_name = 'the_unique_caller_xyz'"
        ).fetchone()[0]
        all_chunks: dict[str, dict] = {
            caller_id: {"chunk_id": caller_id, "content": "stub"}
        }
        snapshot_before = dict(all_chunks)

        querier._graph_expand(store, [caller_id], all_chunks, "q")

        assert all_chunks == snapshot_before, (
            "_graph_expand mutated the caller's all_chunks dict in place "
            "(M-5: should use a local dict)"
        )
        querier.close()


class TestReaderSelfHeal:
    def test_query_self_heals_after_atomic_swap(self, tmp_path: Path) -> None:
        """A long-lived querier sees fresh results after the db is swapped.

        The querier caches an IndexStore connection. When a build swaps
        ``index.db`` (a new inode via atomic rename), the cached reader must
        detect the replacement and reopen — without an explicit close — so
        it never serves stale results.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def alpha_marker():\n    return 1\n")
        Index(repo).build()

        config = resolve_config(str(repo))
        querier = IndexQuerier(repo, config, embedder=None)
        try:
            # Warm the cached connection against the original db file.
            before = querier.query("alpha_marker")
            assert any(r.symbol_name == "alpha_marker" for r in before)

            # Swap the db underneath the open querier via a full rebuild.
            (repo / "a.py").write_text("def beta_marker():\n    return 2\n")
            Index(repo).build()

            # Same querier, no explicit close — results must self-heal.
            after = querier.query("beta_marker")
            assert any(r.symbol_name == "beta_marker" for r in after), (
                "querier served stale results after the db was swapped"
            )
            assert not querier.query("alpha_marker"), (
                "stale alpha_marker still visible after swap"
            )
        finally:
            querier.close()

    def test_index_query_self_heals_after_swap(self, tmp_path: Path) -> None:
        """A live Index reopens its cached querier after a db swap.

        Exercises the write-lock reopen path so in-flight readers drain
        before the stale connection is torn down (M-2 invariant).
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def alpha_marker():\n    return 1\n")
        Index(repo).build()

        reader = Index(repo, embedder=None)
        try:
            assert any(
                r.symbol_name == "alpha_marker" for r in reader.query("alpha_marker")
            )

            (repo / "a.py").write_text("def beta_marker():\n    return 2\n")
            Index(repo).build()

            assert any(
                r.symbol_name == "beta_marker" for r in reader.query("beta_marker")
            )
            assert not reader.query("alpha_marker")
        finally:
            reader.close()

    def test_failed_reopen_does_not_cache_closed_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reopen that fails must not leave a closed store cached.

        Otherwise the closed store's file_replaced() reads False and it is
        served as if live, silently breaking every later query.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("def alpha_marker():\n    return 1\n")
        Index(repo).build()

        querier = IndexQuerier(repo, resolve_config(str(repo)), embedder=None)
        querier.query("alpha_marker")  # opens + caches the store
        old_store = querier._store
        assert old_store is not None

        # Bump the db mtime so file_replaced() fires on the next query.
        db = get_db_path(repo)
        st = db.stat()
        os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        # Force the reopen to fail after the old store is closed.
        def _boom() -> object:
            raise RuntimeError("reopen failed")

        monkeypatch.setattr(querier, "_open_store", _boom)

        with pytest.raises(RuntimeError):
            querier.query("alpha_marker")

        assert querier._store is None, (
            "a failed reopen must clear the cached store, not keep the closed one"
        )
        assert old_store._conn is None, "the old store connection must be closed"
        querier.close()
