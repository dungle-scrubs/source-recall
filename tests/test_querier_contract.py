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

from pathlib import Path

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
