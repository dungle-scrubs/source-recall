"""Tests for vector search: store vec_chunks, RRF merge, hybrid query."""

from __future__ import annotations

from pathlib import Path

import pytest

from source_recall.embedder import BagOfWordsEmbedder
from source_recall.models import ChunkData, SymbolType
from source_recall.querier import _rrf_merge
from source_recall.store import IndexStore


@pytest.fixture
def vec_store(tmp_path: Path) -> IndexStore:
    """Create a store with schema + vec_chunks table.

    @returns: IndexStore with vectors enabled.
    """
    db_path = tmp_path / "test_vec.db"
    s = IndexStore(db_path)
    s.open()
    s.create_schema()
    ok = s.ensure_vec_table(dimensions=256)
    assert ok, "sqlite-vec must be available for these tests"
    return s


@pytest.fixture
def embedder() -> BagOfWordsEmbedder:
    """256-d BagOfWords embedder for tests (higher dims = fewer collisions)."""
    return BagOfWordsEmbedder(dimensions=256)


# ---------------------------------------------------------------------------
# Store: vec_chunks CRUD
# ---------------------------------------------------------------------------


class TestVecChunksStore:
    def test_insert_and_count(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """Inserted vectors are countable."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="foo",
                symbol_type=SymbolType.FUNCTION,
                content="def foo(): pass",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="a.py",
                symbol_name="bar",
                symbol_type=SymbolType.FUNCTION,
                content="def bar(): return 42",
                start_line=3,
                end_line=3,
            ),
        ]
        vec_store.insert_chunks(chunks)

        ids = [c.chunk_id for c in chunks]
        vecs = embedder.embed_chunks([c.content for c in chunks])
        vec_store.insert_vectors(ids, vecs)

        assert vec_store.get_vector_count() == 2

    def test_search_returns_results(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """Vector search returns results ordered by distance."""
        # Use highly distinctive vocabulary to avoid hash collisions.
        chunks = [
            ChunkData(
                file_path="auth.py",
                symbol_name="authenticate",
                symbol_type=SymbolType.FUNCTION,
                content="authenticate login password credentials session token verify",
                start_line=1,
                end_line=2,
            ),
            ChunkData(
                file_path="render.py",
                symbol_name="draw_canvas",
                symbol_type=SymbolType.FUNCTION,
                content="canvas pixels shader render opengl graphics framebuffer",
                start_line=1,
                end_line=2,
            ),
        ]
        vec_store.insert_chunks(chunks)

        ids = [c.chunk_id for c in chunks]
        vecs = embedder.embed_chunks([c.content for c in chunks])
        vec_store.insert_vectors(ids, vecs)

        # Query shares vocabulary with auth chunk.
        query_vec = embedder.embed_query("authenticate login password credentials")
        results = vec_store.search_vectors(query_vec, top_k=2)

        assert len(results) == 2
        # Auth chunk should rank first (lower distance).
        assert results[0]["symbol_name"] == "authenticate"

    def test_delete_vectors_by_file(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """delete_vectors_by_file removes vectors for all chunks in a file."""
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="f1",
                symbol_type=SymbolType.FUNCTION,
                content="def f1(): pass",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="b.py",
                symbol_name="f2",
                symbol_type=SymbolType.FUNCTION,
                content="def f2(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        vec_store.insert_chunks(chunks)

        ids = [c.chunk_id for c in chunks]
        vecs = embedder.embed_chunks([c.content for c in chunks])
        vec_store.insert_vectors(ids, vecs)
        assert vec_store.get_vector_count() == 2

        vec_store.delete_vectors_by_file("a.py")
        assert vec_store.get_vector_count() == 1

    def test_delete_vectors_by_ids_is_atomic(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """delete_vectors_by_ids wraps all deletes in one transaction.

        H-2 fix: a failure mid-loop must roll back the whole batch so we
        don't leave orphaned/partial vector state.  We force a failure on
        the second DELETE and confirm none of the rows were removed.
        """
        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name=f"f{i}",
                symbol_type=SymbolType.FUNCTION,
                content=f"def f{i}(): pass",
                start_line=i,
                end_line=i,
            )
            for i in range(3)
        ]
        vec_store.insert_chunks(chunks)
        ids = [c.chunk_id for c in chunks]
        vecs = embedder.embed_chunks([c.content for c in chunks])
        vec_store.insert_vectors(ids, vecs)
        assert vec_store.get_vector_count() == 3

        # Wrap the apsw connection so the second DELETE fails mid-batch.
        real_conn = vec_store._get_vec_conn()  # type: ignore[union-attr]
        delete_count = {"n": 0}

        class _FlakyConn:
            def execute(self, sql: str, *args: object) -> object:
                if isinstance(sql, str) and sql.strip().upper().startswith("DELETE"):
                    delete_count["n"] += 1
                    if delete_count["n"] == 2:
                        msg = "simulated mid-batch failure"
                        raise RuntimeError(msg)
                return real_conn.execute(sql, *args)

        vec_store._vec_conn = _FlakyConn()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="simulated"):
            vec_store.delete_vectors_by_ids(ids)

        # Restore the real connection and confirm the rollback left all
        # three rows intact.
        vec_store._vec_conn = real_conn  # type: ignore[assignment]
        assert vec_store.get_vector_count() == 3

    def test_no_vec_table_returns_zero(self, tmp_path: Path) -> None:
        """get_vector_count returns 0 when vec_chunks doesn't exist."""
        db_path = tmp_path / "no_vec.db"
        s = IndexStore(db_path)
        s.open()
        s.create_schema()
        assert s.get_vector_count() == 0

    def test_insert_empty_is_noop(self, vec_store: IndexStore) -> None:
        """Inserting empty lists is a no-op."""
        vec_store.insert_vectors([], [])
        assert vec_store.get_vector_count() == 0

    def test_upsert_replaces_vector(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """Re-inserting same chunk_id replaces the vector (delete+insert)."""
        chunk = ChunkData(
            file_path="a.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        vec_store.insert_chunks([chunk])

        vec1 = embedder.embed_chunks(["def foo(): pass"])
        vec_store.insert_vectors([chunk.chunk_id], vec1)
        assert vec_store.get_vector_count() == 1

        vec2 = embedder.embed_chunks(["def foo(): return 42"])
        vec_store.insert_vectors([chunk.chunk_id], vec2)
        assert vec_store.get_vector_count() == 1  # Replaced, not duplicated.


# ---------------------------------------------------------------------------
# Schema migration to v2
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_migration_v1_to_latest(self, tmp_path: Path) -> None:
        """A v1 store migrates to the latest version."""
        db_path = tmp_path / "migrate.db"
        s = IndexStore(db_path)
        s.open()
        s.create_schema()
        # Force schema_version to 1 to simulate a Phase 1 index.
        s.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '1')"
        )
        s.conn.commit()
        s.close()

        # Re-open and migrate.
        s2 = IndexStore(db_path)
        s2.open()
        s2.run_migrations()

        version = s2.get_meta("schema_version")
        assert int(version) >= 3
        s2.close()


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_single_list(self) -> None:
        """RRF with empty second list returns scores from first."""
        fts = [
            {"chunk_id": "a", "score": 10},
            {"chunk_id": "b", "score": 5},
        ]
        scores = _rrf_merge(fts, [], k=15)
        assert "a" in scores
        assert "b" in scores
        assert scores["a"] > scores["b"]

    def test_two_lists_overlap(self) -> None:
        """Chunks appearing in both lists get boosted."""
        fts = [
            {"chunk_id": "a", "score": 10},
            {"chunk_id": "b", "score": 5},
        ]
        vec = [
            {"chunk_id": "b", "distance": 0.1},
            {"chunk_id": "c", "distance": 0.5},
        ]
        scores = _rrf_merge(fts, vec, k=15)

        # b appears in both → boosted above a (which only appears in fts).
        assert scores["b"] > scores["a"]
        assert "c" in scores

    def test_k_parameter_affects_scores(self) -> None:
        """Smaller k gives more weight to top ranks."""
        fts = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        vec: list[dict[str, object]] = []

        scores_k5 = _rrf_merge(fts, vec, k=5)
        scores_k50 = _rrf_merge(fts, vec, k=50)

        # With k=5, rank-1 vs rank-2 difference is larger.
        gap_k5 = scores_k5["a"] - scores_k5["b"]
        gap_k50 = scores_k50["a"] - scores_k50["b"]
        assert gap_k5 > gap_k50

    def test_empty_inputs(self) -> None:
        """Empty inputs return empty scores."""
        scores = _rrf_merge([], [], k=15)
        assert scores == {}


# ---------------------------------------------------------------------------
# Hybrid query (integration with BagOfWords)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_build_continues_when_embedder_crashes(self, tmp_path: Path) -> None:
        """If embedder.embed_chunks raises, chunks are still indexed (FTS works)."""
        import shutil

        from source_recall.builder import IndexBuilder
        from source_recall.config import SRConfig
        from source_recall.embedder import BagOfWordsEmbedder
        from source_recall.store import IndexStore, get_db_path

        # Copy py-app fixture.
        src = Path(__file__).parent / "fixtures" / "py-app"
        repo = tmp_path / "repo"
        shutil.copytree(src, repo)

        class ExplodingEmbedder(BagOfWordsEmbedder):
            """Embedder that always raises on embed_chunks."""

            def embed_chunks(self, _texts: list[str]) -> list[list[float]]:
                msg = "GPU exploded"
                raise RuntimeError(msg)

        config = SRConfig()
        builder = IndexBuilder(repo, config, embedder=ExplodingEmbedder())
        builder.build()

        # FTS still works — chunks were inserted.
        store = IndexStore(get_db_path(repo))
        store.open()
        assert store.get_chunk_count() > 0

        # But no vectors.
        assert store.get_vector_count() == 0

        # FTS search returns results.
        results = store.fts_search("validate")
        assert len(results) > 0
        store.close()

    def test_embed_enabled_false_skips_vectors(self, tmp_path: Path) -> None:
        """Index with embed_enabled=false produces zero vectors."""
        import shutil

        from source_recall import Index

        src = Path(__file__).parent / "fixtures" / "py-app"
        repo = tmp_path / "repo2"
        shutil.copytree(src, repo)

        idx = Index(repo, embedder=None)
        idx.build()

        s = idx.status()
        assert s.chunk_count > 0
        assert s.vector_count == 0
        assert s.embed_model == ""


class TestHybridQuery:
    def test_vector_search_finds_semantic_match(
        self, vec_store: IndexStore, embedder: BagOfWordsEmbedder
    ) -> None:
        """Vector search finds semantically similar code that FTS misses."""
        # Use highly distinctive vocabulary for clear BoW separation.
        chunks = [
            ChunkData(
                file_path="auth.py",
                symbol_name="check_auth",
                symbol_type=SymbolType.FUNCTION,
                content="authenticate login password credentials session token verify",
                start_line=1,
                end_line=1,
            ),
            ChunkData(
                file_path="math.py",
                symbol_name="add_numbers",
                symbol_type=SymbolType.FUNCTION,
                content="calculate sum multiply divide arithmetic integer float",
                start_line=1,
                end_line=1,
            ),
        ]
        vec_store.insert_chunks(chunks)

        ids = [c.chunk_id for c in chunks]
        vecs = embedder.embed_chunks([c.content for c in chunks])
        vec_store.insert_vectors(ids, vecs)

        # Query shares vocabulary with auth chunk.
        query_vec = embedder.embed_query("authenticate login password credentials")
        results = vec_store.search_vectors(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0]["symbol_name"] == "check_auth"

    def test_fts_fallback_when_no_vectors(self, tmp_path: Path) -> None:
        """Querier falls back to FTS-only when vec_chunks doesn't exist."""
        from source_recall.config import SRConfig
        from source_recall.querier import IndexQuerier

        db_path = tmp_path / "fts_only.db"
        s = IndexStore(db_path)
        s.open()
        s.create_schema()

        chunks = [
            ChunkData(
                file_path="a.py",
                symbol_name="process_payment",
                symbol_type=SymbolType.FUNCTION,
                content="def process_payment(amount): charge(amount)",
                start_line=1,
                end_line=1,
            ),
        ]
        s.insert_chunks(chunks)
        s.close()

        # Query with embedder but no vec_chunks table.
        config = SRConfig()
        querier = IndexQuerier.__new__(IndexQuerier)
        querier.repo_path = tmp_path
        querier.config = config
        querier.embedder = BagOfWordsEmbedder()
        querier.reranker = None
        querier._store = IndexStore(db_path)
        querier._store.open()

        results = querier.query("payment")
        assert len(results) >= 1
        assert results[0].symbol_name == "process_payment"
        assert "bm25" in results[0].match_reason

        querier.close()
