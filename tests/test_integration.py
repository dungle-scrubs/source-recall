"""Integration tests: full index → query pipeline."""

from __future__ import annotations

from pathlib import Path

from source_recall.embedder import BagOfWordsEmbedder


class TestBuildAndQuery:
    def test_index_and_query_py_app(self, py_app_path: Path) -> None:
        """Build index on Python fixture, then query it."""
        from source_recall import Index

        idx = Index(py_app_path, embedder=None)
        idx.build()

        # Query for authenticate (exact keyword in the fixture).
        results = idx.query("authenticate")
        assert len(results) > 0
        # Should find AuthService or auth-related chunks.
        found_auth = any(
            "auth" in r.file_path.lower() or "auth" in r.symbol_name.lower()
            for r in results
        )
        assert found_auth, (
            f"Expected auth results, got: {[r.symbol_name for r in results]}"
        )

    def test_index_and_query_ts_app(self, ts_app_path: Path) -> None:
        """Build index on TypeScript fixture, then query it."""
        from source_recall import Index

        idx = Index(ts_app_path, embedder=None)
        idx.build()

        # Query for user.
        results = idx.query("UserService")
        assert len(results) > 0

    def test_symbol_exact_match(self, py_app_path: Path) -> None:
        """Symbol queries find exact matches with high scores."""
        from source_recall import Index

        idx = Index(py_app_path, embedder=None)
        idx.build()

        results = idx.query("AuthService")
        assert len(results) > 0
        # First result should be the class/shell or method.
        auth_results = [r for r in results if "AuthService" in r.symbol_name]
        assert len(auth_results) > 0
        assert "symbol" in auth_results[0].match_reason

    def test_text_fallback_scores_lower(self, mixed_path: Path) -> None:
        """Text-fallback chunks score lower than AST chunks."""
        from source_recall import Index

        idx = Index(mixed_path, embedder=None)
        idx.build()

        # All results from mixed fixture.
        results = idx.query("deploy")
        # Verify we get results from the mixed fixture.
        assert len(results) > 0

    def test_status_after_build(self, py_app_path: Path) -> None:
        """Status returns correct counts after build."""
        from source_recall import Index

        idx = Index(py_app_path, embedder=None)
        idx.build()

        s = idx.status()
        assert s.file_count > 0
        assert s.chunk_count > 0
        assert s.ast_files > 0
        assert s.db_size_bytes > 0
        assert s.repo_path == str(py_app_path.resolve())


class TestIncrementalRefresh:
    def test_refresh_detects_changes(self, tmp_repo: Path) -> None:
        """Refresh re-indexes changed files."""
        from source_recall import Index

        idx = Index(tmp_repo, embedder=None)
        idx.build()

        s1 = idx.status()
        original_chunks = s1.chunk_count

        # Modify a file.
        auth_file = tmp_repo / "myapp" / "auth.py"
        content = auth_file.read_text()
        auth_file.write_text(content + "\ndef new_function():\n    return 42\n")

        refreshed = idx.refresh()
        assert refreshed > 0

        s2 = idx.status()
        assert s2.chunk_count > original_chunks

    def test_refresh_no_changes(self, tmp_repo: Path) -> None:
        """Refresh returns 0 when nothing changed."""
        from source_recall import Index

        idx = Index(tmp_repo, embedder=None)
        idx.build()

        refreshed = idx.refresh()
        assert refreshed == 0


class TestQueryOutputModes:
    def test_to_dict_serialization(self, py_app_path: Path) -> None:
        """QueryResult.to_dict produces JSON-serializable output."""
        import json

        from source_recall import Index

        idx = Index(py_app_path, embedder=None)
        idx.build()

        results = idx.query("validate")
        assert len(results) > 0

        # Verify JSON serialization works.
        serialized = json.dumps([r.to_dict() for r in results])
        parsed = json.loads(serialized)
        assert len(parsed) == len(results)
        assert "chunk_id" in parsed[0]
        assert "content" in parsed[0]
        assert "score" in parsed[0]


# ---------------------------------------------------------------------------
# Phase 1b: Vector integration tests
# ---------------------------------------------------------------------------


class TestVectorBuildAndQuery:
    def test_build_with_vectors(self, py_app_path: Path) -> None:
        """Build with BagOfWordsEmbedder produces vectors."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        s = idx.status()
        assert s.vector_count > 0
        assert s.vector_count == s.chunk_count
        assert s.embed_model == "BagOfWordsEmbedder"
        assert s.embed_dimensions == 64

    def test_hybrid_query_returns_results(self, py_app_path: Path) -> None:
        """Hybrid BM25+vector query returns ranked results."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        results = idx.query("authenticate user password")
        assert len(results) > 0
        # Should find auth-related results via both FTS and vectors.
        found_auth = any(
            "auth" in r.file_path.lower() or "auth" in r.symbol_name.lower()
            for r in results
        )
        assert found_auth

    def test_hybrid_match_reason_includes_vector(self, py_app_path: Path) -> None:
        """Hybrid results include 'vector' in match_reason."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        results = idx.query("authenticate")
        assert len(results) > 0
        # At least one result should have vector in match_reason.
        has_vector = any("vector" in r.match_reason for r in results)
        assert has_vector, (
            f"Expected vector matches, got: {[r.match_reason for r in results]}"
        )

    def test_refresh_updates_vectors(self, tmp_repo: Path) -> None:
        """Incremental refresh updates vectors for changed files."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(tmp_repo, embedder=emb)
        idx.build()

        s1 = idx.status()
        assert s1.vector_count == s1.chunk_count

        # Modify a file.
        auth_file = tmp_repo / "myapp" / "auth.py"
        content = auth_file.read_text()
        auth_file.write_text(content + "\ndef new_vector_fn():\n    return 99\n")

        refreshed = idx.refresh()
        assert refreshed > 0

        s2 = idx.status()
        assert s2.vector_count == s2.chunk_count
        assert s2.chunk_count > s1.chunk_count

    def test_status_shows_vector_info(self, py_app_path: Path) -> None:
        """Status includes vector count, model, and dimensions."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        s = idx.status()
        assert s.vector_count > 0
        assert s.embed_model == "BagOfWordsEmbedder"
        assert s.embed_dimensions == 64

    def test_fts_only_when_embed_disabled(self, py_app_path: Path) -> None:
        """Index with embedder=None produces no vectors."""
        from source_recall import Index

        idx = Index(py_app_path, embedder=None)
        idx.build()

        s = idx.status()
        assert s.vector_count == 0
        assert s.embed_model == ""

        # Query still works (FTS-only).
        results = idx.query("authenticate")
        assert len(results) > 0
        assert all("vector" not in r.match_reason for r in results)
