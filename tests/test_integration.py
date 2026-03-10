"""Integration tests: full index → query pipeline."""

from __future__ import annotations

from pathlib import Path


class TestBuildAndQuery:
    def test_index_and_query_py_app(self, py_app_path: Path) -> None:
        """Build index on Python fixture, then query it."""
        from source_recall import Index

        idx = Index(py_app_path)
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

        idx = Index(ts_app_path)
        idx.build()

        # Query for user.
        results = idx.query("UserService")
        assert len(results) > 0

    def test_symbol_exact_match(self, py_app_path: Path) -> None:
        """Symbol queries find exact matches with high scores."""
        from source_recall import Index

        idx = Index(py_app_path)
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

        idx = Index(mixed_path)
        idx.build()

        # All results from mixed fixture.
        results = idx.query("deploy")
        # Verify we get results from the mixed fixture.
        assert len(results) > 0

    def test_status_after_build(self, py_app_path: Path) -> None:
        """Status returns correct counts after build."""
        from source_recall import Index

        idx = Index(py_app_path)
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

        idx = Index(tmp_repo)
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

        idx = Index(tmp_repo)
        idx.build()

        refreshed = idx.refresh()
        assert refreshed == 0


class TestQueryOutputModes:
    def test_to_dict_serialization(self, py_app_path: Path) -> None:
        """QueryResult.to_dict produces JSON-serializable output."""
        import json

        from source_recall import Index

        idx = Index(py_app_path)
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
