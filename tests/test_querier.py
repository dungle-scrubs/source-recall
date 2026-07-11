"""Tests for querier.py — query classification heuristics."""

from __future__ import annotations

from pathlib import Path

from source_recall.models import SearchRow
from source_recall.querier import (
    _compute_symbol_weight,
    _extract_symbol_candidates,
    _filter_by_branch,
)


def _row(chunk_id: str, branches: str = "") -> SearchRow:
    """Build a minimal SearchRow for branch-filter tests."""
    return SearchRow(
        chunk_id=chunk_id,
        file_path=f"{chunk_id}.py",
        symbol_name=chunk_id,
        symbol_type="function",
        content="",
        start_line=1,
        end_line=1,
        search_quality="ast",
        branches=branches,
    )


class TestComputeSymbolWeight:
    def test_pascal_case_positive(self) -> None:
        """PascalCase tokens increase symbol weight."""
        weight = _compute_symbol_weight("UserService")
        assert weight >= 0.3

    def test_snake_case_positive(self) -> None:
        """snake_case tokens increase symbol weight."""
        weight = _compute_symbol_weight("validate_email")
        assert weight >= 0.2

    def test_camel_case_positive(self) -> None:
        """camelCase tokens increase symbol weight."""
        weight = _compute_symbol_weight("processPayment")
        assert weight >= 0.1

    def test_dot_qualified_positive(self) -> None:
        """Dot-qualified names increase symbol weight."""
        weight = _compute_symbol_weight("AuthService.validate")
        assert weight >= 0.2

    def test_question_words_reduce_weight(self) -> None:
        """Question words reduce symbol weight."""
        plain = _compute_symbol_weight("UserService")
        with_question = _compute_symbol_weight("what is UserService")
        assert with_question < plain

    def test_natural_language_low_weight(self) -> None:
        """Plain English queries have low/zero weight."""
        weight = _compute_symbol_weight("how does authentication work")
        assert weight <= 0.0

    def test_mixed_query(self) -> None:
        """Mixed queries combine signals."""
        weight = _compute_symbol_weight("find UserService validate_email")
        assert weight >= 0.5  # PascalCase + snake_case

    def test_empty_query(self) -> None:
        """Empty query has zero weight."""
        assert _compute_symbol_weight("") == 0.0


class TestQueryDeduplication:
    def test_bm25_and_symbol_merged(self, tmp_path: Path) -> None:
        """A chunk found by both FTS and symbol search has combined reason."""
        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "svc.py").write_text(
            "class UserService:\n    def find_user(self): pass\n"
        )

        idx = Index(repo)
        idx.build()

        # "UserService" triggers both FTS (keyword) and symbol (PascalCase).
        results = idx.query("UserService")
        assert len(results) > 0
        user_results = [r for r in results if r.symbol_name == "UserService"]
        assert len(user_results) > 0
        # Should be combined: bm25+symbol or symbol_exact.
        assert "symbol" in user_results[0].match_reason

    def test_dot_qualified_symbol_query(self, tmp_path: Path) -> None:
        """Dot-qualified queries like 'AuthService.validate' find methods."""
        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "auth.py").write_text(
            "class AuthService:\n    def validate(self):\n        return True\n"
        )

        idx = Index(repo)
        idx.build()

        results = idx.query("AuthService.validate")
        assert len(results) > 0
        # Should find the method via symbol search.
        method_results = [r for r in results if "validate" in r.symbol_name]
        assert len(method_results) > 0


class TestExtractSymbolCandidates:
    def test_extracts_pascal_case(self) -> None:
        """PascalCase tokens are extracted as candidates."""
        candidates = _extract_symbol_candidates("find UserService")
        assert "UserService" in candidates

    def test_extracts_snake_case(self) -> None:
        """snake_case tokens are extracted as candidates."""
        candidates = _extract_symbol_candidates("show validate_email")
        assert "validate_email" in candidates

    def test_extracts_dot_qualified(self) -> None:
        """Dot-qualified names are extracted as candidates."""
        candidates = _extract_symbol_candidates("AuthService.validate")
        assert "AuthService.validate" in candidates

    def test_strips_punctuation(self) -> None:
        """Surrounding punctuation is stripped before matching."""
        candidates = _extract_symbol_candidates("(UserService)")
        assert "UserService" in candidates

    def test_plain_words_not_extracted(self) -> None:
        """Plain lowercase words are not extracted."""
        candidates = _extract_symbol_candidates("how does auth work")
        assert len(candidates) == 0

    def test_multiple_symbols(self) -> None:
        """Multiple symbol candidates extracted from one query."""
        candidates = _extract_symbol_candidates(
            "UserService.validate and process_payment"
        )
        assert "UserService.validate" in candidates
        assert "process_payment" in candidates


class TestFilterByBranch:
    def test_filters_to_matching_branch(self) -> None:
        """Only results containing the target branch pass through."""
        results = [
            _row("a", "main"),
            _row("b", "main,feature"),
            _row("c", "feature"),
        ]
        filtered = _filter_by_branch(results, "main")
        ids = {r.chunk_id for r in filtered}
        assert ids == {"a", "b"}

    def test_empty_branches_passes_through(self) -> None:
        """Legacy chunks with empty branches are always included."""
        results = [
            _row("a", ""),
            _row("b", "main"),
        ]
        filtered = _filter_by_branch(results, "main")
        ids = {r.chunk_id for r in filtered}
        assert ids == {"a", "b"}

    def test_no_false_positive_on_substring(self) -> None:
        """'feature' must not match 'feature-old'."""
        results = [
            _row("a", "feature-old"),
            _row("b", "feature"),
        ]
        filtered = _filter_by_branch(results, "feature")
        ids = {r.chunk_id for r in filtered}
        assert ids == {"b"}

    def test_default_empty_branches_passes(self) -> None:
        """A row with default (empty) branches passes through (backward compat)."""
        results = [
            _row("a"),
            _row("b", "main"),
        ]
        filtered = _filter_by_branch(results, "main")
        ids = {r.chunk_id for r in filtered}
        assert ids == {"a", "b"}
