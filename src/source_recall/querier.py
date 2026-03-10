"""IndexQuerier: FTS search, symbol matching, scoring, and status."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from source_recall.config import SRConfig
from source_recall.models import IndexNotFoundError, IndexStatus, QueryResult
from source_recall.store import IndexStore, get_db_path

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_PASCAL_RE = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+")
_SNAKE_RE = re.compile(r"[a-z]+(?:_[a-z]+)+")
_CAMEL_RE = re.compile(r"[a-z]+(?:[A-Z][a-z]+)+")
_DOT_QUALIFIED_RE = re.compile(r"\w+\.\w+")
_QUESTION_WORDS = frozenset(
    {"how", "why", "what", "where", "when", "which", "does", "is"}
)

# Quality multipliers for search_quality values.
_QUALITY_MULTIPLIER: dict[str, float] = {
    "ast": 1.0,
    "regex": 0.85,
    "text_fallback": 0.7,
}


def _compute_symbol_weight(query: str) -> float:
    """Compute a symbol-likelihood score for a query.

    Multi-signal scoring:
    - +0.3 for PascalCase tokens
    - +0.2 for snake_case tokens
    - +0.2 for dot-qualified tokens
    - +0.1 for camelCase tokens
    - −0.1 for question words

    @param query: User query string.
    @returns: Symbol weight (higher = more likely a symbol query).
    """
    tokens = query.split()
    weight = 0.0

    for token in tokens:
        clean = token.strip("()[]{}:;,.")
        if not clean:
            continue
        lower = clean.lower()

        if lower in _QUESTION_WORDS:
            weight -= 0.1
        if _PASCAL_RE.fullmatch(clean):
            weight += 0.3
        if _SNAKE_RE.fullmatch(clean):
            weight += 0.2
        if _DOT_QUALIFIED_RE.fullmatch(clean):
            weight += 0.2
        if _CAMEL_RE.fullmatch(clean):
            weight += 0.1

    return weight


def _extract_symbol_candidates(query: str) -> list[str]:
    """Extract likely symbol names from a query.

    @param query: User query string.
    @returns: List of candidate symbol names.
    """
    candidates: list[str] = []
    tokens = query.split()

    for token in tokens:
        clean = token.strip("()[]{}:;,.'\"?")
        if not clean:
            continue
        if (
            _PASCAL_RE.fullmatch(clean)
            or _SNAKE_RE.fullmatch(clean)
            or _CAMEL_RE.fullmatch(clean)
            or _DOT_QUALIFIED_RE.fullmatch(clean)
        ):
            candidates.append(clean)

    return candidates


# ---------------------------------------------------------------------------
# IndexQuerier
# ---------------------------------------------------------------------------


class IndexQuerier:
    """Queries an existing index via FTS and symbol matching.

    @param repo_path: Absolute path to the repository root.
    @param config: Resolved SRConfig.
    """

    def __init__(self, repo_path: Path, config: SRConfig) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self._store: IndexStore | None = None

    def _get_store(self) -> IndexStore:
        """Get or open the IndexStore.

        @returns: Active IndexStore.
        @raises IndexNotFoundError: If no index exists.
        """
        if self._store is not None:
            return self._store

        db_path = get_db_path(self.repo_path)
        if not db_path.exists():
            raise IndexNotFoundError(str(self.repo_path))

        store = IndexStore(db_path)
        store.open()
        store.run_migrations()
        self._store = store
        return store

    def close(self) -> None:
        """Close the underlying store."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Search the index.

        Combines FTS BM25 results with symbol exact matches,
        deduplicates, applies quality multipliers, and returns
        ranked results.

        @param question: Natural language or symbol query.
        @param top_k: Override number of results (default: config.top_k).
        @returns: Ranked list of QueryResult.
        """
        store = self._get_store()
        k = top_k if top_k is not None else self.config.top_k

        # FTS BM25 search.
        fts_results = store.fts_search(question, limit=30)

        # Symbol search if query looks like it references symbols.
        symbol_weight = _compute_symbol_weight(question)
        symbol_results: list[dict[str, Any]] = []

        if symbol_weight >= 0.3:
            candidates = _extract_symbol_candidates(question)
            for candidate in candidates:
                symbol_results.extend(store.symbol_search(candidate, limit=5))

        # Merge and deduplicate.
        seen: dict[str, dict[str, Any]] = {}

        for row in fts_results:
            cid = row["chunk_id"]
            if cid not in seen:
                row["match_reason"] = "bm25"
                seen[cid] = row
            # Keep the higher score.
            elif row["score"] > seen[cid]["score"]:
                seen[cid]["score"] = row["score"]

        for row in symbol_results:
            cid = row["chunk_id"]
            if cid not in seen:
                row["match_reason"] = "symbol_exact"
                seen[cid] = row
            else:
                # Boost existing entry + mark combined match.
                seen[cid]["score"] = max(seen[cid]["score"], row["score"])
                if "symbol" not in seen[cid]["match_reason"]:
                    seen[cid]["match_reason"] += "+symbol"

        # Apply quality multipliers.
        for row in seen.values():
            multiplier = _QUALITY_MULTIPLIER.get(row["search_quality"], 1.0)
            row["score"] *= multiplier

        # Rank by score descending.
        ranked = sorted(seen.values(), key=lambda r: r["score"], reverse=True)

        # Return top_k.
        return [
            QueryResult(
                chunk_id=r["chunk_id"],
                file_path=r["file_path"],
                symbol_name=r["symbol_name"],
                symbol_type=r["symbol_type"],
                content=r["content"],
                score=round(r["score"], 4),
                start_line=r["start_line"],
                end_line=r["end_line"],
                search_quality=r["search_quality"],
                match_reason=r["match_reason"],
            )
            for r in ranked[:k]
        ]

    def status(self) -> IndexStatus:
        """Get index status information.

        @returns: IndexStatus with all metrics.
        @raises IndexNotFoundError: If no index exists.
        """
        store = self._get_store()
        db_path = get_db_path(self.repo_path)

        indexed_at = store.get_meta("indexed_at") or ""
        last_commit = store.get_meta("last_commit") or ""
        chunk_count = store.get_chunk_count()

        mode_counts = store.get_file_count_by_mode()
        ast_files = mode_counts.get("ast", 0)
        regex_files = mode_counts.get("regex", 0)
        text_files = mode_counts.get("text_fallback", 0)
        file_count = ast_files + regex_files + text_files

        try:
            db_size = db_path.stat().st_size
        except OSError:
            db_size = 0

        return IndexStatus(
            repo_path=str(self.repo_path),
            db_path=str(db_path),
            db_size_bytes=db_size,
            indexed_at=indexed_at,
            last_commit=last_commit,
            file_count=file_count,
            chunk_count=chunk_count,
            ast_files=ast_files,
            regex_files=regex_files,
            text_fallback_files=text_files,
        )
