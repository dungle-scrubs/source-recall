"""IndexQuerier: FTS search, vector search, RRF merge, and status."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from source_recall.config import SRConfig
from source_recall.models import IndexNotFoundError, IndexStatus, QueryResult
from source_recall.store import IndexStore, get_db_path

if TYPE_CHECKING:
    from source_recall.embedder import Embedder
    from source_recall.reranker import Reranker

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

# RRF parameter — optimized for short lists (not the standard k=60).
_RRF_K = 15


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
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


def _filter_by_branch(
    results: list[dict[str, Any]], branch: str
) -> list[dict[str, Any]]:
    """Filter search results to those belonging to a specific branch.

    Uses exact match against the comma-separated branches field.
    Results without a branches key or with empty branches pass through
    (backward compatibility with pre-v4 indexes).

    @param results: Search results with optional 'branches' key.
    @param branch: Target branch name.
    @returns: Filtered results.
    """
    filtered: list[dict[str, Any]] = []
    for row in results:
        branches_csv = row.get("branches", "")
        if not branches_csv:
            # Legacy chunk (no branch info) — include by default.
            filtered.append(row)
        elif branch in branches_csv.split(","):
            filtered.append(row)
    return filtered


def _rrf_merge(
    fts_results: list[dict[str, Any]],
    vec_results: list[dict[str, Any]],
    *,
    k: int = 15,
) -> dict[str, float]:
    """Merge FTS and vector results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list containing the chunk.
    Using k=15 (not 60) because our lists are short (≤30 items).

    @param fts_results: BM25-ranked results (highest score first).
    @param vec_results: Distance-ranked results (lowest distance first).
    @param k: RRF constant.
    @returns: Dict mapping chunk_id to RRF score.
    """
    scores: dict[str, float] = {}

    for rank, row in enumerate(fts_results, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, row in enumerate(vec_results, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    return scores


# ---------------------------------------------------------------------------
# IndexQuerier
# ---------------------------------------------------------------------------


class IndexQuerier:
    """Queries an existing index via FTS, vectors, and symbol matching.

    @param repo_path: Absolute path to the repository root.
    @param config: Resolved SRConfig.
    @param embedder: Optional embedder for vector search. None = FTS-only.
    """

    def __init__(
        self,
        repo_path: Path,
        config: SRConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self.embedder = embedder
        self.reranker = reranker
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
        branch: str | None = None,
    ) -> list[QueryResult]:
        """Search the index using FTS + vectors + symbol matching.

        When vectors are available, results from FTS and vector search
        are merged via Reciprocal Rank Fusion (k=15). When vectors are
        unavailable, falls back to FTS-only (Phase 1 behavior).

        @param question: Natural language or symbol query.
        @param top_k: Override number of results (default: config.top_k).
        @param branch: Filter results to this branch. None = use active_branch
            from meta; empty string = no filtering (all branches).
        @returns: Ranked list of QueryResult.
        """
        store = self._get_store()
        k = top_k if top_k is not None else self.config.top_k

        # Resolve branch: None → active_branch from meta.
        if branch is None:
            branch = store.get_meta("active_branch") or ""

        # FTS BM25 search.
        fts_results = store.fts_search(question, limit=30)

        # Vector search if embedder and vec_chunks available.
        # Skip for empty/whitespace queries — no meaningful embedding.
        vec_results: list[dict[str, Any]] = []
        if self.embedder is not None and store.has_vec_table() and question.strip():
            try:
                query_vec = self.embedder.embed_query(question)
                vec_results = store.search_vectors(query_vec, top_k=30)
            except Exception:
                pass  # Vector search failure is non-fatal.

        # Symbol search if query looks like it references symbols.
        symbol_weight = _compute_symbol_weight(question)
        symbol_results: list[dict[str, Any]] = []

        if symbol_weight >= 0.3:
            candidates = _extract_symbol_candidates(question)
            for candidate in candidates:
                symbol_results.extend(store.symbol_search(candidate, limit=5))

        # Post-filter by branch if specified.
        if branch:
            fts_results = _filter_by_branch(fts_results, branch)
            vec_results = _filter_by_branch(vec_results, branch)
            symbol_results = _filter_by_branch(symbol_results, branch)

        # Build chunk data lookup.
        all_chunks: dict[str, dict[str, Any]] = {}
        for row in fts_results:
            all_chunks.setdefault(row["chunk_id"], row)
        for row in vec_results:
            all_chunks.setdefault(row["chunk_id"], row)
        for row in symbol_results:
            all_chunks.setdefault(row["chunk_id"], row)

        # Determine match reasons.
        fts_ids = {r["chunk_id"] for r in fts_results}
        vec_ids = {r["chunk_id"] for r in vec_results}
        sym_ids = {r["chunk_id"] for r in symbol_results}

        if vec_results:
            # Hybrid mode: RRF merge of FTS + vector results.
            scores = _rrf_merge(fts_results, vec_results, k=_RRF_K)

            # Symbol exact matches get a rank-1 bonus.
            for row in symbol_results:
                cid = row["chunk_id"]
                bonus = 1.0 / (_RRF_K + 1)  # Rank-1 RRF score.
                scores[cid] = scores.get(cid, 0.0) + bonus

            # Apply quality multipliers.
            for cid in scores:
                chunk = all_chunks.get(cid)
                if chunk:
                    multiplier = _QUALITY_MULTIPLIER.get(
                        chunk.get("search_quality", "ast"), 1.0
                    )
                    scores[cid] *= multiplier

            # Build match reasons.
            reasons: dict[str, str] = {}
            for cid in scores:
                parts: list[str] = []
                if cid in fts_ids:
                    parts.append("bm25")
                if cid in vec_ids:
                    parts.append("vector")
                if cid in sym_ids:
                    parts.append("symbol")
                reasons[cid] = "+".join(parts) if parts else "rrf"

            # Rank by RRF score descending.
            ranked_ids = sorted(scores, key=lambda c: scores[c], reverse=True)

        else:
            # FTS-only mode (no vectors available).
            seen: dict[str, float] = {}
            reasons = {}

            for row in fts_results:
                cid = row["chunk_id"]
                if cid not in seen or row["score"] > seen[cid]:
                    seen[cid] = row["score"]
                    reasons[cid] = "bm25"

            for row in symbol_results:
                cid = row["chunk_id"]
                if cid not in seen or row["score"] > seen[cid]:
                    seen[cid] = max(seen.get(cid, 0.0), row["score"])
                    r = reasons.get(cid, "")
                    if "symbol" not in r:
                        reasons[cid] = f"{r}+symbol" if r else "symbol_exact"

            # Apply quality multipliers.
            for cid in seen:
                chunk = all_chunks.get(cid)
                if chunk:
                    multiplier = _QUALITY_MULTIPLIER.get(
                        chunk.get("search_quality", "ast"), 1.0
                    )
                    seen[cid] *= multiplier

            scores = seen
            ranked_ids = sorted(scores, key=lambda c: scores[c], reverse=True)

        # Rerank top candidates if reranker is available.
        candidates = ranked_ids[: k * 3]  # Rerank 3x top_k for quality.
        if self.reranker is not None and candidates and question.strip():
            try:
                items = [
                    {"chunk_id": cid, "content": all_chunks[cid]["content"]}
                    for cid in candidates
                    if cid in all_chunks
                ]
                reranked = self.reranker.rerank(question, items)
                # Replace ranking with reranker scores.
                ranked_ids = [item["chunk_id"] for item, _score in reranked]
                for item, score in reranked:
                    scores[item["chunk_id"]] = score
                    reason = reasons.get(item["chunk_id"], "")
                    if reason:
                        reasons[item["chunk_id"]] = f"{reason}+reranked"
                    else:
                        reasons[item["chunk_id"]] = "reranked"
            except Exception:
                pass  # Reranking failure is non-fatal.

        # Graph expansion: expand top results along ref edges.
        top_ids = ranked_ids[:5]
        expanded: list[str] = []
        if top_ids:
            expanded = self._graph_expand(store, top_ids, all_chunks, question)

        # Merge expanded into ranked results.
        final_ids = ranked_ids[:k]
        expansion_slots = k - len(final_ids)
        if expansion_slots <= 0:
            expansion_slots = min(3, k)  # Always allow a few expansions.
        for eid in expanded[:expansion_slots]:
            if eid not in final_ids:
                final_ids.append(eid)
                reasons[eid] = "graph_expansion"

        # Return top_k.
        return [
            QueryResult(
                chunk_id=cid,
                file_path=all_chunks[cid]["file_path"],
                symbol_name=all_chunks[cid]["symbol_name"],
                symbol_type=all_chunks[cid]["symbol_type"],
                content=all_chunks[cid]["content"],
                score=round(scores.get(cid, 0.0), 4),
                start_line=all_chunks[cid]["start_line"],
                end_line=all_chunks[cid]["end_line"],
                search_quality=all_chunks[cid]["search_quality"],
                match_reason=reasons.get(cid, ""),
            )
            for cid in final_ids[: k + expansion_slots]
            if cid in all_chunks
        ]

    def _graph_expand(
        self,
        store: IndexStore,
        top_ids: list[str],
        all_chunks: dict[str, dict[str, Any]],
        _question: str,
    ) -> list[str]:
        """Expand top results along the ref graph.

        For each top result, look up its outgoing refs, resolve targets
        via symbol_lookup, and include chunks that aren't already in results.

        @param store: Active IndexStore.
        @param top_ids: Top chunk IDs to expand from.
        @param all_chunks: Already-loaded chunk data.
        @param _question: Original query (reserved for future score-gating).
        @returns: List of expanded chunk IDs.
        """
        existing = set(all_chunks.keys())
        expanded: list[str] = []

        for cid in top_ids:
            refs = store.get_refs_for_chunk(cid)
            for ref in refs:
                # Resolve target symbol to chunk(s).
                targets = store.lookup_symbol(ref.target_symbol)
                for target in targets:
                    tcid = target["chunk_id"]
                    if tcid not in existing and tcid not in expanded:
                        # Add to all_chunks so we can build QueryResult.
                        all_chunks[tcid] = target
                        expanded.append(tcid)

        return expanded[:10]  # Cap at 10 expansion slots.

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

        # Vector info.
        vector_count = store.get_vector_count()
        embed_model = store.get_meta("embed_model") or ""
        embed_dimensions = int(store.get_meta("embed_dimensions") or "0")

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
            vector_count=vector_count,
            embed_model=embed_model,
            embed_dimensions=embed_dimensions,
        )
