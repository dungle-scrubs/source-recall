"""IndexQuerier: FTS search, vector search, RRF merge, and status."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.concurrency import ReaderWriterLock
from source_recall.config import SRConfig
from source_recall.models import (
    IndexNotFoundError,
    IndexStatus,
    QueryResult,
    SearchRow,
)
from source_recall.store import IndexStore, get_db_path

logger = logging.getLogger(__name__)

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

# Per-name cap on ref-target definitions during graph expansion.
# Applied per requested symbol (not globally) so one hot symbol name
# cannot starve the others; expansion still contributes at most 10 slots
# overall (bounded again by top_k).  Generous enough to leave headroom for
# dedup against results already ranked.
_GRAPH_LOOKUP_LIMIT = 25


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


def _filter_by_branch(results: list[SearchRow], branch: str) -> list[SearchRow]:
    """Filter search results to those belonging to a specific branch.

    Uses exact match against the comma-separated branches field.
    Results with empty branches pass through (backward compatibility
    with pre-v4 indexes).

    @param results: Search rows carrying a 'branches' field.
    @param branch: Target branch name.
    @returns: Filtered results.
    """
    filtered: list[SearchRow] = []
    for row in results:
        branches_csv = row.branches
        if not branches_csv:
            # Legacy chunk (no branch info) — include by default.
            filtered.append(row)
        elif branch in branches_csv.split(","):
            filtered.append(row)
    return filtered


def _rrf_merge(
    fts_results: list[SearchRow],
    vec_results: list[SearchRow],
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
        cid = row.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, row in enumerate(vec_results, start=1):
        cid = row.chunk_id
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
        # Governs the cached store's *lifetime*.  Concurrent queries hold
        # the read lock (running in parallel); a swap-triggered reopen
        # takes the write lock, which drains in-flight readers before the
        # stale connection is closed.  Without this a reopen could close a
        # sqlite connection another thread is mid-query on.
        self._store_rwlock = ReaderWriterLock()

    def _enter_store(self) -> IndexStore:
        """Acquire the store read lock and return a live IndexStore.

        On return the caller holds the read lock and MUST release it once
        it is finished using the store (see ``query``/``status``).

        Self-heal: a build may have atomic-swapped ``index.db`` out from
        under our long-lived connection.  When the file changed (new
        inode/mtime) the store is reopened under the *write* lock so
        concurrent readers drain first — we never close a connection a
        reader is mid-query on.

        @returns: Active IndexStore (read lock held).
        @raises IndexNotFoundError: If no index exists.
        """
        while True:
            self._store_rwlock.acquire_read()
            store = self._store
            if store is not None and not store.file_replaced():
                return store
            # (Re)open needed — escalate to the write lock (drains readers).
            self._store_rwlock.release_read()
            self._store_rwlock.acquire_write()
            try:
                if self._store is not None and self._store.file_replaced():
                    # Drop the reference *before* reopening: if _open_store
                    # fails we must not leave a closed store cached (its
                    # file_replaced() would read False and be served as
                    # live).  The next _enter_store then reopens cleanly.
                    old = self._store
                    self._store = None
                    old.close()
                if self._store is None:
                    self._open_store()
            finally:
                self._store_rwlock.release_write()
            # Loop back: re-acquire the read lock and re-verify.

    def _get_store(self) -> IndexStore:
        """Ensure the store is open (self-healing) and return it.

        Convenience for callers that use the store synchronously without
        needing the lifetime read lock held (tests, single-threaded use).
        ``query``/``status`` use ``_enter_store`` so the read lock is held
        for the full operation instead.

        @returns: Active IndexStore.
        @raises IndexNotFoundError: If no index exists.
        """
        store = self._enter_store()
        self._store_rwlock.release_read()
        return store

    def _open_store(self) -> IndexStore:
        """Open, migrate, and validate a fresh store (caller holds write lock).

        Caches the store on ``self._store`` only on full success; on any
        initialization failure the candidate connection is closed so it is
        not leaked and no half-open store is cached.

        @returns: Active IndexStore.
        @raises IndexNotFoundError: If no index exists.
        """
        db_path = get_db_path(self.repo_path)
        if not db_path.exists():
            raise IndexNotFoundError(str(self.repo_path))

        store = IndexStore(db_path)
        try:
            store.open()
            store.run_migrations()
            self._validate_embed_dimensions(store)
        except BaseException:
            store.close()
            raise

        self._store = store
        return store

    def _validate_embed_dimensions(self, store: IndexStore) -> None:
        """Warn if the embedder's dimensions differ from the stored index.

        A mismatch silently degrades vector search to empty results because
        vec_chunks' schema is fixed at creation time (H-3).

        @param store: The freshly opened store to inspect.
        """
        if self.embedder is None:
            return
        stored_dim_str = store.get_meta("embed_dimensions")
        if not stored_dim_str:
            return
        try:
            stored_dim = int(stored_dim_str)
        except ValueError:
            stored_dim = 0
        if stored_dim and stored_dim != self.embedder.dimensions:
            logger.warning(
                "Embedding dimension mismatch: index was built with "
                "%d dimensions but the current embedder uses %d. "
                "Vector search is disabled until the index is "
                "rebuilt with a matching embedder. Re-run "
                "'sr index' to fix.",
                stored_dim,
                self.embedder.dimensions,
            )

    def close(self) -> None:
        """Close the underlying store.

        Takes the write lock so it cannot tear the connection down while a
        concurrent query holds the read lock.
        """
        self._store_rwlock.acquire_write()
        try:
            if self._store is not None:
                self._store.close()
                self._store = None
        finally:
            self._store_rwlock.release_write()

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        branch: str | None = None,
        query_vec: list[float] | None = None,
    ) -> list[QueryResult]:
        """Search the index using FTS + vectors + symbol matching.

        When vectors are available, results from FTS and vector search
        are merged via Reciprocal Rank Fusion (k=15). When vectors are
        unavailable, falls back to FTS-only (Phase 1 behavior).

        @param question: Natural language or symbol query.
        @param top_k: Override number of results (default: config.top_k).
        @param branch: Filter results to this branch. None = use active_branch
            from meta; empty string = no filtering (all branches).
        @param query_vec: Precomputed query embedding. When provided, the
            embedder is not re-invoked for vector search — a multi-repo
            fan-out embeds the query once and reuses the vector across every
            repo (they share the same embedder). Must have been produced by
            an embedder with the same dimensions as this index.
        @returns: Ranked list of QueryResult.
        """
        # Hold the store read lock for the whole query so a concurrent
        # swap-triggered reopen cannot close the connection mid-flight.
        store = self._enter_store()
        try:
            return self._run_query(
                store, question, top_k=top_k, branch=branch, query_vec=query_vec
            )
        finally:
            self._store_rwlock.release_read()

    def _run_query(
        self,
        store: IndexStore,
        question: str,
        *,
        top_k: int | None,
        branch: str | None,
        query_vec: list[float] | None = None,
    ) -> list[QueryResult]:
        """Execute a query against an already-acquired store.

        Caller holds the store read lock for the duration.

        @param store: Live IndexStore (read lock held by caller).
        @param question: Natural language or symbol query.
        @param top_k: Override number of results (default: config.top_k).
        @param branch: Branch filter (see ``query``).
        @param query_vec: Precomputed query embedding (see ``query``).
        @returns: Ranked list of QueryResult.
        """
        k = top_k if top_k is not None else self.config.top_k

        # Resolve branch: None → active_branch from meta.
        if branch is None:
            branch = store.get_meta("active_branch") or ""

        # FTS BM25 search.
        fts_results = store.fts_search(question, limit=30)

        # Vector search if embedder and vec_chunks available.
        # Skip for empty/whitespace queries — no meaningful embedding.
        vec_results: list[SearchRow] = []
        if self.embedder is not None and store.has_vec_table() and question.strip():
            try:
                # Reuse a caller-supplied embedding (multi-repo fan-out
                # computes it once) instead of re-embedding per repo.
                vec = (
                    query_vec
                    if query_vec is not None
                    else self.embedder.embed_query(question)
                )
                vec_results = store.search_vectors(vec, top_k=30)
            except Exception:
                logger.warning("Vector search failed", exc_info=True)

        # Symbol search if query looks like it references symbols.
        symbol_weight = _compute_symbol_weight(question)
        symbol_results: list[SearchRow] = []

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
        all_chunks: dict[str, SearchRow] = {}
        for row in fts_results:
            all_chunks.setdefault(row.chunk_id, row)
        for row in vec_results:
            all_chunks.setdefault(row.chunk_id, row)
        for row in symbol_results:
            all_chunks.setdefault(row.chunk_id, row)

        # Determine match reasons.
        fts_ids = {r.chunk_id for r in fts_results}
        vec_ids = {r.chunk_id for r in vec_results}
        sym_ids = {r.chunk_id for r in symbol_results}

        if vec_results:
            # Hybrid mode: RRF merge of FTS + vector results.
            scores = _rrf_merge(fts_results, vec_results, k=_RRF_K)

            # Symbol exact matches get a rank-1 bonus.
            for row in symbol_results:
                cid = row.chunk_id
                bonus = 1.0 / (_RRF_K + 1)  # Rank-1 RRF score.
                scores[cid] = scores.get(cid, 0.0) + bonus

            # Apply quality multipliers.
            for cid in scores:
                chunk = all_chunks.get(cid)
                if chunk:
                    multiplier = _QUALITY_MULTIPLIER.get(chunk.search_quality, 1.0)
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
                cid = row.chunk_id
                if cid not in seen or row.score > seen[cid]:
                    seen[cid] = row.score
                    reasons[cid] = "bm25"

            for row in symbol_results:
                cid = row.chunk_id
                if cid not in seen or row.score > seen[cid]:
                    seen[cid] = max(seen.get(cid, 0.0), row.score)
                    r = reasons.get(cid, "")
                    if "symbol" not in r:
                        reasons[cid] = f"{r}+symbol" if r else "symbol_exact"

            # Apply quality multipliers.
            for cid in seen:
                chunk = all_chunks.get(cid)
                if chunk:
                    multiplier = _QUALITY_MULTIPLIER.get(chunk.search_quality, 1.0)
                    seen[cid] *= multiplier

            scores = seen
            ranked_ids = sorted(scores, key=lambda c: scores[c], reverse=True)

        # Rerank top candidates if reranker is available.
        candidates = ranked_ids[: k * 3]  # Rerank 3x top_k for quality.
        if self.reranker is not None and candidates and question.strip():
            try:
                items = [
                    {"chunk_id": cid, "content": all_chunks[cid].content}
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
                logger.warning("Reranking failed", exc_info=True)

        # Graph expansion: expand top results along ref edges.
        # _graph_expand returns (chunk_ids, chunk_data) and does NOT
        # mutate the caller's all_chunks (M-5 fix).
        top_ids = ranked_ids[:5]
        expanded_ids: list[str] = []
        expanded_chunks: dict[str, SearchRow] = {}
        # Gate expansion behind config (like the reranker/vector gates):
        # when disabled we never touch the ref graph — zero overhead.
        if top_ids and self.config.graph_expand_enabled:
            expanded_ids, expanded_chunks = self._graph_expand(
                store, top_ids, all_chunks, question
            )

        # Merge expanded into ranked results, strictly bounded by top_k.
        # Expansion fills remaining slots up to k; never exceeds k.
        final_ids = ranked_ids[:k]
        for eid in expanded_ids:
            if len(final_ids) >= k:
                break
            if eid not in final_ids:
                final_ids.append(eid)
                reasons[eid] = "graph_expansion"
                # Merge expansion data into all_chunks for result building.
                if eid in expanded_chunks:
                    all_chunks[eid] = expanded_chunks[eid]

        # Return strictly at most top_k results.
        return [
            QueryResult(
                chunk_id=cid,
                file_path=all_chunks[cid].file_path,
                symbol_name=all_chunks[cid].symbol_name,
                symbol_type=all_chunks[cid].symbol_type,
                content=all_chunks[cid].content,
                score=round(scores.get(cid, 0.0), 4),
                start_line=all_chunks[cid].start_line,
                end_line=all_chunks[cid].end_line,
                search_quality=all_chunks[cid].search_quality,
                match_reason=reasons.get(cid, ""),
            )
            for cid in final_ids[:k]
            if cid in all_chunks
        ]

    def _graph_expand(
        self,
        store: IndexStore,
        top_ids: list[str],
        all_chunks: dict[str, SearchRow],
        _question: str,
    ) -> tuple[list[str], dict[str, SearchRow]]:
        """Expand top results along the ref graph.

        For each top result, look up its outgoing refs, resolve targets
        via symbol_lookup, and include chunks that aren't already in results.

        Does NOT mutate ``all_chunks`` — expansion data is returned in a
        separate dict that the caller merges explicitly (M-5 fix).

        @param store: Active IndexStore.
        @param top_ids: Top chunk IDs to expand from.
        @param all_chunks: Already-loaded chunk data (read-only here).
        @param _question: Original query (reserved for future score-gating).
        @returns: (expanded chunk IDs, expanded chunk data keyed by chunk_id).
        """
        existing = set(all_chunks.keys())
        expanded_ids: list[str] = []
        expanded_chunks: dict[str, SearchRow] = {}

        # Collect every ref target across the top results in encounter
        # order, then resolve them all in a single batched query instead
        # of one lookup_symbol round-trip per ref (N+1 fix).
        target_names: list[str] = []
        for cid in top_ids:
            for ref in store.get_refs_for_chunk(cid):
                target_names.append(ref.target_symbol)

        if not target_names:
            return [], {}

        targets = store.lookup_symbols(target_names, limit=_GRAPH_LOOKUP_LIMIT)
        for target in targets:
            tcid = target.chunk_id
            if tcid not in existing and tcid not in expanded_chunks:
                expanded_chunks[tcid] = target
                expanded_ids.append(tcid)

        return expanded_ids[:10], expanded_chunks  # Cap at 10 expansion slots.

    def status(self) -> IndexStatus:
        """Get index status information.

        @returns: IndexStatus with all metrics.
        @raises IndexNotFoundError: If no index exists.
        """
        store = self._enter_store()
        try:
            return self._collect_status(store)
        finally:
            self._store_rwlock.release_read()

    def _collect_status(self, store: IndexStore) -> IndexStatus:
        """Gather status metrics from an already-acquired store.

        @param store: Live IndexStore (read lock held by caller).
        @returns: IndexStatus with all metrics.
        """
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
