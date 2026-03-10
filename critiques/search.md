# Search & Retrieval Pipeline — Adversarial Critique

---

## 1. Voyage Code 3 Embedding Model + 1024 Dimensions

**The decision**: Voyage Code 3 at 1024 dimensions, batched in groups of 128, with no alternative model path.

**What's wrong**:

Voyage Code 3 is a strong choice on the MTEB code retrieval benchmarks as of late 2024, but several compounding problems undermine "best":

**Model lock-in is not acknowledged as a risk.** The plan hard-codes Voyage semantics into the schema itself (`embedding float[1024]` in `vec_chunks`). Switching models requires a full rebuild — not just a config change. The plan mentions `embedding_dimensions` in meta and validates it at open, which is good, but the real cost of switching is never surfaced. Voyage has already made one breaking API change between v2 and v3; betting the schema on a single vendor's versioning with no migration path beyond "rebuild everything" is fragile.

**1024 dimensions is an arbitrary middle ground that may be wrong in both directions.** Voyage Code 3 supports output truncation to 256, 512, 1024, or 2048 dimensions. The plan uses 1024 with zero empirical justification. For a local SQLite deployment with sqlite-vec doing exact cosine search (no ANN approximation), the dimensionality directly multiplies storage cost: 1024 × 4 bytes × 100k chunks = ~400MB of raw vector data. The plan acknowledges ~400MB for vectors in comments but doesn't question whether that's necessary. Meanwhile, Matryoshka representation learning (MRL) — which Voyage Code 3 supports — means 512d vectors often achieve 95%+ of the retrieval quality of 1024d for code retrieval. The plan doesn't test this tradeoff at all; it just picks 1024 and moves on.

**No local model path exists even as a fallback.** For users who can't or won't send proprietary code to Voyage's API (enterprise, air-gapped, regulated industries), there's nothing. CodeBERT, StarEncoder, Nomic-Embed-Code, and Jina Code v2 (768d, Apache 2.0, runs on a MacBook M2) all exist as ONNX-exportable alternatives. The plan already handles "Voyage down" with FTS-only degradation — but a local embedding model would be a dramatically better degradation path. This isn't a nice-to-have; for the enterprise developer who is the target MCP user, it's a blocker.

**The comparison field is moving fast and the choice is underdocumented.** As of early 2026, OpenAI `text-embedding-3-large` with `dimensions=1024` competes directly with Voyage Code 3 on code retrieval and is cheaper at scale ($0.00013/1K tokens vs Voyage's ~$0.01/1M = $0.00001/1K — actually Voyage is cheaper, but the gap is narrower than assumed). Cohere Embed v4 (released mid-2025) has strong multilingual code performance and notably produces int8-quantizable embeddings natively, which could cut sqlite-vec storage by 4×. The plan doesn't acknowledge any of these as evaluated-and-rejected; it just says "best code embeddings" without citing evidence.

**Concrete alternatives with tradeoffs**:
- **512d Voyage Code 3 via MRL**: Half the storage, ~3% quality loss on code retrieval benchmarks. Should be the default with 1024d as opt-in. Zero vendor switching cost.
- **Nomic Embed Code (local, ONNX)**: 768d, runs locally, no API cost, ~Apache 2.0. Roughly 10-15% worse than Voyage on hard code retrieval but zero latency and zero data exposure. Should be the FTS degradation fallback, not keyword-only.
- **Cohere Embed v4 with int8 quantization**: Reduces vector storage by 4× at ~1-2% recall cost. Important for repos with 500k+ chunks.

**Specific recommendation**: Add a `local` embedder using `onnxruntime` + Nomic Embed Code as the graceful degradation path instead of FTS-only. Let users configure dimension via `embedding.dimensions` with 512 as default, 1024 as quality mode. Document why Voyage Code 3 was chosen over alternatives.

---

## 2. Cohere Rerank 3.5

**The decision**: Cohere Rerank 3.5 over the merged top-50, returning top-8.

**What's wrong**:

**The reranker is the highest-latency single point in the query path, and it's the one users feel most.** A Cohere Rerank 3.5 call over 50 documents takes ~400-800ms on a good day, often 1-1.5s. Combined with the Voyage embed call (~150-300ms), the user waits 600ms-2s just on API calls before seeing any result. The plan quotes "~3s" for a query and doesn't break this down — the reranker alone is almost certainly 50-70% of that time.

**Cohere Rerank 3.5 is a cross-encoder, but so are cheaper alternatives.** The plan doesn't evaluate:
- **Voyage Rerank 2** (released Q4 2024): Slightly better on code-specific reranking benchmarks, similar latency, different vendor (reducing single-vendor dependency — you already use Voyage for embed, so a consolidated billing relationship could cut admin overhead).
- **Jina Reranker v2** (late 2024): Available as both API and local ONNX model. The local path is critical: a MacBook M2 runs Jina Reranker v2 over 50 chunks in ~80-150ms — faster than a network round-trip. This is the most significant gap in the plan.
- **BGE-Reranker-v2-M3**: Best open-weight reranker on BEIR as of early 2025, runs on-device via ONNX, 568MB model weight. At 50 docs of typical code chunk length (~200-500 tokens), local inference is ~100-200ms on CPU, ~30-50ms on M-series GPU. This obliterates the Cohere latency argument.

**The "skip reranking, use RRF" fallback is actually not bad**, but the plan frames it as a degradation rather than asking whether reranking adds enough value to justify the cost and latency. For queries where vector and FTS agree (both rank the same chunks highly), the reranker adds little. For queries where they disagree, the reranker matters. The plan never measures this split.

**Vendor concentration**: Using Voyage for embeddings AND Cohere for reranking means two external API calls per query, two API keys to manage, two points of failure, and two billing relationships. The plan's "graceful degradation" section handles each independently, but the combined failure probability (Voyage down OR Cohere down) is higher than either alone.

**Concrete alternative with tradeoffs**:
- **Local Jina Reranker v2 via ONNX**: 80-150ms local inference, zero API cost, no data exposure. Slightly behind Cohere on general benchmarks but comparable on code. This should replace the "skip reranking" fallback when Cohere is down — not RRF scores.
- **Voyage Rerank 2**: Better code-specific performance, consolidated vendor, similar API structure — easy drop-in.
- **Conditional reranking**: Only invoke the reranker when vector and FTS disagree significantly (measure rank-correlation of their top-10; if Kendall's τ > 0.8, skip rerank). This could cut reranker calls by 30-40% at minimal quality cost.

---

## 3. Reciprocal Rank Fusion (k=60)

**The decision**: RRF with k=60 as the merge strategy for vector, FTS, and symbol results.

**What's wrong**:

**k=60 is not a "standard constant" — it's a historically suggested default** from the original Cormack & Clarke (2009) paper for web search result fusion. It has never been empirically validated for code retrieval. Code search has different rank distribution properties than web search: BM25 scores for exact symbol matches (e.g., `AuthService`) are extremely high relative to semantic misses, while vector similarity for natural language intent queries (e.g., "how does authentication work") compresses scores into a narrow 0.7-0.85 cosine range. The k=60 constant determines how aggressively RRF penalizes low ranks. For a list of 30 vector results, `1/(60+1)` through `1/(60+30)` spans a very narrow range. k=60 was designed for lists of hundreds to thousands of documents; at 20-30 results, it under-differentiates. **A lower k (e.g., 10-20) would give more spread to the rank differences that actually exist in these short lists.**

**RRF treats all retriever lists as equally reliable**, which is wrong for this system. A symbol exact match is categorically more reliable for symbol-lookup queries than a vector result; a BM25 match on a precise identifier is more reliable than a cosine similarity of 0.71 for a generic semantic query. The plan partially compensates with "symbol exact matches get a bonus: rank=1 regardless of position" — but that's a hack layered on top of a fundamentally flat fusion, not a principled solution.

**No learned weights**: The system has enough query structure (classified as symbol vs. semantic) to apply retriever-specific weights. For symbol queries, the FTS/symbol retriever should dominate; for semantic queries, the vector retriever should dominate. A trivial learned fusion — even just two hardcoded weight sets indexed by query class — would outperform flat RRF on classified queries. CombMNZ (sum of normalized scores × number of retrievers that returned a result) is empirically competitive with RRF and handles score normalization differently, potentially better for this score distribution.

**The deduplication strategy isn't specified clearly.** When the same chunk appears in both the vector and FTS lists, RRF sums their `1/(k+rank)` scores. But what happens to the score value reported to the reranker? The plan says "keep the RRF sum" — fine, but the reranker doesn't use this score; it re-scores everything. So the RRF score only matters for the 50-to-8 cut when the reranker is down. In the happy path, RRF determines the 50-candidate slate but then the reranker takes over. This means the quality of RRF only meaningfully affects quality when Cohere is down — a case that probably gets less attention in testing.

**Concrete alternatives with tradeoffs**:
- **RRF with k=10**: Better rank discrimination for short candidate lists. Test on your actual query distribution. Near-zero implementation change.
- **Weighted RRF by query class**: `score = α·(1/(k+rank_vector)) + β·(1/(k+rank_fts)) + γ·(1/(k+rank_symbol))` where (α,β,γ) = (0.7,0.3,1.0) for symbol queries and (0.9,0.4,0.1) for semantic queries. Cheap to implement, meaningfully better.
- **CombMNZ with min-max normalization**: Better theoretical grounding for heterogeneous score distributions. More complex to implement (~30 extra lines) but no additional dependencies.

---

## 4. Candidate Counts (Vector top-30, FTS top-20, Cap at 50, Return 8)

**The decision**: Retrieve 30 vector candidates + 20 FTS candidates, merge and cap at 50, rerank to 8.

**What's wrong**:

**These numbers are not derived from anything — they're plausible-sounding guesses.** The plan offers no empirical justification: no recall@50 measurement, no analysis of how often the "right" answer falls outside top-30 vector or top-20 FTS, no study of reranker accuracy degradation as candidate set grows. This isn't a design decision; it's a guess wearing design clothing.

**The asymmetry between vector (30) and FTS (20) is questionable.** The rationale given is "semantic search has broader recall but noisier precision" vs "keyword matches are more precise but narrower recall." This is true for natural language queries against document corpora, but for code search it's often inverted: a precise identifier like `process_payment` gets a near-perfect BM25 match and should dominate, while the vector search for the same query may return 20 semantically similar but wrong functions. For *semantic* queries ("how does error handling work"), vector dominates. The fixed asymmetry 30/20 is wrong for at least half the query types.

**Returning top-8 is too conservative.** The primary consumer is an LLM context window (via MCP), and modern models (Claude 3.5 Sonnet, GPT-4o) have 200k token context windows. Eight code chunks averaging ~300 tokens each = 2400 tokens, which is less than 1.5% of available context. The plan's `sr ask` CLI prints them to a terminal, where 8 is reasonable for human consumption — but the MCP path should probably default to 12-15, or let the LLM request more. The hard-coded top_k=8 default conflates CLI ergonomics with API ergonomics.

**The 50-candidate cap is almost certainly binding at the wrong time.** After RRF dedup, having more than 50 unique chunks from 30+20=50 raw candidates is only possible if vector and FTS have zero overlap — which happens when the query is in a regime where the two retrievers strongly disagree. That's exactly when you *want* more candidates for the reranker, not fewer. A cap of 50 on inputs of 50 is effectively no cap at all in the common case, and a too-tight cap in the edge case where it would matter.

**The symbol search result count is not specified.** "Exact match on symbol_name, if applicable" — how many symbol results? The plan doesn't say. If there are 5 classes named `BaseModel` across the codebase, do all 5 enter the candidate pool? If so, what rank? The symbol path interacts with RRF through the "rank=1 bonus" hack, but the number of results from the symbol index is undefined.

**Concrete recommendations**:
- **Profile actual recall@50 on your own fixture repos** before shipping. This is a 2-hour exercise that would validate or invalidate every number here.
- **Make candidate counts query-class-aware**: symbol queries → vector-20/FTS-30/symbol-all; semantic queries → vector-40/FTS-15/symbol-0.
- **MCP default top_k=12, CLI default top_k=8**: Different consumers have different ergonomics.
- **Cap at 60, not 50**: The extra 10 candidates cost ~2% more reranker time and meaningfully improve recall@8.

---

## 5. Query Classification Heuristic (PascalCase/snake_case Detection)

**The decision**: A regex heuristic on PascalCase and snake_case patterns determines whether to include symbol search.

**What's wrong**:

**This is maximally naive and the plan knows it but doesn't address the consequences.** PascalCase detection will fire on natural English words like "JavaScript", "TypeScript", "StackOverflow", "GitHub", "PayPal" — all of which appear constantly in code comments and developer queries. A question like "How does TypeScript handle module resolution?" would trigger symbol search for "TypeScript" and waste a retrieval slot on a symbol that doesn't exist (or worse, matches an unrelated symbol).

**It misses real symbol queries that don't use naming conventions.** `process_payment` in snake_case is correctly caught, but `PROCESS_PAYMENT_MAX_RETRIES` (screaming snake case for constants) may or may not be caught depending on the regex. `authHandler` (camelCase, common in JS/TS) won't trigger PascalCase detection. Single-word symbols (`handler`, `user`, `auth`) are never caught regardless of how centrally important they are to the query.

**The classification result affects retrieval strategy but not weighting.** The plan says "include symbol search" vs "don't include symbol search" — binary. It never considers: what if the query is *mostly* semantic but mentions a symbol? ("How does the `AuthService` interact with the session store?"). You should run symbol search AND semantic search, but weight symbol results differently.

**No feedback loop**: If the classifier misfires, there's no way to know. The reranker might recover by deprioritizing bad symbol results, but you're wasting a reranker slot either way. And there's no logging of which classification was applied per query.

**The bar for "should we use an LLM classifier?" is higher than the plan acknowledges.** The plan dismisses this without saying so — it just doesn't mention it. For a query classifier that runs on every search, a 7B-parameter local model (Phi-3 Mini, LLaMA-3.2-3B via ONNX/llama.cpp) could classify queries in ~50ms locally with much higher accuracy. But there's a simpler middle ground: a **tiny trained classifier** (logistic regression on simple token features) that doesn't require an LLM at all. Train on 500 labeled queries (takes an afternoon), get 95%+ accuracy, run in <1ms. The current heuristic probably gets 70-80% on real developer queries.

**Better yet**: stop treating classification as binary. Produce three scores: `(symbol_weight, semantic_weight, keyword_weight)` and use these to weight the RRF fusion. The heuristic detection of PascalCase becomes a soft signal that boosts symbol weighting, not a hard switch.

**Concrete alternative**:
- **Multi-signal scoring**: +0.3 to symbol_weight for each PascalCase token, +0.2 for each snake_case-with-dot token, -0.1 for question words ("how", "why", "what"), +0.1 for camelCase. Normalize. Use as RRF weights. Takes 20 lines. Much more robust than binary switch.
- **Add a "dot-qualified" detection rule**: `foo.bar`, `module.ClassName` patterns are much higher precision than raw PascalCase.

---

## 6. Graph Expansion (1 Level, Cap at 5 Per Result, 15 Total)

**The decision**: After retrieval, pull direct references (1 level, max 5 chunks per result, 15 total expanded chunks).

**What's wrong**:

**One level of expansion is often useless for the questions this system is designed to answer.** "How does the payment flow work?" — the answer likely involves: `process_payment()` → calls `validate_card()` → calls `charge_stripe()` → handles `StripeError`. One level of expansion from `process_payment` gets you `validate_card`, but not `charge_stripe`. The flow is 3 levels deep; you've surfaced level 2 and stopped. The user still doesn't have the answer.

**The cap of 5 per result and 15 total is too aggressive given top_k=8.** You have 8 top results, each capped at 5 expansions = potential 40 expansions, but globally capped at 15. This means if the first 3 top results all expand to 5 chunks, the remaining 5 top results get zero expansion. The cap strategy is FIFO on the results list — the 4th-ranked result might be the one that most needs context from its callees, but it gets nothing. This is probably wrong.

**The expansion uses the refs table (callers/callees), but the direction is not specified.** When you "expand" a result, do you pull what it *calls* (callees — to understand implementation) or what *calls it* (callers — to understand usage)? The answer depends on the query. "How does auth work?" → you want callees (what auth calls). "What calls process_payment?" → you want callers. The plan doesn't distinguish these; it just says "callers, callees, type definitions" — presumably pulling both directions. Pulling both from 8 results at 5 each before the 15-total cap is hit means the expansion is a noisy mixture of context types.

**Expanded chunks don't get reranked.** The top-8 are reranked by Cohere; the 15 expansion chunks are appended without reranking. This means the expansion results are ordered by their RRF score in the refs table (which is... undefined, since refs are resolved, not scored). The user sees 23 chunks where the last 15 are unranked filler. An LLM consumer receiving 23 chunks will treat them as roughly equal, which they aren't.

**The deduplication logic is correct** (expanded chunks that appear in top-8 are excluded once). This is a genuine good decision.

**Concrete alternatives with tradeoffs**:
- **2-level expansion, caller-only for "what calls X" queries, callee-only for "how does X work" queries**: Better coverage for flow questions. Requires query classification (which you already have a heuristic for).
- **Score-gate expanded chunks**: Only include expanded chunks that themselves have a vector similarity > 0.65 to the query. This filters noisy cross-references that happen to exist structurally but aren't semantically relevant. Costs one vector distance computation per expansion candidate (fast — already have the embeddings in sqlite-vec).
- **Re-score expanded chunks with Cohere Rerank**: Add them to the reranker input as a second tier (lower priority). The reranker already handles 50 docs; 15 more costs marginally more.
- **Allocate expansion budget proportionally to result rank**: Top result gets 5 expansions, second gets 4, third gets 3... stopping at 15. Ensures the most relevant result gets the most context.

---

## 7. Graceful Degradation — FTS-Only When Voyage Is Down

**The decision**: When Voyage API is unavailable, fall back to FTS5/BM25 + symbol search only.

**What's wrong**:

**FTS-only is not "graceful degradation" — it's a qualitatively different product.** The system's core value proposition is semantic search (finding "how auth works" even when the word "auth" doesn't appear in the function body). FTS-only can't do this. For the most common developer queries — conceptual questions, "what does X do", intent-based search — FTS-only returns results nearly as bad as `grep`. Calling this "graceful degradation" sets a misleading expectation.

**The backfill path compounds the problem.** If Voyage is down during `build()`, chunks are stored without embeddings. When Voyage comes back, `refresh()` auto-backfills — but only for chunks stored *after* the outage began. If the user built a full index during a Voyage outage, all chunks lack embeddings and `refresh()` doesn't know which ones to backfill (the backfill query looks for chunks with no `vec_chunks` row, which is correct, but it's silent about how many are affected). The user may not realize their index is in FTS-only mode until they run `sr status` or notice poor query quality. **The `sr status` output should prominently flag vector coverage percentage** (chunks with embeddings / total chunks) — this is not mentioned in the plan.

**No local embedding model fallback means "Voyage down" permanently degrades a repo** until the user notices and runs `sr backfill` manually. In an MCP workflow where the user is relying on source-recall for codebase Q&A, a 2-hour Voyage outage could leave a frequently-updated repo partially unindexed for days.

**The "skip reranking" fallback when Cohere is down is actually fine**, because RRF scores are a reasonable proxy for relevance when the two retrievers mostly agree. But the "skip embeddings" fallback is not fine — it removes the system's primary differentiator.

**Concrete alternative**:
- **Local embedding model as degradation path**: When Voyage API fails, switch to a local ONNX model (Nomic Embed Code or similar). FTS-only should be the last resort, only when even the local model fails (e.g., model files not present). The plan already tracks `available()` on the embedder — a `LocalONNXEmbedder` implementation fits this interface perfectly. The storage overhead of supporting a local model path is one additional module (~100 lines) and a soft dependency on `onnxruntime`.
- **At minimum, surface vector coverage prominently in `sr status`**: "Coverage: 47,231/50,000 chunks (94.5%). 2,769 chunks missing vectors — run `sr backfill`." Currently status returns `indexed_at, commit, file_count, chunk_count, stale_files, db_size_bytes, ast_parsed_files, text_fallback_files` — vector coverage is absent.

---

## 8. No Query Rewriting (No Expansion, No Synonyms, No Decomposition)

**The decision**: Queries are passed to the embedding and FTS pipeline verbatim, with no rewriting.

**What's wrong**:

**This is the most significant gap in retrieval quality for a code search system.** Consider these real developer queries and their failure modes:

- "how does authentication work" → FTS misses because no function is named `authentication`; vector search may surface the right auth module, but only if the embedding space aligns well
- "where is the database configured" → FTS scores "database" and "configured" separately; no BM25 document contains both; the config file is probably named `db.py` or `settings.py`
- "what happens when a user logs in" → zero overlap between query tokens and likely code tokens ("user", maybe, but "logs in" maps to `authenticate`, `create_session`, `issue_token`)
- "find the stripe integration" → correct if someone named something `stripe`; completely broken if it's called `payment_gateway` or `billing_service`

**BM25 cannot handle vocabulary mismatch.** This is the fundamental limitation of keyword search, and the plan never acknowledges it as a retrieval quality problem — only as motivation for semantic search. But for *conceptual* queries (the hardest and most valuable queries), even the vector search may struggle if the embedding model's training distribution doesn't include your specific domain terminology.

**No query expansion means compound queries are broken.** "How does the payment retry logic interact with the notification system?" is a multi-hop query. The embedding of this full sentence may not surface good results for either the payment retry module OR the notification module — it may find something in the middle that matches neither well. Decomposing to ["payment retry logic", "notification system"] and running two retrieval passes would dramatically improve coverage. The plan doesn't even mention this as a known limitation.

**The counterargument the plan would make** — that synthesis is the caller's job, not the library's — does not apply here. Query rewriting is a retrieval technique, not synthesis. It happens before retrieval, it doesn't produce a summarized answer, and it's entirely within the scope of a retrieval library. The plan correctly rejects LLM synthesis but should not conflate that with query reformulation.

**Concrete alternatives with tradeoffs**:
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical code snippet that would answer the query (using a fast LLM call), embed *that*, use as the vector query. Significant quality gains for semantic queries. Cost: one LLM call per query (~$0.001 with GPT-4o-mini or Claude Haiku). Latency: +300-500ms. The plan's "retrieval only, no LLM" philosophy would oppose this — but HyDE is retrieval machinery, not synthesis.
- **Vocabulary expansion via code glossary**: Maintain a small static mapping (or BM25 top-terms per symbol) that expands "authentication" → ["auth", "login", "jwt", "session", "token", "authenticate"]. Add these as FTS OR terms. Costs ~50ms per query. No API dependency. Works only for common patterns but covers a lot of real queries.
- **Query decomposition for "and" queries**: Detect connective words ("and", "how X interacts with Y") and issue parallel sub-queries. Merge results with additional RRF pass. No LLM needed for simple decomposition heuristics.
- **At minimum: document this gap.** The plan should explicitly state "we don't do query expansion; for multi-hop or vocabulary-mismatch queries, results may be poor."

---

## 9. The Retrieval-Only Philosophy (No LLM Synthesis in the Library)

**The decision**: The library returns ranked chunks; callers synthesize. No LLM calls for synthesis. The plan is explicit: "the library does retrieval, not synthesis — that's the consumer's job."

**What's wrong** (and what's right):

**The boundary is correctly drawn for the wrong reason.** The plan draws the line at "retrieval vs synthesis" — retrieval stops at returning chunks. This is architecturally clean and the right call. But the *reason* given is separation of concerns and letting consumers control LLM choice. The plan doesn't acknowledge that the boundary also prevents several *retrieval-level* LLM uses that are completely legitimate: query rewriting, HyDE, classification, and expansion scoring. By conflating "no LLM synthesis" with "no LLM in the pipeline," the plan leaves significant retrieval quality gains on the table without even acknowledging the tradeoff.

**The philosophy is correct for the MCP use case**, where tool-proxy is already running inside an LLM-orchestrated workflow. The LLM calling `source_recall:ask` can synthesize the returned chunks. Adding synthesis inside source-recall would create redundant synthesis (the outer LLM already synthesizes). For MCP, retrieval-only is exactly right.

**The philosophy is incomplete for the CLI `sr ask` use case.** The plan says "users pipe results to an LLM if they want a synthesized answer." This is a design cop-out. The CLI already depends on Rich for formatting — it's not a "dumb" output pipe. A `sr ask --explain` flag that passes retrieved chunks to a configurable LLM for a brief (3-sentence) synthesis would make the CLI dramatically more useful for the stated use case of "AI coding tools." This doesn't compromise the SDK or MCP boundaries at all.

**The deeper issue**: retrieval-only is the right *library* philosophy, but it means the system has no way to know whether its retrieval is actually good. Without synthesis, there's no end-to-end signal. You get 8 chunks — was the answer in there? You don't know. A synthesis step (even just "did these chunks answer the question — yes/no?") would enable basic quality measurement. Without this, the only way to evaluate retrieval quality is manual inspection of fixture outputs, which the plan relies on but doesn't systematize.

**Concrete recommendations**:
- **Maintain the retrieval-only library boundary** — this is correct.
- **Add `--explain` to CLI**: Optional, configurable LLM (via `SR_LLM_API_KEY` + `SR_LLM_MODEL`). 3-sentence synthesis over retrieved chunks. This is a CLI concern, not a library concern, and doesn't violate the library's philosophy.
- **Add `include_relevance_reason` to QueryResult**: A short (1-sentence) string explaining why the chunk was ranked where it was (e.g., "vector similarity 0.89 + FTS match on 'process_payment' + graph expansion from direct caller"). No LLM needed — deterministic from retrieval metadata. This helps consumers and helps debugging.
- **Track end-to-end retrieval quality** via a `tests/live/` benchmark: 20 representative queries against the fixture repos with known ground-truth results (chunk IDs that should appear in top-8). Run this as `--run-live` test to catch quality regressions when you change RRF params, switch embedding models, or tune candidate counts. The plan has `test_voyage_cohere.py` as a live test but doesn't specify what it measures.

---

## Summary: Priority of Issues

| Issue | Severity | Effort to fix |
|-------|----------|---------------|
| No local embedding fallback (FTS-only degradation is misleading) | **High** | Medium (add ONNX embedder) |
| No query rewriting/expansion (vocabulary mismatch silent failures) | **High** | Low-Medium |
| k=60 RRF constant unjustified for short lists; no weighted fusion | **Medium-High** | Low |
| 1024d dimension unjustified; 512d likely sufficient | **Medium** | Low (config change) |
| Graph expansion: 1-level too shallow; budget allocation wrong | **Medium** | Medium |
| Query classifier: PascalCase fires on English words; misses camelCase | **Medium** | Low |
| Reranker latency unacknowledged; no local reranker path | **Medium** | Medium |
| Candidate counts unjustified; top_k=8 conflates CLI and MCP ergonomics | **Low-Medium** | Low |
| Retrieval-only boundary correct but prevents query-level LLM use | **Low** | Low (architectural clarity) |
| Vector coverage not surfaced in `sr status` | **Low** | Low |

The three biggest retrieval quality gaps in order: (1) vocabulary mismatch from no query expansion, (2) FTS-only degradation masking as "graceful", (3) flat RRF with magic constants instead of query-class-aware weighted fusion. Everything else is tuning. Fix these three and the system goes from "plausible" to "demonstrably correct."
