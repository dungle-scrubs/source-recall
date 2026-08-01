# Adversarial Critique: Parsing & Indexing Strategy

---

### 1. tree-sitter + tree-sitter-language-pack — ABI & Parser Choice

**The decision**: Use `tree-sitter>=0.24,<0.25` + `tree-sitter-language-pack>=0.24,<0.25` pinned to the same minor version to avoid ABI segfaults.

**What's wrong / risky**:

**ABI pinning is insufficient.** The plan says "ABI compat is minor-version-scoped (0.24.x), not patch-level — confirmed by tree-sitter docs." This is optimistic. Tree-sitter's Rust/C ABI has historically broken *within* minor versions before the project stabilized. More critically, `tree-sitter-language-pack` bundles pre-compiled `.so` files built against a specific tree-sitter C ABI version. The `>=0.24,<0.25` pin allows `0.24.0` of the core and `0.24.9` of the language pack — different patch levels. On Python 3.12 on Linux (where manylinux wheels embed the ABI into the `.so` name), this can produce immediate segfaults if the pre-compiled grammar `.so` has a different `TREE_SITTER_LANGUAGE_VERSION` constant than what the Python bindings expect. The plan says "if patch-level ABI breaks surface, tighten to exact `==` pins" — but this is a reactive mitigation, not a proactive one. The symptom is a segfault, not a graceful parse failure. It will manifest differently on macOS vs Linux due to dylib loading differences.

**Language-pack is a heavy dependency with zero selectivity.** The plan acknowledges ~15MB for 20 grammars when 3 are used. More importantly, `tree-sitter-language-pack` is maintained by a different team than `tree-sitter` and often lags by several weeks on version bumps. When tree-sitter releases `0.25`, the language pack may not immediately follow, forcing the project to hold back on tree-sitter updates. It's also a single point of failure: if the language pack maintainer abandons it, you're stuck.

**tree-sitter has real-world error recovery gaps.** tree-sitter's error recovery is designed for IDE use (partial re-parse on keystrokes), not for static analysis. It never *fails* — it produces an ERROR node and continues. This sounds good (no crashes) but means the plan's "if tree-sitter returns an error node covering the entire file" heuristic is under-specified. In practice, tree-sitter can produce a partially-invalid parse tree with scattered ERROR nodes that *looks* valid but yields wrong chunk boundaries. A function whose signature contains an unsupported syntax pattern will have its body misattributed to the wrong parent node. The plan has no validation that error nodes aren't scattered throughout an "AST-parsed" file — `parse_mode = 'ast'` in `file_hashes` can be misleading.

**The alternatives rejected without analysis**:

- **LSP (Language Server Protocol)**: Dismissed implicitly. For Python specifically, `pylsp` or `pyright`'s language server provides *semantically accurate* symbol resolution — it knows that `from foo import bar` where `bar` is a re-export resolves to the actual definition. tree-sitter only sees tokens/syntax; it cannot resolve that `from . import utils` in a deeply nested package resolves correctly without additional path computation. LSP is overkill for MVP, but the plan should acknowledge this gap explicitly. **Tradeoff**: LSP requires language servers running per-repo, introduces process management complexity, and is slow (seconds for initial analysis). tree-sitter is correct to use for MVP.
- **`ast` module for Python**: Python's built-in `ast` module produces a semantically richer tree than tree-sitter for Python — type comments, constant folding awareness, etc. It's zero-dependency. **Tradeoff**: only covers Python; not worth maintaining two parse paths when tree-sitter covers multiple languages.
- **SWC for TypeScript**: SWC's TypeScript parser (via `swc-node` or the Python-accessible `swc_py`) is the parser TypeScript itself uses-adjacent tooling prefers. It handles all TS syntax including decorators (stage 3 vs stage 2 semantics), `satisfies` operator, const type parameters — features that tree-sitter's TS grammar may lag on. **Tradeoff**: SWC is Rust-based with no first-class Python bindings, requires subprocess or FFI, adds complexity.

**Concrete recommendation**: Pin to exact `==` versions (`tree-sitter==0.24.4`, `tree-sitter-language-pack==0.24.4`) from day one. Add an integration test that imports the language pack and runs a known parse to catch ABI mismatches at CI time, not at user runtime. Add an error node density check: if >10% of nodes in a parse tree are ERROR type, demote to `parse_mode='partial_ast'` and fall back to text chunking for that file. This is a 5-line addition to the parser that prevents "ghost" AST-parsed files that are actually garbage-chunked.

---

### 2. Chunking Granularity — Function/Class/Method Level

**The decision**: Chunk at function, class (shell), and method level for supported languages.

**What's wrong / risky**:

**The granularity is wrong for retrieval, not just for representation.** Function-level chunking assumes the *function* is the unit of semantic meaning. This is true for "what does `validate_user` do?" but breaks down in several common patterns:

1. **Co-located helper functions**: A file with 10 small 3-line functions all serving one larger workflow gets chunked into 10 separate embeddings. Each embedding is too short to carry meaningful semantic signal — Voyage Code 3 will produce near-identical vectors for small helper functions that are semantically similar but used differently. The retrieval recall for "how does the payment flow work" will surface individual tiny functions rather than the coherent flow.

2. **Large coordinating functions**: A 300-line orchestrating function that calls 15 sub-functions will be sub-chunked (due to the 6000-char cap) into several chunks, each without the call-flow context of its position within the function. The sub-chunking adds the function signature as a prefix, but "def process_checkout(cart, user, payment_method)" doesn't tell you whether you're looking at the error-handling section or the success path.

3. **Class shell vs. methods**: Splitting class declarations into "shell + individual methods" means the class-level docstring, class variables, and `__init__` are in one chunk, while each method is in another. A query about "how UserService is initialized" may miss that `__init__` calls a private `_setup_db_connection` method — the inheritance context is in the shell chunk, the initialization logic is in the method chunk, and the connection logic is in a third. The cross-reference extraction is supposed to bridge this, but cross-references only help if they're resolved (which is best-effort).

**What the research says**: The seminal work on chunking for code retrieval (CodeSearchNet, CodeBERT, UniXcoder) primarily benchmarks at function level — so the choice is not unreasonable. However, newer work (RepoFusion, ReACC) shows that *multi-granularity* retrieval — indexing at both function AND file level — significantly improves recall for questions about architectural patterns vs. implementation details. The plan only uses function-level chunking.

**Alternative — Hierarchical multi-granularity chunking**: Index each class as a standalone chunk *in addition to* its methods. Index each file as a standalone chunk *in addition to* its functions. Use a `granularity` field in the `chunks` table. During retrieval, deduplicate: if both a function and its containing class are retrieved, score-weight toward the more specific one (function), but use the class chunk for context expansion. **Tradeoff**: ~3-4x more chunks (larger DB, higher embedding cost on initial index), more complex deduplication in retrieval. For a 5k-chunk repo this becomes ~15-20k chunks. The embedding cost increases from ~$0.03 to ~$0.10 for full index — still trivial.

**Alternative — Sliding window over file content**: Rather than AST-aware chunking, use 512-token windows with 50% overlap. Simple, language-agnostic, preserves context continuity. **Tradeoff**: Loses all structural signal (symbol names, types), makes cross-reference extraction impossible, worsens symbol search precision. Clearly inferior for a code-intelligence tool, but worth noting that the overlapping window is used by several production RAG systems for exactly the context-continuity reason the plan ignores.

**The real gap**: The plan has no overlapping chunks. When a function is sub-chunked, sub-chunk 1 has the signature + first 1500 tokens, sub-chunk 2 has tokens 1501-3000 (with only the signature prepended as context). There's no overlap window between sub-chunks. For a complex function, the logic at the boundary between sub-chunks 1 and 2 is contextually orphaned — a query targeting that boundary logic will retrieve a chunk starting at an arbitrary boundary point. **Concrete fix**: when sub-chunking, include a 5-10 line overlap between consecutive sub-chunks. The cost is ~5% extra tokens stored; the benefit is eliminating the context-cliff at sub-chunk boundaries.

---

### 3. 6000-Character Chunk Cap (~1500 Tokens)

**The decision**: Cap at 6000 characters using a 1 char ≈ 4 chars/token heuristic, targeting ~1500 tokens. "Sits 10x below Voyage Code 3's 16k token input limit."

**What's wrong / risky**:

**The heuristic is wrong in the stated direction.** The plan claims the cap is "deliberately conservative" but the math is inverted. If 1 token ≈ 4 characters (a Python/English estimate), then 6000 chars ≈ 1500 tokens. The plan then says this is "10x below" the 16k limit. 1500 is not "10x below 16k" — it's 10x *smaller*. If the intent is to use 1/10th of the available context, fine — but calling it "conservative" while meaning "small" is muddled reasoning. Voyage Code 3 is explicitly designed for long-context code embedding; using only 9% of its capacity means you're leaving significant semantic richness on the table, particularly for complex class methods that need their full context to be differentiated from similar-looking code elsewhere.

**The heuristic is systematically wrong for TypeScript.** The plan acknowledges "minified JS ~2 chars/token." TypeScript generic types and decorators have a *higher* token density than prose: `export const handler: RequestHandler<Params, ResBody, ReqBody, Query>` tokenizes at roughly 2.5 chars/token for the generic portions. A 6000-char TypeScript function with heavy generics and decorators is actually ~2400 tokens, not 1500. This doesn't exceed Voyage's limit, but it does mean your token estimates for cost calculation (the ~$0.03 figure) could be off by 60% for TypeScript-heavy repos.

**6000 chars is too small for realistic class methods.** A Java-style (or TypeScript verbose-style) controller method with:
- Method signature with 5+ parameters
- JSDoc comment block (50-200 lines in enterprise code)
- Input validation (20-40 lines)
- Business logic (50-100 lines)
- Error handling (20-50 lines)

...easily reaches 400-600 lines = 12,000-18,000 characters. This gets sub-chunked into 3 pieces, each with only the method signature prepended. A query about the error handling behavior retrieves sub-chunk 3, which has zero context about *why* those particular errors are handled that way (the reason is in the business logic in sub-chunk 2).

**What the research says**: Voyage AI's own benchmarks for Voyage Code 3 suggest optimal performance at 512-2048 tokens for retrieval tasks — shorter chunks retrieve more precisely but miss context; longer chunks provide more context but dilute the embedding's specificity. The 1500-token target is reasonable, but it should be a *soft* target (split at natural boundaries near 1500 tokens) rather than a hard character-count ceiling applied mechanically.

**Concrete alternative**: Replace the character-count cap with an AST-aware splitting that prefers natural sub-block boundaries (at the granularity of try/except blocks, if/else branches, or comment-delimited sections within a function), targeting 1000-2000 tokens rather than a hard 1500-token cut. When no natural boundaries exist within a 100-line band around the target, fall back to line splitting. This requires a more sophisticated sub-chunker but produces semantically coherent sub-chunks. **Tradeoff**: significantly more complex sub-chunking logic; harder to reason about boundary cases; increases chunker.py from ~450 lines to ~600 lines.

---

### 4. Cross-Reference Extraction

**The decision**: Extract imports, function calls, inheritance relationships from AST. Record as `ref_type` ∈ {import, type_import, call, inherit, source}.

**What's wrong / risky**:

**Call extraction is under-specified.** The plan shows import/type_import/inherit extraction with concrete examples, but "call" extraction is mentioned in the ref_type enum without any specification of what constitutes a call. Does it track:
- Direct function calls: `foo(x)`?
- Method calls: `self.foo(x)` or `obj.method()`?
- Chained calls: `service.get_user(id).validate()`?
- Calls through variables: `handler = get_handler(); handler(req)`?

The chunking rules say "extract cross-references (imports, calls, source includes)" but the ref extraction section only documents imports and inheritance. This is a significant gap — without call extraction being fully specified, the "what calls X" query type (explicitly listed as a Phase 1 capability) has no implementation foundation. Dynamic dispatch and method chaining are essentially impossible to track correctly without type inference.

**Type references are entirely missing.** TypeScript code is saturated with type references that aren't imports:
- `const user: User = ...` — `User` is used as a type annotation
- `function process(req: Request, res: Response): Promise<void>` — 3 type references, none captured
- `type Handler = (event: Event) => void` — `Event` type reference not captured

These type references are critical for "what type is used for the request object" or "what implements the Handler interface" queries. The plan captures `type_import` (import-level) but not type-annotation-level references within function bodies and signatures. This means the graph edges are incomplete for TypeScript codebases.

**Dynamic imports are explicitly un-handled.** Modern TypeScript/JavaScript codebases use dynamic imports extensively:
- `const module = await import('./plugins/' + name)` — path not statically determinable
- `const { handler } = await import(config.handlerPath)` — path from config
- `React.lazy(() => import('./HeavyComponent'))` — lazy loading pattern
- `require(process.env.ADAPTER_MODULE)` — environment-based module selection

The plan handles `require('./foo')` → `ref(type=import, target=foo)`, but only static string arguments. Dynamic import patterns with variables or expressions are silently dropped — no warning, no partial capture. In a typical React/Node codebase, 10-30% of imports may be dynamic. The cross-reference graph will have systematic blind spots in exactly the places that matter most (plugin systems, lazy-loaded components, environment-configured adapters).

**Decorator usage is missing.** Python decorator usage creates critical implicit relationships:
- `@app.route('/users')` — links a function to a Flask/FastAPI app instance
- `@require_permissions('admin')` — links a function to a permissions system
- `@cached_property` — changes the semantics of what looks like a regular property
- `@pytest.fixture` — marks test infrastructure

Similarly for TypeScript decorators:
- `@Injectable()` — Angular/NestJS dependency injection
- `@Entity()`, `@Column()` — TypeORM schema definitions
- `@Controller('/users')` — NestJS routing

None of these are captured as cross-references. A query about "what endpoints are available" in a NestJS app will find nothing from the dependency graph — it requires embedding-level retrieval only.

**Concrete additions**: Add `ref_type = "decorator"` for Python decorators and TS decorator applications. Add `ref_type = "type_ref"` for type annotation references (parseable from tree-sitter's `type_annotation` nodes). For dynamic imports, add a `ref_type = "dynamic_import"` with `target_symbol = "<dynamic>"` as a sentinel — at least indicate that a dynamic dependency exists even if its target is unknowable. Add a `ref_type = "call"` specification with clear scope: track direct function calls and same-object method calls only; explicitly exclude chained calls and variable-dispatch (document this limitation rather than pretending calls are fully tracked).

---

### 5. symbol_lookup Resolution Algorithm

**The decision**: When multiple chunks define the same symbol, prefer: (a) same directory as source, (b) same top-level package, (c) alphabetical by file_path.

**What's wrong / risky**:

**Same-directory preference is wrong for the most common case.** In a monorepo or well-structured project, the convention is that you import from *other* directories (shared utilities, services, models) — not from within the same directory. If `src/auth/handlers.py` calls `validate_token(token)`, and there's also a `validate_token` in `src/auth/helpers.py` (same directory) AND in `src/security/tokens.py` (different directory), the algorithm will always prefer `src/auth/helpers.py` — even if `src/auth/handlers.py` explicitly imports from `src/security/tokens`. The resolution algorithm doesn't consult the *actual import statements* in the source file; it uses directory proximity as a proxy. For Python, this is particularly wrong because `from . import helpers` and `from security import tokens` are syntactically distinct — the resolution algorithm should use the import statements already captured in the `refs` table to resolve call targets, not directory proximity.

**Barrel files / re-exports break the algorithm completely.** Consider:
```
src/
  index.ts            (re-exports: export { UserService } from './services/user')
  services/
    index.ts          (re-exports: export { UserService } from './user')
    user.ts           (defines: class UserService)
```

`UserService` appears in `symbol_lookup` three times (one per file that mentions it as an export). The alphabetical tiebreaker resolves to `src/index.ts` if it sorts earlier — the barrel file, not the implementation. Searching "what is UserService" returns the re-export barrel, not the actual class definition. This is a fundamental failure mode for any TypeScript project using barrel-based exports, which is essentially all large TypeScript projects.

**Aliased imports are not handled.**
```typescript
import { UserService as US } from './services/user'
US.validate(user)
```
The call `US.validate(user)` generates a ref with target `US.validate`. The `symbol_lookup` table has no entry for `US` — it has `UserService`. The alias is not tracked anywhere. This ref will always resolve to NULL. Similarly for Python `import numpy as np` — `np.array(x)` will never resolve.

**Top-level package is ambiguously defined for flat structures.** "Same top-level package (first path component)" means: for a file at `src/auth/handlers.py`, the top-level package is `src`. For a file at `auth/handlers.py`, it's `auth`. For a flat project with all files at root level, every file is its own "top-level package" and the heuristic degrades to alphabetical immediately. Monorepos where everything lives under `packages/` or `apps/` will have a single top-level package that encompasses the entire monorepo — the heuristic provides zero disambiguation there.

**Concrete alternative**: Use import-graph-aware resolution. For each call ref in chunk C, check if C's file has an import statement (already in the refs table) that imports the target symbol. If a direct import exists, use that import's resolved file path to look up the symbol. This is O(n) per ref resolution but uses the explicitly declared dependencies instead of proximity guessing. Only fall back to directory proximity for calls to symbols that are never imported (same-file utility functions). **Tradeoff**: requires two-pass resolution (imports must be resolved before calls), adds ~50 lines to the resolution pass. For the barrel file problem specifically: add a `is_barrel = true` flag (already tracked) and deprioritize barrel files in `symbol_lookup` — always prefer non-barrel definitions.

---

### 6. Incremental Refresh via Git Diff

**The decision**: `git diff --name-only <last_indexed_commit>` (no HEAD — diffs against working tree). 500-commit fallback threshold. File hash comparison as supplemental.

**What's wrong / risky**:

**The 500-commit threshold is an arbitrary magic number with no analysis.** A 500-commit range in a fast-moving repo with small commits (e.g., squash-merge PR workflow with 2-3 files changed per commit) is fine. A 500-commit range in a repo with large refactoring commits (e.g., an automated `sed` replacement across 800 files) means `git diff` against commit 499 will already touch 90% of the repo — but the algorithm only falls back at commit 501. The threshold should be based on *changed file count*, not commit count. `git diff --name-only <last> HEAD | wc -l` gives the actual file change count in one command, and is more reliable than commit count for deciding whether incremental is worthwhile.

**Working tree diff approach has a race condition.** `git diff --name-only <last_indexed_commit>` (without HEAD) includes *unstaged working tree changes*. This is intentional per the plan — "catches uncommitted edits." But if the user is actively editing files while indexing runs, the file hash captured at the start of the refresh may not match the file content at the time of reading. The plan addresses this partially with the file hash comparison step, but the sequence is:
1. Get list of changed files from git diff
2. For each file, read content and compute hash
3. Compare hash to stored hash
4. If different, re-index

Step 2 happens at the time of reading — if the file is modified between step 1 and step 2, the content indexed is the "in-progress" version, not the committed version. For long refresh operations on large repos, this window can be minutes. The indexed chunk may contain half-edited code that is syntactically invalid. tree-sitter's error recovery will parse something, but the embedding of half-edited code is semantically poisoned until the next refresh. **This is probably acceptable** for an interactive tool, but should be documented as a known limitation.

**`git diff` against a non-ancestor commit is silently wrong.** After a rebase or interactive history rewrite, `last_indexed_commit` may no longer be an ancestor of HEAD. `git diff --name-only <orphaned_commit>` will either fail with a non-zero exit code (if the commit doesn't exist) or produce an incorrect diff (if the SHAs were reused/recycled by force-push). The plan mentions that `file_hashes` comparison "catches edge cases like amended commits, rebases" — but this is only true if the file *content* changed. A rebase that rewrites commit metadata without touching file content will produce an incorrect `last_indexed_commit` with zero changed files detected. The index will report itself as fresh while the commit SHA is wrong. **Concrete fix**: before running `git diff`, verify `git merge-base --is-ancestor <last_indexed_commit> HEAD`. If it fails, skip the git diff step entirely and fall back to file hash comparison only. This is a two-command check that costs <5ms.

**Non-git mode staleness is purely mtime-based by omission.** The plan says "file hashes + mtime only" for non-git mode but the schema stores `content_hash` only — no mtime column. The text says "mtime" but there's nowhere to store it. If non-git mode re-hashes every file on every refresh (which is the only option without mtime), it becomes O(n) I/O for every refresh in a large non-git directory. Stated as "slightly slower" but 100KB max per file × 1000 files = 100MB of reads just to check staleness. An mtime cache column in `file_hashes` would allow skipping the hash recomputation for files with unchanged mtime, reducing this to O(n) stat() calls instead of O(n) reads.

---

### 7. Text Fallback for Unsupported Languages

**The decision**: Blank-line splitting for unsupported extensions (Go, Rust, Ruby, Java, SQL, YAML, Dockerfile, etc.). `symbol_type = "block"`, no symbol name, still embedded and searchable.

**What's wrong / risky**:

**Blank-line splitting is semantically incoherent for most languages.** Consider:
- **Go**: A struct definition followed by 6 methods typically has blank lines between methods but the struct and its methods form one semantic unit. Blank-line splitting will chunk the struct separately from its methods.
- **YAML/TOML config files**: Often have no blank lines at all (compact format) → entire file becomes one chunk. Or dense blank lines between every stanza → 30 tiny one-line chunks each with a single config key.
- **SQL**: A stored procedure with blank-line-separated sections (declarations, body, exception handling) gets split into 3 fragments, each meaningless in isolation.
- **Dockerfile**: Each `RUN apt-get install ...` block separated by blank lines creates individual chunks with no context about which image stage they belong to.
- **Rust**: `impl` blocks often have blank lines between associated functions — splitting by blank lines separates the `impl` header from its methods, producing headless method chunks.

The result is chunks with `symbol_type = "block"` that contaminate the vector index with low-quality embeddings. When a user asks "how does the Go service handle authentication," the retrieved blocks will be random fragments of Go code with no symbol names, no cross-references, and potentially cut in semantically meaningless places. These results will score well in BM25 (they contain the right keywords) but provide poor context. **There is no mechanism to rank down or deprioritize text-fallback blocks vs. properly parsed AST chunks.** The plan says they're "still embedded and searchable" as if this is neutral — but noisy fallback chunks actively hurt precision.

**The fallback is applied to file types where it produces pure noise.** YAML configuration files, Dockerfile, and Makefile are indexed as text blocks with embeddings. A query about "what Docker base image does the API service use" will retrieve a block containing `FROM node:18-alpine` — great! But it will also retrieve 40 other blocks from YAML files that mention "api" and "service" in config keys, with no way to signal that the Dockerfile block is more relevant. The absence of `symbol_type` meaning forces the retriever to treat all blocks equally.

**Alternative 1 — Language-specific simple chunkers for high-value unsupported languages**: Go, Rust, Ruby, and Java are not tree-sitter-hard — simple regex-based chunkers that split on `func `, `fn `, `def `, `public class ` patterns would produce meaningful symbol-named chunks without full AST support. These are ~20 lines per language. Not as accurate as tree-sitter but infinitely better than blank-line splitting. **Tradeoff**: maintenance burden, edge cases (multiline function signatures, anonymous functions), false positives on comments containing these patterns.

**Alternative 2 — Skip YAML/TOML/Dockerfile entirely by default, or use structured chunkers**: YAML/TOML are structured — use Python's `yaml`/`tomllib` parsers to extract top-level keys as named chunks (`symbol_name = "database.host"`, `symbol_type = "config_key"`). Dockerfile has only ~20 instruction types — a simple line-by-line parser that groups each instruction into a chunk with `symbol_name = "FROM"` or `symbol_name = "RUN apt-get"` is trivial and produces searchable named chunks. **Tradeoff**: adds YAML/TOML parse dependencies; edge cases in YAML anchors and complex structures.

**Concrete recommendation**: Add a `search_quality` field to chunks: `ast` > `regex` > `text_fallback`. During RRF scoring, apply a 0.7x multiplier to text-fallback chunks to bias toward better-parsed results when scores are comparable. This prevents blank-line blocks from ranking above properly-named functions on keyword overlap alone. This is a 5-line change to the retriever.

---

### 8. Sub-chunk Ref Attribution

**The decision**: When a chunk is split into sub-chunks, assign refs to the sub-chunk whose line range contains the reference site. If line ranges are unavailable, assign all refs to the first sub-chunk as fallback.

**What's wrong / risky**:

**The fallback is wrong in the most common case.** The "first sub-chunk as fallback" means: if ref extraction can't determine line ranges (described as "rare"), all import statements, all call references, and all inheritance references for the entire original function are pinned to sub-chunk 1. Sub-chunk 2 and beyond appear to have no dependencies. This breaks the "what calls X" query type for large functions — callers in sub-chunk 2+ will never appear in reverse-lookup queries. Since tree-sitter nodes always have line positions, "rare" may mean "rare in practice" but it should mean "impossible by construction" — if it can't be guaranteed, it will happen in production.

**Import statements are almost always in the file header, not in function bodies.** The import refs for a Python module-level chunk or a TypeScript class are in the top-level `import` statements, which belong to the *file* scope, not to any particular function. When a file is chunked into 10 functions + a "module" chunk, the import refs are correctly attributed to the module chunk. But when a *function* is sub-chunked, and the function body contains an inline `import()` (TypeScript dynamic import) or a `__import__()` call (Python), those are typically in the *early* part of the function body — sub-chunk 1 gets them correctly. The fallback never fires for this case because imports have clear line positions.

The real risk is **call refs in deeply-nested code paths**. Consider a 400-line function where the first 50 lines are setup, lines 51-250 are the happy path calling various services, and lines 251-400 are error handling calling logging and alerting services. The happy-path calls are in sub-chunk 2 (lines 51-250); error handling calls are in sub-chunk 3 (lines 251-400). Both have clear line positions and will be attributed correctly. The fallback only fires if the ref extraction fails to produce line positions for some refs.

**The deeper issue**: attributing refs to sub-chunks is the wrong level of granularity. The sub-chunk is an artifact of the 6000-char ceiling, not a semantically meaningful unit. The original function is the semantic unit. The parent chunk's ID should be queryable to retrieve all refs for the logical unit, regardless of sub-chunking. The current schema has no `parent_chunk_id` column — sub-chunks are first-class chunks with no structural link back to their parent. This means:
- "What does `process_checkout` call?" will only retrieve refs attributed to the sub-chunk named `process_checkout:1`, not sub-chunks `process_checkout:2` through `process_checkout:N`.
- The symbol_lookup table has the full function name for all sub-chunks (since the signature is prepended), so multiple sub-chunks map to the same symbol — the resolution algorithm will pick one arbitrarily.

**Concrete fix**: Add `parent_chunk_id TEXT` and `sub_chunk_index INTEGER` columns to the `chunks` table. When resolving "what does X call," union the refs for all sub-chunks where `parent_chunk_id = X's root chunk`. This requires a two-hop query but gives complete ref coverage for sub-chunked functions. The schema change is non-breaking (nullable columns).

---

### 9. Language Coverage — TS, Python, Bash Only

**The decision**: Full AST support for TypeScript/JS, Python, Bash. All other languages get text fallback.

**What's wrong / risky**:

**Bash is the wrong third language.** Bash scripting is increasingly a legacy choice — modern infra scripting uses Python, Go, or Makefile+shell. More critically, Bash AST chunking provides minimal semantic value because Bash functions are typically short (10-30 lines), rarely have deep call hierarchies, and the cross-reference extraction (just `source` includes) provides little graph value. The only Bash-specific value-add in the plan is `env_var` extraction and `source` ref tracking, which could be done with 10 lines of regex without tree-sitter.

**The three languages with the most value to add beyond TS/Python are Go, Ruby, and Rust** (in that order for the probable user base of a developer tool):
- **Go**: Extremely regular syntax, highly amenable to tree-sitter parsing. All function definitions start with `func `, struct definitions with `type X struct`, interfaces with `type X interface`. The tree-sitter Go grammar is stable and well-maintained. Go has a huge developer tooling market (GitHub Copilot, Cursor, etc.) — users with Go repos are primary targets. The value of "what methods does type X implement" in Go is extremely high.
- **Ruby**: The Rails ecosystem is enormous. `has_many`, `belongs_to`, ActiveRecord associations are implicit cross-references that tree-sitter can detect. Controller actions (`def index`, `def show`) are natural chunk boundaries. Many Rails shops index their entire codebase — Bash support is irrelevant to them.
- **Rust**: Rust's `impl` blocks, `trait` definitions, `fn` functions, `struct` definitions are all clean tree-sitter targets. The Rust developer community is highly technical and more likely to use a tool like this.

**Java and Kotlin deserve mention**: Large enterprise codebases are almost always Java or Kotlin. Not supporting them limits the market significantly.

**The 15MB grammar overhead argument misses the real cost**: The plan justifies the language-pack's 15MB overhead because "there's no selective install option." But the *real* cost of the current approach is developer time spent on Bash chunking (with its heredoc truncation, section comment regex, multi-heuristic splitting) that could instead be spent on Go AST chunking that would serve 10x more users. The Bash chunker is clearly the most complex per-language implementation (3 fallback heuristics, heredoc handling, env_var extraction) for the language with the lowest ROI.

**Concrete recommendation**: Replace Bash full support with Bash regex-level support (regex-based function boundary detection, env_var extraction) and use tree-sitter for Go instead. The tree-sitter Go grammar (`tree-sitter-go`, included in language-pack) is mature. Go function/method/struct/interface chunking is ~80 lines of tree-sitter node matching — simpler than Bash chunking's 3-tier heuristic fallback. The effort trade-off favors Go over Bash by roughly 3:1 in user value per line of implementation.

---

### 10. React Component Detection

**The decision**: Arrow functions returning JSX tagged as `symbol_type = "component"`. Detected via `export_statement → lexical_declaration → variable_declarator → arrow_function`.

**What's wrong / risky**:

**The detection pattern misses 40-60% of React components in real codebases:**

1. **Non-exported components**: Many components are defined locally within a file and not exported from the top level:
   ```tsx
   const ListItem = ({ item }) => <li>{item.name}</li>  // no 'export' keyword
   function renderItems(items) { return items.map(i => <ListItem key={i.id} item={i} />) }
   ```
   `ListItem` is a component but the pattern requires `export_statement` as the root. Non-exported arrow function components — which are extremely common for local sub-components — are silently misclassified as regular functions.

2. **Function declaration components**: React components can be (and often are) regular function declarations:
   ```tsx
   export function UserCard({ user }: UserCardProps) {
     return <div className="card">{user.name}</div>
   }
   ```
   The plan's detection pattern only covers `arrow_function` nodes. `function_declaration` returning JSX is not tagged as `symbol_type = "component"`. In codebases that use function declarations (which ESLint rules like `react/function-component-definition` often enforce), component detection fails entirely.

3. **`React.forwardRef` wrapping**:
   ```tsx
   export const Input = React.forwardRef<HTMLInputElement, InputProps>(
     ({ label, ...props }, ref) => <div><label>{label}</label><input ref={ref} {...props} /></div>
   )
   ```
   The outer node is a `call_expression` (`React.forwardRef(...)`), not an `arrow_function` directly under a `lexical_declaration`. The inner arrow function never returns JSX at the top level (it's inside the `forwardRef` call). This pattern — used for essentially every reusable UI component library — produces components tagged as regular functions.

4. **`React.memo` wrapping**:
   ```tsx
   export const ExpensiveComponent = React.memo(function({ data }) {
     return <div>{data.map(d => <Item key={d.id} {...d} />)}</div>
   })
   ```
   Same issue: the outer wrapper is a `call_expression`.

5. **`React.lazy` for code-splitting**:
   ```tsx
   const LazyDashboard = React.lazy(() => import('./Dashboard'))
   ```
   This is a component reference but its "body" is in another file. Detection would need to understand that `React.lazy` wraps a component reference.

6. **Higher-order components (HOCs)**:
   ```tsx
   export const withAuth = (WrappedComponent) => (props) =>
     isAuthenticated() ? <WrappedComponent {...props} /> : <Redirect to="/login" />
   ```
   The outer arrow function `(WrappedComponent) => ...` doesn't return JSX — it returns another function that returns JSX. The inner function isn't a top-level `variable_declarator`.

7. **Class components**: Although less common in new code, large codebases still have class components:
   ```tsx
   export class ErrorBoundary extends React.Component<Props, State> {
     render() { return <div>{this.props.children}</div> }
   }
   ```
   These are `class_declaration` nodes, not `arrow_function` nodes. Class components aren't detected as components at all.

**The JSX-return detection itself is fragile.** tree-sitter identifies JSX nodes (`jsx_element`, `jsx_fragment`, `jsx_self_closing_element`). But a function that conditionally returns JSX:
```tsx
const MaybeComponent = ({ show, children }) => {
  if (!show) return null  // null return first!
  return <div>{children}</div>
}
```
...requires finding *any* return statement in the function body that returns JSX, not just the final expression. If the detection only checks the immediate return value of the arrow function expression (which tree-sitter represents as the body of `arrow_function`), conditional returns won't be detected.

**Concrete alternative**: Use a two-signal approach:
1. **Naming convention**: PascalCase function/variable names that are exported are almost certainly React components. This is the React convention and is enforced by all popular linters (`react/display-name` rule, `react/naming-convention`). A simple check `symbol_name[0].isupper()` for exported functions in `.tsx`/`.jsx` files is ~90% accurate with zero AST complexity.
2. **JSX presence**: Scan the function body (and nested arrow functions) for any `jsx_element` or `jsx_fragment` node as a confirmation signal.

Combining PascalCase + JSX anywhere in body (including in sub-functions) catches `forwardRef`, `memo`, HOCs, and class component `render()` methods. Add explicit detection for `React.memo(...)`, `React.forwardRef(...)`, and `React.lazy(...)` call expressions as `symbol_type = "component"` regardless of what they wrap. This is ~30 additional lines in the TypeScript chunker.

---

## Summary Table

| Decision | Severity | Key Risk | Recommended Fix |
|---|---|---|---|
| tree-sitter ABI pinning | **HIGH** | Segfaults from minor-version mismatch | Use exact `==` pins; add CI ABI smoke test; add error-node-density check |
| Chunking granularity | **MEDIUM** | Context cliff at sub-chunk boundaries; no file-level chunks | Add 5-10 line overlap between sub-chunks; add file-level chunks for semantic search |
| 6000-char cap heuristic | **MEDIUM** | TS token density underestimated by ~60%; arbitrary cut mid-logic | Soft target at AST block boundaries; acknowledge cost estimate error for TS |
| Cross-ref extraction | **HIGH** | Calls under-specified; type refs missing; dynamic imports silently dropped | Add type_ref and decorator ref types; document dynamic import limitation; specify call extraction scope |
| symbol_lookup resolution | **HIGH** | Directory proximity is wrong; barrel files break it; aliases untracked | Import-graph-aware resolution; deprioritize barrel files; track import aliases |
| Incremental refresh | **MEDIUM** | 500-commit threshold arbitrary; rebase orphans silently wrong; no mtime cache | Change threshold to file-count-based; add ancestor check before git diff |
| Text fallback | **MEDIUM** | Blank-line splitting is incoherent; noisy blocks hurt precision | Add 0.7x quality multiplier for text-fallback chunks in RRF scoring |
| Sub-chunk ref attribution | **MEDIUM** | First-sub-chunk fallback silently breaks "what calls X"; no parent_chunk_id | Add parent_chunk_id column; union sub-chunk refs in call lookups |
| Language coverage (Bash) | **LOW-MEDIUM** | Wrong ROI; Go/Ruby would serve more users | Replace Bash AST support with Go; keep regex-level Bash for env_var/source detection |
| React component detection | **MEDIUM-HIGH** | Misses forwardRef, memo, function declarations, non-exported components | PascalCase + JSX-in-body heuristic; explicit wrapping-function detection |

---

## Addendum: Blade Strategy Selection (feat/php-blade-vue-chunking)

Why Blade templates get a structural directive scanner instead of a
tree-sitter grammar. Recorded here so the survey and measurements do not
live as a block comment in `chunker.py`.

tree-sitter-language-pack 0.13.0 ships no Blade grammar (checked against
its full 173-language list; it has `php`, `html`, `twig` and `vue`, but
nothing for Blade). Neither neighbouring grammar is usable as a stand-in:

- `php` "succeeds" on any Blade file: everything outside `<?php` tags is a
  single opaque `text` node, so it reports zero parse errors while yielding
  one file-sized chunk. That is worse than a real fallback because the
  error-density guard cannot detect it.
- `html` chokes on Blade's control flow: directives routinely open a tag in
  one branch and close it in another. Measured over the 82 Blade templates
  in a production Laravel app, 22 (27%) exceeded the chunker's 10%
  error-node threshold, several above 95%.

So Blade gets a structural directive scanner: Blade's own block directives
(`@section`/`@endsection`, `@push`, `@component`, `@if`, ...) are the
boundaries an author already writes, and they nest properly. The scanner
tracks that nesting, so a `@section` is chunked whole no matter how much
control flow it contains, and descends into a section only when the section
itself exceeds `max_chars`. Quality is REGEX, matching the Bash and
Markdown strategies.
