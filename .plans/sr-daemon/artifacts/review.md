# RFC-02 Review: Persistent daemon with multi-repo registry

## Structural Validation

Validator reports 6 errors — all false positives (TOML `[[repos]]`
and `[daemon]` table headers inside fenced code blocks detected as
unfilled RFC placeholders). No real structural issues.

- Abstract stands alone ✓
- Scope clearly states in/out ✓
- RFC 2119 keywords used consistently ✓
- Error handling covers failure modes ✓
- Security section addresses trust boundaries ✓
- Open questions are specific and actionable ✓
- Implementation plan is phased with dependencies ✓

## Internal Consistency

1. **Duplicate `/refresh` row in API table.** Lines 181-182 list
   `POST /refresh` twice — once as "unchanged" and once as
   "extended with files parameter." These are the same endpoint
   with an optional field, not two endpoints. Should be one row.

2. **`query_timeout_s` defined in two places.** The daemon config
   block (line ~375) shows it as part of the `[daemon]` table.
   The Query Timeout section (line ~450) shows it again in its own
   `[daemon]` block. Same value (10), but the duplication could
   drift. Consolidate to one authoritative config block.

3. **`shutdown_timeout_s` vs `ExitTimeOut`.** The config file uses
   `shutdown_timeout_s = 10` and the launchd plist uses
   `ExitTimeOut = 15`. These SHOULD match or the RFC should explain
   why the plist gives 5s extra (answer: to allow uvicorn drain +
   lifespan shutdown). The 5s gap is intentional but not documented.

4. **State `unindexed` vs auto-index behavior.** The Auto-Index
   section says the daemon MUST queue a background build when a repo
   has no index. But the state machine has `unindexed` as a state.
   When does a repo stay in `unindexed`? If auto-index always runs,
   the state would be `indexing` immediately, never `unindexed`.
   Either remove `unindexed` or clarify when it applies (e.g.,
   daemon started without embedding model, repo added but auto-index
   disabled).

5. **Open Question 4 is already answered.** "Should the daemon block
   startup until indexed or build in background?" The Auto-Index
   section already says "queue a background full index build" and
   "serve queries immediately for already-indexed repos." The
   question is resolved by the design — remove it or mark as decided.

## Codebase Alignment

Checked: `server.py`, `builder.py`, `store.py`, `querier.py`,
`config.py`, `cli.py`, `__init__.py`, `models.py`.

1. **`create_app()` takes `repo_paths` as a list — compatible.**
   The RFC's multi-repo design matches the existing multi-repo
   support in `server.py`. The `state["indexes"]` dict already
   keys by repo name. Adding/removing repos at runtime requires
   mutating this dict, which is straightforward.

2. **`SRConfig` is per-repo, not global.** `config.py` resolves
   config per-repo (reads `.source-recall.toml` from the repo root).
   The daemon config (`repos.toml`) is a new global config that
   doesn't conflict — it lists repos and daemon-level settings,
   while `SRConfig` handles per-repo chunking/search options.
   Alignment is good.

3. **`IndexBuilder.build()` acquires a PID-file lock.** This means
   background indexing from the daemon will block if another process
   (e.g., manual `sr index`) is also building the same repo. The
   RFC doesn't mention this interaction. The lock timeout is 5s —
   if the daemon's background indexer can't acquire the lock, it
   should retry later, not fail permanently.

4. **`IndexBuilder.refresh()` also acquires the PID lock.** Two
   concurrent refresh calls (e.g., tallow and afk both fire
   refresh-on-write) will serialize on the lock. The second caller
   waits 5s then raises `IndexLockError`. The RFC's rate limiter
   (429) partially addresses this but doesn't cover the case where
   two different consumers refresh the same repo simultaneously.

5. **`_resolve_index()` in server.py does repo name lookup.**
   Adding runtime repo add/remove means this function and the
   `state["indexes"]` dict need thread-safe mutation. FastAPI runs
   sync endpoints in a threadpool — concurrent requests could read
   the dict while another thread modifies it. Need a lock or
   copy-on-write pattern.

6. **No `DELETE` support in existing server.** The RFC proposes
   `DELETE /repos/{name}` but the current server only has GET and
   POST routes. This is new code, not a conflict — just noting it
   requires a new endpoint, not modification of existing ones.

## Adversarial Review

### High severity

1. **Concurrent refresh from multiple consumers is under-specified.**
   Scenario: tallow writes `auth.ts` and fires refresh. Simultaneously,
   afk writes `docs/auth.md` and fires refresh for the same repo. Both
   hit `IndexBuilder.refresh()` which acquires the PID lock. The second
   caller gets `IndexLockError` after 5s timeout.
   Consequence: one consumer's refresh silently fails. The 429 rate
   limiter is per-repo per-time, not per-consumer — it might not catch
   this. The daemon should serialize refreshes per-repo internally
   (queue, not reject) so both consumers' changes get indexed.

2. **Periodic refresh overlapping with targeted refresh.**
   Scenario: the 5-minute periodic refresh starts. 2 seconds in, tallow
   fires a targeted refresh for 3 files. The targeted refresh tries to
   acquire the PID lock, fails (periodic holds it), returns 429 or
   errors. The agent's changes aren't indexed until the next cycle.
   Fix: the daemon should own all refresh serialization internally,
   not rely on the PID-file lock (which was designed for multi-process,
   not multi-request-in-one-process).

3. **Config file written by API while daemon is reading it.**
   Scenario: `POST /repos` writes to `repos.toml`. Simultaneously,
   the daemon's config watcher reads the file. Partial write = corrupt
   TOML parse. This is a classic write-during-read race.
   Fix: atomic write (write to tmp, rename) or use the API as the
   sole mutator and only read config at startup.

### Medium severity

4. **Repo path deleted while indexing.**
   Scenario: user deletes `~/dev/old-project`. The daemon's periodic
   refresh tries to git-diff, fails, logs an error. But does it
   transition the repo to `error` state? The RFC says broken repos
   are "marked as error" but doesn't specify the transition trigger.
   If the path is missing, the daemon should detect this in the
   periodic refresh loop and transition to `error` with a clear
   message, not just log and silently skip.

5. **SSE client disconnect during indexing.**
   Scenario: tallow subscribes to `/repos/marrow/progress` SSE stream.
   Tallow session ends (user exits). The SSE generator keeps running
   in the server, writing to a closed connection. FastAPI/Starlette
   handles this with `asyncio.CancelledError` but only if the
   generator checks for disconnection. If not, the generator leaks
   until indexing completes.
   Fix: the SSE generator MUST check `request.is_disconnected()` or
   use Starlette's built-in disconnect detection.

6. **Model load failure at startup.**
   Scenario: the HuggingFace cache is corrupted or disk is full.
   `CodeRankEmbedder.__init__` doesn't fail (lazy load), but the first
   query or index triggers the load, which raises. If the daemon starts
   with a broken model, all repos will fail on first access. The RFC's
   startup resilience section handles per-repo failures but doesn't
   address model-level failure.
   Fix: probe-load the model during lifespan startup. If it fails,
   log an error and start in FTS-only mode (queries work, just no
   vector search). Surface this in `/health` as `degraded`.

### Low severity

7. **`sr ask` CLI assumes daemon is running.**
   The RFC says `sr ask` "talks to daemon" but the current CLI
   (`cli.py`) loads the model in-process. Migration path is unclear —
   does `sr ask` become a thin HTTP client? What if the daemon is
   down? Should it fall back to in-process mode (slow but works)?

8. **`repos.toml` uses `~` in paths.**
   The RFC shows `path = "~/dev/marrow"` but Python's `tomllib`
   returns the literal string `~/dev/marrow`. The daemon must
   expand `~` — the RFC says SHOULD resolve but doesn't specify
   the expansion mechanism. Use `Path.expanduser()`.

9. **No log rotation.**
   The launchd plist writes stdout/stderr to fixed files. Without
   rotation, these grow unbounded. Phase 2 lists "log rotation config"
   as a deliverable but the RFC doesn't specify the mechanism
   (newsyslog, logrotate, or app-level).

## Summary

- **Structural issues**: 0 real errors (6 false positives from TOML syntax)
- **Inconsistencies**: 5 (1 duplicate API row, 1 duplicate config block,
  1 undocumented timeout gap, 1 unused state, 1 pre-answered question)
- **Alignment issues**: 6 (PID lock interaction, thread safety, new
  endpoints needed — all manageable, no blockers)
- **Adversarial findings**: 9 (3 high, 3 medium, 3 low)

The two highest-severity findings both point to the same root cause:
the PID-file lock was designed for multi-process isolation but the
daemon is now single-process multi-request. Refresh serialization
should move from PID-file locks to an internal per-repo queue.
