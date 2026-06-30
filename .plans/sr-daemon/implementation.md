# source-recall Daemon — Implementation Plan

## Architecture

### Constraints

- <!-- D-001 --> Single process, per-repo internal refresh lock
  (not PID-file for intra-daemon serialization)
- <!-- D-003 --> Must start in FTS-only degraded mode if embedding
  model fails to load
- <!-- D-004 --> Background index builds; serve healthy repos
  immediately
- <!-- D-005 --> launchd ExitTimeOut (15s) > shutdown_timeout_s
  (10s) to allow lifespan cleanup
- <!-- D-006 --> `sr ask` becomes daemon-first HTTP client with
  in-process fallback
- SQLite per repo (existing), apsw for vec (existing)
- Embedding model loaded once, shared across all repos (~500MB)
- Localhost only (`127.0.0.1:7249`), CORS restricted

### Topology

```
repos.toml ──► DaemonConfig ──► create_app()
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼               ▼
               RepoManager    EmbedModel      FastAPI app
               (registry,     (shared,        (routes,
                locks,         lazy-load,       SSE,
                scheduler)     fts-fallback)    lifespan)
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
       RepoSlot  RepoSlot  RepoSlot
       (marrow)  (tallow)  (afk)
       state,    state,    state,
       Index,    Index,    Index,
       lock      lock      lock
```

### Key Abstractions

- **DaemonConfig** — global `[daemon]` settings + `[[repos]]` list,
  parsed from `repos.toml`
- **RepoManager** — owns the dict of `RepoSlot`s, handles add/remove,
  periodic refresh scheduling
- **RepoSlot** — per-repo state machine (`queued` → `indexing` →
  `ready` / `error`), owns the `Index` instance and per-repo
  `threading.Lock` (sync endpoints run in threadpool)
- **Existing `Index` facade** — unchanged; RepoSlot wraps it

---

## Phases

### Phase 1: Repo registry and daemon mode (Week 1-2)

#### M1.1: DaemonConfig parser
- **Dependencies:** none
- **Effort:** S (1-2d)
- **Test spec:** Config loads from TOML, resolves `~`, validates
  paths, merges `[daemon]` defaults, rejects invalid TOML
- **Tasks:**
  1. RED: Test that `DaemonConfig.from_toml()` parses a multi-repo
     config with `[daemon]` section and per-repo overrides
  2. GREEN: Implement `DaemonConfig` pydantic model + `from_toml()`
  3. RED: Test `~` expansion and symlink resolution
  4. GREEN: Add `Path.expanduser().resolve()` in parser
  5. RED: Test missing/invalid TOML raises `ConfigError`
  6. GREEN: Add validation

#### M1.2: RepoManager and RepoSlot
- **Dependencies:** M1.1
- **Effort:** M (3-5d)
- **Test spec:** RepoManager adds/removes repos, transitions
  states, exposes slot status. Slots hold per-repo Index + lock.
- **Tasks:**
  1. RED: Test `RepoManager.add()` creates a RepoSlot in `queued`
     state for a valid path
  2. GREEN: Implement RepoManager + RepoSlot dataclass
  3. RED: Test `RepoManager.add()` rejects nonexistent path
  4. GREEN: Add path validation
  5. RED: Test `RepoManager.remove()` closes Index and removes slot
  6. GREEN: Implement remove with cleanup
  7. RED: Test `RepoManager.load_from_config()` initializes all
     repos from DaemonConfig
  8. GREEN: Wire config → manager

#### M1.3: Daemon server with registry endpoints
- **Dependencies:** M1.2
- **Effort:** M (3-5d)
- **Test spec:** `create_app()` accepts `DaemonConfig`, serves
  `GET /repos`, `POST /repos`, `DELETE /repos/{name}`. Existing
  `/query` and `/refresh` routes work against registered repos.
- **Tasks:**
  1. RED: Test `GET /repos` returns registered repos with state
  2. GREEN: Refactor `create_app()` to accept DaemonConfig,
     initialize RepoManager in lifespan
  3. RED: Test `POST /repos` adds a repo and returns 201
  4. GREEN: Implement add endpoint with auto-index trigger
  5. RED: Test `DELETE /repos/{name}` removes and returns 200
  6. GREEN: Implement delete endpoint
  7. RED: Test `POST /repos` writes atomically to repos.toml
  8. GREEN: <!-- D-002 --> Implement atomic write (tmp + rename)

#### M1.4: CLI commands
- **Dependencies:** M1.3
- **Effort:** S (1-2d)
- **Test spec:** `sr daemon run` starts foreground server from
  config. `sr add`, `sr remove`, `sr repos` call daemon HTTP API.
- **Tasks:**
  1. RED: Test `sr daemon run` loads config and starts uvicorn
  2. GREEN: Add `daemon` command group to cli.py with `run`
     subcommand
  3. RED: Test `sr add ~/dev/foo` POSTs to daemon and prints status
  4. GREEN: Implement `add`/`remove`/`repos` as HTTP client commands
  5. REFACTOR: Extract shared HTTP client helper for CLI → daemon

### Gate 1→2

- [ ] `sr daemon run` serves all repos from `repos.toml`
- [ ] `POST /repos` adds a repo and triggers indexing
- [ ] `DELETE /repos/{name}` removes a repo
- [ ] `GET /repos` shows all repos with state
- [ ] Atomic config write on `POST /repos`
- [ ] All existing tests still pass (no regression)

---

### Phase 2: launchd, lifecycle, and reliability (Week 3)

#### M2.1: launchd plist generation and management
- **Dependencies:** M1.4
- **Effort:** S (1-2d)
- **Existing work:** `launchd/dev.source-recall.serve.plist` and
  justfile recipes (`launchd-install`, `launchd-start`,
  `launchd-stop`, `launchd-logs`) already exist. The current
  approach hardcodes repo paths in the plist — the daemon replaces
  this with `sr daemon run` which reads repos from `repos.toml`.
  The generated plist invokes `sr daemon run` instead of
  `sr serve <paths>`. Existing justfile recipes should be updated
  to wrap `sr daemon start/stop/status`.
- **Test spec:** `sr daemon start` writes plist, loads via
  launchctl, waits for health. `sr daemon stop` unloads.
  `sr daemon status` shows running/stopped + repo summary.
- **Tasks:**
  1. RED: Test plist generation contains `sr daemon run` (not
     `sr serve`) and correct PATH/log paths
  2. GREEN: Implement plist template generation from DaemonConfig,
     replacing the static `launchd/` template
  3. RED: Test `sr daemon stop` unloads the plist
  4. GREEN: Implement stop with launchctl bootout
  5. RED: Test `sr daemon status` reports running state
  6. GREEN: Implement status via PID check + `/health` probe
  7. REFACTOR: Update justfile — replace `launchd-*` recipes with
     `daemon-*` wrappers around `sr daemon start/stop/status/logs`

#### M2.2: Graceful shutdown
- **Dependencies:** M2.1
- **Effort:** S (1-2d)
- **Test spec:** SIGTERM triggers drain → close sequence. In-flight
  refresh commits partial progress. No WAL corruption.
- **Tasks:**
  1. RED: Test that in-flight refresh completes on SIGTERM (not
     killed mid-write)
  2. GREEN: Configure uvicorn `--timeout-graceful-shutdown`,
     ensure lifespan shutdown closes all connections
  3. RED: Test that DB connections are closed after shutdown
  4. GREEN: Add connection cleanup in lifespan shutdown

#### M2.3: Startup resilience
- **Dependencies:** M1.3
- **Effort:** S (1-2d)
- **Test spec:** Bad repo doesn't crash daemon. Model failure
  starts FTS-only mode. `/health` reports degraded mode.
- **Tasks:**
  1. RED: Test daemon starts when one repo has corrupt/missing index
  2. GREEN: Wrap per-repo init in try/except, set `state: error`
  3. RED: Test daemon starts in fts_only mode when model load fails
  4. GREEN: <!-- D-003 --> Catch model load error, set mode flag,
     report in `/health`
  5. RED: Test `/health` reports `mode: fts_only` when degraded
  6. GREEN: Add mode field to HealthResponse

### Gate 2→3

- [ ] `sr daemon start` / `stop` / `status` work
- [ ] Daemon survives reboot (launchd restarts it)
- [ ] SIGTERM drains cleanly (no corrupt WAL)
- [ ] Bad repo doesn't crash daemon
- [ ] Model failure → FTS-only mode reported in `/health`

---

### Phase 3: Progress, targeted refresh, serialization (Week 4-5)

#### M3.1: Per-repo status endpoint
- **Dependencies:** M1.2 (RepoSlot state machine)
- **Effort:** S (1-2d)
- **Test spec:** `GET /repos/{name}/status` returns state,
  progress, timestamps. State transitions are correct.
- **Tasks:**
  1. RED: Test status endpoint returns `ready` for indexed repo
  2. GREEN: Implement `/repos/{name}/status` route
  3. RED: Test status shows `indexing` with progress during build
  4. GREEN: Wire progress callback into RepoSlot state

#### M3.2: SSE progress stream
- **Dependencies:** M3.1
- **Effort:** S (1-2d)
- **Test spec:** `/repos/{name}/progress` streams events during
  indexing, closes on completion. Client disconnect doesn't leak.
- **Tasks:**
  1. RED: Test SSE stream emits progress events during build
  2. GREEN: Implement SSE endpoint with StreamingResponse
  3. RED: Test stream closes when indexing completes
  4. GREEN: Add completion event + stream close
  5. RED: Test no resource leak on client disconnect
  6. GREEN: Check `request.is_disconnected()` in generator

#### M3.3: Refresh serialization
- **Dependencies:** M1.2 (RepoSlot lock)
- **Effort:** M (3-5d)
- **Test spec:** <!-- D-001 --> Concurrent refreshes serialize via
  per-repo lock. Targeted refreshes coalesce queued file lists.
  Periodic skips if running. External PID lock still works.
- **Tasks:**
  1. RED: Test two concurrent `/refresh` calls serialize (second
     waits, doesn't 429)
  2. GREEN: Add per-repo `threading.Lock` in RepoSlot, serialize
     refresh calls
  3. RED: Test targeted refresh with `files` parameter re-indexes
     only those files
  4. GREEN: Extend refresh handler to accept `files`, pass to
     builder change detection
  5. RED: Test rapid targeted refreshes coalesce file lists
  6. GREEN: Implement queue + merge in RepoSlot
  7. RED: Test external `sr index` still works (PID lock)
  8. GREEN: Daemon acquires PID lock before refresh, retries on
     contention

#### M3.4: Periodic background refresh
- **Dependencies:** M3.3
- **Effort:** S (1-2d)
- **Test spec:** Periodic refresh runs every N seconds for each
  repo. Skips if refresh already running. Long refresh doesn't
  cause backlog.
- **Tasks:**
  1. RED: Test periodic refresh triggers after interval elapses
  2. GREEN: Add background timer task in RepoManager
  3. RED: Test periodic skips when refresh is already running
  4. GREEN: Check lock state before periodic refresh
  5. RED: Test `vec_dirty` files are re-embedded on periodic refresh
  6. GREEN: Wire `vec_dirty` check into periodic refresh path

### Gate 3→4

- [ ] `/repos/{name}/status` returns correct state and progress
- [ ] SSE stream works and doesn't leak on disconnect
- [ ] Concurrent refreshes serialize, not reject
- [ ] Targeted refresh re-indexes only specified files
- [ ] Periodic refresh runs on schedule, skips when busy
- [ ] `vec_dirty` files re-embedded on next refresh

---

### Phase 4: tallow extension (Week 6-7)

#### M4.1: Extension scaffold
- **Dependencies:** Phase 3 complete, tallow extension API
- **Effort:** S (1-2d)
- **Test spec:** Extension loads, probes `/health`, registers
  tools when daemon available, no-ops when unavailable.
- **Tasks:**
  1. Scaffold extension in `tallow-plugins/base/extensions/sr/`
  2. Implement health probe on session start
  3. Register `sr_search` tool when daemon is available
  4. Inject system prompt fragment about `sr_search` availability

#### M4.2: `sr_search` tool
- **Dependencies:** M4.1
- **Effort:** S (1-2d)
- **Test spec:** Tool calls `/query`, formats results for agent
  consumption. Handles timeout and daemon-down gracefully.
- **Tasks:**
  1. Implement tool that accepts query string + optional repo
  2. Call daemon `/query`, format results as structured output
  3. Handle 504 timeout, 503 indexing, connection refused

#### M4.3: Refresh-on-write hook
- **Dependencies:** M4.1, M3.3 (targeted refresh)
- **Effort:** S (1-2d)
- **Test spec:** After `write` or `edit` tool calls, extension
  fires non-blocking targeted refresh. Failures don't block agent.
- **Tasks:**
  1. Register post-tool-use hook for write/edit tools
  2. Extract changed file paths from tool call args
  3. Fire `POST /refresh` with file list (non-blocking)

#### M4.4: Footer status widget
- **Dependencies:** M4.1, M3.1 (status endpoint)
- **Effort:** S (1-2d)
- **Test spec:** Footer shows per-repo status (✓/⟳/%/✗). Updates
  on state transitions.
- **Tasks:**
  1. Poll `/repos` on interval (every 5s)
  2. Render compact status line: `sr: marrow ✓  tallow ⟳ 45%`
  3. Handle daemon-down gracefully (show `sr: offline`)

### Gate 4→5

- [ ] tallow agent can call `sr_search` tool successfully
- [ ] Post-write refresh fires without blocking edits
- [ ] Footer shows live index status
- [ ] Extension degrades gracefully when daemon is down

---

### Phase 5: Cross-repo query (Week 8)

#### M5.1: Multi-repo query routing
- **Dependencies:** M1.3 (multi-repo daemon)
- **Effort:** M (3-5d)
- **Test spec:** `/query` without `repo` searches all registered
  repos. Results tagged with repo name. Ranking is reasonable.
- **Tasks:**
  1. RED: Test `/query` without `repo` returns results from
     multiple repos
  2. GREEN: Fan out query to all ready repos, merge results
  3. RED: Test results include repo name in each result
  4. GREEN: Add `repo_name` field to QueryResultResponse
  5. RED: Test ranking across repos is reasonable (not biased
     toward larger repos)
  6. GREEN: Normalize scores per-repo before merging (divide by
     max score per repo)

#### M5.2: `sr ask` daemon-first CLI
- **Dependencies:** M5.1
- **Effort:** S (1-2d)
- **Test spec:** <!-- D-006 --> `sr ask` tries daemon first, falls
  back to in-process. `--repo` filter works in both modes.
- **Tasks:**
  1. RED: Test `sr ask` queries daemon when running
  2. GREEN: Probe `/health`, if up → HTTP query, format results
  3. RED: Test `sr ask` falls back to in-process when daemon is down
  4. GREEN: Catch connection error, fall back to current behavior

### Gate 5→done

- [ ] `sr ask "auth flow"` searches all repos via daemon
- [ ] `sr ask --repo marrow` filters to one repo
- [ ] `sr ask` falls back to in-process when daemon is down
- [ ] Results from multiple repos are reasonably ranked

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Model load takes >30s, launchd health check times out | medium | Increase health wait timeout; model is cached after first download |
| 20+ repos exhaust memory (SQLite connections + cached queries) | medium | Monitor RSS; add repo limit to config if needed |
| Periodic refresh on large repo blocks worker thread for minutes | medium | Query timeout (10s) applies; periodic runs in background thread |
| Atomic config write fails (disk full) | low | Write to tmp first; if tmp fails, leave original intact |
| Consumer fires refresh for wrong repo (path mismatch) | low | Daemon resolves paths; 404 if repo not found |
| tallow extension adds latency to every write/edit | low | Refresh-on-write is fire-and-forget (non-blocking) |

---

## Open Questions (deferred to Phase 5+)

1. **Config file location** — `~/.config/source-recall/repos.toml`
   vs `~/.local/share/source-recall/repos.toml`. Phase 1 picks one
   and ships; can migrate later if wrong.

2. **Cross-repo ranking** — when searching all repos, RRF scores
   aren't comparable across different corpus sizes. Options:
   (a) re-rank merged results with cross-encoder, (b) normalize
   by max score per repo, (c) return per-repo results separately.
   Phase 5 decides based on real usage.

3. **Tallow extension tool design** — one `sr_search` tool or two
   (`sr_search` + `sr_symbols`). Phase 4 starts with one tool;
   split if agent feedback shows it needs finer control.

4. **Config hot-reload vs restart** — watch `repos.toml` for
   changes vs require `sr daemon reload`. Start with manual reload
   (SIGHUP or `POST /reload`); add file watching only if the
   manual step proves annoying.

---

## Security Considerations

- Daemon listens on `127.0.0.1` only — MUST NOT bind `0.0.0.0`
  by default. Warning printed if overridden with `--host`.
- CORS restricted to localhost origins (already implemented).
- `POST /repos` validates path exists + is directory before
  indexing. SHOULD NOT follow symlinks outside `$HOME`.
- No authentication — acceptable for localhost single-user.
- Config file is user-owned; daemon never writes with elevated
  privileges.

## Alternatives Rejected

- **Per-consumer server instances** — wastes ~500MB RAM each
  for duplicate model loads.
- **Library import (no server)** — puts torch/sentence-transformers
  into every consumer process.
- **MCP server only (via tool-proxy)** — doesn't help ears (raw
  HTTP); adds proxy hop. MAY be added later as additional front door.
- **fswatch for auto-refresh** — high CPU, platform-specific,
  can't distinguish agent writes from build artifacts.

## References

- [RFC-02](.tallow/rfc/02_persistent-daemon-with-multi-repo-registry-and-refresh-on-write.rfc.md) — full specification
- [RFC-01](.tallow/rfc/01_git-object-based-branch-aware-indexing.rfc.md) — index architecture
- [ears SOURCE-RECALL.md](~/dev/ears/SOURCE-RECALL.md) — HTTP boundary decision
- [afk ROADMAP.md](~/dev/afk/ROADMAP.md) — planned integration
- [Existing justfile](justfile) — current task runner with launchd recipes
- [Existing launchd plist](launchd/dev.source-recall.serve.plist) — template to replace
