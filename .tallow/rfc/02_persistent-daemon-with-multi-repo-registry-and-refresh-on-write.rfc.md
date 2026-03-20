---
number: 02
title: "Persistent daemon with multi-repo registry and refresh-on-write"
type: feature
status: Draft
author: Marcus
date: 2026-03-20
---

# RFC-02: Persistent daemon with multi-repo registry and refresh-on-write

## Abstract

source-recall today is a CLI tool that indexes one or more repos on
demand. Multiple consumers (ears, tallow agents, afk) each need fast
semantic code search, but there is no persistent process managing
indexes or serving queries. This RFC proposes evolving `sr serve`
into a single persistent daemon with a repo registry, auto-indexing,
and a refresh-on-write protocol that any writing agent can call to
keep indexes current. The daemon is the sole runtime — all consumers
share one process, one model load, and one set of warm indexes.

## Introduction

### Problem

source-recall has three usage patterns emerging:

1. **ears** — voice-driven codebase queries via HTTP
2. **tallow agents** — fast code navigation during coding sessions
3. **afk** — doc-phase semantic search over changed code (planned)

Each consumer wants sub-100ms queries across multiple repos. Today
this requires manually running `sr index` per repo, then
`sr serve ~/dev/a ~/dev/b ...` with the right paths. There is no
process supervision, no auto-indexing, and no way for agents to
signal that code has changed.

### Scope

**In scope:**

- Persistent daemon with launchd supervision
- Repo registry (config file + runtime API)
- Auto-index on repo registration
- Refresh-on-write protocol for writing agents
- Consumer integration points (tallow extension, ears, afk)
- CLI commands for daemon management

**Out of scope:**

- Git hook integration (future — refresh-on-write covers
  agent-driven changes; human commits can use periodic refresh)
- Remote/networked operation (localhost only)
- Multi-user access control
- Replacing ears' fallback ChromaDB path

## Terminology

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD,
SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to
be interpreted as described in RFC 2119.

- **daemon** — the long-running `sr serve` process managed by launchd
- **repo registry** — the set of repos the daemon knows about,
  persisted in a config file
- **refresh-on-write** — a consumer calls `POST /refresh` after
  modifying files in an indexed repo, triggering incremental
  re-indexing
- **consumer** — any process that queries the daemon (ears, tallow,
  afk, CLI)
- **warm index** — an index whose embedding model is loaded and
  ready for sub-100ms queries

## Motivation

The embedding model takes ~12s to load. Without a persistent daemon,
every `sr ask` invocation pays this cost. The server mode (`sr serve`)
solves this but is manually managed — no config file, no supervision,
no way to add repos at runtime.

Agents that modify code (tallow, afk) make the index stale within
minutes. Without a refresh signal, queries return results from code
that no longer exists. The consumer shouldn't need to know about
source-recall internals — it should fire a single HTTP call after
writing files.

## Design

### Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  tallow  │  │   ears   │  │   afk    │
│(extension│  │ (httpx)  │  │ (http)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │
     └──────┬──────┘──────────────┘
            │  HTTP :7249
     ┌──────▼──────────────────────────┐
     │  sr daemon (single process)     │
     │                                 │
     │  ┌───────────────────────────┐  │
     │  │  embedding model (768d)   │  │
     │  │  loaded once, shared      │  │
     │  └───────────────────────────┘  │
     │                                 │
     │  ┌─────────┐ ┌─────────┐       │
     │  │ marrow  │ │ tallow  │ ...   │
     │  │ index   │ │ index   │       │
     │  └─────────┘ └─────────┘       │
     └─────────────────────────────────┘
              managed by launchd
```

### Repo Registry

A TOML config file at `~/.config/source-recall/repos.toml`:

```toml
# Repos to index and serve. The daemon watches this file
# and picks up changes without restart.

[[repos]]
path = "~/dev/marrow"

[[repos]]
path = "~/dev/tallow"
no_embed = false       # default: use vectors

[[repos]]
path = "~/dev/afk"

[[repos]]
path = "~/dev/source-recall"
```

The registry MUST support:

- Adding repos via config file edit (daemon watches for changes)
- Adding repos via `POST /repos` API (writes back to config)
- Removing repos via API or config edit
- Per-repo overrides (`no_embed`, `exclude_patterns`)

The daemon MUST resolve `~` via `Path.expanduser()` and resolve
symlinks via `Path.resolve()` at load time, storing the canonical
absolute path internally. Config files use `~` for portability;
internal state uses absolute paths for correctness.

### CLI Commands

```bash
# Daemon lifecycle (wraps launchctl)
sr daemon start          # load launchd plist, start daemon
sr daemon stop           # unload plist, stop daemon
sr daemon status         # show running state, loaded repos, uptime
sr daemon logs           # tail daemon logs

# Repo management (talks to running daemon via HTTP)
sr add ~/dev/project     # register + auto-index
sr remove project        # unregister, optionally delete index
sr repos                 # list registered repos with status

# Query (unchanged — talks to daemon)
sr ask "auth flow" --repo marrow
sr ask "auth flow"       # searches all repos, returns with repo context
```

`sr serve` SHOULD remain as an alias / compatibility mode that
starts the daemon in the foreground without launchd.

`sr ask` SHOULD become a thin HTTP client that queries the
running daemon. If the daemon is not running, it SHOULD fall
back to the current in-process behavior (load model, query,
exit) so the CLI always works — just slower without the daemon.

### HTTP API Changes

New and modified endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Unchanged — liveness check |
| `GET` | `/repos` | List registered repos with index status |
| `POST` | `/repos` | Add a repo: `{"path": "/abs/path"}` |
| `DELETE` | `/repos/{name}` | Remove a repo |
| `POST` | `/query` | Unchanged — add optional `repo` filter |
| `POST` | `/refresh` | Incremental re-index; optional `repo` and `files` fields |
| `GET` | `/repos/{name}/status` | Per-repo index status with progress |

When `repo` is omitted from `/query`, the daemon SHOULD search
all registered repos and return results tagged with the repo name.
This is the default mode for consumers that don't know or care
which repo contains the answer.

### Targeted Refresh

The current `/refresh` endpoint re-indexes all changed files in a
repo. For refresh-on-write, consumers often know exactly which
files they changed. A targeted refresh avoids the full git-diff:

```
POST /refresh
{
  "repo": "tallow",
  "files": ["src/tallow/agent.ts", "src/tallow/tools.ts"]
}
```

The daemon MUST still verify content hashes — if the file hasn't
actually changed (e.g. the consumer speculatively refreshes), it
SHOULD skip re-chunking.

When `files` is omitted, the current behavior (full git-diff
detection) applies.

### Auto-Index on Registration

When a repo is added (via API or config), the daemon MUST:

1. Check if an index already exists for that repo path
2. If yes: load it and run migrations
3. If no: queue a background full index build
4. Serve queries immediately for already-indexed repos
5. Return 503 for repos still being indexed, with an
   `X-Index-Status: building` header and ETA if available

The full index build SHOULD run in a background thread/process
so it doesn't block queries on other repos.

### Index Progress

The daemon MUST expose per-repo index status and progress so
consumers can surface it to users (tallow footer widget, ears
status, afk job logs).

#### Endpoint: `GET /repos/{name}/status`

```json
{
  "name": "marrow",
  "path": "/Users/marcus/dev/marrow",
  "state": "indexing",
  "index": {
    "files_total": 758,
    "files_indexed": 342,
    "chunks": 4210,
    "vectors": 4210,
    "progress_pct": 45.1,
    "elapsed_s": 48.2,
    "eta_s": 58.6
  },
  "last_indexed_at": "2026-03-20T10:39:25Z",
  "last_refreshed_at": "2026-03-20T11:15:02Z",
  "vec_dirty": false
}
```

The `state` field MUST be one of:

| State | Meaning |
|-------|---------|
| `ready` | Index is current, queries return results |
| `indexing` | Full build in progress (first index or rebuild) |
| `refreshing` | Incremental refresh in progress |
| `error` | Indexing failed — `error_detail` field explains why |
| `queued` | Repo registered, auto-index queued but not yet started |

During `indexing` or `refreshing`, the `index` object MUST
include `files_total`, `files_indexed`, and `progress_pct`.
`eta_s` is OPTIONAL — the daemon SHOULD compute it from the
rolling average per-file time.

The `GET /repos` endpoint (listing all repos) SHOULD include
a summary `state` per repo so consumers can get a quick
overview without polling each one:

```json
{
  "repos": [
    {"name": "marrow", "state": "ready", "chunks": 8849},
    {"name": "tallow", "state": "indexing", "progress_pct": 45.1},
    {"name": "afk", "state": "ready", "chunks": 2100}
  ]
}
```

#### SSE Stream: `GET /repos/{name}/progress`

For consumers that want live updates (tallow footer, web
dashboards), the daemon SHOULD offer a Server-Sent Events
stream:

```
GET /repos/marrow/progress
Accept: text/event-stream

data: {"files_indexed": 342, "files_total": 758, "progress_pct": 45.1}
data: {"files_indexed": 343, "files_total": 758, "progress_pct": 45.3}
...
data: {"state": "ready", "chunks": 8849, "vectors": 8849}
```

The stream MUST close when the operation completes (state
transitions to `ready` or `error`). Consumers that don't
need live updates can poll `/repos/{name}/status` instead.

#### Consumer Usage

**tallow extension** — polls `/repos/{name}/status` or
subscribes to the SSE stream. Renders in the TUI footer:

```
 sr: marrow ✓  tallow ⟳ 45%  afk ✓
```

When a repo is indexing, the agent MAY be informed via system
prompt that search results for that repo could be incomplete.

**ears** — checks `state` field at startup. If the configured
codebase is `indexing`, logs a warning and falls back to
ChromaDB until the daemon reports `ready`.

**afk** — checks `state` before querying during the doc phase.
If `indexing`, the job can either wait (with a bounded timeout)
or skip the source-recall query and note it in the job log.

### Process Management

#### launchd plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.source-recall.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/marcus/.local/bin/sr</string>
        <string>daemon</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/marcus/.local/share/source-recall/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/marcus/.local/share/source-recall/daemon.err</string>
</dict>
</plist>
```

`sr daemon start` MUST:

1. Write the plist to `~/Library/LaunchAgents/`
2. Call `launchctl load`
3. Wait for `/health` to respond (timeout 30s for model load)
4. Print status

`sr daemon stop` MUST:

1. Call `launchctl unload`
2. Verify the process exited
3. Leave the plist in place for next start

### Refresh Serialization

The existing `IndexBuilder` uses a PID-file lock for
multi-process isolation (prevents two `sr index` commands from
colliding). In a single-process daemon handling concurrent HTTP
requests, this lock causes problems:

- Two consumers fire `POST /refresh` for the same repo → the
  second caller gets `IndexLockError` after 5s timeout
- Periodic refresh is running when a targeted refresh arrives →
  same collision

The daemon MUST own refresh serialization internally via a
per-repo `asyncio.Lock` (or threading lock). All refresh
requests — targeted, periodic, and auto-index — go through
this lock. The behavior when a refresh is already in progress:

- **Targeted refresh** — queue the file list. When the current
  refresh completes, merge all queued file lists and run one
  combined refresh. This coalesces rapid-fire writes from an
  agent editing multiple files.
- **Periodic refresh** — skip if a refresh is already running
  for that repo. The next cycle will catch any remaining changes.
- **Auto-index (full build)** — blocks targeted/periodic until
  complete. Full builds are rare (only on first registration).

The PID-file lock SHOULD remain for external `sr index` CLI
calls that run outside the daemon process. The daemon MUST
acquire the PID lock before starting any refresh to prevent
collision with external builds. If the PID lock is held by an
external process, the daemon's refresh SHOULD retry after a
delay rather than failing the HTTP request.

### Config File Safety

The `POST /repos` endpoint writes to `repos.toml`. To prevent
a read-during-write race with the config watcher, all writes
MUST use atomic write (write to a temporary file in the same
directory, then `os.rename`). This guarantees the watcher
always reads a complete file.

### Periodic Background Refresh

In addition to refresh-on-write, the daemon SHOULD run a
periodic refresh cycle (default: every 5 minutes) that checks
all registered repos for changes. This catches changes made by
humans (git pull, manual edits) that no agent signaled.

If a periodic refresh takes longer than the interval, the next
cycle MUST be skipped (not queued). This prevents unbounded
refresh backlogs on slow repos.

The interval SHOULD be configurable:

```toml
[daemon]
refresh_interval_seconds = 300   # default: 5 minutes
port = 7249                      # default
host = "127.0.0.1"              # default, localhost only
query_timeout_s = 10             # default: per-request timeout
shutdown_timeout_s = 10          # default: graceful drain window
```

### Daemon Reliability

These three concerns are specific to a long-running daemon and
do not apply to the CLI tool. They prevent data corruption,
boot loops, and consumer starvation.

#### Graceful Shutdown

launchd sends SIGTERM when stopping the daemon. If the daemon
is mid-write (inserting vectors, flushing chunks during a
refresh), an unhandled SIGTERM can corrupt the WAL or leave
partial state in the index.

The daemon MUST install a SIGTERM handler that:

1. Stops accepting new requests (close the listening socket)
2. Waits for in-flight `/refresh` and `/query` requests to
   complete (bounded by `shutdown_timeout_s`, default 10s)
3. Closes all IndexStore and apsw connections cleanly
4. Exits with code 0

FastAPI's lifespan shutdown hook handles step 3, but uvicorn
needs `--timeout-graceful-shutdown` configured to drain
in-flight requests before the lifespan runs. The launchd plist
SHOULD set `ExitTimeOut` to match:

```xml
<key>ExitTimeOut</key>
<integer>15</integer>
```

The 5s gap between `shutdown_timeout_s` (10s) and `ExitTimeOut`
(15s) is intentional: uvicorn drains in-flight requests during
the shutdown timeout, then FastAPI's lifespan shutdown hook runs
to close DB connections. The extra 5s ensures launchd doesn't
SIGKILL the process before lifespan cleanup completes.

If a refresh is in progress when SIGTERM arrives, the current
batch SHOULD be committed (not rolled back) so partial progress
is preserved. The next startup will detect remaining changes
and continue.

#### Startup Resilience

If one repo has a corrupt index, a missing path, or a schema
version from a newer source-recall release, the daemon MUST
start anyway and serve the healthy repos. A single bad repo
MUST NOT crash the process or trigger a launchd restart loop.

On startup, the daemon MUST:

1. Load the repo registry
2. For each repo, attempt to open and migrate the index
3. If a repo fails: log the error, mark it as `state: error`
   with `error_detail`, and continue to the next repo
4. Load the embedding model (shared across all repos)
5. If the model fails to load (corrupt cache, disk full,
   missing weights): log the error and start in FTS-only
   degraded mode. The `/health` endpoint MUST report
   `"mode": "fts_only"` so consumers know vector search
   is unavailable. Queries still work via BM25.
6. Begin serving — healthy repos return results, errored repos
   return 503 with the error detail

The `sr daemon status` command and `GET /repos` endpoint MUST
surface errored repos and the daemon mode (`hybrid` or
`fts_only`) so the user can diagnose and fix them (rebuild,
remove, fix the path, or clear the HuggingFace cache).

#### Query Timeout

A pathological query on a large repo (bad FTS match pattern,
huge result set, slow embedding) can block a uvicorn worker
thread. In a daemon serving multiple consumers, one slow query
MUST NOT starve others.

The daemon MUST enforce a per-request timeout (configured via
`query_timeout_s` in the `[daemon]` config block above).

If a query exceeds the timeout, the daemon MUST return:

```json
{"detail": "Query timed out after 10.0s", "code": "timeout"}
```

with HTTP 504. The underlying SQLite query is cancelled via
`sqlite3.Connection.interrupt()` (stdlib) or the apsw
equivalent.

The timeout applies to `/query` and `/refresh`. The `/repos`
and `/health` endpoints are metadata-only and SHOULD NOT need
timeouts.

### Consumer Integration

#### tallow (extension in tallow-plugins)

A tallow extension that:

1. On session start, probes `GET /health` on the daemon
2. If available, registers custom tools:
   - `sr_search` — semantic code search across repos
   - `sr_navigate` — find symbol definitions and references
3. After any `write` or `edit` tool call, fires
   `POST /refresh {"repo": "<current>", "files": [<changed>]}`
4. Falls back to grep/find if daemon is unavailable

The extension SHOULD inject a system prompt fragment telling
the agent that `sr_search` is available and faster than grep
for semantic queries.

The refresh-on-write hook SHOULD be non-blocking — fire and
forget. If the refresh fails, the agent continues. Stale results
are better than blocked edits.

#### ears

ears already probes `/health` and falls back to ChromaDB. With
the daemon always running, the fallback path becomes rare. No
changes needed beyond updating the health check to verify the
specific repo ears cares about is loaded:

```
GET /health → check repos list includes the configured codebase
```

#### afk

afk's document phase can query the daemon once it's stable:

```
POST /query {"question": "docs related to auth", "repo": "marrow"}
```

No special integration needed — standard HTTP client.

## Error Handling

| Code | Condition | Recovery |
|------|-----------|----------|
| `200` | Query/refresh succeeded | — |
| `400` | Missing required field | Return validation error |
| `404` | Repo not found in registry | Suggest `sr add <path>` |
| `429` | Refresh rate limited | Retry after `Retry-After` header |
| `503` | Repo still indexing | Return `X-Index-Status: building`, client retries |
| `503` | Model still loading (startup) | Return `X-Status: loading`, client waits |

The daemon MUST NOT crash on indexing failures for individual
repos. A broken repo (missing path, parse errors) SHOULD be
marked as `error` in the registry and skip future refresh cycles
until the path exists again.

## Security Considerations

- The daemon listens on `127.0.0.1` only — MUST NOT bind to
  `0.0.0.0` by default
- CORS is restricted to localhost origins (implemented in
  this sprint)
- The `POST /repos` endpoint accepts arbitrary filesystem paths.
  The daemon MUST verify the path exists and is a directory
  before indexing. It SHOULD NOT follow symlinks outside the
  user's home directory to prevent traversal attacks in
  multi-user environments
- No authentication — acceptable for a localhost-only single-user
  service. If `--host 0.0.0.0` is used, a warning MUST be printed
- The config file at `~/.config/source-recall/repos.toml` is
  user-owned. No daemon should write to it with elevated
  privileges

## Alternatives Considered

### Per-consumer server instances

Each consumer starts its own `sr serve`. Rejected: wastes ~500MB
RAM per instance for the embedding model, and indexes would need
separate refresh coordination.

### Library import (no server)

Consumers import source-recall as a Python dependency. Rejected:
puts torch/sentence-transformers/sqlite-vec into every consumer's
process. The ears migration doc explicitly calls this out as
the reason for the HTTP boundary.

### MCP server only (via tool-proxy)

Register source-recall as a tool-proxy app. This handles the
agent-tool pattern but doesn't help ears (which uses raw HTTP)
and adds a proxy hop. The daemon should be the primary interface;
tool-proxy registration MAY be added later as an additional
front door for MCP-native consumers.

### fswatch / inotify for auto-refresh

Watch the filesystem for changes instead of refresh-on-write.
Rejected for now: high CPU on large repos, doesn't distinguish
agent writes from build artifacts, and platform-specific. The
periodic background refresh plus agent-driven refresh-on-write
covers the important cases. Can revisit if the 5-minute periodic
window is too long for human edits.

## Implementation Plan

### Phase 1: Repo registry and daemon mode

- Add `~/.config/source-recall/repos.toml` config parsing
- Add `sr daemon run` (foreground, reads registry)
- Add `POST /repos` and `DELETE /repos/{name}` endpoints
- Add `sr add` / `sr remove` / `sr repos` CLI commands
- Auto-index on repo add (blocking for Phase 1)
- Tests for registry CRUD and config watching

**Deliverable:** `sr daemon run` serves all configured repos.

### Phase 2: launchd and lifecycle

- Generate and install launchd plist
- `sr daemon start` / `sr daemon stop` / `sr daemon status`
- Log rotation config
- Graceful shutdown (close all DB connections)
- Health check waits for model load

**Deliverable:** `sr daemon start` survives reboots.

### Phase 3: Index progress and targeted refresh

- Add `GET /repos/{name}/status` endpoint with progress fields
- Add `GET /repos/{name}/progress` SSE stream
- Extend `GET /repos` with per-repo summary state
- Extend `/refresh` with `files` parameter
- Add periodic background refresh loop (configurable interval)
- `vec_dirty` integration — re-embed degraded files on next
  periodic refresh

**Deliverable:** Indexes stay current, consumers can observe progress.

### Phase 4: tallow extension

- tallow-plugins extension that registers `sr_search` tool
- Post-write hook fires targeted refresh
- Footer widget showing per-repo index state (✓ ready, ⟳ 45%)
- System prompt injection for agent awareness
- Graceful fallback when daemon is down

**Deliverable:** tallow agents can `sr_search` instead of grep,
with live index status in the footer.

### Phase 5: Cross-repo query

- `/query` without `repo` searches all registered repos
- Results tagged with repo name and path
- Ranking across repos (normalize scores)

**Deliverable:** "how does auth work" searches everything.

## Open Questions

1. **Config file location** — `~/.config/source-recall/repos.toml`
   vs `~/.local/share/source-recall/repos.toml` (next to indexes).
   Config conventions say `~/.config`, but co-locating with data
   simplifies backup. Which?

2. **Cross-repo ranking** — when searching all repos, how should
   scores be normalized? RRF scores aren't comparable across
   indexes with different corpus sizes. Options: (a) re-rank
   merged results with the cross-encoder, (b) normalize by
   corpus size, (c) return per-repo results separately and let
   the consumer merge.

3. **Tallow extension tool design** — should `sr_search` be one
   tool that returns code chunks, or two tools (`sr_search` for
   semantic and `sr_symbols` for exact symbol lookup)? One tool
   is simpler for the agent; two tools give it more control.

4. **Config hot-reload vs restart** — should the daemon watch
   `repos.toml` for changes and auto-reload, or require
   `sr daemon reload`? Hot-reload is convenient but adds
   complexity and a class of bugs. A manual reload signal
   (SIGHUP or API call) might be simpler.

## References

### Normative

- [source-recall README](../README.md) — current CLI and server docs
- [RFC-01: Git-object-based branch-aware indexing](01_git-object-based-branch-aware-indexing.rfc.md) — index architecture

### Informative

- [ears SOURCE-RECALL.md](~/dev/ears/SOURCE-RECALL.md) — ears migration plan documenting the HTTP boundary decision
- [afk ROADMAP.md](~/dev/afk/ROADMAP.md) — planned source-recall integration for doc-phase search
- [tallow extensions docs](~/dev/tallow/docs/extensions.md) — extension API for Phase 4
