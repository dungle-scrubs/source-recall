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

The daemon SHOULD resolve `~` and symlinks at load time and
store the canonical absolute path internally.

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

### HTTP API Changes

New and modified endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Unchanged — liveness check |
| `GET` | `/repos` | List registered repos with index status |
| `POST` | `/repos` | Add a repo: `{"path": "/abs/path"}` |
| `DELETE` | `/repos/{name}` | Remove a repo |
| `POST` | `/query` | Unchanged — add optional `repo` filter |
| `POST` | `/refresh` | Unchanged — accepts optional `repo` |
| `POST` | `/refresh` | Extended: `{"files": ["src/a.py"]}` for targeted refresh |

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

### Periodic Background Refresh

In addition to refresh-on-write, the daemon SHOULD run a
periodic refresh cycle (default: every 5 minutes) that checks
all registered repos for changes. This catches changes made by
humans (git pull, manual edits) that no agent signaled.

The interval SHOULD be configurable:

```toml
[daemon]
refresh_interval_seconds = 300   # default: 5 minutes
port = 7249                      # default
host = "127.0.0.1"              # default, localhost only
```

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

### Phase 3: Targeted refresh and background refresh

- Extend `/refresh` with `files` parameter
- Add periodic background refresh loop (configurable interval)
- `vec_dirty` integration — re-embed degraded files on next
  periodic refresh

**Deliverable:** Indexes stay current via periodic + targeted refresh.

### Phase 4: tallow extension

- tallow-plugins extension that registers `sr_search` tool
- Post-write hook fires targeted refresh
- System prompt injection for agent awareness
- Graceful fallback when daemon is down

**Deliverable:** tallow agents can `sr_search` instead of grep.

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

4. **Index build during daemon startup** — if a configured repo
   has no index yet, should the daemon block startup until it's
   built, or start serving other repos immediately and build in
   the background? Background is better UX but more complex.

5. **Config hot-reload vs restart** — should the daemon watch
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
