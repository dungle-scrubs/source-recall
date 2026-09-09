---
number: 01
title: "Shared repo resolution and error contract"
type: refactor
status: Draft
author: Kevin Frilot
date: 2026-09-06
---

# RFC-01: Shared repo resolution and error contract

## Abstract

The daemon and the query server both answer one question on every request:
which repo does this request mean. That question is answered by four separate
copies of the same branch ladder, and the copies have drifted apart in status
code, message text, and readiness checks. This RFC defines one resolution
operation over a registry protocol that both servers satisfy, a typed error set
for the three failure modes, and a single HTTP mapping per server. It covers the
resolution seam and the client-visible error contract for repo addressing. It
does not merge the two servers, change the retrieval pipeline, or alter the
response bodies of successful requests.

## Introduction

### Problem statement

`source-recall` ships two HTTP servers. `create_app` (`src/source_recall/server.py`)
serves a fixed set of repos loaded at startup with no authentication.
`create_daemon_app` (`src/source_recall/daemon.py`) serves a dynamic registry
behind a token. Both accept an optional `repo` name on their endpoints, and both
have to turn that optional name into an `Index` or into an error.

That logic exists in four places:

| Copy | Location | Registry it reads |
|---|---|---|
| `_resolve_index` | `daemon.py:416-455` | `RepoManager.slots` |
| inline, in `/query` | `daemon.py:846-891` | `RepoManager.slots` |
| inline, in `/refresh` | `daemon.py:936-958` | `RepoManager.slots` |
| `_resolve_index` | `server.py:384-404` | `state["indexes"]` |

The two daemon endpoints cannot call the daemon's own `_resolve_index` because
it returns only the `Index`: `/refresh` needs the slot for its per-repo lock and
its rate-limit key, and `/query` needs the whole usable set for fan-out. So each
grew its own copy, and the copies diverged. Four defects follow directly, all
confirmed by reading the cited lines.

**1. One client condition, two answers.** With no `repo` supplied and no repo
usable, `_resolve_index` and `/query` return 503 `No repos ready`
(`daemon.py:430-434`, `:890-891`). `/refresh` reaches `elif repo is None` first
and returns 400 `Multiple repos loaded. Specify 'repo': one of []`
(`daemon.py:941-945`) — a 400 that names an empty list and tells the client to
pick from nothing.

**2. Three wordings for one state.** `Repo '{name}' is in state '{state}' (not ready)`
(`daemon.py:448-453`), `Repo '{name}' not ready (state: {state})`
(`daemon.py:877-880`), and the same again at `daemon.py:955-958`. `cli.py` reads
`detail` back out of these responses at `cli.py:300`, `:1268`, `:1293`.

**3. One copy omits the recovery hint.** The 404 in `/refresh`
(`daemon.py:946-950`) is `Repo '{name}' not found.` with no `Available: [...]`
suffix. The other three include the list.

**4. Only one copy checks that the `Index` is present.** `/query` filters on
`s.state == SlotState.READY and s.index is not None` (`daemon.py:846-850`).
`_resolve_index` and `/refresh` filter on state alone. `RepoSlot.close()` sets
`self.index = None` and leaves `state` at `READY` (`repo_manager.py:172-179`),
so between `close_all` and process exit a `/status` request resolves a slot and
calls `None.status()`, returning 500 instead of 503.

Separately, all four copies iterate `manager.slots` outside the lock that
`RepoManager.add` and `RepoManager.remove` take (`repo_manager.py:231-238`,
`:252-257`), while the class docstring at `repo_manager.py:184-186` states the
registry is thread-safe. FastAPI dispatches these sync endpoints on a
threadpool, so `POST /repos` can resize the dict during a `/query` comprehension
and raise `RuntimeError: dictionary changed size during iteration`. The lifespan
loop and the periodic-refresh loop already take a `list(...)` snapshot
(`daemon.py:258`, `:286`); the request handlers do not.

### Motivation

The audit of 2026-09-05 found these as one design candidate and three separate
correctness findings. The divergence is not stable: three of the four copies
were added after the first, each because the shared one could not express what
the endpoint needed. Any new repo-addressable endpoint adds a fifth copy on the
same trajectory. Fixing the four defects one at a time leaves the shape that
produced them.

### Scope

In scope:

- One resolution operation and one usable-set operation, over a registry
  protocol satisfied by both servers.
- A typed error set for the three failure modes, and one HTTP mapping per
  server.
- The canonical `detail` string for each failure mode.
- Thread-safe reads of the daemon registry from request handlers.
- The `index is not None` condition, applied uniformly.
- The `/refresh` rate-limit key on the query server, which currently buckets one
  repo twice (`server.py:508`, `repo or "_default"`, so `repo=None` and
  `repo="<name>"` are separate buckets for the same index; the daemon keys on
  the resolved `slot.name` at `daemon.py:963`).

Out of scope, with the reason each was declined:

- **Merging `create_app` and `create_daemon_app`.** They are two products: an
  unauthenticated server over a fixed repo set, and a token-gated server over a
  dynamic registry. Merging them is a product decision that changes what both
  offer, and no user has asked for it. The duplication inside them is what this
  RFC addresses.
- **Not-ready semantics for the query server.** `create_app` builds every
  `Index` synchronously in its lifespan (`server.py:304-317`), so a loaded repo
  is always usable and there is no not-ready state to report. Adding one would
  add a second code path for a case no current user has. The registry protocol
  therefore lets an adapter declare every registered repo usable, and the query
  server never raises `RepoNotReady`.
- **Making the query server use `RepoManager`.** That would import slot
  lifecycle, background indexing, and cancellation into a server that loads
  eagerly and never removes a repo. The narrower option, a protocol both
  registries satisfy, serves the same purpose.
- **The response-shape duplication** (`QueryResult` and `IndexStatus` projected
  by hand on eleven surfaces). A real finding, a separate seam, and no
  dependency on this one.
- **`ParseFidelity`, the `ReaderWriterLock` borrow protocol, the `DaemonClient`,
  and the `_DDL`/`_MIGRATIONS` duplication.** Each is its own design decision
  from the same audit and warrants its own RFC or, in the case of
  `_DDL`/`_MIGRATIONS`, a direct change with no RFC.
- **The audit's critical and high findings** (the test that unloads the user's
  LaunchAgent, targeted refresh bypassing exclusions, dirty-file detection
  dropping files, porcelain path quoting, the discarded vector-flush result,
  the per-file commit storm). Those are bugs. The design is "make it correct",
  so they are fixed directly, not specified here. Three findings that live
  inside the code this RFC rewrites are folded in and named in scope above.

## Terminology

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as
described in RFC 2119.

**Repo** — a repository root that a server can answer queries about. Addressed
by a short name, which defaults to the directory basename.

**Slot** (`RepoSlot`, `repo_manager.py:34`) — the daemon's per-repo record. Owns
the `Index`, the per-repo lock, the slot state, and the progress fields.

**Slot state** (`SlotState`, `repo_manager.py:20-30`) — `queued`, `indexing`,
`ready`, or `error`. A daemon concept; the query server has no equivalent.

**Registry** — the collection a server resolves names against: `RepoManager.slots`
for the daemon, `state["indexes"]` for the query server.

**Handle** — whatever the registry returns for one repo. `RepoSlot` for the
daemon; a record carrying the name and the `Index` for the query server.

**Usable** — a repo that can serve a request right now. For the daemon: slot
state is `ready` AND `slot.index is not None`. For the query server: registered.

**Resolution** — turning an optional repo name plus a registry into one handle,
or into one typed error.

**Detail** — the `detail` field of a FastAPI `HTTPException`, which reaches the
client as the JSON body `{"detail": "..."}` and which `cli.py` parses.

## Current State

### Behavior today, by condition and call site

`n` is the number of usable repos. Cells marked in bold are the divergences.

| Condition | `daemon._resolve_index` | daemon `/query` | daemon `/refresh` | `server._resolve_index` |
|---|---|---|---|---|
| no name, n = 1 | handle | handle | handle | handle |
| no name, n = 0 | 503 `No repos ready` | 503 `No repos ready` | **400 `Multiple repos loaded. Specify 'repo': one of []`** | **400, same text** (unreachable from `sr serve`, which defaults `paths` to `["."]` at `cli.py:900`; reachable via `create_app([])`) |
| no name, n > 1 | 400 `Multiple repos loaded...` | fans out | 400 `Multiple repos loaded...` | 400 `Multiple repos loaded...` |
| unknown name | 404 `... not found. Available: [...]` | 404 `... not found. Available: [...]` | **404 `... not found.`** | 404 `... not found. Available: [...]` |
| known name, not ready | 503 **`is in state '{s}' (not ready)`** | 503 **`not ready (state: {s})`** | 503 **`not ready (state: {s})`** | n/a, no state concept |
| known name, ready, `index is None` | **returns `None`, caller raises 500** | 503 `not ready` | **returns a slot with no index** | n/a |

### Registry shapes

The daemon registry is `RepoManager.slots: dict[str, RepoSlot]`
(`repo_manager.py:189`), public and mutable, guarded for writes by
`RepoManager._lock` and read directly by five request handlers
(`daemon.py:424`, `:535`, `:552`, `:846`, `:938`).

The query server registry is `state["indexes"]: dict[str, dict[str, Any]]`,
built at `server.py:317` as `{"index": idx, "path": repo_path}` and read at
`server.py:391-404`.

### Consequences

- A client cannot write one handler for "repo not usable" because the status
  code depends on which endpoint it called.
- A client cannot match on `detail` because the wording depends on which
  endpoint it called. `cli.py:300`, `:1268`, and `:1293` read `detail`.
- The registry's stated thread-safety is not delivered to request handlers.
- Adding an endpoint means choosing which of four copies to imitate.

## Proposed Changes

### New module: `src/source_recall/repo_resolution.py`

A registry protocol, two operations, and no I/O.

```python
H = TypeVar("H")

class RepoRegistry(Protocol[H]):
    """A named collection of repos that can be resolved against."""

    def names(self) -> list[str]:
        """Every registered repo name, usable or not. Snapshot."""

    def usable(self) -> Mapping[str, H]:
        """Registered repos that can serve a request now. Snapshot."""


def resolve(registry: RepoRegistry[H], name: str | None) -> H:
    """Resolve an optional repo name to exactly one handle."""


def usable_or_raise(registry: RepoRegistry[H]) -> Mapping[str, H]:
    """The usable set, or RepoNotReady when it is empty."""
```

Both operations MUST call `names()` and `usable()` at most once per invocation,
and MUST NOT retain either result. Both MUST be free of HTTP types: this module
MUST NOT import from `fastapi`.

### Resolution semantics

`resolve` MUST apply these conditions in order and MUST NOT reorder them:

1. `name` is given and is absent from `names()` → raise `RepoNotFound(name, available=names())`.
2. `name` is given and is absent from `usable()` → raise `RepoNotReady(name)`.
3. `name` is given → return `usable()[name]`.
4. `name` is omitted and `usable()` is empty → raise `RepoNotReady(None)`.
5. `name` is omitted and `usable()` holds exactly one entry → return it.
6. `name` is omitted and `usable()` holds more than one → raise `RepoAmbiguous(available=sorted(usable()))`.

Condition 1 MUST precede condition 2 so that an unknown name and a known but
unusable name stay distinguishable. Condition 4 MUST precede condition 6; that
ordering is the fix for the `/refresh` 400-on-empty defect, which exists only
because the current code tests ambiguity before emptiness.

`usable_or_raise` MUST raise `RepoNotReady(None)` on an empty usable set and
MUST otherwise return the mapping unchanged. Fan-out endpoints use it; they MUST
NOT treat more than one usable repo as ambiguous.

### Error set

Three errors, added to `src/source_recall/models.py` beside the existing
hierarchy, all deriving from `SourceRecallError`:

```python
class RepoResolutionError(SourceRecallError): ...

class RepoNotFound(RepoResolutionError):
    name: str
    available: list[str]

class RepoAmbiguous(RepoResolutionError):
    available: list[str]

class RepoNotReady(RepoResolutionError):
    name: str | None      # None when no name was supplied
    state: str | None     # slot state when the adapter has one
```

Each error MUST carry its data as attributes, not only in the message, so a
caller can act without parsing strings. `RepoNotReady.state` is OPTIONAL and is
`None` for registries with no state concept.

### HTTP mapping

One function maps the error set to a status and a detail. Each server registers
it once as a FastAPI exception handler; route bodies call `resolve` and let the
error propagate. No route SHOULD wrap `resolve` in `try`/`except`.

| Error | Status | Canonical detail |
|---|---|---|
| `RepoNotFound` | 404 | `Repo '{name}' not found. Available: {available}` |
| `RepoAmbiguous` | 400 | `Multiple repos loaded. Specify 'repo': one of {available}` |
| `RepoNotReady`, `name is None` | 503 | `No repos ready` |
| `RepoNotReady`, `name` given | 503 | `Repo '{name}' not ready (state: {state})` |

Three of these four strings are already the dominant wording in the current
code, so the mapping keeps them verbatim. The fourth, the by-name not-ready
string, is the wording used by two of the three daemon copies; the third copy's
`is in state '{state}' (not ready)` is dropped. When `state` is `None`, the
suffix ` (state: {state})` MUST be omitted rather than rendering `None`.

### Adapters

**Daemon.** `RepoManager` gains two methods that satisfy the protocol and take
`_lock`:

```python
def names(self) -> list[str]:
    with self._lock:
        return list(self.slots)

def usable(self) -> Mapping[str, RepoSlot]:
    with self._lock:
        return {
            n: s for n, s in self.slots.items()
            if s.state == SlotState.READY and s.index is not None
        }
```

Both return a new object built under the lock, so a caller iterating the result
cannot observe a concurrent `add` or `remove`. Request handlers MUST use these
methods and MUST NOT read `RepoManager.slots` directly. `RepoManager.slots`
SHOULD become `_slots` once no caller outside the class reads it. That rename is
sequenced last so the behavioral change and the mechanical one land in separate
diffs, not because it is large: `.slots` appears 19 times in `daemon.py`, 10
times inside `repo_manager.py` itself, and 16 times across four test files
(12 in `tests/test_repo_manager.py`, 2 in `tests/test_remove_repo_shutdown.py`,
1 each in `tests/test_audit_fixes.py` and `tests/test_periodic_refresh.py`).

`RepoNotReady.state` for this adapter is the slot state, except that a slot in
state `ready` with `index is None` reports state `closed`. Reporting `ready` for
a repo the resolver just rejected would be self-contradictory.

**Query server.** A small adapter over the existing dict, added in `server.py`:

```python
class _EagerRegistry:
    """Every registered repo is usable: create_app builds them all at startup."""

    def __init__(self, indexes: dict[str, dict[str, Any]]) -> None:
        self._indexes = indexes

    def names(self) -> list[str]:
        return list(self._indexes)

    def usable(self) -> Mapping[str, RepoHandle]:
        return {n: RepoHandle(name=n, index=e["index"]) for n, e in self._indexes.items()}
```

`RepoHandle` is a frozen dataclass carrying `name` and `index`. `RepoSlot`
already carries both attributes, so both adapters yield handles with a common
readable surface without a shared base class.

### Call sites after the change

| Call site | Before | After |
|---|---|---|
| `daemon._resolve_index` | 40-line ladder | `resolve(manager, repo).index` |
| daemon `/query`, named | inline ladder | `resolve(manager, req.repo)` |
| daemon `/query`, fan-out | inline `ready` comprehension | `usable_or_raise(manager)` |
| daemon `/refresh` | inline ladder | `slot = resolve(manager, repo)` |
| daemon `/health`, `/repos` | direct `slots` iteration | `manager.names()` / `manager.usable()` |
| `server._resolve_index` | 20-line ladder | `resolve(_EagerRegistry(state["indexes"]), repo).index` |
| `server` `/refresh` limiter key | `repo or "_default"` | the resolved handle's `name` |

### API surface changes

- Added: `repo_resolution.RepoRegistry`, `resolve`, `usable_or_raise`,
  `RepoHandle`; `models.RepoResolutionError`, `RepoNotFound`, `RepoAmbiguous`,
  `RepoNotReady`; `RepoManager.names`, `RepoManager.usable`.
- Modified: `RepoManager.slots` becomes internal by convention in phase 1 and
  private by rename in phase 5.
- Removed: `daemon._resolve_index`'s ladder body and both inline ladders;
  `server._resolve_index`'s ladder body. The two `_resolve_index` names remain
  as one-line wrappers.
- Client-visible: one status code changes and two `detail` strings change. See
  Migration Strategy.

## Error Handling

Error codes below are documentation identifiers for this RFC. They are not
emitted on the wire; the wire contract is the status code plus `detail`.

```
E001 - RepoNotFound: the supplied name is not registered (severity: warning)
       Wire: 404, detail includes the available names.
       Recovery: client picks a name from `available`, or from GET /repos.
       Retry: not retryable without changing the request.
       Escalation: none. A human chose a wrong name.

E002 - RepoAmbiguous: no name supplied, more than one repo usable (severity: warning)
       Wire: 400, detail includes the usable names.
       Recovery: client resends with `repo` set.
       Retry: not retryable without changing the request.
       Escalation: none.

E003 - RepoNotReady, no name supplied: nothing is usable yet (severity: info)
       Wire: 503, detail `No repos ready`.
       Recovery: transient during daemon startup and background indexing.
       Retry: client SHOULD retry with backoff, starting at 1s, capped at 30s.
       Escalation: a repo stuck in `error` state never becomes usable; the
       client SHOULD stop retrying after 5 attempts and read GET /repos, whose
       per-repo `error` field carries the reason.

E004 - RepoNotReady, name supplied: that repo is not usable (severity: info)
       Wire: 503, detail names the repo and its state.
       Recovery: as E003, and the state distinguishes the cases: `queued` and
       `indexing` resolve on their own; `error` does not; `closed` means the
       server is shutting down.
       Retry: as E003, and a client MUST NOT retry on state `error`.
       Escalation: as E003.
```

Transient versus permanent: E003 and E004 are transient except for state
`error`. E001 and E002 are permanent for an unchanged request.

The resolution operations raise domain errors only. A registry method that
itself fails is a programming error and MUST propagate as a 500; it MUST NOT be
mapped to any of the four codes above, because reporting an internal fault as
"not ready" would send clients into a retry loop against a broken server.

`RepoSlot.close()` SHOULD set slot state to a terminal value rather than leaving
`ready`, so the `index is None` condition and the slot state agree. Until it
does, the `usable()` filter is the guard and the resolver reports `closed`.

## Security Considerations

**Trust boundaries.** The daemon authenticates every route with a bearer or
`X-SR-Token` header (`daemon.py:193-213`, `:375`). The query server has no
authentication by design (`server.py:251-258`) and relies on a loopback bind
plus `TrustedHostMiddleware`. This RFC does not change either posture. The
shared resolution module is reached only after each server's own authentication
layer has run, so moving the logic into it removes no check.

**Input validation.** `name` is client-supplied on both servers and reaches the
resolver as an arbitrary string. The resolver MUST treat it as an opaque key: it
MUST NOT use it to build a filesystem path, and MUST NOT use it in a message
without the surrounding quoting the canonical detail strings already apply. The
resolver reflects the supplied name back in the `RepoNotFound` detail, so the
HTTP layer MUST bound the reflected name: a name longer than 200 characters MUST
be truncated in the detail. Without that bound a client can inflate an error
body with a multi-megabyte name.

**Name disclosure.** `RepoNotFound` and `RepoAmbiguous` list the registered repo
names, which are directory basenames and therefore weak information about the
operator's filesystem. On the daemon this is behind the token. On the
unauthenticated query server the same names are already returned by `/health`
and `/repos` to any caller that reaches the port, and its CORS policy admits any
localhost origin (`server.py:372-382`), so withholding them from the 404 would
hide nothing. The list is therefore kept on both servers. This is a decision,
not an oversight; narrowing the query server's disclosure is a separate change
that has to cover `/health` and `/repos` as well.

**Blast radius.** The resolver sits in front of every repo-addressed endpoint on
both servers. A defect that resolves the wrong handle would serve one repo's
indexed source in answer to a query naming another. That is the worst case, and
it is why condition ordering is normative above and why the contract test in
Testing Strategy asserts identity of the returned handle rather than only the
status code. A defect that raises where it should resolve is a denial of
service, bounded by the client's retry policy.

**Data sensitivity.** No credentials pass through this module. The daemon token
is validated before the resolver runs and is never an input to it.

**Concurrency as a security property.** Replacing unlocked dict iteration with
snapshots removes a `RuntimeError` that currently surfaces as a 500 on an
unrelated request whenever a repo is added concurrently. That is availability,
not confidentiality, but it is reachable by any token holder that calls
`POST /repos` while queries are in flight.

## Alternatives Considered

**Keep four copies, fix the four defects in place.** Attractive because it is
the smallest diff and touches no interface. Rejected because it leaves the shape
that produced the divergence: the copies exist precisely because the shared
`_resolve_index` cannot return a slot or a set, and nothing in the fix changes
that. The next endpoint adds a fifth copy.

**Put `resolve` on `RepoManager` and have the query server construct one.** This
was the audit's own suggestion. Attractive because it needs no new protocol.
Rejected because `RepoManager` carries slot lifecycle, background index threads,
cancellation, and bounded shutdown, none of which the query server has any use
for; adopting it there would mean maintaining a second calling convention for
that machinery. The protocol keeps the query server's eager dict.

**One FastAPI dependency (`Depends`) shared by both apps.** Attractive because
FastAPI would inject the handle and map errors without an explicit exception
handler. Rejected because the dependency would have to reach the registry
through app state, which differs between the two apps, and because `/refresh`
and `/query` need the handle in the body of the route for locking and fan-out,
not merely as a parameter. The plain function plus one exception handler keeps
the resolution testable without a request object.

**Return a result object instead of raising.** Attractive because it makes every
failure mode visible in the return type and avoids exceptions for control flow.
Rejected because every call site would then have to branch, which is the branch
ladder this RFC removes, and because the existing codebase already models
domain failures as exceptions under `SourceRecallError`.

**Normalize the status codes without extracting a module.** Attractive as a
one-day change that fixes the client-visible contract. Rejected because the
contract would then be maintained by convention across four sites, which is the
condition the audit recorded; the second and third divergence arrived that way.

## Migration Strategy

Two of the three client-visible changes are strictly additive. One is breaking.

**Breaking: `/refresh` on the daemon, no `repo` supplied, nothing usable.**
400 becomes 503, and the detail changes from
`Multiple repos loaded. Specify 'repo': one of []` to `No repos ready`. A client
that treats 4xx as permanent and 5xx as retryable changes behavior: it starts
retrying where it previously gave up. That is the intended behavior, since the
condition does clear once indexing finishes. No test pins the current behavior. A
scan of `tests/` for the affected strings finds only
`tests/test_audit_daemon.py:161`, which asserts `No repos ready` on `/query` and
stays green. The two tests that call `/refresh` without `repo` both run against
exactly one usable repo and take the `n = 1` path:
`tests/test_e2e_daemon.py:445` and `tests/test_audit_fixes.py:1050-1053`. The
second is also the rate-limit-key case from Testing Strategy item 6, and stays
green because both of its calls resolve to the same single repo under either
key.

**Additive: `/refresh` 404 gains `Available: [...]`.** A client matching on the
prefix is unaffected. A client matching the full string breaks; none is known,
and `cli.py` prints `detail` rather than matching it.

**Additive: the by-name not-ready detail** changes at one of three sites, from
`Repo 'x' is in state 'indexing' (not ready)` to
`Repo 'x' not ready (state: indexing)`. The status code does not change.

Backward compatibility guarantees:

- Status codes for every condition except the one named above MUST be unchanged.
- Successful response bodies MUST be unchanged; this RFC does not touch them.
- The `repo` query parameter and body field keep their names and meaning.
- `daemon._resolve_index` and `server._resolve_index` keep their names and
  signatures as wrappers, so any caller outside the routes is unaffected.

No dual-write, feature flag, or data migration is required: the change is
in-process and stateless. Rollback is a revert of the phase in question, since
no phase writes persistent state.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A condition is reordered during extraction, so an unknown name reports not-ready or the reverse | low | high: wrong status for a permanent error, and clients retry forever | The ordering is normative above; the contract test enumerates all six conditions against both adapters. |
| The resolver returns the wrong handle under a name collision | very low | critical: one repo's source answers a query naming another | The contract test asserts handle identity, not just status. Names are dict keys, so collision is not representable. |
| A snapshot taken under `_lock` is held while an `Index` operation runs, extending lock hold time | low | medium: `add`/`remove` block behind a query | `usable()` builds and returns a new mapping and does no I/O; the route uses the snapshot after the lock is released. |
| `usable()` allocates a dict per request on a hot path | medium | low: one small dict per repo per request | The current code already builds the same comprehension per request at `daemon.py:846-850`. No regression. |
| A client depends on the `/refresh` 400 | low | medium | Named in Migration Strategy; no test pins it; the CLI does not match on it. |
| The `RepoManager.slots` rename breaks many tests at once | high if done early | medium: a large mechanical diff obscuring the behavioral one | Sequenced last, as its own phase, with no behavior change in it. |

Blast radius if the whole change is wrong: every repo-addressed endpoint on both
servers returns wrong errors or wrong handles. That is the entire HTTP surface
except `/health`. This is why phase 1 lands the module and its tests before any
call site moves, and why each later phase moves one server at a time.

## Testing Strategy

The point of the tests is that the four copies became four behaviors. The
regression guard is therefore one contract test parameterized over both
adapters, not per-endpoint assertions.

1. **Resolution contract, both adapters.** One parameterized suite over a fake
   registry and a real `RepoManager`, asserting all six conditions from
   Resolution semantics: returned handle identity for the two success cases, and
   error type plus attributes for the four failure cases. The daemon adapter
   additionally covers `ready` with `index is None`.
2. **Ordering.** Explicit cases for the two orderings that are normative:
   unknown-and-unusable resolves to `RepoNotFound`, not `RepoNotReady`; and
   no-name-with-zero-usable resolves to `RepoNotReady`, not `RepoAmbiguous`.
   The second is the regression test for the `/refresh` defect.
3. **HTTP mapping.** A table test over the four errors asserting status and the
   canonical detail, including the `state is None` case that omits the suffix
   and the long-name truncation from Security Considerations.
4. **Endpoint parity.** For each condition, assert that `/query`, `/refresh`,
   and `/status` on the daemon return the same status and the same detail. This
   test would fail today on three of six conditions, which is the point.
5. **Concurrency.** A test that calls `POST /repos` from one thread while
   another loops `/query`, asserting no 500. It fails today with
   `RuntimeError: dictionary changed size during iteration` given enough
   iterations; it is inherently probabilistic, so it MUST be written with a
   bounded iteration count and MUST NOT gate CI on a timing threshold.
6. **Rate-limit key.** Assert that `POST /refresh` and `POST /refresh?repo=<the
   only repo>` share one bucket on the query server.
7. **Existing tests.** `tests/test_server.py:231`, `:260`, `:287` (ambiguity
   returns 400) and `tests/test_audit_daemon.py:144`, `:160` (zero ready returns
   503 with `No repos ready`) MUST stay green unmodified. They are the
   backward-compatibility check.

The full suite (`just test`) is green at 648 passed as of 2026-09-05 and MUST
stay green at every phase boundary.

## Implementation Plan

Each phase is independently revertible and leaves the suite green.

**Phase 1 — the module, no call sites.** Add `repo_resolution.py`, the three
errors in `models.py`, and `RepoHandle`. Add tests 1, 2, and 3. Nothing imports
the module in production yet.
Verify: new tests pass; full suite unchanged.
Rollback: delete the module. Nothing depends on it.

**Phase 2 — daemon adapter and registry snapshots.** Add `RepoManager.names`
and `usable`. Move the five daemon handlers off direct `slots` iteration. No
resolution behavior changes yet.
Verify: full suite green; test 5 added and passing.
Rollback: revert; handlers return to direct iteration.

**Phase 3 — daemon resolution.** Replace `_resolve_index`'s body and both inline
ladders with `resolve` / `usable_or_raise`. Register the exception handler. This
phase contains the one breaking change and the two detail changes.
Verify: tests 4 and 7; the `/refresh` zero-usable case now returns 503.
Go/no-go: if endpoint parity (test 4) cannot be made to pass without changing a
status code not named in Migration Strategy, stop and amend this RFC.
Rollback: revert. The client contract returns to its current divergent state.

**Phase 4 — query server.** Add `_EagerRegistry`, replace
`server._resolve_index`'s body, register the exception handler, fix the
rate-limit key.
Verify: tests 1 and 6; `tests/test_server.py` ambiguity tests green unmodified.
Rollback: revert; the daemon keeps phase 3.

**Phase 5 — seal the registry.** Rename `RepoManager.slots` to `_slots` and
update the test references. No behavior change.
Verify: full suite green; no production reader of `_slots` outside
`RepoManager`.
Rollback: revert the rename.

Phases 1 and 2 have no client-visible effect and can land together. Phase 3 is
the only phase that changes the wire contract and SHOULD land alone. Phase 5 is
optional and can be deferred indefinitely without leaving the codebase in an
inconsistent state.

Dependencies: 2 depends on 1; 3 depends on 2; 4 depends on 1; 5 depends on 3.
Phase 4 does not depend on phase 3 and can land first if the breaking change
needs to wait.

## Open Questions

1. **Does the `/refresh` status change need a deprecation window?**
   Options: (a) change it in phase 3 as specified; (b) emit 503 only when a
   request header opts in, for one release. Criterion: whether any client
   outside this repository calls `POST /refresh` without `repo`. The CLI does
   not (`cli.py` always resolves a name first). Recommended: (a), because the
   current 400 names an empty list and is not a contract any client can be
   relying on deliberately. Decided (a) by the drafting agent, unattended;
   confirm before phase 3 lands. Decider: repository owner.

2. **Should `RepoSlot.close()` move the slot out of `ready`?**
   Options: (a) leave it and let `usable()` be the guard, as specified; (b) add
   a `CLOSED` member to `SlotState`. Criterion: whether any consumer reads slot
   state for a closed slot; `/repos` and the CLI icon map both would render an
   unknown state. Recommended: (b) in a follow-up, since it makes state and
   usability agree at the source rather than at the filter, but it widens
   `SlotState`, which is a product-visible enum. Not decided.

3. **Does the query server keep the `Available:` list on 404?**
   Options: (a) keep it, as specified; (b) drop it on the unauthenticated
   server only. Criterion: whether the same names are obtainable elsewhere on
   that server; today they are, from `/health` and `/repos`. Recommended: (a),
   with any narrowing handled as one change across all three endpoints.
   Decided (a) by the drafting agent, unattended. Decider: repository owner.

4. **Is the 200-character truncation bound right?**
   Options: (a) 200; (b) reject names over the bound with 400 before resolution;
   (c) no bound. Criterion: the longest legitimate repo name, which is a
   directory basename and so is bounded by the filesystem at 255 bytes.
   Recommended: (b) with the bound at 255, since a name that cannot be a
   basename cannot be registered and rejecting it is cheaper than reflecting it.
   Not decided.

5. **Which RFC comes next from the same audit?**
   Options: `ParseFidelity` (one enum for a concept spread over twelve
   surfaces), the `ReaderWriterLock` borrow protocol (highest risk, touches the
   concurrency core), or the `DaemonClient` (removes twelve type suppressions
   and the token-to-arbitrary-host leak). Criterion: whether the concurrency
   core is being touched for other reasons in the same window; if it is, the
   borrow protocol goes first to avoid two passes. Not decided. Decider:
   repository owner.

## References

### Normative

- `src/source_recall/daemon.py` — `_resolve_index` at 416-455, `/query`
  resolution at 846-891, `/refresh` resolution at 936-958, auth dependency at
  193-213. The three daemon copies this RFC replaces.
- `src/source_recall/server.py` — `_resolve_index` at 384-404, registry
  construction at 304-317, refresh limiter at 495-520, CORS at 372-382. The
  fourth copy and the query server's trust posture.
- `src/source_recall/repo_manager.py` — `SlotState` at 20-30, `RepoSlot` at
  34-59, `RepoSlot.close` at 172-179, `RepoManager` and its lock at 181-238.
  The daemon registry and the adapter's home.
- `src/source_recall/models.py` — the `SourceRecallError` hierarchy the three
  new errors join.
- [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) — the keyword definitions
  this document uses.

### Informative

- `docs/history/AUDIT_2026-09-05.md` — the audit that produced this RFC.
  Design candidate 3.1 is the seam; findings 2.15 (unlocked registry
  iteration), 2.61 (refresh limiter key), and 2.67 (`close()` leaves `ready`
  with no index) are folded in; candidate 3.9 and findings 2.14 and 2.83 are
  adjacent and out of scope.
- `AGENTS.md` — the repository's load-bearing constraints. None of the six
  constrain this seam, which is why no invariant section appears above.
- `docs/history/critiques/architecture.md` — sections 2 and 4, the original
  argument for decomposing the god object and for treating daemon concurrency
  as a day-one requirement rather than a contingency.
- `tests/test_server.py:231-305`, `tests/test_audit_daemon.py:135-162` — the
  existing tests that pin the parts of the contract this RFC preserves.
