# Open-Source Release Readiness Audit

Date: 2026-06-30
Auditor: Trevor (post-host-restart sweep)
Tree: `f4ba7e4` (post-fix)
Scope: license, README, packaging, CI, install path, dependency hygiene,
secrets, test infrastructure, repo hygiene. Reads only — no code changes
proposed beyond this report.

Verdict: **Not ready to publish.** Three blockers (license, CI, test
isolation contract for `sentence-transformers`); one high-severity
issue (release artifact distribution); ~12 medium items; ~8 low.

---

## 🔴 Blockers (must fix before publishing)

### B-1. No LICENSE file

`LICENSE` does not exist at the repo root. `pyproject.toml` lists an
author email (`dungle-scrubs@users.noreply.github.com`) but no `license`
field in `[project]`. Without a license, the code is "all rights
reserved" by default in most jurisdictions — contributors have no
permission to use, modify, or distribute. This blocks publishing on
PyPI and GitHub.

Recommended action: add an MIT or Apache-2.0 `LICENSE` file and a
matching `license = { text = "MIT" }` (or SPDX expression) entry in
`[project]` of `pyproject.toml`. The README already advertises
CodeRankEmbed as MIT-licensed.

### B-2. No CI configuration

`.github/` does not exist. No `.circleci/`, no `.gitlab-ci.yml`. There
is no automated check that `uv run pytest` or `uv run ruff check` pass
on a fresh checkout. For an open-source repo this is table stakes — a
PR that breaks the suite should fail CI before a maintainer reads it.

Recommended action: add `.github/workflows/ci.yml` that runs `mise
install && uv sync && uv run pytest -m "not slow" && uv run ruff check
src/ tests/` on Python 3.12 and 3.13. The matrix is cheap because the
default tests use the `BagOfWordsEmbedder` test double and don't pull
the real model (see B-3 for the caveat).

### B-3. `pyproject.toml` lacks an optional `embed` extra for the heavy ML deps

`sentence-transformers`, `onnxruntime`, `einops` (and their transitive
`torch`) are listed as required dependencies. The README says
"`SR_EMBED_ENABLED=false` for pure FTS keyword search (no model
download, instant startup)". But the *import-time* requirement means
`uv tool install -e .` always installs ~2 GB of torch even for users
who never want vector search.

Furthermore, `sentence-transformers` is lazily imported inside
`embedder.py` and `reranker.py`, and tests use `BagOfWordsEmbedder`,
so the dependency is only needed at runtime when the embedder is
actually loaded. It should be optional.

Recommended action: split `dependencies` into core + an `embed` extra:

```toml
[project.optional-dependencies]
embed = ["sentence-transformers>=3.0.0,<4.0.0", "onnxruntime>=1.18.0,<2.0.0", "einops>=0.8.0,<1.0.0"]

[dependency-groups]
dev = ["pytest>=8.0.0", "ruff>=0.9.0", "source-recall[embed]"]
```

Update README install: `uv tool install -e .[embed]` or
`uv tool install -e .` for FTS-only mode.

---

## 🟠 High-Severity

### H-1. No release artifact / distribution story

There is no `dist/`, no GitHub Release workflow, no PyPI publishing
config. `pyproject.toml` uses the `uv_build` backend — fine — but no
`[tool.uv_build]` block, no version bump workflow (`release-please` is
in the available skills roster), no changelog. Users who want to pin a
specific version have no released artifacts to pin against.

Recommended action: pick a versioning policy (CalVer or SemVer),
add a `CHANGELOG.md`, and either (a) wire up `release-please` or (b)
publish to PyPI on tag.

### H-2. README does not document Python version requirement

`pyproject.toml` requires `>=3.12`. `mise.toml` pins `python = "3.13"`.
`.python-version` is `3.13` (one byte — I assume it's `3.13`; file
shows 5 bytes which is plausible). The README has no "Requirements"
section. New users will install under 3.11 and get a cryptic error
from uv.

Recommended action: add a `## Requirements` section in README
stating Python ≥ 3.12 and that mise is recommended.

### H-3. `apsw` listed as a hard dependency even though it's only used on macOS for sqlite-vec

`store.py:27` says "sqlite-vec availability (requires apsw for
extension loading on macOS)". Yet `apsw` is in core `dependencies`,
not an `[os-specific]` extra. On Linux the stdlib `sqlite3` can load
extensions via `conn.enable_load_extension(True) + sqlite_vec.load()`,
which would let the project drop `apsw` on Linux entirely. As-is,
Linux users pay for an unused dependency.

Recommended action: gate `apsw` on `sys.platform == "darwin"` either
via a marker (`"apsw>=3.49.0.0,<4.0.0.0; sys_platform == 'darwin'"`)
or a separate `[os-darwin]` extra. Verify `sqlite-vec` works on Linux
without `apsw` first; the design comment suggests it should.

---

## 🟡 Medium-Severity

### M-1. `.gitignore` excludes `uv.lock`

Line: `.gitignore:19`
The lock file is committed in the current tree (verified: `uv.lock`
exists at 296 KB) but is listed in `.gitignore`. Either remove the
line (if the project commits the lock) or document the decision.
Most open-source Python projects commit `uv.lock` for reproducible
installs; the current tree follows that practice but the gitignore
suggests the original intent was to not commit it.

### M-2. `.gitignore` allows `.claude/` to be ignored but doesn't add `.cursor/`, `.aider*`, etc.

Line: `.gitignore:20`
`.claude/` is gitignored, which is appropriate. But there's no entry
for other AI assistant state directories (`.aider*`, `.continue/`,
`AGENTS.local.md`, etc.). If a contributor runs a different AI tool,
its state files will leak into the PR.

Recommended action: ignore common AI-tool state: `.aider*`,
`.continue/`, `AGENTS.local.md`, `.cursor/`, `*.swp`.

### M-3. `pyproject.toml` `authors` uses a GitHub noreply email

Line: `pyproject.toml:6`
`authors = [{ name = "Kevin", email = "dungle-scrubs@users.noreply.github.com" }]`
For PyPI and `pip show`, the noreply email is fine. For the public
repo, contributors can't reach a maintainer via this address. If
support is intended, list a real contact channel (GitHub Discussions,
Discord, mailing list). Otherwise document that the project is
maintained on a best-effort basis.

### M-4. No `CODE_OF_CONDUCT.md`

GitHub displays a `CODE_OF_CONDUCT.md` link in the community
profile; missing it signals "we haven't thought about governance".
A simple `Contributor Covenant v2.1` is the minimum bar for OSS
publishing.

### M-5. No `CONTRIBUTING.md`

The README has a `## Development` section with the basics, but no
dedicated CONTRIBUTING guide for how to file issues, run a single
test, write a regression test, etc. AGENTS.md covers the TDD
discipline well but is internal documentation — a CONTRIBUTING.md
should reference it.

### M-6. No SECURITY.md / vulnerability disclosure policy

Standard for OSS. One-paragraph "report via GitHub Security Advisories
or email X with a 90-day disclosure window" is enough.

### M-7. `tests/test_reranker.py::TestRerankerProtocol::test_cross_encoder_reranker_reorders` is marked `@pytest.mark.slow` but is the ONLY model-loading test

Line: `tests/test_reranker.py:27`
The mark exists but no other test in the suite actually loads the
real embedder or reranker. This means CI without the `slow` mark
will never exercise the production embedding path. Either add a
small smoke test for `CodeRankEmbedder` (using a tiny fixture) or
note in CI that the slow suite runs nightly, not per-PR.

### M-8. `scripts/smoke.sh` is the only smoke test; no equivalent for daemon mode

Line: `scripts/smoke.sh`
The script exercises the CLI + HTTP server smoke path. It does not
exercise the daemon (`sr daemon run`). The daemon is documented as
the multi-repo / refresh story; it deserves its own smoke check.

### M-9. `launchd/dev.source-recall.serve.plist` hardcodes `/Users/kevin`

Line: `launchd/dev.source-recall.serve.plist:11` and `:18`
Hardcoded path `/Users/kevin/.local/bin/sr` and a comment "Edit these
paths to match your setup". This plist is checked into the repo,
which means anyone who clones gets a non-working plist. Either:
- Move the plist to a template (`launchd/dev.source-recall.serve.plist.template`)
  and have `just launchd-install` substitute the user's path, or
- Document loudly in README that the plist is a template requiring
  editing.

### M-10. `justfile` has 12+ deprecated recipes (`launchd-*`) that print deprecation warnings

Line: `justfile:88-127`
The `launchd-install/start/stop/logs` recipes print "DEPRECATED" but
remain in the file. For an OSS release, dead code in user-facing
recipes is confusing. Either remove them or move them to a
deprecated.md justfile target.

### M-11. No `ty` check in CI / `justfile`

`pyproject.toml` declares `[tool.ty]` (the Astral type checker) but
neither `just lint` nor any command runs `ty check`. If the project
intends to use `ty`, add a `just typecheck` recipe and run it in CI.
If not, remove the `[tool.ty]` block.

### M-12. README claim "30ms queries" is unverified on a multi-repo cold start

Line: `README.md:13` and `README.md:79`
The README states `sr serve` is "~30ms queries instead of 12s". The
12s model-load is plausible. The 30ms claim is not annotated with
conditions (single repo? cold or warm? query length?). For OSS
marketing it would be more credible to say "<100ms in practice;
depends on query complexity and index size".

### M-13. No example / demo

For a code-search tool, a 30-second demo (terminal recording or
screencast) is the highest-ROI documentation. The README has curl
examples but no GIF or asciinema. Optional but high-leverage.

---

## 🔵 Low / Nit

### L-1. `critiques/` directory contents are design rationale, not user-facing

Lines: `critiques/architecture.md`, `parsing.md`, `search.md`, `storage.md`
These are interesting read for a contributor but not for an end user.
Consider linking them from `docs/decisions/` or moving them to a
`docs/design/` path so the repo root stays clean.

### L-2. `PHASE1.md`, `PHASE1B.md`, `PHASE1C.md`, `PLAN.md`, `ROADMAP.md` mix roadmap status

`ROADMAP.md` says "Phase 1c — Reranking + cross-references" is
"In Progress" but the implementation is not present in `src/`. Either
ROADMAP is out of date or the feature is half-built. Update or mark
stale.

### L-3. `.claude/` is in `.gitignore` but `.claude/commands/audit.md` is checked in (5 KB)

Inconsistent — verify the intent. If the audit prompt is meant to
ship with the repo, remove `.claude/` from gitignore. If it's local
only, ensure it's not committed (current tree has it).

### L-4. `.tallow/rules/` exists but `.tallow/` itself is a custom tooling directory

Verify whether this is meant to ship. If it's a build artifact,
gitignore it.

### L-5. `pyproject.toml` `name = "source-recall"` will collide on PyPI with any pre-existing project

Search PyPI before publishing. `source-recall` is short and generic;
the name may be taken.

### L-6. README "Auto-Start (macOS)" section says launchd-only

No systemd / Windows Task Scheduler equivalent. For multi-platform
OSS users, this is a limitation worth acknowledging or implementing.

### L-7. `just smoke` accepts a path arg but defaults to `.`, which is the *test target* — confusing

Line: `justfile:9-10`
The `smoke` recipe defaults to `.` which runs the smoke test against
the source-recall repo itself. That's fine for development, but a
new user reading the justfile doesn't know this — they might run
`just smoke /path/to/their/repo` and get a different result. Document
or rename to `smoke-self` vs `smoke-against <path>`.

### L-8. No `MISE_VERSION` pin, only Python version

`.mise.toml` pins `python = "3.13"` but uv and just are pinned without
a strict version constraint. `uv = "0.11.15"` and `just = "1.51.0"`
are exact, good — verify they're not stale.

---

## ✅ What's already in good shape

- **Atomicity** — `atomic_swap` is well-tested (`test_atomic_swap_guard`,
  `test_data_fsync_precedes_rename`, `test_dir_fsync_follows_rename`).
- **Savepoint cleanup** — fixed in `81a93e3` with regression tests.
- **BG-thread tracking** — fixed in `f4ba7e4` with regression test.
- **Schema migrations** — wrapped in savepoints, forward-version
  rejection, indexed triggers.
- **Identity verification** — path-first per AGENTS.md.
- **Test count** — 506 tests, 1 known flake (`test_periodic_triggers_after_interval`).
- **No secrets in repo** — scanned, none found.
- **No TODO/FIXME in src/** — clean.
- **Pre-1.0 dependency pins noted in `pyproject.toml`** — sqlite-vec,
  sentence-transformers, apsw all have inline comments explaining
  why they are constrained.
- **Test fixtures are auto-cleaned** via `clean_index_dir` autouse
  fixture in `conftest.py` — no user-data pollution.

---

## Suggested fix order

1. **B-1**: Add LICENSE + `license` field. Five minutes.
2. **B-2**: Add `.github/workflows/ci.yml`. ~30 min including matrix.
3. **B-3**: Split `embed` extra. ~1 hour including test verification.
4. **H-1**: Pick version policy + first PyPI release. Half a day.
5. **H-2/H-3**: README + apsw gating. ~1 hour.
6. **M-1..M-13**: drive-by sweep. Half a day total.

Total to publishable: ~1 focused day of work. After that, point the
`release-please` skill at the repo and cut a v0.1.0.