# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **packaging:** Trove classifiers, keywords, and `[project.urls]`
  (Homepage, Repository, Issues, Changelog) in `pyproject.toml`. The
  PyPI listing now carries Python-version, topic, and audience
  classifiers plus direct links back to the project.
- **packaging:** `license-files = ["LICENSE"]` so the MIT LICENSE
  ships inside the sdist (`License-File: LICENSE` in METADATA) and is
  discoverable by license scanners.
- **cli:** `sr --version` (and `-v`) reports the installed version, derived
  from package metadata so it tracks release-please bumps automatically.
- **license:** MIT LICENSE file and SPDX license expression in `pyproject.toml`.
- **ci:** GitHub Actions workflow (`.github/workflows/ci.yml`) with Python 3.12
  and 3.13 matrix, a nightly slow-test job (model-loading), and an
  end-to-end smoke job.
- **packaging:** Optional `embed` install extra so FTS-only users can
  `uv tool install -e .` without pulling ~2 GB of torch. Full install
  is now `uv tool install -e .[embed]`.
- **docs:** `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and
  `SECURITY.md` (private disclosure policy).
- **tests:** `test_packaging.py` pins the embed-extra split contract
  (lazy import, extra declaration, dev-group inclusion).
- **tests:** `test_savepoint_cleanup.py` regression tests for the
  savepoint-release failure path.
- **tests:** `test_bg_index_shutdown.py` regression test for completed
  background-thread cleanup.
- **tests:** `test_periodic_refresh.py` regression test for the
  module-level stop-event bleed.
- **tests:** `test_version.py` covers the `--version` flag.

### Fixed
- **store:** `run_migrations`, `_transaction`, and nested `batch_mode`
  now explicitly `RELEASE` the savepoint after `ROLLBACK TO` on
  failure paths. SQLite retains the savepoint in the transaction
  stack after `ROLLBACK TO`; without the explicit `RELEASE`, repeated
  failures in the same transaction window accumulated nested
  savepoints of the same name.
- **daemon:** Background index threads are removed from the
  `bg_threads` tracking list after they finish. Previously the list
  was append-only, so a long-lived daemon accumulated dead `Thread`
  objects and shutdown joined every historical thread.
- **daemon:** `_stop_event` is cleared at lifespan startup. The
  module-level event was set on shutdown but never cleared, so a
  second daemon in the same process (e.g. the next test) saw the
  event already set and its periodic refresh loop exited before the
  first tick.

### Changed
- **launchd:** Removed the static `launchd/dev.source-recall.serve.plist`
  (which hardcoded `/Users/kevin`) and the deprecated `launchd-*`
  justfile recipes. `sr daemon start` already generates a path-resolved
  plist dynamically via `launchd.generate_plist()`.
- **deps:** `uv.lock` is now tracked (un-ignored via `!uv.lock`) for
  reproducible CI installs. `*.lock` still catches runtime `.db.lock`
  PID-file sidecars.
- **deps:** `apsw` stays in core dependencies on all platforms.
  `store.py` uses its extension-loading API as the load path for
  sqlite-vec because stdlib `sqlite3` is built without
  `SQLITE_ENABLE_LOAD_EXTENSION` on most distros.
- **config:** Renamed `.mise.toml` to the modern `mise.toml` form.
- **lint:** Added `tests/*` per-file-ignores for `SIM117`,
  `ARG001`, and `ARG002` (conventional test-only relaxations).
  Applied `ruff format` across the tree.
- **docs:** Relocated `PHASE1*.md`, `PLAN.md`, `OPEN_SOURCE_READINESS.md`,
  and `critiques/` under `docs/history/`. Gitignored author-local
  `.plans/` and `.tallow/`. Removed the unused `[tool.ty]` block.
- **docs:** README gains a Requirements section, a Support section,
  and a qualified query-latency claim.

### Fixed
- **build:** `build-system.requires` upper bound bumped from
  `uv_build<0.11.0` to `<0.12.0`. The previous constraint excluded the
  uv_build series that CI (`uv==0.11.x`) ships; `uv build` only warned,
  but PEP 517 frontends honoring `build-system.requires` (pip, `build`)
  would have selected an incompatible backend.
- **tracking:** `.plans/` and `.tallow/` were listed in `.gitignore`
  but never untracked. `git rm --cached` removes the 7 author-local
  scratch files from the index; they remain on disk, ignored.
- **toolchain:** Bumped the uv pin from `0.11.15` to `0.11.25` in
  `mise.toml`, `.github/workflows/ci.yml`, and
  `.github/workflows/release.yml` so local dev and CI agree on the uv
  binary version.
