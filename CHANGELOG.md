# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **license:** MIT LICENSE file and SPDX license expression in `pyproject.toml`.
- **ci:** GitHub Actions workflow (`.github/workflows/ci.yml`) with Python 3.12
  and 3.13 matrix, a nightly slow-test job (model-loading), and an
  end-to-end smoke job.
- **packaging:** Optional `embed` install extra so FTS-only users can
  `uv tool install -e .` without pulling ~2 GB of torch. Full install
  is now `uv tool install -e .[embed]`.
- **docs:** `OPEN_SOURCE_READINESS.md` audit report and `ROADMAP.md`
  release-readiness section.
- **tests:** `test_packaging.py` pins the embed-extra split contract
  (lazy import, extra declaration, dev-group inclusion).
- **tests:** `test_savepoint_cleanup.py` regression tests for the
  savepoint-release failure path.
- **tests:** `test_bg_index_shutdown.py` regression test for completed
  background-thread cleanup.
- **tests:** `test_periodic_refresh.py` regression test for the
  module-level stop-event bleed.

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
- **deps:** `uv.lock` is now tracked (un-ignored via `!uv.lock`) for
  reproducible CI installs. `*.lock` still catches runtime `.db.lock`
  PID-file sidecars.
- **deps:** `apsw` stays in core dependencies on all platforms.
  `store.py` uses its extension-loading API as the load path for
  sqlite-vec because stdlib `sqlite3` is built without
  `SQLITE_ENABLE_LOAD_EXTENSION` on most distros.
- **lint:** Added `tests/*` per-file-ignores for `SIM117`,
  `ARG001`, and `ARG002` (conventional test-only relaxations).
  Applied `ruff format` across the tree.
- **docs:** README gains a Requirements section and clarifies that
  vector-search config flags require the `embed` extra.
