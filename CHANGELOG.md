# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2](https://github.com/dungle-scrubs/source-recall/compare/v0.1.1...v0.1.2) (2026-09-09)


### Documentation

* brand marks, social preview, 2026-09-05 audit, RFC-01 ([#25](https://github.com/dungle-scrubs/source-recall/issues/25)) ([4397d91](https://github.com/dungle-scrubs/source-recall/commit/4397d91a98a3079a597a0d175c9dd933bcc1fd30))


### Maintenance

* relock after v0.1.1 ([#26](https://github.com/dungle-scrubs/source-recall/issues/26)) ([7c1d442](https://github.com/dungle-scrubs/source-recall/commit/7c1d442d74dec3efc9f53c9d4f4724a3a61ab0b8))

## [0.1.1](https://github.com/dungle-scrubs/source-recall/compare/v0.1.0...v0.1.1) (2026-09-09)


### Added

* **cli:** add sr --version ([b8f726e](https://github.com/dungle-scrubs/source-recall/commit/b8f726ef1b5b70c825d04c9f81310abcc2c2c8ee))


### Fixed

* **builder,daemon:** close refresh read-path data loss and bound shutdown ([b670b8a](https://github.com/dungle-scrubs/source-recall/commit/b670b8a6ed82252df150d96b7b4979cdfc225b85))
* **builder,store:** prevent refresh data loss and cross-device sidecar corruption ([90b93b3](https://github.com/dungle-scrubs/source-recall/commit/90b93b3b86fa5f5cfc892ca148f64e3839751109))
* **chunker,builder,embedder:** stage 5 chunker & builder correctness ([d461ea1](https://github.com/dungle-scrubs/source-recall/commit/d461ea10f8f193fba835f6e88901bac0bd50e963))
* **daemon:** authenticate routes, validate Host, contain repo paths ([6e736b8](https://github.com/dungle-scrubs/source-recall/commit/6e736b80efa5779d1601c55e02dc0b7a669014cf))
* **daemon:** clear _stop_event on lifespan startup ([0ae41b1](https://github.com/dungle-scrubs/source-recall/commit/0ae41b143e1603a9f39d4bb2fed20b8503c3fb2f))
* **daemon:** remove completed bg index threads from tracking list ([f4ba7e4](https://github.com/dungle-scrubs/source-recall/commit/f4ba7e42b499de26e72bb46ce520c50c8e37555e))
* **daemon:** warm models at startup, embed fan-out once, honor shutdown timeout ([acd0987](https://github.com/dungle-scrubs/source-recall/commit/acd0987940ae1bbe69891a52689c67fe8c8b8136))
* **embedder:** verify config.json checksum before trusting remote model ([04c4beb](https://github.com/dungle-scrubs/source-recall/commit/04c4bebefce585f0cf09c85838c75f18fb6265ad))
* fail closed on git-status detection failure and bound shutdown ([1862aa7](https://github.com/dungle-scrubs/source-recall/commit/1862aa70b4bc2d01ee74a655609f0ae712a80d5c))
* **packaging:** complete PyPI metadata and fix stale build pin ([70880b0](https://github.com/dungle-scrubs/source-recall/commit/70880b07e77e8d2b67cc6ec4e10ff49bbef38286))
* **querier:** typed search boundary, graph-expand gate, reader self-heal ([1fe955a](https://github.com/dungle-scrubs/source-recall/commit/1fe955a516828f3fb200fec25ff26cc984534f78))
* **refresh:** preserve index entries on transient access failures ([d43b373](https://github.com/dungle-scrubs/source-recall/commit/d43b373cc5cb7c4460d2c1833c9b095f5c62175a))
* **security:** harden daemon auth and model-integrity gate ([68a9e8e](https://github.com/dungle-scrubs/source-recall/commit/68a9e8e7447d052aa02717ec64314bb9f477be86))
* **store:** batch symbol IN-list and make per-name cap quality-aware ([91b00dd](https://github.com/dungle-scrubs/source-recall/commit/91b00dd5e0f0a4d72fe2c2a4af89d349a4563cfc))
* **store:** harden atomic_swap, memoize vec-table check, bound symbol lookup ([53f5c8d](https://github.com/dungle-scrubs/source-recall/commit/53f5c8dd2ced0a9ec56749f4316e18e79c9dcc10))
* **store:** release savepoints after ROLLBACK TO on failure paths ([81a93e3](https://github.com/dungle-scrubs/source-recall/commit/81a93e3528c90e78674ef5ab9e01fd5c13e51cb8))
* **verify:** reconcile flaky bg-index thread-join test ([04b2e1c](https://github.com/dungle-scrubs/source-recall/commit/04b2e1c12124fed19cbab4486f4e73b6133c389e))


### Documentation

* add CODE_OF_CONDUCT.md and SECURITY.md ([f709385](https://github.com/dungle-scrubs/source-recall/commit/f70938591c01f3a96add32d5d63ea46a2a9da40a))
* **audit:** remove incorrect findings H-3 and old M-1 ([4cf1ac1](https://github.com/dungle-scrubs/source-recall/commit/4cf1ac1b03c99d9103e2714ae08a0e1c95a10e0b))
* **changelog:** record packaging metadata and toolchain fixes ([c263335](https://github.com/dungle-scrubs/source-recall/commit/c2633353f89d8f3f643bf30708e4cb98426c832c))
* **changelog:** record version flag and launchd cleanup ([e6218b9](https://github.com/dungle-scrubs/source-recall/commit/e6218b9d85e0bd86123f48b5ee7080da447fdcf5))
* open-source release-readiness audit ([ba5d008](https://github.com/dungle-scrubs/source-recall/commit/ba5d0089d3be0f01c7f71a8b882810eded111eb7))
* relocate internal docs to docs/history, add CoC and SECURITY ([1962051](https://github.com/dungle-scrubs/source-recall/commit/1962051f71633a9a3b77cc4df2f653e52176c312))
* remove old root copies of relocated history docs ([67b5719](https://github.com/dungle-scrubs/source-recall/commit/67b571952238d597ac665b8656abdd980c1366a8))
* **roadmap:** dedupe Phase 2 section, drop stale blocker list ([adfa222](https://github.com/dungle-scrubs/source-recall/commit/adfa222f18e3724cc2806673d86f287884d05f2d))
* **roadmap:** mark Phase 1c as Done; add release-readiness section ([2e31e19](https://github.com/dungle-scrubs/source-recall/commit/2e31e19d2d87ea919959697f2c4d75a0bf46df38))


### Maintenance

* add GitHub Actions workflow; fix lint; track uv.lock ([f23cfaf](https://github.com/dungle-scrubs/source-recall/commit/f23cfaf267e4f467b458920959f4177e84ff6249))
* **deps-dev:** update uv-build requirement ([#17](https://github.com/dungle-scrubs/source-recall/issues/17)) ([760fa5d](https://github.com/dungle-scrubs/source-recall/commit/760fa5df2bc3a5f0d4a4b04f69ed5a584d81df63))
* **deps:** bump actions/cache from 4.3.0 to 6.1.0 ([#22](https://github.com/dungle-scrubs/source-recall/issues/22)) ([88b85de](https://github.com/dungle-scrubs/source-recall/commit/88b85def1c7a477f39b377577f5f74b61fad14d4))
* **deps:** bump sqlite-vec in the pip-minor-patch group ([#23](https://github.com/dungle-scrubs/source-recall/issues/23)) ([aecbfca](https://github.com/dungle-scrubs/source-recall/commit/aecbfca2923aa195f67c7e1236b5f7168134afce))
* harden for public release — templates, CI concurrency, deps, topics ([ec5333c](https://github.com/dungle-scrubs/source-recall/commit/ec5333cdaeb92d7b999d7a53a80f4cb1e69700df))
* **launchd:** remove static plist and deprecated recipes ([37a9c87](https://github.com/dungle-scrubs/source-recall/commit/37a9c8706bb87c7ffe2a5751fd8b1f991ac21917))
* **pins,launchd:** guard tree-sitter pins, cover launchd, de-flake refresh ([4b53844](https://github.com/dungle-scrubs/source-recall/commit/4b53844cb836e2ff55f1a3bb91b540de4b429533))
* **pyproject:** remove unused [tool.ty] block ([d98584f](https://github.com/dungle-scrubs/source-recall/commit/d98584fbbb40c6246c29cb29d0c6b222537d3f1a))
* release/publish parity with scraper ([#21](https://github.com/dungle-scrubs/source-recall/issues/21)) ([2d7ed20](https://github.com/dungle-scrubs/source-recall/commit/2d7ed20e4b27ff25fc8e5353ec9d017de2140aa3))
* **release:** add install smoke before uv build ([47c6be3](https://github.com/dungle-scrubs/source-recall/commit/47c6be376f2ca7b2d6810a6db57ee752b311080e))
* rename .mise.toml to mise.toml ([f50740a](https://github.com/dungle-scrubs/source-recall/commit/f50740ab0598c8b78c911a2f37e8288f73060eb8))
* **toolchain:** align uv pin to 0.11.25 across mise and CI ([3a0a07c](https://github.com/dungle-scrubs/source-recall/commit/3a0a07c38cae70ecadd1d2a0a078d08129d873a7))
* untrack author-local .plans/ and .tallow/ scratch dirs ([7219375](https://github.com/dungle-scrubs/source-recall/commit/721937577708468f8d48846c47d9b2cc1cab2a8a))

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
