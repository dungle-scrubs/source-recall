# Contributing to source-recall

Thanks for your interest in contributing. This guide covers the
basics. For the full design rationale and the load-bearing
constraints the codebase depends on, read
[`AGENTS.md`](./AGENTS.md) and the [`critiques/`](./critiques/)
directory.

## Requirements

- Python ≥ 3.12 (3.13 recommended)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [mise](https://mise.jdx.dev/) optional but recommended — pins
  Python, uv, and just to known-good versions

```bash
mise install        # optional: pins toolchain versions
uv sync             # installs core + dev dependencies (incl. embed extra)
```

## Development workflow

### Running tests

```bash
just test           # full suite (includes slow/model-loading tests)
just t              # short output

# or directly:
uv run pytest -m "not slow"    # fast suite, no model download (~70s)
uv run pytest                  # full suite including model-loading tests
uv run pytest tests/test_store.py -k "test_atomic_swap"  # single test
```

The `slow` marker selects tests that download ~500 MB of model
weights (CodeRankEmbed / the cross-encoder reranker). CI runs the
fast suite per-PR and the slow suite nightly on `main`.

### Lint and format

```bash
just lint           # check
just fix            # auto-fix
```

### Test-first discipline

Every code change follows **RED → GREEN → REFACTOR**. Write one
failing test, implement the minimum to make it pass, then clean up.
Never write more than one test before implementing. Never skip
running the full suite between phases. See `AGENTS.md` for the full
rationale — this is mandatory, not optional.

Skip TDD only for exploratory spikes or trivial glue code.

### Index storage during tests

Tests never touch `~/.local/share/source-recall/`. The
`clean_index_dir` autouse fixture in `conftest.py` redirects all
index storage to `tmp_path`. Any new code path that calls
`get_index_dir` or `get_db_path` automatically uses the
monkeypatched version.

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
for automatically generating the changelog and bumping versions via
[release-please](https://github.com/googleapis/release-please).

Format:

```
<type>(<scope>): <description>

<optional body>
```

| Type | Maps to changelog | Example |
|------|-------------------|---------|
| `feat` | Features | `feat(chunker): add Rust support` |
| `fix` | Bug Fixes | `fix(store): release savepoint on rollback` |
| `perf` | Performance | `perf(querier): cache symbol lookups` |
| `docs` | Documentation | `docs(readme): add install instructions` |
| `refactor` | *(hidden)* | `refactor(daemon): extract lifespan setup` |
| `test` | *(hidden)* | `test(store): add migration rollback test` |
| `ci` | *(hidden)* | `ci: add Python 3.13 to matrix` |
| `chore` | *(hidden)* | `chore(deps): bump ruff` |

**Breaking changes:** add `!` after the type/scope, and include a
`BREAKING CHANGE:` footer. Release-please will bump the major version.

```
feat(api)!: rename Index.query to Index.search

BREAKING CHANGE: Index.query is renamed to Index.search for clarity.
```

`feat` and `fix` are the only types that produce user-visible
changelog entries by default. The others are hidden because they
don't matter to someone reading release notes.

## Filing issues

- **Bugs:** include the exact `sr` command, the error output, your
  Python version, and OS.
- **Features:** describe the use case before the solution. A PR
  without an issue is fine for small fixes; larger changes benefit
  from discussion first.

## Release process

Releases are automated. When conventional-commits land on `main`,
release-please opens a PR bumping the version and updating
`CHANGELOG.md`. Merging that PR creates a tag, which triggers the
publish workflow (`.github/workflows/release.yml`) to build the
wheel and sdist and publish to PyPI.

You do not need to manually bump versions or edit the changelog.
