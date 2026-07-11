"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def ts_app_path() -> Path:
    """Path to the TypeScript fixture app."""
    return FIXTURES_DIR / "ts-app"


@pytest.fixture
def py_app_path(tmp_path: Path) -> Path:
    """Path to an isolated copy of the Python fixture app."""
    dst = tmp_path / "py-app"
    shutil.copytree(FIXTURES_DIR / "py-app", dst)
    return dst


@pytest.fixture
def mixed_path() -> Path:
    """Path to the mixed-language fixture."""
    return FIXTURES_DIR / "mixed"


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repo with some files.

    @returns: Path to the temp repo root.
    """
    # Copy py-app fixture into tmp_path.
    src = FIXTURES_DIR / "py-app"
    dst = tmp_path / "repo"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(autouse=True)
def clean_index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect index storage to tmp_path to avoid polluting user data."""
    index_base = tmp_path / "source-recall-indexes"
    index_base.mkdir()

    # Monkeypatch the get_index_dir function.
    import source_recall.store as store_mod

    def patched_get_index_dir(repo_path: Path) -> Path:
        import hashlib

        real = str(repo_path.resolve())
        path_hash = hashlib.sha256(real.encode()).hexdigest()[:16]
        repo_name = repo_path.resolve().name
        return index_base / f"{repo_name}-{path_hash}"

    def patched_get_db_path(repo_path: Path) -> Path:
        return patched_get_index_dir(repo_path) / "index.db"

    monkeypatch.setattr(store_mod, "get_index_dir", patched_get_index_dir)
    monkeypatch.setattr(store_mod, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(store_mod, "get_index_base", lambda: index_base)

    # Some modules import get_db_path directly. Patch those aliases too,
    # otherwise tests can leak paths across test cases after module import.
    import source_recall.builder as builder_mod
    import source_recall.cli as cli_mod
    import source_recall.querier as querier_mod

    monkeypatch.setattr(builder_mod, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(querier_mod, "get_db_path", patched_get_db_path)
    # sr list / sr clean call get_index_base() directly.
    monkeypatch.setattr(cli_mod, "get_index_base", lambda: index_base)


@pytest.fixture(autouse=True)
def _auth_test_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``TestClient`` speak the daemon's auth + host contract.

    The daemon now (a) requires a per-instance auth token on every route
    and (b) validates the Host header via TrustedHostMiddleware. Rather
    than touch ~40 client construction sites, this autouse fixture wraps
    ``TestClient.__init__`` to:

    * pin ``base_url`` to ``http://127.0.0.1`` so the Host header passes
      TrustedHost (default is the non-loopback ``testserver``), and
    * forward the app's token (published on ``app.state.sr_token`` by the
      daemon factory) as ``X-SR-Token`` so requests are authorized.

    Tests that want to exercise the negative paths override per request:
    strip the header for a 401, or send a bogus ``host`` for a 400.
    """
    from fastapi.testclient import TestClient

    orig_init = TestClient.__init__

    def patched_init(
        self: TestClient, app: object = None, *args: object, **kwargs: object
    ) -> None:
        if not args:  # base_url is the first positional after app.
            kwargs.setdefault("base_url", "http://127.0.0.1")
        token = getattr(getattr(app, "state", None), "sr_token", None)
        if token is not None:
            headers = dict(kwargs.get("headers") or {})  # type: ignore[arg-type]
            headers.setdefault("X-SR-Token", token)
            kwargs["headers"] = headers
        orig_init(self, app, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(TestClient, "__init__", patched_init)
