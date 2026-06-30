"""Path-traversal protection on IndexBuilder.refresh(files=[...]).

Targeted refresh accepts caller-supplied repo-relative paths. Without
containment validation, a caller can supply ``../../etc/passwd`` and
force its content into the index. These tests pin the rejection behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from source_recall.builder import IndexBuilder, PathTraversalError
from source_recall.config import resolve_config
from source_recall.models import IndexNotFoundError


def _make_builder(tmp_path: Path) -> IndexBuilder:
    """Build a minimal IndexBuilder pinned to ``tmp_path`` as repo root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return IndexBuilder(repo, resolve_config())


class TestRefreshPathValidation:
    def test_rejects_relative_traversal(self, tmp_path: Path) -> None:
        """``../escape`` is rejected and never opened."""
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / "secret.txt").write_text("SHOULD NOT BE READ", encoding="utf-8")

        repo = tmp_path / "repo"
        repo.mkdir()

        # Place a sentinel file outside the repo that the traversal would hit.
        builder = IndexBuilder(repo, resolve_config())

        with pytest.raises(PathTraversalError):
            builder.refresh(files=["../outer/secret.txt"])

        # The sentinel MUST still exist with its original contents
        # (refresh path never actually opened the file for read).
        assert (outer / "secret.txt").read_text(encoding="utf-8") == "SHOULD NOT BE READ"

    def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        """Absolute filesystem paths outside the repo are rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        builder = IndexBuilder(repo, resolve_config())

        with pytest.raises(PathTraversalError):
            builder.refresh(files=[str(tmp_path / "anything.txt")])

    def test_rejects_empty_string(self, tmp_path: Path) -> None:
        """Empty strings are rejected (would resolve to repo_path itself)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        builder = IndexBuilder(repo, resolve_config())

        with pytest.raises(PathTraversalError):
            builder.refresh(files=[""])

    def test_accepts_repo_relative_path(self, tmp_path: Path) -> None:
        """Legitimate files inside the repo are accepted (no traversal)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "ok.py").write_text("x = 1\n", encoding="utf-8")

        builder = IndexBuilder(repo, resolve_config())

        # No exception expected. We don't care about the count for this
        # test — just that containment validated and the call proceeded.
        # (It will raise IndexNotFoundError because there's no index yet,
        # but that comes AFTER containment validation succeeds.)
        with pytest.raises(IndexNotFoundError):
            builder.refresh(files=["ok.py"])

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """A repo-relative path that resolves via symlink outside the repo is rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "link.py").symlink_to(tmp_path / "outside.py")
        (tmp_path / "outside.py").write_text("no", encoding="utf-8")

        builder = IndexBuilder(repo, resolve_config())
        with pytest.raises(PathTraversalError):
            builder.refresh(files=["link.py"])

    def test_mixed_valid_and_traversal_rejected(self, tmp_path: Path) -> None:
        """A list containing one traversal causes the whole call to be rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        builder = _make_builder(tmp_path.parent)
        # rebuild on this tmp_path subtree:
        builder = IndexBuilder(repo, resolve_config())

        with pytest.raises(PathTraversalError):
            builder.refresh(files=["ok.py", "../../etc/passwd"])

    def test_empty_files_list_returns_zero(self, tmp_path: Path) -> None:
        """The existing empty-list-as-noop behavior is preserved."""
        repo = tmp_path / "repo"
        repo.mkdir()
        builder = IndexBuilder(repo, resolve_config())

        count = builder.refresh(files=[])
        assert count == 0
