"""Tests for targeted file-list refresh."""

from __future__ import annotations

from pathlib import Path

from source_recall.embedder import BagOfWordsEmbedder


class TestTargetedRefresh:
    def test_refresh_with_files_only_reindexes_those(self, py_app_path: Path) -> None:
        """Index.refresh(files=[...]) only re-indexes named files."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        # Modify one file.
        auth_file = py_app_path / "myapp" / "auth.py"
        original = auth_file.read_text()
        auth_file.write_text(original + "\n# targeted change\n")

        # Targeted refresh — only auth.py.
        count = idx.refresh(files=["myapp/auth.py"])
        assert count == 1

    def test_refresh_with_empty_files_is_noop(self, py_app_path: Path) -> None:
        """Index.refresh(files=[]) does nothing."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        count = idx.refresh(files=[])
        assert count == 0

    def test_refresh_without_files_does_full(self, py_app_path: Path) -> None:
        """Index.refresh() without files does full change detection."""
        from source_recall import Index

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        # No changes — full refresh returns 0.
        count = idx.refresh()
        assert count == 0
