"""Tests for _get_extension (M-4).

Before the fix, ``_get_extension`` used ``file_path.rfind('.')`` over the
whole path, so a directory component containing a dot (``my.app/src/file``)
corrupted the extension.  The fix uses ``Path(file_path).suffix.lower()``.
"""

from __future__ import annotations

from source_recall.chunker import _get_extension


class TestGetExtension:
    def test_simple_extension(self) -> None:
        assert _get_extension("foo.py") == ".py"

    def test_nested_path(self) -> None:
        assert _get_extension("src/lib/foo.ts") == ".ts"

    def test_no_extension(self) -> None:
        assert _get_extension("README") == ""

    def test_dotted_directory_no_extension(self) -> None:
        """A directory with a dot must not leak into the extension."""
        assert _get_extension("my.app/src/file") == ""

    def test_dotted_directory_with_extension(self) -> None:
        """The file's own extension wins even if a dir has a dot."""
        assert _get_extension("my.app/src/file.py") == ".py"

    def test_uppercase_extension_lowercased(self) -> None:
        assert _get_extension("Foo.PY") == ".py"

    def test_double_extension(self) -> None:
        assert _get_extension("archive.tar.gz") == ".gz"
