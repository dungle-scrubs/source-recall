"""Tests for models.py."""

from source_recall.models import (
    ChunkData,
    ConfigError,
    FileDiscoveryError,
    IndexIdentityError,
    IndexLockError,
    IndexNotFoundError,
    SchemaVersionError,
    SourceRecallError,
    SymbolType,
)


class TestChunkData:
    def test_chunk_id_deterministic(self) -> None:
        """Same inputs produce the same chunk ID."""
        c1 = ChunkData(
            file_path="src/auth.py",
            symbol_name="AuthService",
            symbol_type=SymbolType.CLASS,
            content="class AuthService: ...",
            start_line=1,
            end_line=10,
        )
        c2 = ChunkData(
            file_path="src/auth.py",
            symbol_name="AuthService",
            symbol_type=SymbolType.CLASS,
            content="class AuthService: ...",
            start_line=1,
            end_line=10,
        )
        assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_unique_for_different_content(self) -> None:
        """Different content produces different IDs."""
        c1 = ChunkData(
            file_path="a.py",
            symbol_name="f",
            symbol_type=SymbolType.FUNCTION,
            content="def f(): pass",
            start_line=1,
            end_line=1,
        )
        c2 = ChunkData(
            file_path="a.py",
            symbol_name="f",
            symbol_type=SymbolType.FUNCTION,
            content="def f(): return 1",
            start_line=1,
            end_line=1,
        )
        assert c1.chunk_id != c2.chunk_id

    def test_chunk_id_no_separator_collision(self) -> None:
        """Length-prefixed encoding prevents colon collisions."""
        c1 = ChunkData(
            file_path="a:b",
            symbol_name="c",
            symbol_type=SymbolType.BLOCK,
            content="x",
            start_line=1,
            end_line=1,
        )
        c2 = ChunkData(
            file_path="a",
            symbol_name="b:c",
            symbol_type=SymbolType.BLOCK,
            content="x",
            start_line=1,
            end_line=1,
        )
        # These would collide without length-prefixing.
        assert c1.chunk_id != c2.chunk_id

    def test_chunk_id_length(self) -> None:
        """Chunk IDs are 32 hex characters."""
        c = ChunkData(
            file_path="test.py",
            symbol_name="",
            symbol_type=SymbolType.BLOCK,
            content="hello",
            start_line=1,
            end_line=1,
        )
        assert len(c.chunk_id) == 32
        assert all(ch in "0123456789abcdef" for ch in c.chunk_id)


class TestErrors:
    def test_error_hierarchy(self) -> None:
        """All custom errors inherit from SourceRecallError."""
        assert issubclass(IndexNotFoundError, SourceRecallError)
        assert issubclass(IndexLockError, SourceRecallError)
        assert issubclass(IndexIdentityError, SourceRecallError)
        assert issubclass(SchemaVersionError, SourceRecallError)
        assert issubclass(ConfigError, SourceRecallError)
        assert issubclass(FileDiscoveryError, SourceRecallError)

    def test_errors_have_structured_attributes(self) -> None:
        """Errors carry machine-readable data."""
        e1 = IndexNotFoundError("/path/to/repo")
        assert e1.repo_path == "/path/to/repo"

        e2 = IndexLockError("/path/lock", pid=1234)
        assert e2.lock_path == "/path/lock"
        assert e2.pid == 1234

        e3 = SchemaVersionError(on_disk=3, expected=2)
        assert e3.on_disk == 3
        assert e3.expected == 2

        e4 = ConfigError("top_k", -1, "must be positive")
        assert e4.field == "top_k"
        assert e4.value == -1

    def test_file_discovery_error_validates_reason(self) -> None:
        """FileDiscoveryError rejects unknown reason codes."""
        import pytest

        with pytest.raises(ValueError, match="Unknown reason"):
            FileDiscoveryError("bad_reason")
