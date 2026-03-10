"""Domain models and error hierarchy for source-recall."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchQuality(enum.StrEnum):
    """How the chunk was parsed — affects retrieval scoring."""

    AST = "ast"
    REGEX = "regex"
    TEXT_FALLBACK = "text_fallback"


class ParseMode(enum.StrEnum):
    """Per-file parse strategy that was used."""

    AST = "ast"
    REGEX = "regex"
    TEXT_FALLBACK = "text_fallback"


class SymbolType(enum.StrEnum):
    """Classification of a chunk's structural role."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CLASS_SHELL = "class_shell"
    COMPONENT = "component"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    MODULE = "module"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkData:
    """A single chunk produced by the chunker.

    @param file_path: Repo-relative path to the source file.
    @param symbol_name: Qualified name of the symbol (empty for blocks).
    @param symbol_type: Structural classification.
    @param content: Full text of the chunk.
    @param start_line: 1-indexed first line in the source file.
    @param end_line: 1-indexed last line in the source file.
    @param search_quality: Parse fidelity used to produce this chunk.
    @param parent_chunk_id: If this is a sub-chunk, the parent's ID.
    @param sub_chunk_index: Ordering within the parent (0-based).
    """

    file_path: str
    symbol_name: str
    symbol_type: SymbolType
    content: str
    start_line: int
    end_line: int
    search_quality: SearchQuality = SearchQuality.AST
    parent_chunk_id: str | None = None
    sub_chunk_index: int | None = None

    @property
    def chunk_id(self) -> str:
        """Deterministic, collision-resistant chunk ID.

        Uses length-prefixed fields to prevent separator collisions
        (e.g. paths containing colons).

        @returns: 32-char hex digest.
        """
        import hashlib

        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        raw = (
            f"{len(self.file_path)}:{self.file_path}"
            f"|{len(self.symbol_name)}:{self.symbol_name}"
            f"|{content_hash}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class QueryResult:
    """A single search result returned to the caller.

    @param chunk_id: Unique identifier for the chunk.
    @param file_path: Repo-relative path.
    @param symbol_name: Name of the symbol (may be empty).
    @param symbol_type: Structural classification.
    @param content: Full text of the chunk.
    @param score: Retrieval score (higher = more relevant).
    @param start_line: 1-indexed first line.
    @param end_line: 1-indexed last line.
    @param search_quality: Parse fidelity.
    @param match_reason: How the result was found.
    """

    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    content: str
    score: float
    start_line: int
    end_line: int
    search_quality: str
    match_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        @returns: Dictionary with all fields.
        """
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "content": self.content,
            "score": self.score,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "search_quality": self.search_quality,
            "match_reason": self.match_reason,
        }


@dataclass(frozen=True, slots=True)
class IndexStatus:
    """Status information for a built index.

    @param repo_path: Absolute path to the indexed repository.
    @param db_path: Absolute path to the index database.
    @param db_size_bytes: Size of the database file in bytes.
    @param indexed_at: ISO timestamp of last index operation.
    @param last_commit: Git commit hash at last index.
    @param file_count: Total files indexed.
    @param chunk_count: Total chunks stored.
    @param ast_files: Files parsed via tree-sitter AST.
    @param regex_files: Files parsed via regex heuristics.
    @param text_fallback_files: Files using text fallback.
    @param stale_files: Files modified since last index.
    @param vector_count: Number of chunks with vector embeddings.
    @param embed_model: Name of the embedding model used.
    @param embed_dimensions: Dimensionality of embedding vectors.
    """

    repo_path: str
    db_path: str
    db_size_bytes: int
    indexed_at: str
    last_commit: str
    file_count: int
    chunk_count: int
    ast_files: int
    regex_files: int
    text_fallback_files: int
    stale_files: int = 0
    vector_count: int = 0
    embed_model: str = ""
    embed_dimensions: int = 0


@dataclass(slots=True)
class FileRecord:
    """Tracks per-file indexing state.

    @param file_path: Repo-relative path.
    @param content_hash: SHA-256 hex of file content.
    @param parse_mode: How the file was parsed.
    @param mtime_ns: Stat mtime in nanoseconds (for non-git fast path).
    """

    file_path: str
    content_hash: str
    parse_mode: ParseMode
    mtime_ns: int | None = None


# ---------------------------------------------------------------------------
# Error hierarchy — flat, with structured attributes
# ---------------------------------------------------------------------------


class SourceRecallError(Exception):
    """Base error for all source-recall operations."""


class IndexNotFoundError(SourceRecallError):
    """No index exists for the given repository.

    @param repo_path: Path that was searched.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        super().__init__(f"No index found for {repo_path}")


class IndexLockError(SourceRecallError):
    """Another process holds the index lock.

    @param lock_path: Path to the lock file.
    @param pid: PID of the process holding the lock.
    """

    def __init__(self, lock_path: str, pid: int) -> None:
        self.lock_path = lock_path
        self.pid = pid
        super().__init__(f"Index locked by PID {pid} ({lock_path})")


class IndexIdentityError(SourceRecallError):
    """Index belongs to a different repository.

    @param stored_commit: Root commit stored in the index.
    @param current_commit: Root commit of the current repo.
    """

    def __init__(self, stored_commit: str, current_commit: str) -> None:
        self.stored_commit = stored_commit
        self.current_commit = current_commit
        super().__init__(
            f"Index identity mismatch: stored={stored_commit}, current={current_commit}"
        )


class SchemaVersionError(SourceRecallError):
    """Index schema is incompatible with this version.

    @param on_disk: Schema version found in the database.
    @param expected: Schema version this code expects.
    """

    def __init__(self, on_disk: int, expected: int) -> None:
        self.on_disk = on_disk
        self.expected = expected
        super().__init__(
            f"Schema version mismatch: on_disk={on_disk}, expected={expected}"
        )


class ConfigError(SourceRecallError):
    """Invalid configuration value.

    @param field: Config field name.
    @param value: The invalid value.
    @param reason: Why the value is invalid.
    """

    def __init__(self, field: str, value: Any, reason: str = "") -> None:
        self.field = field
        self.value = value
        self.reason = reason
        msg = f"Invalid config: {field}={value!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class FileDiscoveryError(SourceRecallError):
    """Failed to discover files for indexing.

    @param reason: Machine-readable reason code.
    @param detail: Human-readable detail.
    """

    REASONS = ("git_not_installed", "path_not_found", "not_a_directory")

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in self.REASONS:
            msg = f"Unknown reason: {reason}"
            raise ValueError(msg)
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason)


# Private — not exported from __init__.py
class _ParseError(Exception):
    """Internal: tree-sitter or regex parse failure."""

    def __init__(self, file_path: str, detail: str = "") -> None:
        self.file_path = file_path
        self.detail = detail
        super().__init__(f"Parse error in {file_path}: {detail}")
