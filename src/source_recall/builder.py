"""IndexBuilder: full build, incremental refresh, file discovery."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.chunker import chunk_file_with_refs, chunk_pdf
from source_recall.config import SRConfig
from source_recall.models import (
    FileDiscoveryError,
    FileRecord,
    IndexIdentityError,
    ParseMode,
)
from source_recall.store import IndexStore, _now_iso, get_db_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from source_recall.embedder import Embedder

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Orchestrates building and refreshing a code index.

    @param repo_path: Absolute path to the repository root.
    @param config: Resolved SRConfig.
    @param on_progress: Optional callback(file_path, current, total).
    @param embedder: Optional embedder for vector search. None disables vectors.
    """

    def __init__(
        self,
        repo_path: Path,
        config: SRConfig,
        on_progress: Callable[[str, int, int], None] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self.on_progress = on_progress
        self.embedder = embedder

    def build(self) -> Path:
        """Full index build with atomic swap.

        @returns: Path to the final index.db.
        @raises FileDiscoveryError: If repo path is invalid.
        @raises IndexLockError: If another build is in progress.
        """
        self._validate_repo_path()
        db_path = get_db_path(self.repo_path)
        IndexStore.acquire_lock(db_path, timeout=5)

        try:
            # Clean stale tmp files from previous crashed builds.
            IndexStore.clean_tmp_files(db_path)

            # Build into a temp file.
            tmp_path = db_path.parent / f"{db_path.name}.tmp.{os.getpid()}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            with IndexStore(tmp_path) as store:
                store.create_schema()

                # Set up vector table if embedder is available.
                vec_enabled = False
                if self.embedder is not None:
                    vec_enabled = store.ensure_vec_table(self.embedder.dimensions)
                    if not vec_enabled:
                        logger.warning(
                            "sqlite-vec unavailable — building FTS-only index"
                        )

                # Detect current branch for branch-aware indexing.
                branch = self._get_current_branch()

                # Discover and index files.
                files = self._discover_files()
                total = len(files)

                # Collect chunks for batch embedding.
                pending_vectors: list[tuple[str, str]] = []  # (chunk_id, content)

                for i, rel_path in enumerate(files):
                    if self.on_progress:
                        self.on_progress(rel_path, i + 1, total)
                    chunk_ids = self._index_file(store, rel_path, branch=branch)

                    if vec_enabled and self.embedder is not None:
                        for cid, content in chunk_ids:
                            pending_vectors.append((cid, content))

                        # Batch embed when we have enough.
                        if len(pending_vectors) >= self.config.embed_batch_size:
                            self._flush_vectors(store, pending_vectors)
                            pending_vectors.clear()

                # Flush remaining vectors.
                if pending_vectors and vec_enabled and self.embedder is not None:
                    self._flush_vectors(store, pending_vectors)

                # Write meta.
                meta = {
                    "repo_path": str(self.repo_path),
                    "indexed_at": _now_iso(),
                    "last_commit": _git_head(self.repo_path) or "",
                    "repo_root_commit": _git_root_commit(self.repo_path) or "",
                    "repo_remote_url_hash": _git_remote_hash(self.repo_path) or "",
                    "active_branch": branch,
                }
                if vec_enabled and self.embedder is not None:
                    meta["embed_model"] = type(self.embedder).__name__
                    meta["embed_dimensions"] = str(self.embedder.dimensions)
                    chunk_count = store.get_chunk_count()
                    vec_count = store.get_vector_count()
                    coverage = vec_count / chunk_count if chunk_count > 0 else 0.0
                    meta["embed_coverage"] = f"{coverage:.4f}"
                store.set_meta_batch(meta)

            # Atomic swap (after store is closed by context manager).
            IndexStore.atomic_swap(tmp_path, db_path)

        finally:
            IndexStore.release_lock(db_path)

        return db_path

    def refresh(self) -> int:
        """Incremental refresh: only re-index changed files.

        @returns: Number of files re-indexed.
        @raises IndexNotFoundError: If no index exists.
        """
        db_path = get_db_path(self.repo_path)
        if not db_path.exists():
            from source_recall.models import IndexNotFoundError

            raise IndexNotFoundError(str(self.repo_path))

        IndexStore.acquire_lock(db_path, timeout=5)
        try:
            with IndexStore(db_path) as store:
                store.run_migrations()

                # Verify identity.
                self._verify_identity(store)

                # Detect current branch.
                branch = self._get_current_branch()

                changed_files = self._detect_changes(store)
                if not changed_files:
                    # Still update active_branch even if no files changed.
                    store.set_meta_batch({"active_branch": branch})
                    return 0

                # If too many changes, full rebuild is more efficient.
                if len(changed_files) > 500:
                    # Close store (via context manager exit) before rebuild.
                    # build() acquires its own lock, so release ours first.
                    store.close()
                    IndexStore.release_lock(db_path)
                    self.build()
                    return len(changed_files)

                # Set up vectors for refresh if embedder available.
                vec_enabled = False
                if self.embedder is not None:
                    vec_enabled = store.ensure_vec_table(self.embedder.dimensions)

                # Incremental update.
                total = len(changed_files)
                pending_vectors: list[tuple[str, str]] = []

                # Phase 1: Delete old vectors (apsw connection — must
                # happen outside the sqlite3 batch to avoid write
                # contention between the two connections in WAL mode).
                if vec_enabled:
                    for rel_path, _action in changed_files:
                        store.delete_vectors_by_file(rel_path)

                # Phase 2: Batch sqlite3 writes (chunks, FTS, refs,
                # file_hashes) in a single transaction.
                with store.batch_mode():
                    for i, (rel_path, action) in enumerate(changed_files):
                        if self.on_progress:
                            self.on_progress(rel_path, i + 1, total)

                        store.delete_chunks_for_file(rel_path)
                        store.delete_file_hash(rel_path)

                        if action != "delete":
                            chunk_ids = self._index_file(store, rel_path, branch=branch)

                            if vec_enabled and self.embedder is not None:
                                for cid, content in chunk_ids:
                                    pending_vectors.append((cid, content))

                # Phase 3: Insert new vectors (apsw connection).
                if pending_vectors and vec_enabled and self.embedder is not None:
                    self._flush_vectors(store, pending_vectors)

                # Update meta.
                meta: dict[str, str] = {
                    "indexed_at": _now_iso(),
                    "last_commit": _git_head(self.repo_path) or "",
                    "active_branch": branch,
                }
                if vec_enabled and self.embedder is not None:
                    chunk_count = store.get_chunk_count()
                    vec_count = store.get_vector_count()
                    coverage = vec_count / chunk_count if chunk_count > 0 else 0.0
                    meta["embed_coverage"] = f"{coverage:.4f}"
                store.set_meta_batch(meta)

        finally:
            IndexStore.release_lock(db_path)

        return len(changed_files)

    # -- File cleanup -------------------------------------------------------

    @staticmethod
    def _remove_file_data(store: IndexStore, rel_path: str, vec_enabled: bool) -> None:
        """Remove all indexed data for a file (vectors, chunks, file hash).

        @param store: Active IndexStore.
        @param rel_path: Repo-relative file path.
        @param vec_enabled: Whether vector table is available.
        """
        if vec_enabled:
            store.delete_vectors_by_file(rel_path)
        store.delete_chunks_for_file(rel_path)
        store.delete_file_hash(rel_path)

    # -- File discovery -----------------------------------------------------

    def _discover_files(self) -> list[str]:
        """Discover all indexable files in the repo.

        @returns: List of repo-relative paths.
        @raises FileDiscoveryError: On errors.
        """
        self._validate_repo_path()

        # Try git ls-files first (respects .gitignore).
        files = self._git_ls_files()
        if files is not None:
            return self._filter_files(files)

        # Fallback: walk the directory.
        result: list[str] = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Prune excluded directories.
            dirs[:] = [d for d in dirs if not self._is_excluded(d + "/")]
            for fname in filenames:
                full = Path(root) / fname
                rel = str(full.relative_to(self.repo_path))
                if not self._is_excluded(rel):
                    result.append(rel)

        return self._filter_files(result)

    def _git_ls_files(self) -> list[str] | None:
        """Use git ls-files to list tracked files.

        @returns: List of relative paths, or None if not a git repo.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            return [f for f in result.stdout.strip().split("\n") if f]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _filter_files(self, files: list[str]) -> list[str]:
        """Filter files by exclusion patterns and size limits.

        @param files: Raw list of repo-relative paths.
        @returns: Filtered list.
        """
        result: list[str] = []
        for rel_path in files:
            if self._is_excluded(rel_path):
                continue
            full = self.repo_path / rel_path
            try:
                size = full.stat().st_size
                if size > self.config.max_file_size:
                    continue
                if size == 0:
                    continue
            except OSError:
                continue
            result.append(rel_path)
        return sorted(result)

    def _is_excluded(self, path: str) -> bool:
        """Check if a path matches any exclusion pattern.

        @param path: Repo-relative path or directory name.
        @returns: True if excluded.
        """
        for pattern in self.config.exclude_patterns:
            if fnmatch(path, pattern):
                return True
            # Also check each path component.
            for part in Path(path).parts:
                if fnmatch(part, pattern) or fnmatch(part + "/", pattern):
                    return True
        return False

    # -- File indexing ------------------------------------------------------

    def _index_file(
        self, store: IndexStore, rel_path: str, branch: str = ""
    ) -> list[tuple[str, str]]:
        """Read, chunk, and store a single file.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        @param branch: Branch name for branch-aware indexing.
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        full = self.repo_path / rel_path

        # PDF files need binary extraction via pymupdf.
        if full.suffix.lower() == ".pdf":
            return self._index_pdf(store, rel_path, full)

        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Chunk the file and extract refs.
        chunks, quality, refs = chunk_file_with_refs(
            rel_path, content, max_chars=self.config.chunk_max_chars
        )

        chunk_pairs: list[tuple[str, str]] = []
        if chunks:
            store.insert_chunks(chunks, branch=branch)
            chunk_pairs = [(c.chunk_id, c.content) for c in chunks]

            # Store refs.
            if refs:
                store.insert_refs(refs)

            # Store symbol_lookup for named symbols.
            sym_entries = [
                (c.chunk_id, c.symbol_name, c.file_path)
                for c in chunks
                if c.symbol_name
            ]
            if sym_entries:
                store.insert_symbol_lookups(sym_entries)

        # Record file hash.
        try:
            mtime_ns = full.stat().st_mtime_ns
        except OSError:
            mtime_ns = None

        parse_mode = ParseMode(quality.value)
        store.upsert_file_hash(
            FileRecord(
                file_path=rel_path,
                content_hash=content_hash,
                parse_mode=parse_mode,
                mtime_ns=mtime_ns,
            ),
            branch=branch,
        )

        return chunk_pairs

    def _index_pdf(
        self, store: IndexStore, rel_path: str, full: Path
    ) -> list[tuple[str, str]]:
        """Extract text from a PDF and index its chunks.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        @param full: Absolute path to the PDF file.
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        try:
            chunks, quality = chunk_pdf(rel_path, full)
        except Exception:
            return []

        # Use file mtime as hash proxy for PDFs.
        try:
            mtime_ns = full.stat().st_mtime_ns
            content_hash = hashlib.sha256(str(mtime_ns).encode()).hexdigest()
        except OSError:
            content_hash = hashlib.sha256(b"pdf").hexdigest()
            mtime_ns = None

        chunk_pairs: list[tuple[str, str]] = []
        if chunks:
            store.insert_chunks(chunks)
            chunk_pairs = [(c.chunk_id, c.content) for c in chunks]

        parse_mode = ParseMode(quality.value)
        store.upsert_file_hash(
            FileRecord(
                file_path=rel_path,
                content_hash=content_hash,
                parse_mode=parse_mode,
                mtime_ns=mtime_ns,
            )
        )

        return chunk_pairs

    def _flush_vectors(
        self,
        store: IndexStore,
        pending: list[tuple[str, str]],
    ) -> None:
        """Embed and insert a batch of chunks into vec_chunks.

        @param store: IndexStore to write to.
        @param pending: List of (chunk_id, content) to embed.
        """
        if not pending or self.embedder is None:
            return
        chunk_ids = [p[0] for p in pending]
        texts = [p[1] for p in pending]
        try:
            embeddings = self.embedder.embed_chunks(texts)
            store.insert_vectors(chunk_ids, embeddings)
        except Exception:
            logger.warning(
                "Embedding failed for batch of %d chunks — skipping vectors",
                len(pending),
                exc_info=True,
            )

    # -- Change detection ---------------------------------------------------

    def _detect_changes(self, store: IndexStore) -> list[tuple[str, str]]:
        """Detect which files need re-indexing.

        @param store: Open IndexStore.
        @returns: List of (rel_path, action) where action is 'update' or 'delete'.
        """
        stored_hashes = store.get_all_file_hashes()
        last_commit = store.get_meta("last_commit")

        # Get changed files from git diff if possible.
        git_changed = self._git_diff_files(last_commit)

        if git_changed is not None:
            changes: list[tuple[str, str]] = []
            for rel_path in git_changed:
                full = self.repo_path / rel_path
                if not full.exists():
                    if rel_path in stored_hashes:
                        changes.append((rel_path, "delete"))
                    continue

                # Check if content actually changed.
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    new_hash = hashlib.sha256(content.encode()).hexdigest()
                except OSError:
                    continue

                stored = stored_hashes.get(rel_path)
                if stored is None or stored.content_hash != new_hash:
                    changes.append((rel_path, "update"))

            # Also check for newly added files not in stored hashes.
            current_files = set(self._discover_files())
            for rel_path in current_files - set(stored_hashes.keys()):
                if (rel_path, "update") not in changes:
                    changes.append((rel_path, "update"))

            # Check for deleted files.
            for rel_path in set(stored_hashes.keys()) - current_files:
                if (rel_path, "delete") not in changes:
                    changes.append((rel_path, "delete"))

            return changes

        # No git — fall back to hash comparison for all files.
        return self._hash_diff(stored_hashes)

    def _git_diff_files(self, last_commit: str | None) -> list[str] | None:
        """Get changed files from git diff.

        Verifies ancestor relationship first to handle rebases.

        @param last_commit: Commit hash of last index.
        @returns: List of changed file paths, or None if git unavailable.
        """
        if not last_commit:
            return None

        # Verify ancestor relationship.
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", last_commit, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                # Not an ancestor (rebase, filter-repo, etc.) — fall back.
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        # Get diff (includes working tree changes).
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", last_commit],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            return [f for f in result.stdout.strip().split("\n") if f]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _hash_diff(self, stored_hashes: dict[str, FileRecord]) -> list[tuple[str, str]]:
        """Full hash comparison for non-git repos.

        @param stored_hashes: Currently stored file records.
        @returns: List of (rel_path, action).
        """
        current_files = set(self._discover_files())
        changes: list[tuple[str, str]] = []

        # Deleted files.
        for rel_path in set(stored_hashes.keys()) - current_files:
            changes.append((rel_path, "delete"))

        # New or modified files.
        for rel_path in current_files:
            full = self.repo_path / rel_path
            stored = stored_hashes.get(rel_path)

            # Fast path: check mtime first.
            if stored and stored.mtime_ns is not None:
                try:
                    current_mtime = full.stat().st_mtime_ns
                    if current_mtime == stored.mtime_ns:
                        continue
                except OSError:
                    pass

            # Slow path: hash comparison.
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                new_hash = hashlib.sha256(content.encode()).hexdigest()
            except OSError:
                continue

            if stored is None or stored.content_hash != new_hash:
                changes.append((rel_path, "update"))

        return changes

    # -- Identity -----------------------------------------------------------

    def _verify_identity(self, store: IndexStore) -> None:
        """Verify the index belongs to this repository.

        Checks repo_path first (catches forks with same root commit and
        index-dir hash collisions), then root commit, then remote URL
        hash as fallback for shallow clones.

        @param store: Open IndexStore.
        @raises IndexIdentityError: If identity doesn't match.
        """
        # First-line check: repo_path must match.
        # This catches forks/clones that share a root commit but live
        # at different filesystem paths, and 48-bit index-dir hash collisions.
        stored_path = store.get_meta("repo_path")
        if stored_path and stored_path != str(self.repo_path):
            stored_root = store.get_meta("repo_root_commit") or "(none)"
            current_root = _git_root_commit(self.repo_path) or "(none)"
            raise IndexIdentityError(
                stored_commit=stored_root,
                current_commit=current_root,
            )

        stored_root = store.get_meta("repo_root_commit")
        if not stored_root:
            return  # No identity stored — skip commit check.

        current_root = _git_root_commit(self.repo_path) or ""

        # Root commit match.
        if current_root and stored_root:
            stored_set = set(stored_root.split(","))
            current_set = set(current_root.split(","))
            if stored_set & current_set:
                return  # At least one root in common.

        # Fallback: remote URL hash (handles shallow clones).
        stored_remote = store.get_meta("repo_remote_url_hash")
        current_remote = _git_remote_hash(self.repo_path)
        if stored_remote and current_remote and stored_remote == current_remote:
            return

        # If both are empty (non-git), skip the check.
        if not stored_root and not current_root:
            return

        raise IndexIdentityError(
            stored_commit=stored_root or "(none)",
            current_commit=current_root or "(none)",
        )

    # -- Validation ---------------------------------------------------------

    def _validate_repo_path(self) -> None:
        """Validate the repository path exists and is a directory.

        @raises FileDiscoveryError: If invalid.
        """
        if not self.repo_path.exists():
            raise FileDiscoveryError("path_not_found", str(self.repo_path))
        if not self.repo_path.is_dir():
            raise FileDiscoveryError("not_a_directory", str(self.repo_path))

    # -- Branch-aware helpers -----------------------------------------------

    def _get_current_branch(self) -> str:
        """Get current branch name, with detached HEAD fallback.

        @returns: Branch name or 'detached-<sha[:8]>'.
        """
        result = _git_cmd(self.repo_path, ["git", "branch", "--show-current"])
        if result:
            return result
        # Detached HEAD — use commit SHA as pseudo-branch.
        head = _git_head(self.repo_path)
        if head:
            return f"detached-{head[:8]}"
        return ""


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_head(repo_path: Path) -> str | None:
    """Get the current HEAD commit hash.

    @param repo_path: Repository root.
    @returns: Commit hash or None.
    """
    return _git_cmd(repo_path, ["git", "rev-parse", "HEAD"])


def _git_root_commit(repo_path: Path) -> str | None:
    """Get the root commit(s) — comma-separated if multiple.

    @param repo_path: Repository root.
    @returns: Comma-separated root commit hashes, or None.
    """
    result = _git_cmd(repo_path, ["git", "rev-list", "--max-parents=0", "HEAD"])
    if result is None:
        return None
    # Multiple roots: join with comma.
    roots = [r.strip() for r in result.strip().split("\n") if r.strip()]
    return ",".join(roots) if roots else None


def _git_remote_hash(repo_path: Path) -> str | None:
    """Get a hash of the origin remote URL (fallback identity).

    @param repo_path: Repository root.
    @returns: SHA-256 hex of remote URL, or None.
    """
    url = _git_cmd(repo_path, ["git", "remote", "get-url", "origin"])
    if url is None:
        return None
    return hashlib.sha256(url.strip().encode()).hexdigest()[:16]


def _git_cmd(repo_path: Path, cmd: list[str]) -> str | None:
    """Run a git command and return stdout.

    @param repo_path: Working directory.
    @param cmd: Command and arguments.
    @returns: stdout string or None on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
