"""IndexBuilder: full build, incremental refresh, file discovery."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time
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
    PathTraversalError,
)
from source_recall.store import IndexStore, _now_iso, get_db_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from source_recall.embedder import Embedder

logger = logging.getLogger(__name__)


class _ChunkFailedError(Exception):
    """A file failed to chunk/extract during a refresh re-index.

    Raised only when ``_index_file``/``_index_pdf`` run with
    ``swallow_chunk_errors=False`` so the refresh caller can distinguish a
    chunking failure (recoverable: keep the file's prior index entries)
    from a storage failure (SQLite error, disk exhaustion) which must
    propagate and abort the whole refresh transaction.
    """


class IndexBuilder:
    """Orchestrates building and refreshing a code index.

    @param repo_path: Absolute path to the repository root.
    @param config: Resolved SRConfig.
    @param on_progress: Optional callback(file_path, current, total).
    @param embedder: Optional embedder for vector search. None disables vectors.
    @param on_phase: Optional callback(phase) for coarse build lifecycle progress.
    @param on_progress_detail: Optional callback(detail) for timing progress detail.
    """

    def __init__(
        self,
        repo_path: Path,
        config: SRConfig,
        on_progress: Callable[[str, int, int], None] | None = None,
        embedder: Embedder | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_progress_detail: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self.on_progress = on_progress
        self.embedder = embedder
        self.on_phase = on_phase
        self.on_progress_detail = on_progress_detail
        self._build_started_at = time.monotonic()
        self._phase_started_at = self._build_started_at

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
            self._build_locked(db_path)
        finally:
            IndexStore.release_lock(db_path)

        return db_path

    def _build_locked(self, db_path: Path) -> None:
        """Build logic that assumes the caller already holds the lock.

        @param db_path: Path to the final index.db.
        """
        # Clean stale tmp files from previous crashed builds.
        IndexStore.clean_tmp_files(db_path)

        # Build into a temp file.
        tmp_path = db_path.parent / f"{db_path.name}.tmp.{os.getpid()}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._build_core(tmp_path, db_path)
        finally:
            # On success, atomic_swap renamed tmp_path → db_path so the
            # tmp no longer exists. On any failure path between creation
            # and atomic_swap, unlink so the next build (by this pid)
            # doesn't see a stale tmp from itself.
            tmp_path.unlink(missing_ok=True)
            # WAL mode opened on the temp file leaves -wal/-shm sidecars
            # that atomic_swap cleans on the happy path.  Mirror that
            # cleanup here so a crashed build doesn't leak them across
            # subsequent build attempts (H-1 audit fix).
            for suffix in ("-wal", "-shm"):
                Path(str(tmp_path) + suffix).unlink(missing_ok=True)

    def _build_core(self, tmp_path: Path, db_path: Path) -> None:
        """Core build: write to ``tmp_path`` then atomic-swap into ``db_path``.

        Split out so the caller can wrap it in a ``try/finally`` that
        cleans up the tmp file on failure without disturbing the happy
        path (atomic_swap already consumes tmp_path on success).

        @param tmp_path: Path to the temp build target.
        @param db_path: Path to the final index.db (swap destination).
        """
        with IndexStore(tmp_path, build_mode=True) as store:
            store.create_schema()

            # Set up vector table if embedder is available.
            vec_enabled = False
            if self.embedder is not None:
                vec_enabled = store.ensure_vec_table(self.embedder.dimensions)
                if not vec_enabled:
                    logger.warning("sqlite-vec unavailable — building FTS-only index")

            # Detect current branch for branch-aware indexing.
            branch = self._get_current_branch()

            # Try git-object discovery for blob-SHA-based indexing.
            # Falls back to filesystem for non-git repos and shallow clones.
            use_git_objects = False
            blob_map: dict[str, str] = {}  # rel_path → blob_sha
            dirty_map: dict[str, str] = {}  # rel_path → synthetic blob_sha

            if not self._is_shallow_clone():
                git_entries = self._discover_files_git_objects()
                if git_entries is not None:
                    use_git_objects = True
                    blob_map = dict(git_entries)
                    dirty_map = self._detect_dirty_files()
                    # Merge dirty files into blob_map (override committed SHAs).
                    blob_map.update(dirty_map)
                    # Also add brand-new untracked files.
                    for path, sha in dirty_map.items():
                        if path not in blob_map:
                            blob_map[path] = sha

            # Discover files — filter by size/exclusions.
            if use_git_objects:
                # Start from blob_map keys, apply size/exclusion filtering.
                files = self._filter_files(list(blob_map.keys()))
            else:
                files = self._discover_files()

            total = len(files)

            # Collect chunks for batch embedding.
            pending_vectors: list[tuple[str, str]] = []  # (chunk_id, content)

            self._emit_phase("scanning")
            # Wrap the per-file loop in a single batch transaction so each
            # file's chunks + refs + symbol_lookup + file_hash commit
            # atomically, and the whole build is one transaction (M-1).
            # Vector flushing happens AFTER this batch commits — the apsw
            # vec connection cannot write while sqlite3 holds the write
            # transaction (see comment below).
            with store.batch_mode():
                for i, rel_path in enumerate(files):
                    if self.on_progress:
                        self.on_progress(rel_path, i + 1, total)

                    if use_git_objects:
                        blob_sha = blob_map.get(rel_path)
                        is_dirty = rel_path in dirty_map
                        chunk_ids = self._index_file(
                            store,
                            rel_path,
                            branch=branch,
                            blob_sha=blob_sha,
                            is_dirty=is_dirty,
                        )
                    else:
                        chunk_ids = self._index_file(store, rel_path, branch=branch)

                    if vec_enabled and self.embedder is not None:
                        for cid, content in chunk_ids:
                            pending_vectors.append((cid, content))

            # Flush ALL vectors after the sqlite3 batch commits (H2).
            # The apsw vec connection is an independent WAL reader and
            # cannot write while the sqlite3 connection holds an open
            # write transaction.  Flushing inside batch_mode() causes
            # apsw.BusyError and silently drops vectors.
            if pending_vectors and vec_enabled and self.embedder is not None:
                self._emit_phase("embedding")
                total_batches = (
                    len(pending_vectors) + self.config.embed_batch_size - 1
                ) // self.config.embed_batch_size
                for batch_index, i in enumerate(
                    range(0, len(pending_vectors), self.config.embed_batch_size),
                    start=1,
                ):
                    batch = pending_vectors[i : i + self.config.embed_batch_size]
                    self._emit_progress_detail(
                        {
                            "span": "embedding.batch",
                            "span_current": batch_index,
                            "span_total": total_batches,
                            "batch_size": len(batch),
                            "embedded_chunks": i,
                            "total_chunks": len(pending_vectors),
                        }
                    )
                    self._flush_vectors(store, batch)

            # Write meta.
            self._emit_phase("finalizing")
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

    def _emit_phase(self, phase: str) -> None:
        """Report coarse build lifecycle progress when a callback is configured."""
        self._phase_started_at = time.monotonic()
        if self.on_phase:
            self.on_phase(phase)

    def _emit_progress_detail(self, detail: dict[str, object]) -> None:
        """Report build timing detail when a callback is configured."""
        if not self.on_progress_detail:
            return
        now = time.monotonic()
        self.on_progress_detail(
            {
                **detail,
                "build_elapsed_ms": round((now - self._build_started_at) * 1000),
                "phase_elapsed_ms": round((now - self._phase_started_at) * 1000),
            }
        )

    def refresh(self, *, files: list[str] | None = None) -> int:
        """Incremental refresh: only re-index changed files.

        @param files: Optional list of repo-relative paths. When provided,
            those files are treated as changed (targeted refresh). When
            omitted, full change detection runs. Each entry is validated
            for containment under the repo root (raises
            ``PathTraversalError`` if not).
        @returns: Number of files re-indexed.
        @raises IndexNotFoundError: If no index exists.
        @raises PathTraversalError: If any caller-supplied path escapes the repo.
        """
        # Targeted refresh has no size_bound discovery — every entry comes
        # from the network. Validate containment BEFORE doing any I/O
        # so an attacker can't probe file existence via timing.
        if files is not None:
            if not files:
                return 0
            for entry in files:
                _validate_repo_relative(self.repo_path, entry)

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

                # Targeted refresh: caller specified exact files.
                if files is not None:
                    changed_files = [(f, "update") for f in files]
                else:
                    # Try git-object-based refresh for blob-SHA diffing.
                    git_refresh_result = self._try_git_object_refresh(store, branch)
                    if git_refresh_result is not None:
                        return git_refresh_result

                    # Fallback: legacy change detection.
                    changed_files = self._detect_changes(store)

                if not changed_files:
                    # Still update active_branch even if no files changed.
                    store.set_meta_batch({"active_branch": branch})
                    return 0

                # If too many changes, full rebuild is more efficient.
                if len(changed_files) > 500:
                    # Close store before rebuild, but keep the lock held
                    # to prevent a race with concurrent builders.
                    store.close()
                    self._build_locked(db_path)
                    return len(changed_files)

                # Set up vectors for refresh if embedder available.
                vec_enabled = False
                if self.embedder is not None:
                    vec_enabled = store.ensure_vec_table(self.embedder.dimensions)

                # Incremental update.
                total = len(changed_files)
                pending_vectors: list[tuple[str, str]] = []
                # OLD chunk IDs whose vectors Phase 2 must delete.  These
                # are captured per-file as each delete happens (before the
                # old chunks are gone) and only recorded once the file's
                # re-index actually succeeds — a file that fails to
                # re-chunk keeps both its chunks and its vectors.
                old_vec_chunk_ids: list[str] = []

                # Phase 1: Batch sqlite3 writes (chunks, FTS, refs,
                # file_hashes) in a single transaction.
                with store.batch_mode():
                    for i, (rel_path, action) in enumerate(changed_files):
                        if self.on_progress:
                            self.on_progress(rel_path, i + 1, total)

                        if action == "delete":
                            if vec_enabled:
                                old_vec_chunk_ids.extend(
                                    self._chunk_ids_for_file(store, rel_path)
                                )
                            store.delete_chunks_for_file(rel_path)
                            store.delete_file_hash(rel_path)
                            continue

                        result = self._reindex_file_preserving(
                            store, rel_path, branch=branch, vec_enabled=vec_enabled
                        )
                        if result is None:
                            # Chunking failed — prior index left intact.
                            continue
                        chunk_ids, old_ids = result
                        if vec_enabled and self.embedder is not None:
                            old_vec_chunk_ids.extend(old_ids)
                            pending_vectors.extend(chunk_ids)

                # Phase 2: Vector cleanup + insert (apsw connection).
                # MUST run AFTER the sqlite3 batch commits.  The apsw
                # connection is an independent WAL reader and cannot see
                # uncommitted rows from the sqlite3 connection.  If you
                # move vector ops inside batch_mode(), JOINs against the
                # chunks table will miss the new rows (H5).
                vec_ok = True
                if vec_enabled and old_vec_chunk_ids:
                    store.delete_vectors_by_ids(old_vec_chunk_ids)

                if pending_vectors and vec_enabled and self.embedder is not None:
                    vec_ok = self._flush_vectors(store, pending_vectors)

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
                # Track vector-phase failures so the next refresh can
                # detect degraded coverage and re-embed (H1 audit fix).
                if not vec_ok:
                    dirty_files = ",".join(
                        rp for rp, action in changed_files if action != "delete"
                    )
                    meta["vec_dirty"] = dirty_files
                elif vec_enabled:
                    # Clear any previous vec_dirty on success.
                    meta["vec_dirty"] = ""
                store.set_meta_batch(meta)

        finally:
            IndexStore.release_lock(db_path)

        return len(changed_files)

    def _try_git_object_refresh(self, store: IndexStore, branch: str) -> int | None:
        """Attempt git-object-based refresh using blob-SHA diffing.

        Compares blob SHAs from ``git ls-tree`` against stored
        ``content_hash`` values. Files with matching SHAs are skipped
        (only branches CSV updated). Files with different SHAs are
        re-indexed. Returns None to signal fallback to legacy refresh
        (non-git repo, shallow clone, etc.).

        @param store: Open IndexStore.
        @param branch: Current branch name.
        @returns: Number of changed files, or None to fall back.
        """
        if self._is_shallow_clone():
            return None

        git_entries = self._discover_files_git_objects()
        if git_entries is None:
            return None

        blob_map: dict[str, str] = dict(git_entries)
        dirty_map = self._detect_dirty_files()
        blob_map.update(dirty_map)

        # Filter by size/exclusions.
        current_files = set(self._filter_files(list(blob_map.keys())))
        stored_hashes = store.get_all_file_hashes()
        stored_paths = set(stored_hashes.keys())

        # Classify files into: unchanged, changed, added, deleted.
        unchanged: list[str] = []
        changed: list[str] = []
        added: list[str] = []
        deleted = stored_paths - current_files

        for rel_path in current_files:
            blob_sha = blob_map.get(rel_path)
            stored = stored_hashes.get(rel_path)
            if stored is None:
                added.append(rel_path)
            elif blob_sha and stored.content_hash == blob_sha:
                unchanged.append(rel_path)
            else:
                changed.append(rel_path)

        files_to_reindex = changed + added
        total_changed = len(files_to_reindex) + len(deleted)

        if total_changed == 0 and unchanged:
            # Only update branches on unchanged chunks + update meta.
            for rel_path in unchanged:
                self._update_branches_only(store, rel_path, branch)
            store.set_meta_batch(
                {
                    "indexed_at": _now_iso(),
                    "last_commit": _git_head(self.repo_path) or "",
                    "active_branch": branch,
                }
            )
            return 0

        if total_changed == 0:
            store.set_meta_batch({"active_branch": branch})
            return 0

        # Too many changes → full rebuild.
        if total_changed > 500:
            store.close()
            self._build_locked(get_db_path(self.repo_path))
            return total_changed

        # Set up vectors.
        vec_enabled = False
        if self.embedder is not None:
            vec_enabled = store.ensure_vec_table(self.embedder.dimensions)

        pending_vectors: list[tuple[str, str]] = []

        # OLD chunk IDs whose vectors Phase 2 must delete (C-1 fix).
        # Captured per-file as each delete happens, and for re-indexed
        # files only recorded once the re-index succeeds so a file that
        # fails to re-chunk keeps both its chunks and its vectors.
        old_vec_chunk_ids: list[str] = []

        # Phase 1: Batch sqlite3 writes.
        with store.batch_mode():
            # Delete removed files.
            for rel_path in deleted:
                if vec_enabled:
                    old_vec_chunk_ids.extend(self._chunk_ids_for_file(store, rel_path))
                store.delete_chunks_for_file(rel_path)
                store.delete_file_hash(rel_path)

            # Re-index changed/added files.
            total = len(files_to_reindex)
            for i, rel_path in enumerate(files_to_reindex):
                if self.on_progress:
                    self.on_progress(rel_path, i + 1, total)

                blob_sha = blob_map.get(rel_path)
                is_dirty = rel_path in dirty_map
                result = self._reindex_file_preserving(
                    store,
                    rel_path,
                    branch=branch,
                    vec_enabled=vec_enabled,
                    blob_sha=blob_sha,
                    is_dirty=is_dirty,
                )
                if result is None:
                    # Chunking failed — prior index left intact.
                    continue
                chunk_ids, old_ids = result
                if vec_enabled and self.embedder is not None:
                    old_vec_chunk_ids.extend(old_ids)
                    pending_vectors.extend(chunk_ids)

            # Update branches on unchanged files.
            for rel_path in unchanged:
                self._update_branches_only(store, rel_path, branch)

        # Phase 2: Vector cleanup + insert.
        vec_ok = True
        if vec_enabled and old_vec_chunk_ids:
            store.delete_vectors_by_ids(old_vec_chunk_ids)

        if pending_vectors and vec_enabled and self.embedder is not None:
            # Filter out chunks that already have vectors — skip redundant
            # embedding for chunks that existed on a previous branch.
            existing_vec_ids = store.get_existing_vector_ids(
                [cid for cid, _ in pending_vectors]
            )
            pending_vectors = [
                (cid, content)
                for cid, content in pending_vectors
                if cid not in existing_vec_ids
            ]
            if pending_vectors:
                vec_ok = self._flush_vectors(store, pending_vectors)

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
        if not vec_ok:
            dirty_files = ",".join(files_to_reindex)
            meta["vec_dirty"] = dirty_files
        elif vec_enabled:
            meta["vec_dirty"] = ""
        store.set_meta_batch(meta)

        return total_changed

    def _update_branches_only(
        self, store: IndexStore, rel_path: str, branch: str
    ) -> None:
        """Update branches CSV on chunks and file_hashes without re-indexing.

        @param store: Open IndexStore.
        @param rel_path: Repo-relative path.
        @param branch: Branch name to add.
        """
        if not branch:
            return
        rows = store.conn.execute(
            "SELECT id, branches FROM chunks WHERE file_path = ?",
            (rel_path,),
        ).fetchall()
        for row in rows:
            cid, current = row[0], row[1]
            current_set = set(current.split(",")) if current else set()
            if branch not in current_set:
                current_set.add(branch)
                new_branches = ",".join(sorted(current_set))
                store.conn.execute(
                    "UPDATE chunks SET branches = ? WHERE id = ?",
                    (new_branches, cid),
                )
        # Update file_hashes branch.
        stored = store.get_file_hash(rel_path)
        if stored is not None:
            store.upsert_file_hash(stored, branch=branch)

    def _chunk_ids_for_file(self, store: IndexStore, rel_path: str) -> list[str]:
        """Return the ids of chunks currently stored for a file.

        @param store: Open IndexStore.
        @param rel_path: Repo-relative path.
        @returns: List of chunk ids.
        """
        rows = store.conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?",
            (rel_path,),
        ).fetchall()
        return [row[0] for row in rows]

    def _reindex_file_preserving(
        self,
        store: IndexStore,
        rel_path: str,
        *,
        branch: str,
        vec_enabled: bool,
        blob_sha: str | None = None,
        is_dirty: bool = False,
    ) -> tuple[list[tuple[str, str]], list[str]] | None:
        """Re-index one file during refresh without risking data loss.

        The destructive delete of the file's existing chunks/hashes and the
        insert of its new chunks run inside a single SQLite SAVEPOINT (via
        the re-entrant ``batch_mode``, which nests as a savepoint under the
        surrounding batch).  If chunking raises, the savepoint is rolled
        back so the file KEEPS its prior index entries — a re-chunk failure
        never commits a delete with no replacement data.

        @param store: Open IndexStore (already inside ``batch_mode``).
        @param rel_path: Repo-relative path.
        @param branch: Branch name for branch-aware indexing.
        @param vec_enabled: Whether vectors are enabled (controls old-id
            capture for downstream vector cleanup).
        @param blob_sha: Git blob SHA for content-address comparison.
        @param is_dirty: True if the file has uncommitted changes.
        @returns: ``(chunk_pairs, old_vec_chunk_ids)`` on success, or None if
            the file failed to chunk (prior chunks/hashes left intact).
        """
        # Capture old vector chunk ids BEFORE the delete — the caller only
        # deletes their vectors when this re-index succeeds, so a failed
        # file keeps both its chunks and its vectors.
        old_ids = self._chunk_ids_for_file(store, rel_path) if vec_enabled else []
        try:
            with store.batch_mode():
                store.delete_chunks_for_file(rel_path)
                store.delete_file_hash(rel_path)
                chunk_pairs = self._index_file(
                    store,
                    rel_path,
                    branch=branch,
                    blob_sha=blob_sha,
                    is_dirty=is_dirty,
                    swallow_chunk_errors=False,
                )
        except _ChunkFailedError:
            # Chunking failed — the savepoint already rolled back the
            # pre-delete, so the file keeps its prior index entries.
            # Storage failures are NOT caught here: they propagate out of
            # the surrounding batch and abort the whole refresh so a partial
            # index is never committed as a success.
            logger.warning(
                "Refresh: preserving prior index for file that failed to re-chunk: %s",
                rel_path,
                exc_info=True,
            )
            return None
        return chunk_pairs, old_ids

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

    def _read_file_content(
        self,
        rel_path: str,
        full: Path,
        *,
        blob_sha: str | None,
        is_dirty: bool,
    ) -> str:
        """Read a file's text from its git blob or the working tree.

        Reading from the git blob is preferred when a clean ``blob_sha`` is
        available; a blob-read miss falls back to disk. A working-tree read
        that fails raises ``OSError`` so the caller can decide whether to skip
        (full build) or preserve prior data (refresh) — this method never
        silently swallows a read error into empty content.

        @param rel_path: Repo-relative path (for logging).
        @param full: Absolute path to the file on disk.
        @param blob_sha: Git blob SHA, or None to read the working tree.
        @param is_dirty: True if the working tree copy differs from the blob.
        @returns: Decoded file content.
        @raises OSError: If the working-tree read fails.
        """
        if blob_sha is not None and not is_dirty:
            content = self._read_git_blob(blob_sha)
            if content is not None:
                return content
            logger.warning(
                "Failed to read git blob for %s — falling back to disk", rel_path
            )
        return full.read_text(encoding="utf-8", errors="replace")

    def _index_file(
        self,
        store: IndexStore,
        rel_path: str,
        branch: str = "",
        blob_sha: str | None = None,
        is_dirty: bool = False,
        swallow_chunk_errors: bool = True,
    ) -> list[tuple[str, str]]:
        """Read, chunk, and store a single file.

        When ``blob_sha`` is provided (git-object mode):
        - If the stored ``content_hash`` matches ``blob_sha`` and chunks
          already exist, the file is skipped (no re-read, no re-chunk).
          Only the ``branches`` CSV column is updated via insert_chunks.
        - If dirty, content is read from the working tree.
        - Otherwise, content is read from the git blob.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        @param branch: Branch name for branch-aware indexing.
        @param blob_sha: Git blob SHA for content-address comparison.
        @param is_dirty: True if the file has uncommitted changes.
        @param swallow_chunk_errors: When True (full-build default), a file
            that fails to chunk is logged and skipped so one malformed file
            can't abort the whole build.  Refresh passes False so the caller
            can roll back its pre-delete and preserve the file's prior index
            entries instead of committing a destructive delete with no
            replacement data.
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        full = self.repo_path / rel_path

        # PDF files need binary extraction via pymupdf.
        if full.suffix.lower() == ".pdf":
            return self._index_pdf(
                store,
                rel_path,
                full,
                branch=branch,
                swallow_chunk_errors=swallow_chunk_errors,
            )

        # --- Blob-SHA fast path: skip re-read if content unchanged ---
        if blob_sha is not None and not is_dirty:
            stored = store.get_file_hash(rel_path)
            if stored is not None and stored.content_hash == blob_sha:
                # Content identical — just update branches on existing
                # chunks.  Select id + branches together (one query, not
                # N+1) and only UPDATE rows whose branch set actually
                # changed, mirroring _update_branches_only.
                if branch:
                    rows = store.conn.execute(
                        "SELECT id, branches FROM chunks WHERE file_path = ?",
                        (rel_path,),
                    ).fetchall()
                    updated = False
                    for cid, current in rows:
                        current_set = set(current.split(",")) if current else set()
                        if branch not in current_set:
                            current_set.add(branch)
                            new_branches = ",".join(sorted(current_set))
                            store.conn.execute(
                                "UPDATE chunks SET branches = ? WHERE id = ?",
                                (new_branches, cid),
                            )
                            updated = True
                    if updated:
                        # Use _auto_commit (not raw conn.commit) so this
                        # path is safe inside batch_mode — a raw commit would
                        # prematurely flush the outer batch transaction (H-1).
                        store._auto_commit()
                # Update branch on file_hashes too.
                store.upsert_file_hash(
                    FileRecord(
                        file_path=rel_path,
                        content_hash=blob_sha,
                        parse_mode=stored.parse_mode,
                        mtime_ns=stored.mtime_ns,
                    ),
                    branch=branch,
                )
                return []  # No new chunks to embed.

        # --- Read content ---
        # A read failure is treated exactly like a chunk failure: in refresh
        # mode (swallow_chunk_errors=False) it MUST raise _ChunkFailedError so
        # the caller's savepoint rolls back the pre-delete and the file keeps
        # its prior chunks/hashes/vectors. Returning [] here would commit the
        # delete with no replacement — permanent data loss via the read path.
        try:
            content = self._read_file_content(
                rel_path, full, blob_sha=blob_sha, is_dirty=is_dirty
            )
        except OSError as exc:
            if not swallow_chunk_errors:
                raise _ChunkFailedError(rel_path) from exc
            logger.warning("Skipping unreadable file: %s", rel_path, exc_info=True)
            return []

        # Use blob SHA as content_hash when available, else sha256.
        if blob_sha is not None:
            content_hash = blob_sha
        else:
            content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Chunk the file and extract refs.  A malformed file (unparseable
        # content, a chunker bug) must be SKIPPED, not fatal — one bad file
        # cannot be allowed to abort the whole build's transaction.  The
        # guard is scoped to chunking only: storage failures below (SQLite
        # errors, disk exhaustion) still propagate so a partial index is
        # never published as a successful build.
        try:
            chunks, quality, refs = chunk_file_with_refs(
                rel_path, content, max_chars=self.config.chunk_max_chars
            )
        except Exception as exc:
            if not swallow_chunk_errors:
                # Refresh path: signal a chunk-only failure so the caller's
                # savepoint rolls back the pre-delete and the file keeps its
                # prior index entries.  Storage failures below are NOT
                # wrapped — they must propagate and abort the refresh.
                raise _ChunkFailedError(rel_path) from exc
            logger.warning(
                "Skipping file that failed to chunk: %s", rel_path, exc_info=True
            )
            return []

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
        self,
        store: IndexStore,
        rel_path: str,
        full: Path,
        branch: str = "",
        swallow_chunk_errors: bool = True,
    ) -> list[tuple[str, str]]:
        """Extract text from a PDF and index its chunks.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        @param full: Absolute path to the PDF file.
        @param branch: Branch name for branch-aware indexing.
        @param swallow_chunk_errors: See ``_index_file``.  Refresh passes
            False so a PDF that fails extraction keeps its prior chunks
            instead of losing them to the pre-delete.
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        try:
            chunks, quality = chunk_pdf(rel_path, full)
        except Exception as exc:
            if not swallow_chunk_errors:
                raise _ChunkFailedError(rel_path) from exc
            return []

        # Hash the first 64 KB of the PDF binary for change detection.
        # This catches content changes even when mtime is restored
        # (e.g. rsync --times, touch -t) while avoiding reading the
        # entire file for large PDFs (M-4 fix).
        try:
            mtime_ns = full.stat().st_mtime_ns
            with open(full, "rb") as f:
                head = f.read(65536)
            content_hash = hashlib.sha256(head).hexdigest()
        except OSError:
            content_hash = hashlib.sha256(b"pdf").hexdigest()
            mtime_ns = None

        chunk_pairs: list[tuple[str, str]] = []
        if chunks:
            store.insert_chunks(chunks, branch=branch)
            chunk_pairs = [(c.chunk_id, c.content) for c in chunks]

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

    def _flush_vectors(
        self,
        store: IndexStore,
        pending: list[tuple[str, str]],
    ) -> bool:
        """Embed and insert a batch of chunks into vec_chunks.

        @param store: IndexStore to write to.
        @param pending: List of (chunk_id, content) to embed.
        @returns: True if all embeddings succeeded, False on failure.
        """
        if not pending or self.embedder is None:
            return True
        chunk_ids = [p[0] for p in pending]
        texts = [p[1] for p in pending]
        try:
            embeddings = self.embedder.embed_chunks(texts)
            store.insert_vectors(chunk_ids, embeddings)
            return True
        except Exception:
            logger.warning(
                "Embedding failed for batch of %d chunks — skipping vectors",
                len(pending),
                exc_info=True,
            )
            return False

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
            seen: set[str] = set()  # O(1) dedup instead of O(n) list scan.
            for rel_path in git_changed:
                full = self.repo_path / rel_path
                if not full.exists():
                    if rel_path in stored_hashes and rel_path not in seen:
                        changes.append((rel_path, "delete"))
                        seen.add(rel_path)
                    continue

                # Check if content actually changed.
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    new_hash = hashlib.sha256(content.encode()).hexdigest()
                except OSError:
                    continue

                stored = stored_hashes.get(rel_path)
                if (
                    stored is None or stored.content_hash != new_hash
                ) and rel_path not in seen:
                    changes.append((rel_path, "update"))
                    seen.add(rel_path)

            # Also check for newly added files not in stored hashes.
            current_files = set(self._discover_files())
            for rel_path in current_files - set(stored_hashes.keys()):
                if rel_path not in seen:
                    changes.append((rel_path, "update"))
                    seen.add(rel_path)

            # Check for deleted files.
            for rel_path in set(stored_hashes.keys()) - current_files:
                if rel_path not in seen:
                    changes.append((rel_path, "delete"))
                    seen.add(rel_path)

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
        # A missing repo_path is itself suspicious on any index that also
        # stores a root commit — treat it as a mismatch rather than silently
        # weakening to commit-hash comparison (H-3 fix).
        stored_path = store.get_meta("repo_path")
        stored_root = store.get_meta("repo_root_commit")
        if not stored_path:
            if stored_root:
                # Index claims a root commit but has no path of record.
                current_root = _git_root_commit(self.repo_path) or "(none)"
                raise IndexIdentityError(
                    stored_commit=stored_root,
                    current_commit=current_root,
                )
            # No identity stored at all — nothing to verify against.
            return
        if stored_path != str(self.repo_path):
            current_root = _git_root_commit(self.repo_path) or "(none)"
            raise IndexIdentityError(
                stored_commit=stored_root or "(none)",
                current_commit=current_root,
            )

        if not stored_root:
            return  # No commit identity stored — skip commit check.

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

    # -- Git-object helpers -------------------------------------------------

    def _discover_files_git_objects(self) -> list[tuple[str, str]] | None:
        """Discover files via git ls-tree, returning (path, blob_sha) tuples.

        Uses committed tree state from HEAD. Returns None for non-git repos
        or repos with no commits (empty HEAD).

        @returns: List of (repo-relative path, blob SHA) or None.
        """
        result = _git_cmd(
            self.repo_path,
            ["git", "ls-tree", "-r", "--format=%(objectname)\t%(path)", "HEAD"],
        )
        if result is None:
            return None

        entries: list[tuple[str, str]] = []
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            blob_sha, path = parts
            if not self._is_excluded(path):
                entries.append((path, blob_sha))
        return entries

    def _read_git_blob(self, blob_sha: str) -> str | None:
        """Read file content from a git blob object.

        Unlike ``_git_cmd``, does NOT strip trailing whitespace — file
        content must be preserved byte-for-byte. Decodes with
        ``errors="replace"`` to handle binary/non-UTF8 blobs gracefully
        (matching how the filesystem path reads files).

        @param blob_sha: 40-char hex SHA of the blob.
        @returns: File content string, or None if blob doesn't exist.
        """
        try:
            result = subprocess.run(
                ["git", "cat-file", "blob", blob_sha],
                cwd=self.repo_path,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            return result.stdout.decode("utf-8", errors="replace")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _detect_dirty_files(self) -> dict[str, str]:
        """Detect uncommitted files and compute synthetic blob SHAs.

        Covers modified tracked files (M), staged additions (A), and
        untracked files (?). For each, computes the blob SHA of the
        working-tree content via ``git hash-object --stdin``.

        @returns: Dict mapping repo-relative path to synthetic blob SHA.
        """
        # Cannot use _git_cmd here — it strips leading whitespace from
        # stdout, destroying the porcelain status columns (e.g. " M" → "M").
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return {}
            output = proc.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}
        if not output:
            return {}

        dirty: dict[str, str] = {}
        for line in output.split("\n"):
            if not line or len(line) < 4:
                continue
            # Porcelain v1 format: XY <path>
            # X = index status, Y = working-tree status
            status = line[:2]
            path = line[3:]
            # Skip deleted files — no content to hash.
            if "D" in status:
                continue
            # M (modified), A (added), ? (untracked), R (renamed — path after ->).
            if any(c in status for c in ("M", "A", "?", "R")):
                # Handle renames: "R  old -> new"
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                full = self.repo_path / path
                if not full.is_file():
                    continue
                # Hash the RAW file bytes so the synthetic blob SHA matches
                # git's real blob SHA. Reading as utf-8-replaced text (the
                # old approach) mangles non-UTF8 bytes, yielding a SHA that
                # never equals the committed blob and forcing a re-index on
                # every refresh.
                sha = _git_cmd(
                    self.repo_path,
                    ["git", "hash-object", str(full)],
                )
                if sha:
                    dirty[path] = sha
        return dirty

    def _is_shallow_clone(self) -> bool:
        """Detect whether this repo is a shallow clone.

        @returns: True if shallow, False otherwise.
        """
        result = _git_cmd(
            self.repo_path,
            ["git", "rev-parse", "--is-shallow-repository"],
        )
        return result == "true"

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
# Path validation (security)
# ---------------------------------------------------------------------------


def _validate_repo_relative(repo_path: Path, supplied: str) -> None:
    """Reject any caller-supplied path that escapes the repo root.

    Used by ``IndexBuilder.refresh(files=[...])`` and any other entry
    point that accepts a network-derived path string. The supplied value
    must be non-empty, contain no NUL bytes, and resolve (after symlink
    follow) to a path contained in ``repo_path.resolve()``.

    @param repo_path: Repo root (already resolved).
    @param supplied: Caller-supplied path string.
    @raises PathTraversalError: If the path escapes the repo root or is
        otherwise unusable.
    """
    if not isinstance(supplied, str) or not supplied:
        raise PathTraversalError(supplied if isinstance(supplied, str) else "")

    # Reject NUL bytes outright (POSIX-safe; pathlib will reject them too).
    if "\x00" in supplied:
        raise PathTraversalError(supplied)

    try:
        candidate = (repo_path / supplied).resolve()
    except (OSError, ValueError):
        raise PathTraversalError(supplied) from None

    if not candidate.is_relative_to(repo_path):
        raise PathTraversalError(supplied, str(candidate))


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


def _git_cmd(
    repo_path: Path, cmd: list[str], *, stdin: str | None = None
) -> str | None:
    """Run a git command and return stdout.

    @param repo_path: Working directory.
    @param cmd: Command and arguments.
    @param stdin: Optional string to pipe to the command's stdin.
    @returns: stdout string or None on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            input=stdin,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
