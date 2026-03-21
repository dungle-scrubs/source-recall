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
                    blob_map = {path: sha for path, sha in git_entries}
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

    def refresh(self, *, files: list[str] | None = None) -> int:
        """Incremental refresh: only re-index changed files.

        @param files: Optional list of repo-relative paths. When provided,
            those files are treated as changed (targeted refresh). When
            omitted, full change detection runs.
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

                # Targeted refresh: caller specified exact files.
                if files is not None:
                    if not files:
                        return 0
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
                # Track files that need vector cleanup (deferred until
                # after the sqlite3 batch commits — see C1 atomicity).
                vec_dirty_files: list[str] = []

                # Phase 1: Batch sqlite3 writes (chunks, FTS, refs,
                # file_hashes) in a single transaction.
                with store.batch_mode():
                    for i, (rel_path, action) in enumerate(changed_files):
                        if self.on_progress:
                            self.on_progress(rel_path, i + 1, total)

                        store.delete_chunks_for_file(rel_path)
                        store.delete_file_hash(rel_path)

                        if vec_enabled:
                            vec_dirty_files.append(rel_path)

                        if action != "delete":
                            chunk_ids = self._index_file(store, rel_path, branch=branch)

                            if vec_enabled and self.embedder is not None:
                                for cid, content in chunk_ids:
                                    pending_vectors.append((cid, content))

                # Phase 2: Vector cleanup + insert (apsw connection).
                # MUST run AFTER the sqlite3 batch commits.  The apsw
                # connection is an independent WAL reader and cannot see
                # uncommitted rows from the sqlite3 connection.  If you
                # move vector ops inside batch_mode(), JOINs against the
                # chunks table will miss the new rows (H5).
                vec_ok = True
                if vec_enabled:
                    for rel_path in vec_dirty_files:
                        store.delete_vectors_by_file(rel_path)

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

        blob_map: dict[str, str] = {path: sha for path, sha in git_entries}
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
        vec_dirty_files: list[str] = []

        # Phase 1: Batch sqlite3 writes.
        with store.batch_mode():
            # Delete removed files.
            for rel_path in deleted:
                store.delete_chunks_for_file(rel_path)
                store.delete_file_hash(rel_path)
                if vec_enabled:
                    vec_dirty_files.append(rel_path)

            # Re-index changed/added files.
            total = len(files_to_reindex)
            for i, rel_path in enumerate(files_to_reindex):
                if self.on_progress:
                    self.on_progress(rel_path, i + 1, total)

                store.delete_chunks_for_file(rel_path)
                store.delete_file_hash(rel_path)
                if vec_enabled:
                    vec_dirty_files.append(rel_path)

                blob_sha = blob_map.get(rel_path)
                is_dirty = rel_path in dirty_map
                chunk_ids = self._index_file(
                    store,
                    rel_path,
                    branch=branch,
                    blob_sha=blob_sha,
                    is_dirty=is_dirty,
                )
                if vec_enabled and self.embedder is not None:
                    for cid, content in chunk_ids:
                        pending_vectors.append((cid, content))

            # Update branches on unchanged files.
            for rel_path in unchanged:
                self._update_branches_only(store, rel_path, branch)

        # Phase 2: Vector cleanup + insert.
        vec_ok = True
        if vec_enabled:
            for rel_path in vec_dirty_files:
                store.delete_vectors_by_file(rel_path)

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
        self,
        store: IndexStore,
        rel_path: str,
        branch: str = "",
        blob_sha: str | None = None,
        is_dirty: bool = False,
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
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        full = self.repo_path / rel_path

        # PDF files need binary extraction via pymupdf.
        if full.suffix.lower() == ".pdf":
            return self._index_pdf(store, rel_path, full, branch=branch)

        # --- Blob-SHA fast path: skip re-read if content unchanged ---
        if blob_sha is not None and not is_dirty:
            stored = store.get_file_hash(rel_path)
            if stored is not None and stored.content_hash == blob_sha:
                # Content identical — just update branches on existing chunks.
                existing_chunks = store.conn.execute(
                    "SELECT id FROM chunks WHERE file_path = ?",
                    (rel_path,),
                ).fetchall()
                if existing_chunks and branch:
                    for row in existing_chunks:
                        cid = row[0]
                        br_row = store.conn.execute(
                            "SELECT branches FROM chunks WHERE id = ?",
                            (cid,),
                        ).fetchone()
                        current = br_row[0] if br_row else ""
                        current_set = set(current.split(",")) if current else set()
                        if branch not in current_set:
                            current_set.add(branch)
                            new_branches = ",".join(sorted(current_set))
                            store.conn.execute(
                                "UPDATE chunks SET branches = ? WHERE id = ?",
                                (new_branches, cid),
                            )
                    store.conn.commit()
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
        if blob_sha is not None and not is_dirty:
            # Read from git blob.
            content = self._read_git_blob(blob_sha)
            if content is None:
                logger.warning(
                    "Failed to read git blob for %s — falling back to disk", rel_path
                )
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    logger.warning(
                        "Skipping unreadable file: %s", rel_path, exc_info=True
                    )
                    return []
        else:
            # Read from working tree (dirty file or no blob_sha).
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Skipping unreadable file: %s", rel_path, exc_info=True)
                return []

        # Use blob SHA as content_hash when available, else sha256.
        if blob_sha is not None:
            content_hash = blob_sha
        else:
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
        self, store: IndexStore, rel_path: str, full: Path, branch: str = ""
    ) -> list[tuple[str, str]]:
        """Extract text from a PDF and index its chunks.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        @param full: Absolute path to the PDF file.
        @param branch: Branch name for branch-aware indexing.
        @returns: List of (chunk_id, content) tuples for embedding.
        """
        try:
            chunks, quality = chunk_pdf(rel_path, full)
        except Exception:
            return []

        # Use file mtime as hash proxy for PDFs (avoids reading
        # entire binary for hashing).  Trade-off: if mtime is restored
        # (e.g. rsync --times, touch -t) after content changes, the
        # file won't be detected as changed during incremental refresh.
        # A full rebuild always catches this (M5).
        try:
            mtime_ns = full.stat().st_mtime_ns
            content_hash = hashlib.sha256(str(mtime_ns).encode()).hexdigest()
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
                if stored is None or stored.content_hash != new_hash:
                    if rel_path not in seen:
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
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                # Compute synthetic blob SHA.
                sha = _git_cmd(
                    self.repo_path,
                    ["git", "hash-object", "--stdin"],
                    stdin=content,
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
