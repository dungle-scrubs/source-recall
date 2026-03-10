"""IndexBuilder: full build, incremental refresh, file discovery."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from source_recall.chunker import chunk_file
from source_recall.config import SRConfig
from source_recall.models import (
    FileDiscoveryError,
    FileRecord,
    IndexIdentityError,
    ParseMode,
)
from source_recall.store import IndexStore, get_db_path

if TYPE_CHECKING:
    from collections.abc import Callable


class IndexBuilder:
    """Orchestrates building and refreshing a code index.

    @param repo_path: Absolute path to the repository root.
    @param config: Resolved SRConfig.
    @param on_progress: Optional callback(file_path, current, total).
    """

    def __init__(
        self,
        repo_path: Path,
        config: SRConfig,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self.on_progress = on_progress

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

            store = IndexStore(tmp_path)
            store.open()
            store.create_schema()

            # Discover and index files.
            files = self._discover_files()
            total = len(files)

            for i, rel_path in enumerate(files):
                if self.on_progress:
                    self.on_progress(rel_path, i + 1, total)
                self._index_file(store, rel_path)

            # Write meta.
            store.set_meta_batch(
                {
                    "repo_path": str(self.repo_path),
                    "indexed_at": _now_iso(),
                    "last_commit": _git_head(self.repo_path) or "",
                    "repo_root_commit": _git_root_commit(self.repo_path) or "",
                    "repo_remote_url_hash": _git_remote_hash(self.repo_path) or "",
                    "schema_version": "1",
                }
            )

            store.close()

            # Atomic swap.
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
            store = IndexStore(db_path)
            store.open()
            store.run_migrations()

            # Verify identity.
            self._verify_identity(store)

            changed_files = self._detect_changes(store)
            if not changed_files:
                store.close()
                return 0

            # If too many changes, full rebuild is more efficient.
            if len(changed_files) > 500:
                store.close()
                IndexStore.release_lock(db_path)
                self.build()
                return len(changed_files)

            # Incremental update.
            total = len(changed_files)
            for i, (rel_path, action) in enumerate(changed_files):
                if self.on_progress:
                    self.on_progress(rel_path, i + 1, total)

                if action == "delete":
                    store.delete_chunks_for_file(rel_path)
                    store.delete_file_hash(rel_path)
                else:
                    store.delete_chunks_for_file(rel_path)
                    store.delete_file_hash(rel_path)
                    self._index_file(store, rel_path)

            # Update meta.
            store.set_meta_batch(
                {
                    "indexed_at": _now_iso(),
                    "last_commit": _git_head(self.repo_path) or "",
                }
            )

            store.close()

        finally:
            IndexStore.release_lock(db_path)

        return len(changed_files)

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

    def _index_file(self, store: IndexStore, rel_path: str) -> None:
        """Read, chunk, and store a single file.

        @param store: IndexStore to write to.
        @param rel_path: Repo-relative path.
        """
        full = self.repo_path / rel_path
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Chunk the file.
        chunks, quality = chunk_file(
            rel_path, content, max_chars=self.config.chunk_max_chars
        )

        if chunks:
            store.insert_chunks(chunks)

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
            )
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


def _now_iso() -> str:
    """Current UTC time as ISO string.

    @returns: ISO-formatted timestamp.
    """
    return datetime.now(UTC).isoformat()
