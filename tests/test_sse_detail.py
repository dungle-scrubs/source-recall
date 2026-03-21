"""Tests for SSE progress with file-level detail."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from source_recall.daemon_config import DaemonConfig
from source_recall.embedder import BagOfWordsEmbedder
from source_recall.repo_manager import RepoSlot


class TestSSEProgressDetail:
    def test_progress_fields_on_slot(self, tmp_path: Path) -> None:
        """RepoSlot tracks progress fields (current, total, file)."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_indexing()
        slot.update_progress("src/auth.py", 3, 10)

        assert slot.progress_current == 3
        assert slot.progress_total == 10
        assert slot.progress_file == "src/auth.py"

    def test_sse_stream_contains_progress_fields(self, tmp_path: Path) -> None:
        """SSE progress events include current/total/file during indexing."""
        from source_recall import Index
        from source_recall.daemon import create_daemon_app

        emb = BagOfWordsEmbedder(dimensions=64)

        # Pre-index existing repo.
        existing = tmp_path / "existing"
        existing.mkdir()
        (existing / "a.py").write_text("x = 1\n")
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(Path.home()),
        }
        subprocess.run(["git", "init", str(existing)], capture_output=True, check=True)
        subprocess.run(
            ["git", "-C", str(existing), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(existing), "commit", "-m", "init"],
            capture_output=True,
            check=True,
            env=env,
        )
        Index(existing, embedder=emb).build()

        config = DaemonConfig(
            repos=[DaemonConfig.RepoEntry(path=existing, name="existing")],
            config_path=tmp_path / "repos.toml",
        )
        app = create_daemon_app(config, embedder=emb, refresh_min_interval=0)
        with TestClient(app) as client:
            # SSE for ready repo has completion event with state.
            resp = client.get("/repos/existing/progress")
            body = resp.text
            # Parse the data field of the complete event.
            for line in body.strip().split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line[len("data:") :].strip())
                    assert "state" in data
                    assert data["state"] == "ready"
