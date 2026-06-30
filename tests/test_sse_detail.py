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
    def test_index_build_reports_post_scan_phases(self, tmp_path: Path) -> None:
        """Index builds report scanning, embedding, and finalizing phases."""
        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("def hello() -> str:\n    return 'hi'\n")
        phases: list[str] = []

        Index(
            repo,
            embedder=BagOfWordsEmbedder(dimensions=64),
            on_phase=phases.append,
        ).build()

        assert phases == ["scanning", "embedding", "finalizing"]

    def test_index_build_reports_embedding_batch_span_detail(
        self, tmp_path: Path
    ) -> None:
        """Index builds report batch-level embedding timing detail."""
        from source_recall import Index

        repo = tmp_path / "repo"
        repo.mkdir()
        for i in range(3):
            (repo / f"app_{i}.py").write_text(
                f"def hello_{i}() -> str:\n    return 'hi {i}'\n"
            )
        details: list[dict[str, object]] = []

        Index(
            repo,
            embedder=BagOfWordsEmbedder(dimensions=64),
            embed_batch_size=2,
            on_progress_detail=details.append,
        ).build()

        batch_details = [
            detail for detail in details if detail.get("span") == "embedding.batch"
        ]
        assert batch_details
        assert batch_details[0]["span_current"] == 1
        assert batch_details[0]["span_total"] == 2
        assert batch_details[0]["batch_size"] == 2
        assert "build_elapsed_ms" in batch_details[0]
        assert "phase_elapsed_ms" in batch_details[0]

    def test_progress_fields_on_slot(self, tmp_path: Path) -> None:
        """RepoSlot tracks progress fields (current, total, file)."""
        slot = RepoSlot(name="test", path=tmp_path)
        slot.set_indexing()
        slot.update_progress("src/auth.py", 3, 10)
        slot.update_progress_phase("embedding")
        slot.update_progress_detail({"span": "embedding.batch", "span_current": 2})

        assert slot.progress_current == 3
        assert slot.progress_total == 10
        assert slot.progress_file == "src/auth.py"
        assert slot.progress_phase == "embedding"
        assert slot.progress_detail == {"span": "embedding.batch", "span_current": 2}

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
                    assert data["phase"] == "ready"
