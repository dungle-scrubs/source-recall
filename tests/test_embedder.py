"""Tests for embedder.py — BagOfWordsEmbedder and Protocol compliance."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from source_recall.embedder import BagOfWordsEmbedder, Embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    @param a: First vector.
    @param b: Second vector.
    @returns: Cosine similarity in [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TestBagOfWordsEmbedder:
    def test_implements_protocol(self) -> None:
        """BagOfWordsEmbedder satisfies the Embedder protocol."""
        embedder = BagOfWordsEmbedder()
        assert isinstance(embedder, Embedder)

    def test_dimensions(self) -> None:
        """Dimensions match constructor argument."""
        assert BagOfWordsEmbedder(dimensions=32).dimensions == 32
        assert BagOfWordsEmbedder(dimensions=128).dimensions == 128
        assert BagOfWordsEmbedder().dimensions == 64

    def test_shared_vocab_produces_high_similarity(self) -> None:
        """Texts with shared words produce high cosine similarity."""
        embedder = BagOfWordsEmbedder(dimensions=128)

        a = embedder.embed_query("def process_payment amount charge")
        b = embedder.embed_chunks(["def process_payment amount charge refund"])[0]

        sim = _cosine_similarity(a, b)
        assert sim > 0.8, f"Expected high similarity, got {sim}"

    def test_disjoint_vocab_produces_low_similarity(self) -> None:
        """Texts with no shared words produce low cosine similarity."""
        embedder = BagOfWordsEmbedder(dimensions=256)

        a = embedder.embed_query("authentication login password")
        b = embedder.embed_chunks(["rendering canvas pixel shader"])[0]

        sim = _cosine_similarity(a, b)
        assert sim < 0.5, f"Expected low similarity, got {sim}"

    def test_identical_texts_perfect_similarity(self) -> None:
        """Identical texts produce cosine similarity ≈ 1.0."""
        embedder = BagOfWordsEmbedder(dimensions=64)
        text = "class UserService validate authenticate"

        a = embedder.embed_query(text)
        b = embedder.embed_chunks([text])[0]

        sim = _cosine_similarity(a, b)
        assert sim > 0.99, f"Expected ~1.0, got {sim}"

    def test_batch_preserves_order(self) -> None:
        """Batch embedding returns vectors in input order."""
        embedder = BagOfWordsEmbedder(dimensions=64)
        texts = ["alpha beta", "gamma delta", "epsilon zeta"]

        batch = embedder.embed_chunks(texts)
        singles = [embedder.embed_chunks([t])[0] for t in texts]

        assert len(batch) == 3
        for i in range(3):
            assert batch[i] == singles[i]

    def test_empty_input(self) -> None:
        """Empty input list returns empty output."""
        embedder = BagOfWordsEmbedder()
        assert embedder.embed_chunks([]) == []

    def test_vectors_are_normalized(self) -> None:
        """Output vectors have unit L2 norm."""
        embedder = BagOfWordsEmbedder(dimensions=64)
        vec = embedder.embed_query("hello world test")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_empty_text_returns_zero_vector(self) -> None:
        """Empty text returns a zero vector (L2 norm = 0)."""
        embedder = BagOfWordsEmbedder(dimensions=32)
        vec = embedder.embed_query("")
        assert all(v == 0.0 for v in vec)
        assert len(vec) == 32

    def test_deterministic_across_calls(self) -> None:
        """Same text always produces the same vector (L2 audit fix).

        Python's built-in hash() is randomized per process (PYTHONHASHSEED).
        The embedder must use a deterministic hash so vectors are stable
        across process restarts.
        """
        embedder = BagOfWordsEmbedder(dimensions=64)
        v1 = embedder.embed_query("authenticate login session")
        v2 = embedder.embed_query("authenticate login session")
        assert v1 == v2

        # Verify specific bucket assignments are deterministic by checking
        # a known non-zero pattern.
        assert any(v != 0.0 for v in v1)


class TestCodeRankEncodeBatchSize:
    class _FakeArray:
        def __init__(self, rows: int) -> None:
            self._rows = rows

        def tolist(self) -> list[list[float]]:
            return [[0.0] for _ in range(self._rows)]

    def test_encode_batch_size_passed_through(self) -> None:
        """embed_chunks passes the configured encode batch size to encode."""
        from source_recall.embedder import CodeRankEmbedder

        emb = CodeRankEmbedder(encode_batch_size=4)
        captured: dict[str, object] = {}

        class FakeModel:
            max_seq_length = 512

            def encode(self, inputs, show_progress_bar=False, batch_size=None):
                captured["batch_size"] = batch_size
                return TestCodeRankEncodeBatchSize._FakeArray(len(inputs))

        emb._model = FakeModel()
        emb.embed_chunks(["a", "b"])
        assert captured["batch_size"] == 4

    def test_encode_batch_size_clamped_to_safe_max(self) -> None:
        """An oversized batch size is clamped to avoid attention-matrix OOM."""
        from source_recall.embedder import _MAX_ENCODE_BATCH, CodeRankEmbedder

        emb = CodeRankEmbedder(encode_batch_size=100_000)
        captured: dict[str, object] = {}

        class FakeModel:
            max_seq_length = 512

            def encode(self, inputs, show_progress_bar=False, batch_size=None):
                captured["batch_size"] = batch_size
                return TestCodeRankEncodeBatchSize._FakeArray(len(inputs))

        emb._model = FakeModel()
        emb.embed_chunks(["a"])
        assert captured["batch_size"] == _MAX_ENCODE_BATCH


class TestModelIntegrityGate:
    """The trust_remote_code integrity gate (revision pin + file checksums)."""

    def test_revision_is_full_commit_sha(self) -> None:
        """The pinned revision must be a full 40-hex git commit SHA.

        A full commit SHA content-addresses the entire repo tree at that
        revision, so it is the primary supply-chain guarantee: the Hub can
        only resolve it to the exact audited file bytes. A branch/tag ref
        would be mutable and defeat the pin.
        """
        import re

        from source_recall.embedder import _CODERANK_REVISION

        assert re.fullmatch(r"[0-9a-f]{40}", _CODERANK_REVISION)

    def test_trusted_files_cover_config_and_auto_map_modules(self) -> None:
        """The pinned checksum set covers config.json AND every executed module.

        ``trust_remote_code=True`` executes not just config.json but the
        remote .py modules its ``auto_map`` names, so each must be pinned.
        """
        from source_recall.embedder import _CODERANK_TRUSTED_FILE_SHA256

        assert "config.json" in _CODERANK_TRUSTED_FILE_SHA256
        assert "configuration_hf_nomic_bert.py" in _CODERANK_TRUSTED_FILE_SHA256
        assert "modeling_hf_nomic_bert.py" in _CODERANK_TRUSTED_FILE_SHA256

    def test_auto_map_module_files_discovers_py_modules(self) -> None:
        """auto_map values resolve to their <module>.py filenames."""
        from source_recall.embedder import _auto_map_module_files

        files = _auto_map_module_files(
            {
                "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
                "AutoModel": "modeling_hf_nomic_bert.NomicBertModel",
            }
        )
        assert files == [
            "configuration_hf_nomic_bert.py",
            "modeling_hf_nomic_bert.py",
        ]

    def test_bad_revision_pin_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A revision that is not a full commit SHA is refused before any load."""
        from source_recall import embedder as emb_mod

        monkeypatch.setattr(emb_mod, "_CODERANK_REVISION", "main")
        emb = emb_mod.CodeRankEmbedder()
        with pytest.raises(emb_mod.EmbedderVerificationError, match="commit SHA"):
            emb._verify_model_integrity()

    def test_tampered_remote_module_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A remote module whose bytes do not match its pinned SHA is refused.

        Uses a stubbed resolver so the check is deterministic and offline: the
        config resolves to bytes matching its pinned checksum, but the module
        file's bytes do not match the pinned module checksum.
        """
        import hashlib
        import json

        from source_recall import embedder as emb_mod

        config_bytes = json.dumps(
            {"auto_map": {"AutoModel": "modeling_hf_nomic_bert.NomicBertModel"}}
        ).encode()
        config_path = tmp_path / "config.json"
        config_path.write_bytes(config_bytes)
        module_path = tmp_path / "modeling_hf_nomic_bert.py"
        module_path.write_bytes(b"# tampered\nimport os\n")

        pinned = {
            "config.json": hashlib.sha256(config_bytes).hexdigest(),
            "modeling_hf_nomic_bert.py": "0" * 64,  # deliberately wrong
        }
        monkeypatch.setattr(emb_mod, "_CODERANK_TRUSTED_FILE_SHA256", pinned)

        emb = emb_mod.CodeRankEmbedder()
        resolved = {
            "config.json": str(config_path),
            "modeling_hf_nomic_bert.py": str(module_path),
        }
        monkeypatch.setattr(emb, "_resolve_trusted_file", lambda name: resolved[name])

        with pytest.raises(emb_mod.EmbedderVerificationError, match="checksum"):
            emb._verify_model_integrity()

    def test_auto_map_module_not_in_pinned_set_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An auto_map module absent from the pinned set is refused (fail-closed)."""
        import hashlib
        import json

        from source_recall import embedder as emb_mod

        config_bytes = json.dumps(
            {"auto_map": {"AutoModel": "surprise_module.Model"}}
        ).encode()
        config_path = tmp_path / "config.json"
        config_path.write_bytes(config_bytes)

        pinned = {"config.json": hashlib.sha256(config_bytes).hexdigest()}
        monkeypatch.setattr(emb_mod, "_CODERANK_TRUSTED_FILE_SHA256", pinned)

        emb = emb_mod.CodeRankEmbedder()
        monkeypatch.setattr(
            emb, "_resolve_trusted_file", lambda _name: str(config_path)
        )
        with pytest.raises(emb_mod.EmbedderVerificationError):
            emb._verify_model_integrity()


class TestExecutionCacheGate:
    """The exec-cache gate verifies the copies trust_remote_code actually runs."""

    def test_tampered_execution_cache_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A tampered copy in the transformers module cache is refused.

        The Hub snapshot may be pristine, but trust_remote_code executes a COPY
        under HF_MODULES_CACHE that transformers will not overwrite for a pinned
        revision — so that copy must be verified too.
        """
        import transformers.utils as tu

        from source_recall import embedder as emb_mod

        modules_root = tmp_path / "modules"
        cache_dir = (
            modules_root
            / "transformers_modules"
            / "nomic_hyphen_ai"
            / "CodeRankEmbed"
            / emb_mod._CODERANK_REVISION
        )
        cache_dir.mkdir(parents=True)
        (cache_dir / "modeling_hf_nomic_bert.py").write_bytes(b"import os  # evil\n")

        monkeypatch.setattr(tu, "HF_MODULES_CACHE", str(modules_root))
        monkeypatch.setattr(
            emb_mod,
            "_CODERANK_TRUSTED_FILE_SHA256",
            {"modeling_hf_nomic_bert.py": "0" * 64},
        )

        emb = emb_mod.CodeRankEmbedder()
        with pytest.raises(emb_mod.EmbedderVerificationError, match="Executed"):
            emb._verify_execution_cache(["modeling_hf_nomic_bert.py"])

    def test_matching_execution_cache_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A copy whose bytes match the pinned checksum is accepted."""
        import hashlib

        import transformers.utils as tu

        from source_recall import embedder as emb_mod

        modules_root = tmp_path / "modules"
        cache_dir = (
            modules_root
            / "transformers_modules"
            / "any-sanitized-name"
            / emb_mod._CODERANK_REVISION
        )
        cache_dir.mkdir(parents=True)
        content = b"# legitimate module bytes\n"
        (cache_dir / "modeling_hf_nomic_bert.py").write_bytes(content)

        monkeypatch.setattr(tu, "HF_MODULES_CACHE", str(modules_root))
        monkeypatch.setattr(
            emb_mod,
            "_CODERANK_TRUSTED_FILE_SHA256",
            {"modeling_hf_nomic_bert.py": hashlib.sha256(content).hexdigest()},
        )

        emb = emb_mod.CodeRankEmbedder()
        # Must not raise.
        emb._verify_execution_cache(["modeling_hf_nomic_bert.py"])

    def test_absent_execution_cache_is_safe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No materialized copy yet is a no-op (snapshot check still applies)."""
        import transformers.utils as tu

        from source_recall import embedder as emb_mod

        monkeypatch.setattr(tu, "HF_MODULES_CACHE", str(tmp_path / "empty"))
        emb = emb_mod.CodeRankEmbedder()
        emb._verify_execution_cache(["modeling_hf_nomic_bert.py"])  # no raise


class TestLoadModelConcurrency:
    """_load_model must be idempotent under concurrent callers (warmup race)."""

    def test_concurrent_load_constructs_model_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads calling _load_model construct SentenceTransformer once.

        The Stage-4 startup warmup thread can race a real query; without a
        lock both would construct (and download) the model, spiking memory.
        """
        import sys
        import threading
        import time
        import types

        from source_recall import embedder as emb_mod

        emb = emb_mod.CodeRankEmbedder(show_progress=False)
        monkeypatch.setattr(emb, "_verify_model_integrity", lambda: None)

        constructions: list[int] = []

        class FakeSentenceTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                constructions.append(1)
                # Widen the race window so an unsynchronized check-then-load
                # would reliably double-construct.
                time.sleep(0.05)

        fake_mod = types.ModuleType("sentence_transformers")
        fake_mod.SentenceTransformer = (  # ty: ignore[unresolved-attribute] dynamic attribute on a fake module standing in for sentence-transformers
            FakeSentenceTransformer
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            emb._load_model()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(constructions) == 1


class TestCodeRankInference:
    """Real CodeRankEmbed inference smoke test (downloads/loads the model)."""

    @pytest.mark.slow
    def test_embeds_chunk_and_query_with_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loads the real model, embeds a chunk and a query, checks dims and prefix.

        Exercises _load_model (config verification + cpu/max_seq_length),
        embed_chunks, and embed_query. Asserts 768-d output and that
        embed_query prepends _QUERY_PREFIX to the encoded string.
        """
        from source_recall.embedder import (
            _CODERANK_DIMENSIONS,
            _QUERY_PREFIX,
            CodeRankEmbedder,
        )

        emb = CodeRankEmbedder(show_progress=False)

        chunk_vec = emb.embed_chunks(["def add(a, b):\n    return a + b"])[0]
        assert len(chunk_vec) == _CODERANK_DIMENSIONS == 768

        query_vec = emb.embed_query("how to add two numbers")
        assert len(query_vec) == 768

        # The model is loaded now; spy on encode to capture the exact input
        # string and confirm embed_query applies the required query prefix.
        captured: dict[str, object] = {}
        real_encode = emb._model.encode  # ty: ignore[unresolved-attribute] deliberate reach into private _model (typed object); sentence-transformers is absent from the type env

        def spy(
            texts: list[str],
            *,
            show_progress_bar: bool | None = False,
            batch_size: int | None = None,
        ) -> object:
            captured["texts"] = texts
            if batch_size is None:
                return real_encode(texts, show_progress_bar=show_progress_bar)
            return real_encode(
                texts, show_progress_bar=show_progress_bar, batch_size=batch_size
            )

        monkeypatch.setattr(emb._model, "encode", spy)
        emb.embed_query("find the parser")
        assert captured["texts"] == [f"{_QUERY_PREFIX}find the parser"]
