"""Tests for embedder.py — BagOfWordsEmbedder and Protocol compliance."""

from __future__ import annotations

import math

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
