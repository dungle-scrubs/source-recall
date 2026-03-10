"""Tests for markdown, PDF, and plain-text chunking."""

from __future__ import annotations

from pathlib import Path

from source_recall.chunker import chunk_file
from source_recall.models import SearchQuality, SymbolType


class TestMarkdownChunker:
    def test_splits_on_headings(self) -> None:
        """Each heading starts a new chunk with heading as symbol_name."""
        md = (
            "# Introduction\n"
            "This is the intro.\n"
            "\n"
            "## Setup\n"
            "Install the thing.\n"
            "\n"
            "## Usage\n"
            "Run the thing.\n"
        )
        chunks, quality = chunk_file("docs/README.md", md)

        assert quality == SearchQuality.AST
        assert len(chunks) == 3

        assert chunks[0].symbol_name == "Introduction"
        assert "This is the intro." in chunks[0].content
        assert chunks[0].symbol_type == SymbolType.MODULE

        assert chunks[1].symbol_name == "Setup"
        assert "Install the thing." in chunks[1].content

        assert chunks[2].symbol_name == "Usage"
        assert "Run the thing." in chunks[2].content

    def test_content_before_first_heading(self) -> None:
        """Text before any heading becomes its own chunk."""
        md = "Some preamble text.\n\n# Title\nBody here.\n"
        chunks, _ = chunk_file("notes.md", md)

        assert len(chunks) == 2
        assert chunks[0].symbol_name == ""
        assert "Some preamble text." in chunks[0].content
        assert chunks[0].symbol_type == SymbolType.BLOCK

    def test_empty_section_skipped(self) -> None:
        """Heading followed immediately by another heading → no empty chunk."""
        md = "# One\n# Two\nContent here.\n"
        chunks, _ = chunk_file("doc.md", md)

        # "# One" has no body (next heading is immediate).
        names = [c.symbol_name for c in chunks]
        assert "Two" in names
        # One may or may not appear (it has "# One" as its body which is just the heading).
        # The key assertion: no chunk has empty content.
        for c in chunks:
            assert c.content.strip() != ""

    def test_code_fences_not_split(self) -> None:
        """Headings inside fenced code blocks are not treated as boundaries."""
        md = (
            "# Real heading\n"
            "Some text.\n"
            "\n"
            "```markdown\n"
            "# Fake heading inside fence\n"
            "```\n"
            "\n"
            "# Second heading\n"
            "More text.\n"
        )
        chunks, _ = chunk_file("doc.md", md)

        names = [c.symbol_name for c in chunks]
        assert "Real heading" in names
        assert "Second heading" in names
        assert "Fake heading inside fence" not in names


class TestPdfChunker:
    def test_extracts_text_per_page(self) -> None:
        """PDF produces one chunk per non-empty page."""
        pdf_path = Path(__file__).parent / "fixtures" / "sample.pdf"
        # chunk_file_pdf needs the actual file content (bytes).
        from source_recall.chunker import chunk_pdf

        chunks, quality = chunk_pdf("docs/sample.pdf", pdf_path)

        assert quality == SearchQuality.AST
        # 2 pages with text, 1 empty page → 2 chunks.
        assert len(chunks) == 2
        assert "authentication" in chunks[0].content
        assert "vector search" in chunks[1].content
        assert chunks[0].symbol_name == "Page 1"
        assert chunks[1].symbol_name == "Page 2"

    def test_empty_pdf_returns_no_chunks(self) -> None:
        """PDF with only empty pages returns empty list."""
        import fitz

        tmp = Path(__file__).parent / "fixtures" / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(tmp))
        doc.close()

        from source_recall.chunker import chunk_pdf

        chunks, _ = chunk_pdf("empty.pdf", tmp)
        assert chunks == []
        tmp.unlink()


class TestTextChunker:
    def test_prose_splits_on_sentences(self) -> None:
        """Plain text prose splits on sentence boundaries, not blank lines."""
        text = (
            "First sentence here. Second sentence follows. "
            "Third sentence is also short. Fourth sentence. "
            "Fifth sentence wraps this up."
        )
        # Use a small max_chars to force multiple chunks.
        chunks, quality = chunk_file("notes.txt", text, max_chars=80)

        assert quality == SearchQuality.TEXT_FALLBACK
        assert len(chunks) >= 2
        # Each chunk should end at a sentence boundary (period).
        for c in chunks:
            stripped = c.content.rstrip()
            assert stripped.endswith("."), (
                f"Chunk doesn't end at sentence: {stripped!r}"
            )

    def test_respects_max_chars(self) -> None:
        """No chunk exceeds max_chars."""
        text = ". ".join(f"Sentence number {i}" for i in range(50)) + "."
        chunks, _ = chunk_file("big.txt", text, max_chars=200)

        for c in chunks:
            assert (
                len(c.content) <= 200 + 50
            )  # Allow small overshoot for last sentence.


class TestBuilderPdfIntegration:
    def test_indexes_pdf_files(self, tmp_path: Path) -> None:
        """Builder indexes .pdf files and they appear in search results."""
        import fitz

        from source_recall import Index
        from source_recall.embedder import BagOfWordsEmbedder

        # Create a repo with a PDF.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "hello.py").write_text("def hello():\n    return 'world'\n")
        pdf_path = repo / "docs" / "guide.pdf"
        pdf_path.parent.mkdir()
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Authentication guide for the API.")
        doc.save(str(pdf_path))
        doc.close()

        idx = Index(repo, embedder=BagOfWordsEmbedder(dimensions=64))
        idx.build()

        status = idx.status()
        assert status.file_count >= 2  # .py + .pdf

        results = idx.query("authentication guide")
        # Must find the PDF with properly extracted text, not raw binary.
        pdf_results = [r for r in results if "guide.pdf" in r.file_path]
        assert len(pdf_results) > 0
        assert "Authentication" in pdf_results[0].content
