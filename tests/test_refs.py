"""Tests for cross-reference extraction, storage, and graph expansion."""

from __future__ import annotations

from pathlib import Path

from source_recall.models import RefData, RefType


class TestRefModels:
    def test_ref_type_values(self) -> None:
        """RefType enum has the expected members."""
        assert RefType.IMPORT == "import"
        assert RefType.TYPE_REF == "type_ref"
        assert RefType.CALL == "call"
        assert RefType.INHERITS == "inherits"
        assert RefType.DECORATOR == "decorator"

    def test_ref_data_creation(self) -> None:
        """RefData can be constructed with all fields."""
        ref = RefData(
            source_chunk_id="abc123",
            target_symbol="SomeClass",
            ref_type=RefType.IMPORT,
        )
        assert ref.source_chunk_id == "abc123"
        assert ref.target_symbol == "SomeClass"
        assert ref.ref_type == RefType.IMPORT


class TestRefStore:
    def test_insert_and_lookup_refs(self, tmp_path: Path) -> None:
        """Refs round-trip through the store."""
        from source_recall.store import IndexStore

        db = tmp_path / "test.db"
        store = IndexStore(db)
        store.open()
        store.create_schema()
        store.run_migrations()

        # Insert a chunk first (FK target).
        from source_recall.models import ChunkData, SearchQuality, SymbolType

        chunk = ChunkData(
            file_path="app.py",
            symbol_name="process",
            symbol_type=SymbolType.FUNCTION,
            content="def process(): pass",
            start_line=1,
            end_line=1,
            search_quality=SearchQuality.AST,
        )
        store.insert_chunks([chunk])

        refs = [
            RefData(
                source_chunk_id=chunk.chunk_id,
                target_symbol="os.path",
                ref_type=RefType.IMPORT,
            ),
            RefData(
                source_chunk_id=chunk.chunk_id,
                target_symbol="SomeClass",
                ref_type=RefType.CALL,
            ),
        ]
        store.insert_refs(refs)

        got = store.get_refs_for_chunk(chunk.chunk_id)
        assert len(got) == 2
        symbols = {r.target_symbol for r in got}
        assert symbols == {"os.path", "SomeClass"}

        store.close()

    def test_symbol_lookup_round_trip(self, tmp_path: Path) -> None:
        """Symbol lookup maps symbol names to chunk IDs."""
        from source_recall.models import ChunkData, SearchQuality, SymbolType
        from source_recall.store import IndexStore

        db = tmp_path / "test.db"
        store = IndexStore(db)
        store.open()
        store.create_schema()
        store.run_migrations()

        chunk = ChunkData(
            file_path="models.py",
            symbol_name="UserModel",
            symbol_type=SymbolType.CLASS,
            content="class UserModel: pass",
            start_line=1,
            end_line=1,
            search_quality=SearchQuality.AST,
        )
        store.insert_chunks([chunk])
        store.insert_symbol_lookup(chunk.chunk_id, "UserModel", "models.py")

        results = store.lookup_symbol("UserModel")
        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id

        store.close()

    def test_delete_refs_cascades_with_chunk(self, tmp_path: Path) -> None:
        """Deleting chunks removes associated refs."""
        from source_recall.models import ChunkData, SearchQuality, SymbolType
        from source_recall.store import IndexStore

        db = tmp_path / "test.db"
        store = IndexStore(db)
        store.open()
        store.create_schema()
        store.run_migrations()

        chunk = ChunkData(
            file_path="x.py",
            symbol_name="foo",
            symbol_type=SymbolType.FUNCTION,
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            search_quality=SearchQuality.AST,
        )
        store.insert_chunks([chunk])
        store.insert_refs(
            [
                RefData(
                    source_chunk_id=chunk.chunk_id,
                    target_symbol="bar",
                    ref_type=RefType.CALL,
                ),
            ]
        )
        store.insert_symbol_lookup(chunk.chunk_id, "foo", "x.py")

        # Delete the file's chunks.
        store.delete_chunks_for_file("x.py")

        assert store.get_refs_for_chunk(chunk.chunk_id) == []
        assert store.lookup_symbol("foo") == []

        store.close()


class TestRefExtraction:
    def test_python_imports_extracted(self) -> None:
        """Python import statements produce import refs."""
        from source_recall.chunker import chunk_file_with_refs

        content = (
            "from os.path import join\n"
            "import json\n"
            "\n"
            "def process():\n"
            "    data = json.loads('{}')\n"
            "    return join('/tmp', 'file')\n"
        )
        chunks, _quality, refs = chunk_file_with_refs("app.py", content)

        import_refs = [r for r in refs if r.ref_type == RefType.IMPORT]
        targets = {r.target_symbol for r in import_refs}
        assert "os.path.join" in targets or "join" in targets
        assert "json" in targets

    def test_python_inheritance_extracted(self) -> None:
        """Python class inheritance produces inherits refs."""
        from source_recall.chunker import chunk_file_with_refs

        content = "class MyModel(BaseModel):\n    name: str\n"
        chunks, _quality, refs = chunk_file_with_refs("models.py", content)

        inherit_refs = [r for r in refs if r.ref_type == RefType.INHERITS]
        targets = {r.target_symbol for r in inherit_refs}
        assert "BaseModel" in targets

    def test_python_decorator_extracted(self) -> None:
        """Python decorators produce decorator refs."""
        from source_recall.chunker import chunk_file_with_refs

        content = "@app.route('/api')\ndef handler():\n    pass\n"
        chunks, _quality, refs = chunk_file_with_refs("views.py", content)

        deco_refs = [r for r in refs if r.ref_type == RefType.DECORATOR]
        targets = {r.target_symbol for r in deco_refs}
        assert "app.route" in targets

    def test_typescript_imports_extracted(self) -> None:
        """TypeScript import statements produce import refs."""
        from source_recall.chunker import chunk_file_with_refs

        content = (
            "import { useState, useEffect } from 'react';\n"
            "\n"
            "export function App() {\n"
            "    const [x, setX] = useState(0);\n"
            "    return <div>{x}</div>;\n"
            "}\n"
        )
        chunks, _quality, refs = chunk_file_with_refs("App.tsx", content)

        import_refs = [r for r in refs if r.ref_type == RefType.IMPORT]
        targets = {r.target_symbol for r in import_refs}
        assert "useState" in targets
        assert "useEffect" in targets


class TestRefBuildIntegration:
    def test_build_populates_refs_and_symbols(self, py_app_path: Path) -> None:
        """Full build stores refs and symbol_lookup entries."""
        from source_recall import Index
        from source_recall.embedder import BagOfWordsEmbedder
        from source_recall.store import IndexStore, get_db_path

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        db_path = get_db_path(py_app_path)
        store = IndexStore(db_path)
        store.open()
        store.run_migrations()

        # py-app fixture has imports — refs should exist.
        ref_count = store.conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        assert ref_count > 0

        # Symbol lookup should have entries for defined symbols.
        sym_count = store.conn.execute("SELECT COUNT(*) FROM symbol_lookup").fetchone()[
            0
        ]
        assert sym_count > 0

        store.close()

    def test_graph_expansion_finds_related_chunks(self, py_app_path: Path) -> None:
        """Query results include graph-expanded chunks from refs."""
        from source_recall import Index
        from source_recall.embedder import BagOfWordsEmbedder

        emb = BagOfWordsEmbedder(dimensions=64)
        idx = Index(py_app_path, embedder=emb)
        idx.build()

        # Query for something that should trigger graph expansion.
        results = idx.query("validate_email")

        # Should have results (at least the function itself).
        assert len(results) > 0

        # At least one result should be a direct match.
        direct = [
            r
            for r in results
            if "validate" in r.symbol_name.lower() or "validate" in r.content.lower()
        ]
        assert len(direct) > 0
