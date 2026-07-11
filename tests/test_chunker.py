"""Tests for chunker.py."""

from __future__ import annotations

from pathlib import Path

from source_recall.chunker import _split_into_sub_chunks, chunk_file
from source_recall.models import SearchQuality, SymbolType


class TestTypeScriptChunking:
    def test_function_declaration(self) -> None:
        """Function declarations are chunked."""
        code = "export function processPayment(amount: number): void {\n    console.log(amount)\n}\n"
        chunks, quality = chunk_file("payment.ts", code)
        assert quality == SearchQuality.AST
        assert len(chunks) >= 1
        assert chunks[0].symbol_type == SymbolType.FUNCTION
        assert chunks[0].symbol_name == "processPayment"

    def test_class_with_methods(self) -> None:
        """Classes are split into shell + methods."""
        code = (
            Path(__file__)
            .parent.joinpath("fixtures/ts-app/src/UserService.ts")
            .read_text()
        )
        chunks, quality = chunk_file("src/UserService.ts", code)
        assert quality == SearchQuality.AST
        # Should have: class shell + multiple methods.
        types = {c.symbol_type for c in chunks}
        assert SymbolType.CLASS_SHELL in types or SymbolType.METHOD in types

    def test_interface_and_type_alias(self) -> None:
        """Interfaces and type aliases are chunked."""
        code = (
            Path(__file__).parent.joinpath("fixtures/ts-app/src/types.ts").read_text()
        )
        chunks, quality = chunk_file("src/types.ts", code)
        assert quality == SearchQuality.AST

        types = {c.symbol_type for c in chunks}
        names = {c.symbol_name for c in chunks}
        assert SymbolType.INTERFACE in types
        assert "User" in names or "CreateUserRequest" in names

    def test_enum(self) -> None:
        """Enums are detected."""
        code = "export enum Color {\n    Red = 'red',\n    Blue = 'blue',\n}\n"
        chunks, quality = chunk_file("color.ts", code)
        assert quality == SearchQuality.AST
        assert any(c.symbol_type == SymbolType.ENUM for c in chunks)


class TestReactComponentDetection:
    def test_arrow_component_detected(self) -> None:
        """Arrow function returning JSX in .tsx → component."""
        code = "export const Card = ({ name }: Props) => {\n    return <div>{name}</div>\n}\n"
        chunks, quality = chunk_file("Card.tsx", code)
        assert quality == SearchQuality.AST
        assert any(c.symbol_type == SymbolType.COMPONENT for c in chunks)

    def test_function_declaration_component(self) -> None:
        """Function declaration returning JSX → component."""
        code = "export function UserList({ users }: Props) {\n    return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>\n}\n"
        chunks, quality = chunk_file("UserList.tsx", code)
        assert quality == SearchQuality.AST
        assert any(c.symbol_type == SymbolType.COMPONENT for c in chunks)

    def test_react_memo_detected(self) -> None:
        """React.memo wrapper detected as component."""
        code = (
            'import React from "react"\n'
            "export const MemoCard = React.memo(function Inner({ name }: Props) {\n"
            "    return <div>{name}</div>\n"
            "})\n"
        )
        chunks, quality = chunk_file("MemoCard.tsx", code)
        assert quality == SearchQuality.AST
        memo_chunks = [c for c in chunks if c.symbol_name == "MemoCard"]
        assert any(c.symbol_type == SymbolType.COMPONENT for c in memo_chunks)

    def test_react_forward_ref_detected(self) -> None:
        """React.forwardRef wrapper detected as component."""
        code = (
            'import React from "react"\n'
            "export const FancyInput = React.forwardRef<HTMLInputElement, Props>(\n"
            "    (props, ref) => <input ref={ref} {...props} />\n"
            ")\n"
        )
        chunks, quality = chunk_file("FancyInput.tsx", code)
        assert quality == SearchQuality.AST
        fi_chunks = [c for c in chunks if c.symbol_name == "FancyInput"]
        assert any(c.symbol_type == SymbolType.COMPONENT for c in fi_chunks)

    def test_non_jsx_ts_file_no_components(self) -> None:
        """PascalCase in .ts (not .tsx) is not a component."""
        code = (
            "export const UserService = () => {\n    return { find: () => null }\n}\n"
        )
        chunks, quality = chunk_file("UserService.ts", code)
        assert quality == SearchQuality.AST
        assert not any(c.symbol_type == SymbolType.COMPONENT for c in chunks)


class TestPythonChunking:
    def test_function(self) -> None:
        """Standalone functions are chunked."""
        code = "def validate_email(email: str) -> bool:\n    return '@' in email\n"
        chunks, quality = chunk_file("utils.py", code)
        assert quality == SearchQuality.AST
        assert len(chunks) >= 1
        assert chunks[0].symbol_name == "validate_email"
        assert chunks[0].symbol_type == SymbolType.FUNCTION

    def test_class_with_methods(self) -> None:
        """Classes produce shell + method chunks."""
        code = (
            Path(__file__).parent.joinpath("fixtures/py-app/myapp/auth.py").read_text()
        )
        chunks, quality = chunk_file("myapp/auth.py", code)
        assert quality == SearchQuality.AST

        names = {c.symbol_name for c in chunks}
        types = {c.symbol_type for c in chunks}

        assert SymbolType.METHOD in types
        assert any("authenticate" in n for n in names)
        assert any("validate_token" in n for n in names)

    def test_decorated_function(self) -> None:
        """Decorated functions include the decorator."""
        code = "@dataclass\nclass Token:\n    user_id: str\n    expires_at: str\n"
        chunks, quality = chunk_file("models.py", code)
        assert quality == SearchQuality.AST
        assert len(chunks) >= 1


class TestBashChunking:
    def test_function_detection(self) -> None:
        """Bash functions are detected via regex."""
        code = Path(__file__).parent.joinpath("fixtures/mixed/deploy.sh").read_text()
        chunks, quality = chunk_file("deploy.sh", code)
        assert quality == SearchQuality.REGEX

        names = {c.symbol_name for c in chunks}
        assert "check_prerequisites" in names
        assert "build_image" in names
        assert "deploy_to_k8s" in names

    def test_regex_quality(self) -> None:
        """Bash chunks have regex search quality."""
        code = "function hello() {\n    echo hello\n}\n"
        chunks, quality = chunk_file("test.sh", code)
        assert quality == SearchQuality.REGEX
        assert all(c.search_quality == SearchQuality.REGEX for c in chunks)


class TestTextFallback:
    def test_yaml_fallback(self) -> None:
        """YAML files use text fallback."""
        code = Path(__file__).parent.joinpath("fixtures/mixed/config.yaml").read_text()
        chunks, quality = chunk_file("config.yaml", code)
        assert quality == SearchQuality.TEXT_FALLBACK
        assert all(c.search_quality == SearchQuality.TEXT_FALLBACK for c in chunks)

    def test_unsupported_extension(self) -> None:
        """Unknown extensions use text fallback."""
        code = "some content\n\nmore content\n"
        chunks, quality = chunk_file("readme.txt", code)
        assert quality == SearchQuality.TEXT_FALLBACK


class TestSubChunking:
    def test_large_function_is_sub_chunked(self) -> None:
        """Functions exceeding max_chars are split into sub-chunks."""
        # Generate a large function.
        lines = ["def big_function():"]
        for i in range(200):
            lines.append(f"    x_{i} = process_item({i})  # line {i}")
        code = "\n".join(lines)

        chunks, quality = chunk_file("big.py", code, max_chars=2000)
        assert quality == SearchQuality.AST
        assert len(chunks) > 1

        # All sub-chunks should have parent_chunk_id set.
        for c in chunks:
            assert c.parent_chunk_id is not None
            assert c.sub_chunk_index is not None

        # Sub-chunk indices should be sequential.
        indices = sorted(c.sub_chunk_index for c in chunks)
        assert indices == list(range(len(chunks)))

    def test_small_function_not_sub_chunked(self) -> None:
        """Functions under max_chars are not sub-chunked."""
        code = "def small():\n    return 1\n"
        chunks, quality = chunk_file("small.py", code)
        assert len(chunks) == 1
        assert chunks[0].parent_chunk_id is None
        assert chunks[0].sub_chunk_index is None


class TestSubChunkOverlap:
    def test_sub_chunks_have_overlapping_lines(self) -> None:
        """Adjacent sub-chunks share overlapping content."""
        lines = ["def big_function():"]
        for i in range(200):
            lines.append(f"    x_{i} = process_item({i})  # line {i}")
        code = "\n".join(lines)

        chunks, quality = chunk_file("big.py", code, max_chars=2000)
        assert quality == SearchQuality.AST
        assert len(chunks) >= 3

        # Check that consecutive sub-chunks overlap: the end of chunk N
        # should appear in the start of chunk N+1.
        for i in range(len(chunks) - 1):
            current_lines = chunks[i].content.splitlines()
            next_lines = chunks[i + 1].content.splitlines()
            # Last few lines of current chunk should appear in next chunk.
            tail = set(current_lines[-5:])
            head = set(next_lines[:10])
            overlap = tail & head
            assert len(overlap) > 0, f"No overlap between sub-chunk {i} and {i + 1}"

    def test_multiline_signature_drops_no_lines(self) -> None:
        """A long multi-line signature must not let the overlap clamp jump
        the cursor forward past unemitted lines and silently drop them."""
        # A 5-line signature whose lines are individually long enough that
        # the first sub-chunk cannot fit them all.  The old clamp rewound
        # ``i`` forward to ``sig_line_count`` (== len(lines)), ending the
        # loop before the remaining lines were ever emitted.
        lines = [
            "def process(",
            "    aaa=" + "x" * 600 + ",",
            "    bbb=" + "y" * 600 + ",",
            "    ccc=" + "z" * 600 + ",",
            "):",
        ]
        content = "\n".join(lines)

        subs = _split_into_sub_chunks(content, 2000, "process")
        union = "\n".join(text for text, _, _ in subs)

        for line in lines:
            assert line in union, f"Source line dropped from sub-chunks: {line[:24]!r}"


class TestEdgeCaseFiles:
    def test_empty_file_returns_no_chunks(self) -> None:
        """Empty content produces no chunks."""
        chunks, quality = chunk_file("empty.py", "")
        assert chunks == []

    def test_whitespace_only_file(self) -> None:
        """Whitespace-only content produces no chunks."""
        chunks, quality = chunk_file("blank.py", "   \n\n  \n")
        assert chunks == []

    def test_binary_like_content(self) -> None:
        """Files with binary-ish content don't crash."""
        # Simulated binary content that's technically valid UTF-8.
        content = "\x00\x01\x02\x03" * 100
        chunks, quality = chunk_file("mystery.bin", content)
        # Should not crash — may return text fallback or empty.
        assert isinstance(chunks, list)

    def test_single_line_file(self) -> None:
        """Single-line files produce at least one chunk."""
        chunks, quality = chunk_file("one.py", "x = 42")
        assert len(chunks) >= 1


class TestErrorNodeDensity:
    def test_high_error_density_falls_back(self) -> None:
        """Files with >10% error nodes fall back to text chunking."""
        # Intentionally broken TypeScript.
        code = "export %%% broken {{{{ syntax !!! @@@\n" * 20
        chunks, quality = chunk_file("broken.ts", code)
        assert quality == SearchQuality.TEXT_FALLBACK


class TestDeepASTResilience:
    def test_deeply_nested_ast_does_not_recurse(self) -> None:
        """Deeply nested nodes must not raise RecursionError in AST scans."""
        from tree_sitter_language_pack import get_parser

        from source_recall.chunker import (
            _contains_any_node_type,
            _contains_node_type,
            _has_react_wrapper,
        )

        depth = 3000
        code = "[" * depth + "1" + "]" * depth
        tree = get_parser("javascript").parse(code.encode())
        root = tree.root_node

        # None of these should raise; a straight recursive walk would blow
        # the interpreter stack well before this depth.
        assert _contains_node_type(root, "array") is True
        assert _contains_node_type(root, "jsx_element") is False
        assert _contains_any_node_type(root, {"array"}) is True
        assert _contains_any_node_type(root, {"jsx_element"}) is False
        assert _has_react_wrapper(root) is False


class TestParserCache:
    def test_parser_reused_for_same_language(self) -> None:
        """Parsers are cached per language to avoid allocation overhead (M5)."""
        from source_recall.chunker import _get_cached_parser

        p1 = _get_cached_parser("python")
        p2 = _get_cached_parser("python")
        assert p1 is p2

    def test_different_languages_get_different_parsers(self) -> None:
        """Different language keys produce distinct parser instances."""
        from source_recall.chunker import _get_cached_parser

        py = _get_cached_parser("python")
        ts = _get_cached_parser("typescript")
        assert py is not ts
