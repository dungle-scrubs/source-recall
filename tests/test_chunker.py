"""Tests for chunker.py."""

from __future__ import annotations

from pathlib import Path

from source_recall.chunker import (
    _is_blade,
    _split_into_sub_chunks,
    chunk_file,
    chunk_file_with_refs,
)
from source_recall.models import RefType, SearchQuality, SymbolType


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
        indices: list[int] = []
        for c in chunks:
            assert c.parent_chunk_id is not None
            idx = c.sub_chunk_index
            assert idx is not None
            indices.append(idx)

        # Sub-chunk indices should be sequential.
        indices.sort()
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


def _braces_balanced(text: str) -> bool:
    """Whether a chunk contains as many closing braces as opening ones.

    A cheap stand-in for "the function body was not cut in half": every
    fixture used here keeps braces out of string literals.
    """
    return text.count("{") == text.count("}")


class TestPhpChunking:
    def test_class_is_split_into_shell_and_methods(
        self, laravel_app_path: Path
    ) -> None:
        """A namespaced Laravel controller yields a class shell + methods."""
        path = laravel_app_path / "app/Http/Controllers/ReviewController.php"
        chunks, quality = chunk_file(
            "app/Http/Controllers/ReviewController.php", path.read_text()
        )

        assert quality == SearchQuality.AST
        names = {c.symbol_name for c in chunks}
        assert "ReviewController" in names
        assert "ReviewController.index" in names
        assert "ReviewController.approve" in names
        assert "ReviewController.cacheKey" in names

        shell = next(c for c in chunks if c.symbol_type == SymbolType.CLASS_SHELL)
        assert shell.symbol_name == "ReviewController"

    def test_no_chunk_splits_a_method_body(self, laravel_app_path: Path) -> None:
        """Every method arrives whole, never cut across two chunks."""
        path = laravel_app_path / "app/Http/Controllers/ReviewController.php"
        chunks, _ = chunk_file(
            "app/Http/Controllers/ReviewController.php", path.read_text()
        )

        methods = [c for c in chunks if c.symbol_type == SymbolType.METHOD]
        assert len(methods) == 5
        for method in methods:
            assert method.parent_chunk_id is None, "method was sub-chunked"
            assert _braces_balanced(method.content)
            assert method.content.rstrip().endswith("}")

    def test_docblock_travels_with_its_method(self, laravel_app_path: Path) -> None:
        """PHP hangs docblocks off a sibling comment node; re-attach them."""
        path = laravel_app_path / "app/Http/Controllers/ReviewController.php"
        chunks, _ = chunk_file(
            "app/Http/Controllers/ReviewController.php", path.read_text()
        )

        index = next(c for c in chunks if c.symbol_name == "ReviewController.index")
        assert index.content.lstrip().startswith("/**")
        assert "@return JsonResponse" in index.content

    def test_preamble_holds_namespace_and_imports(self, laravel_app_path: Path) -> None:
        """Imports get their own chunk, not the first symbol's."""
        path = laravel_app_path / "app/Http/Controllers/ReviewController.php"
        chunks, _ = chunk_file(
            "app/Http/Controllers/ReviewController.php", path.read_text()
        )

        preamble = chunks[0]
        assert preamble.symbol_type == SymbolType.MODULE
        assert "namespace App\\Http\\Controllers;" in preamble.content
        assert "use App\\Models\\Review;" in preamble.content
        # And it is not duplicated into per-statement chunks.
        assert sum("namespace App" in c.content for c in chunks) == 1

    def test_interface_trait_and_enum_keep_their_kind(
        self, laravel_app_path: Path
    ) -> None:
        """Non-class declarations are not flattened into CLASS_SHELL."""
        path = laravel_app_path / "app/Models/Review.php"
        chunks, quality = chunk_file("app/Models/Review.php", path.read_text())

        assert quality == SearchQuality.AST
        by_name = {c.symbol_name: c for c in chunks}
        assert by_name["Moderatable"].symbol_type == SymbolType.INTERFACE
        assert by_name["RecordsModeration"].symbol_type == SymbolType.TRAIT
        assert by_name["ReviewStatus"].symbol_type == SymbolType.ENUM
        assert by_name["Review"].symbol_type == SymbolType.CLASS_SHELL
        assert by_name["normalizeRating"].symbol_type == SymbolType.FUNCTION

    def test_mixed_html_and_php_is_handled(self, laravel_app_path: Path) -> None:
        """A template-style .php file still yields whole functions."""
        path = laravel_app_path / "public/legacy-report.php"
        chunks, quality = chunk_file("public/legacy-report.php", path.read_text())

        assert quality == SearchQuality.AST
        fn = next(c for c in chunks if c.symbol_name == "formatRow")
        assert fn.symbol_type == SymbolType.FUNCTION
        assert _braces_balanced(fn.content)
        assert "return sprintf" in fn.content
        # The surrounding HTML is indexed too, as block chunks.
        assert any("<table>" in c.content for c in chunks)

    def test_php_refs_are_extracted(self, laravel_app_path: Path) -> None:
        """`use` imports and `extends` become graph edges."""
        path = laravel_app_path / "app/Http/Controllers/ReviewController.php"
        _, _, refs = chunk_file_with_refs(
            "app/Http/Controllers/ReviewController.php", path.read_text()
        )

        targets = {(r.target_symbol, r.ref_type) for r in refs}
        assert ("App\\Models\\Review", RefType.IMPORT) in targets
        assert ("Controller", RefType.INHERITS) in targets

    def test_non_ascii_php_keeps_method_text_aligned(self) -> None:
        """Byte-offset slicing holds for the PHP grammar too."""
        code = (
            "<?php\n"
            "/** Résumé du contrôleur, accents non ASCII. */\n"
            "class Café\n"
            "{\n"
            "    public function première(): string\n"
            "    {\n"
            "        return 'naïve';\n"
            "    }\n"
            "}\n"
        )
        chunks, quality = chunk_file("Cafe.php", code)

        assert quality == SearchQuality.AST
        method = next(c for c in chunks if c.symbol_name == "Café.première")
        assert method.content.startswith("public function première()")
        assert method.content.rstrip().endswith("}")
        assert "return 'naïve';" in method.content

    def test_unparseable_php_falls_back(self) -> None:
        """Error-dense input drops to text chunking, as for every grammar."""
        code = "<?php\n" + ")))} ??? <<< ((( \n" * 40
        _, quality = chunk_file("broken.php", code)
        assert quality == SearchQuality.TEXT_FALLBACK


class TestBladeChunking:
    def test_blade_suffix_beats_php_extension(self, laravel_app_path: Path) -> None:
        """`.blade.php` routes to the Blade strategy, not the PHP grammar.

        The PHP grammar accepts a Blade template without errors - it just
        sees one opaque ``text`` node - so a suffix collision would look
        like a successful AST parse while producing a single file-sized
        chunk.  REGEX quality is the observable proof Blade won.
        """
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        chunks, quality = chunk_file(
            "resources/views/reviews/index.blade.php", path.read_text()
        )

        assert quality == SearchQuality.REGEX
        names = {c.symbol_name for c in chunks}
        assert "section:content" in names
        assert "push:scripts" in names

    def test_is_blade_matches_only_the_compound_suffix(self) -> None:
        """Routing predicate, independent of the chunking strategy."""
        assert _is_blade("resources/views/home.blade.php") is True
        assert _is_blade("resources/views/Home.Blade.PHP") is True
        assert _is_blade("app/Models/User.php") is False
        assert _is_blade("blade.php/app.py") is False

    def test_section_body_is_one_chunk(self, laravel_app_path: Path) -> None:
        """Nested control flow does not break a section apart."""
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        chunks, _ = chunk_file(
            "resources/views/reviews/index.blade.php", path.read_text()
        )

        section = next(c for c in chunks if c.symbol_name == "section:content")
        assert section.parent_chunk_id is None
        assert section.content.startswith("@section('content')")
        assert section.content.rstrip().endswith("@endsection")
        # The whole @if/@else and its @foreach live inside that one chunk.
        assert section.content.count("@foreach") == section.content.count("@endforeach")
        assert section.content.count("@if") == section.content.count("@endif")

    def test_directive_inside_a_blade_comment_is_ignored(
        self, laravel_app_path: Path
    ) -> None:
        """`{{-- @section --}}` must not open a block."""
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        chunks, _ = chunk_file(
            "resources/views/reviews/index.blade.php", path.read_text()
        )

        # The commented-out directive stays in the leading markup run, and
        # only the two real named blocks are detected.
        named = [c.symbol_name for c in chunks if c.symbol_name]
        assert named == ["section:content", "push:scripts"]

    def test_two_argument_section_is_inline(self, laravel_app_path: Path) -> None:
        """`@section('title', 'Reviews')` has no body to chunk."""
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        chunks, _ = chunk_file(
            "resources/views/reviews/index.blade.php", path.read_text()
        )

        assert not any(c.symbol_name == "section:title" for c in chunks)
        assert any("@section('title', 'Reviews')" in c.content for c in chunks)

    def test_escaped_at_sign_is_not_a_directive(self) -> None:
        """`@@if` renders a literal `@if` and opens nothing."""
        content = "<p>@@if is literal</p>\n@if ($x)\n<b>y</b>\n@endif\n"
        chunks, _ = chunk_file("t.blade.php", content)

        assert len(chunks) == 2
        assert chunks[0].content == "<p>@@if is literal</p>"
        assert chunks[1].content.startswith("@if ($x)")

    def test_unclosed_directive_still_chunks(self) -> None:
        """A malformed template closes open blocks at EOF instead of failing."""
        content = "@section('a')\n<p>one</p>\n@section('b')\n<p>two</p>\n"
        chunks, quality = chunk_file("t.blade.php", content)

        assert quality == SearchQuality.REGEX
        assert [c.symbol_name for c in chunks] == ["section:a"]

    def test_oversized_section_splits_on_inner_directives(self) -> None:
        """A section past max_chars descends to its nested blocks."""
        filler = "        <p>row</p>\n" * 20
        content = (
            "@section('content')\n"
            + filler
            + "@foreach ($rows as $row)\n"
            + filler
            + "@endforeach\n"
            + "@endsection\n"
        )
        chunks, _ = chunk_file("t.blade.php", content, max_chars=600)

        # Every chunk is attributed to the section it came from, and the
        # @foreach body was not cut mid-directive.
        assert all(c.symbol_name == "section:content" for c in chunks)
        foreach = next(c for c in chunks if c.content.startswith("@foreach"))
        assert foreach.content.rstrip().endswith("@endforeach")

    def test_beats_blank_line_splitting(self, laravel_app_path: Path) -> None:
        """The Blade strategy is chosen over the text fallback."""
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        chunks, quality = chunk_file(
            "resources/views/reviews/index.blade.php", path.read_text()
        )

        assert quality != SearchQuality.TEXT_FALLBACK
        assert all(c.search_quality == SearchQuality.REGEX for c in chunks)

    def test_blade_produces_no_refs(self, laravel_app_path: Path) -> None:
        """Blade has no import syntax to mine; ref extraction stays quiet."""
        path = laravel_app_path / "resources/views/reviews/index.blade.php"
        _, _, refs = chunk_file_with_refs(
            "resources/views/reviews/index.blade.php", path.read_text()
        )
        assert refs == []


class TestVueChunking:
    def test_sfc_blocks_and_script_symbols(self, laravel_app_path: Path) -> None:
        """A Vue SFC splits by block, and `<script setup>` gets TS chunks."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        chunks, quality = chunk_file("resources/js/ReviewCard.vue", path.read_text())

        assert quality == SearchQuality.AST
        names = {c.symbol_name for c in chunks}
        assert "approve" in names
        assert "reject" in names
        assert "Review" in names
        assert "template" in names
        assert "style" in names

    def test_script_chunks_match_typescript_quality(
        self, laravel_app_path: Path
    ) -> None:
        """Script-block chunks carry the same types a .ts file would."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        chunks, _ = chunk_file("resources/js/ReviewCard.vue", path.read_text())

        by_name = {c.symbol_name: c for c in chunks}
        assert by_name["Review"].symbol_type == SymbolType.INTERFACE
        assert by_name["approve"].symbol_type == SymbolType.FUNCTION
        assert by_name["canModerate"].symbol_type == SymbolType.FUNCTION
        assert all(
            c.search_quality == SearchQuality.AST
            for c in (by_name["Review"], by_name["approve"])
        )

    def test_script_function_bodies_stay_whole(self, laravel_app_path: Path) -> None:
        """No chunk boundary lands inside a function in the script block."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        chunks, _ = chunk_file("resources/js/ReviewCard.vue", path.read_text())

        approve = next(c for c in chunks if c.symbol_name == "approve")
        assert approve.parent_chunk_id is None
        assert _braces_balanced(approve.content)
        assert "finally" in approve.content
        assert approve.content.rstrip().endswith("}")

    def test_script_line_numbers_are_rebased_onto_the_sfc(
        self, laravel_app_path: Path
    ) -> None:
        """Script chunk lines point at the SFC, not at the script body."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        source = path.read_text()
        chunks, _ = chunk_file("resources/js/ReviewCard.vue", source)

        lines = source.split("\n")
        approve = next(c for c in chunks if c.symbol_name == "approve")
        assert lines[approve.start_line - 1].startswith("async function approve")
        assert lines[approve.end_line - 1] == "}"

    def test_template_and_style_are_separate_blocks(
        self, laravel_app_path: Path
    ) -> None:
        """The three SFC blocks do not bleed into each other."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        chunks, _ = chunk_file("resources/js/ReviewCard.vue", path.read_text())

        template = next(c for c in chunks if c.symbol_name == "template")
        style = next(c for c in chunks if c.symbol_name == "style")
        assert template.content.startswith("<template>")
        assert template.content.rstrip().endswith("</template>")
        assert "<script" not in template.content
        assert style.content.startswith("<style scoped>")
        assert "review-card__header" in style.content

    def test_oversized_template_splits_on_element_boundaries(self) -> None:
        """A big template descends into child elements, never mid-tag."""
        row = '      <li class="row"><span>value</span></li>\n'
        content = (
            "<template>\n  <ul>\n"
            + row * 60
            + "  </ul>\n  <footer>done</footer>\n</template>\n"
        )
        chunks, quality = chunk_file("Big.vue", content, max_chars=800)

        assert quality == SearchQuality.AST
        assert len(chunks) > 1
        for chunk in chunks:
            if chunk.parent_chunk_id is None:
                assert chunk.content.count("<li") == chunk.content.count("</li>")

    def test_vue_script_imports_become_refs(self, laravel_app_path: Path) -> None:
        """SFC imports are mined by the TypeScript ref extractor."""
        path = laravel_app_path / "resources/js/ReviewCard.vue"
        _, _, refs = chunk_file_with_refs(
            "resources/js/ReviewCard.vue", path.read_text()
        )

        targets = {r.target_symbol for r in refs}
        assert "computed" in targets
        assert "StarRating" in targets


class TestNodeTextByteOffsets:
    def test_non_ascii_python_does_not_shift_chunk_text(self) -> None:
        """tree-sitter reports byte offsets; Python strings index code points.

        Without decoding through UTF-8, every node after a multi-byte
        character is sliced at the wrong place.
        """
        code = (
            '"""Résumé du module, accents non ASCII."""\n'
            "\n"
            "class Café:\n"
            "    def première(self) -> str:\n"
            "        return 'naïve'\n"
        )
        chunks, quality = chunk_file("cafe.py", code)

        assert quality == SearchQuality.AST
        method = next(c for c in chunks if c.symbol_name == "Café.première")
        assert method.content.startswith("def première(self)")
        assert "return 'naïve'" in method.content

    def test_non_ascii_typescript_does_not_shift_chunk_text(self) -> None:
        """Same guard on the TS strategy, which slices the same way."""
        code = (
            "// Contrôleur, commentaire accentué\n"
            "export function première(): string {\n"
            "    return 'naïve'\n"
            "}\n"
        )
        chunks, quality = chunk_file("cafe.ts", code)

        assert quality == SearchQuality.AST
        fn = next(c for c in chunks if c.symbol_name == "première")
        assert fn.content.startswith("function première()")
        assert "return 'naïve'" in fn.content
        assert fn.content.rstrip().endswith("}")


class TestVueOptionsApi:
    """Vue 2 / Options API components, which are still the common shape.

    `export default { … }` used to unwrap to an unrecognised node type and
    arrive as one file-sized block, so every method in the component was
    only reachable through a line-window sub-chunk.
    """

    def test_methods_become_individual_chunks(self, laravel_app_path: Path) -> None:
        """Each entry under `methods:` is its own retrievable chunk."""
        path = laravel_app_path / "resources/js/StatsPanel.vue"
        chunks, quality = chunk_file("resources/js/StatsPanel.vue", path.read_text())

        assert quality == SearchQuality.AST
        by_name = {c.symbol_name: c for c in chunks}
        assert by_name["methods.fetchStats"].symbol_type == SymbolType.METHOD
        assert by_name["methods.formatValue"].symbol_type == SymbolType.METHOD
        assert by_name["computed.title"].symbol_type == SymbolType.METHOD
        # Lifecycle hooks sit directly on the object.
        assert by_name["data"].symbol_type == SymbolType.METHOD
        assert by_name["mounted"].symbol_type == SymbolType.METHOD

    def test_method_bodies_stay_whole(self, laravel_app_path: Path) -> None:
        """An Options API method is never cut across chunks."""
        path = laravel_app_path / "resources/js/StatsPanel.vue"
        chunks, _ = chunk_file("resources/js/StatsPanel.vue", path.read_text())

        fetch = next(c for c in chunks if c.symbol_name == "methods.fetchStats")
        assert fetch.parent_chunk_id is None
        assert _braces_balanced(fetch.content)
        assert "finally" in fetch.content

    def test_non_function_options_form_a_shell(self, laravel_app_path: Path) -> None:
        """`name` and `props` travel together as the component's shape."""
        path = laravel_app_path / "resources/js/StatsPanel.vue"
        chunks, _ = chunk_file("resources/js/StatsPanel.vue", path.read_text())

        shell = next(
            c
            for c in chunks
            if "name: 'stats-panel'" in c.content and not c.symbol_name
        )
        assert shell.symbol_type == SymbolType.BLOCK

    def test_empty_option_object_is_dropped(self) -> None:
        """`components: { }` carries nothing worth indexing."""
        code = "export default {\n  components: { },\n  mounted() { this.load() },\n}\n"
        chunks, _ = chunk_file("Widget.ts", code)

        assert not any(c.symbol_name == "components" for c in chunks)
        assert any(c.symbol_name == "mounted" for c in chunks)

    def test_plain_config_object_is_not_shredded(self) -> None:
        """An object with no functions in it stays whole."""
        code = "export default {\n  a: 1,\n  nested: { b: 2, c: 3 },\n}\n"
        chunks, _ = chunk_file("config.ts", code)

        assert len(chunks) == 1
        assert "nested" in chunks[0].content

    def test_shorthand_members_are_not_dropped(self, laravel_app_path: Path) -> None:
        """`components: { StatTile }` has no value node; keep it in the shell."""
        path = laravel_app_path / "resources/js/StatsPanel.vue"
        chunks, _ = chunk_file("resources/js/StatsPanel.vue", path.read_text())

        assert any("StatTile" in c.content for c in chunks)

    def test_spread_members_are_not_dropped(self) -> None:
        """A `...mapState()` spread is neither a method nor a nested object."""
        code = (
            "export default {\n"
            "  ...mapState(['shop']),\n"
            "  mounted() { this.load() },\n"
            "}\n"
        )
        chunks, _ = chunk_file("Widget.ts", code)

        assert any("mapState(['shop'])" in c.content for c in chunks)
        assert any(c.symbol_name == "mounted" for c in chunks)


class TestVueTemplateSpans:
    def test_trailing_markup_is_folded_into_its_chunk(self) -> None:
        """Descent must not leave a bare closing tag as its own chunk."""
        row = '      <li class="row"><span>value</span></li>\n'
        content = (
            "<template>\n  <ul>\n"
            + row * 60
            + "  </ul>\n  <footer>done</footer>\n</template>\n"
        )
        chunks, _ = chunk_file("Big.vue", content, max_chars=800)

        assert len(chunks) > 1
        assert all(len(c.content) >= 40 for c in chunks)
        # Nothing was dropped on the way down.
        assert sum(c.content.count("<li ") for c in chunks) == 60
        assert any("</ul>" in c.content for c in chunks)
        assert any("<footer>done</footer>" in c.content for c in chunks)
