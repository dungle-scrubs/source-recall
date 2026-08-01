"""Tree-sitter chunking for TS/Python/PHP/Vue, regex for Bash/Blade, text fallback."""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from source_recall.models import ChunkData, RefData, RefType, SearchQuality, SymbolType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# File extensions mapped to their language key.
_TS_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".jsx"})
_PY_EXTENSIONS = frozenset({".py"})
_PHP_EXTENSIONS = frozenset({".php", ".phtml"})
_VUE_EXTENSIONS = frozenset({".vue"})
# Compound extension - see ``_is_blade``.
_BLADE_SUFFIX = ".blade.php"
_BASH_EXTENSIONS = frozenset({".sh", ".bash"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".mdx", ".markdown"})
_TEXT_EXTENSIONS = frozenset({".txt", ".text", ".rst", ".log"})

# Sentence-ending punctuation followed by whitespace.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Overlap lines between consecutive sub-chunks.
_SUB_CHUNK_OVERLAP = 8

# Resource-exhaustion guards for PDF extraction. A hostile or degenerate
# PDF can carry an enormous page count or per-page text volume; bound both
# so a single file cannot exhaust memory during a build.
_PDF_MAX_PAGES = 5000
_PDF_MAX_CHARS = 20_000_000


def chunk_file(
    file_path: str,
    content: str,
    *,
    max_chars: int = 6000,
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk a source file into semantic units.

    @param file_path: Repo-relative path to the file.
    @param content: Full text of the file.
    @param max_chars: Character threshold for sub-chunking.
    @returns: Tuple of (chunks, search_quality used).
    """
    # Blade must be tested before the extension table: ``Path.suffix`` of
    # ``foo.blade.php`` is ``.php``, so a plain extension lookup would hand
    # Blade templates to the PHP grammar, which sees the whole template as
    # one opaque ``text`` node and produces a single file-sized chunk.
    if _is_blade(file_path):
        return _chunk_blade(file_path, content, max_chars)

    ext = _get_extension(file_path)

    if ext in _TS_EXTENSIONS:
        return _chunk_typescript(file_path, content, max_chars)
    if ext in _PY_EXTENSIONS:
        return _chunk_python(file_path, content, max_chars)
    if ext in _PHP_EXTENSIONS:
        return _chunk_php(file_path, content, max_chars)
    if ext in _VUE_EXTENSIONS:
        return _chunk_vue(file_path, content, max_chars)
    if ext in _BASH_EXTENSIONS:
        return _chunk_bash_regex(file_path, content, max_chars)
    if ext in _MARKDOWN_EXTENSIONS:
        return _chunk_markdown(file_path, content, max_chars)
    if ext in _TEXT_EXTENSIONS:
        return _chunk_prose(file_path, content, max_chars)
    return _chunk_text_fallback(file_path, content, max_chars)


def chunk_file_with_refs(
    file_path: str,
    content: str,
    *,
    max_chars: int = 6000,
) -> tuple[list[ChunkData], SearchQuality, list[RefData]]:
    """Chunk a file and extract cross-references.

    @param file_path: Repo-relative path to the file.
    @param content: Full text of the file.
    @param max_chars: Character threshold for sub-chunking.
    @returns: Tuple of (chunks, search_quality, refs).
    """
    chunks, quality = chunk_file(file_path, content, max_chars=max_chars)
    ext = _get_extension(file_path)

    refs: list[RefData] = []
    if _is_blade(file_path):
        refs = []
    elif ext in _PY_EXTENSIONS:
        refs = _extract_python_refs(chunks, content)
    elif ext in _TS_EXTENSIONS or ext in _VUE_EXTENSIONS:
        # A Vue SFC's imports live in its ``<script>`` block; the line index
        # built over the SFC's chunks maps them to the right script chunk.
        refs = _extract_ts_refs(chunks, content)
    elif ext in _PHP_EXTENSIONS:
        refs = _extract_php_refs(chunks, content)

    return chunks, quality, refs


# ---------------------------------------------------------------------------
# TypeScript / JavaScript chunking
# ---------------------------------------------------------------------------

# Node types that represent top-level chunk boundaries in TS/TSX.
_TS_TOP_LEVEL = frozenset(
    {
        "function_declaration",
        "export_statement",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
    }
)

# Node types that represent methods inside classes.
_TS_METHOD_TYPES = frozenset(
    {
        "method_definition",
        "public_field_definition",
    }
)


def _chunk_typescript(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk TypeScript/JavaScript via tree-sitter.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    ext = _get_extension(file_path)
    lang = "tsx" if ext in {".tsx", ".jsx"} else "typescript"
    return _chunk_ts_source(file_path, content, max_chars, lang)


def _chunk_ts_source(
    file_path: str, content: str, max_chars: int, lang: str
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk a TypeScript/JavaScript source string with an explicit grammar.

    Split out from ``_chunk_typescript`` so a Vue SFC's ``<script>`` body,
    which has no extension of its own, can go through the same strategy.

    @param file_path: Repo-relative path (used for symbol classification).
    @param content: Source text to chunk.
    @param max_chars: Sub-chunk threshold.
    @param lang: tree-sitter language key.
    @returns: (chunks, quality).
    """
    tree, quality = _parse_with_fallback(file_path, content, lang)
    if quality == SearchQuality.TEXT_FALLBACK:
        return _chunk_text_fallback(file_path, content, max_chars)

    root = tree.root_node
    chunks: list[ChunkData] = []

    for node in root.children:
        if not node.is_named:
            continue

        if node.type == "export_statement":
            # Unwrap: the interesting node is the child.
            inner = _ts_unwrap_export(node)
            if inner is not None:
                _ts_process_node(
                    inner, file_path, content, max_chars, chunks, exported=True
                )
            else:
                # Bare export (e.g. `export { foo }`) — treat as block.
                _add_chunk(
                    chunks,
                    file_path,
                    "",
                    SymbolType.BLOCK,
                    _node_text(node, content),
                    node.start_point[0] + 1,
                    node.end_point[0] + 1,
                    SearchQuality.AST,
                    max_chars,
                )
        elif node.type in _TS_TOP_LEVEL:
            _ts_process_node(
                node, file_path, content, max_chars, chunks, exported=False
            )
        # Skip non-named or unrecognized nodes.

    if not chunks:
        # No recognizable structure — file-level block.
        return _chunk_text_fallback(file_path, content, max_chars)

    return chunks, SearchQuality.AST


def _ts_unwrap_export(export_node: Node) -> Node | None:
    """Get the meaningful child of an export_statement.

    @param export_node: An export_statement node.
    @returns: The inner declaration/expression node, or None.
    """
    for child in export_node.named_children:
        if child.type not in {"comment", "decorator"}:
            return child
    return None


def _ts_process_node(
    node: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
    *,
    exported: bool,  # noqa: ARG001 — reserved for Phase 1c (cross-ref tracking)
) -> None:
    """Process a single top-level TS node into chunks.

    @param node: tree-sitter Node.
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator list.
    @param exported: Whether the node is exported (used in Phase 1c).
    """
    ntype = node.type

    if ntype == "class_declaration":
        _ts_chunk_class(node, file_path, content, max_chars, chunks)
    elif ntype == "function_declaration":
        name = _ts_get_name(node)
        sym_type = _ts_classify_function(node, name, file_path)
        _add_chunk(
            chunks,
            file_path,
            name,
            sym_type,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
    elif ntype == "lexical_declaration":
        _ts_chunk_lexical(node, file_path, content, max_chars, chunks)
    elif ntype == "interface_declaration":
        name = _ts_get_name(node)
        _add_chunk(
            chunks,
            file_path,
            name,
            SymbolType.INTERFACE,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
    elif ntype == "type_alias_declaration":
        name = _ts_get_name(node)
        _add_chunk(
            chunks,
            file_path,
            name,
            SymbolType.TYPE_ALIAS,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
    elif ntype == "enum_declaration":
        name = _ts_get_name(node)
        _add_chunk(
            chunks,
            file_path,
            name,
            SymbolType.ENUM,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
    else:
        # Generic fallback for unrecognized node types.
        _add_chunk(
            chunks,
            file_path,
            "",
            SymbolType.BLOCK,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )


def _ts_chunk_class(
    node: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Chunk a class into shell + individual methods.

    @param node: class_declaration node.
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    class_name = _ts_get_name(node)

    # Find the class body.
    body = node.child_by_field_name("body")
    if body is None:
        _add_chunk(
            chunks,
            file_path,
            class_name,
            SymbolType.CLASS,
            _node_text(node, content),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
        return

    # Build shell: everything except method bodies.
    shell_lines = _build_class_shell(node, body, content)
    if shell_lines:
        _add_chunk(
            chunks,
            file_path,
            class_name,
            SymbolType.CLASS_SHELL,
            shell_lines,
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )

    # Individual methods.
    for child in body.named_children:
        if child.type in _TS_METHOD_TYPES:
            method_name = _ts_get_name(child)
            full_name = f"{class_name}.{method_name}" if method_name else class_name
            _add_chunk(
                chunks,
                file_path,
                full_name,
                SymbolType.METHOD,
                _node_text(child, content),
                child.start_point[0] + 1,
                child.end_point[0] + 1,
                SearchQuality.AST,
                max_chars,
            )


def _ts_chunk_lexical(
    node: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Chunk a lexical_declaration (const/let/var).

    Detects arrow functions and React components.

    @param node: lexical_declaration node.
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    for child in node.named_children:
        if child.type == "variable_declarator":
            name = _ts_get_name(child)
            sym_type = _ts_classify_lexical(child, name, file_path)
            _add_chunk(
                chunks,
                file_path,
                name,
                sym_type,
                _node_text(node, content),
                node.start_point[0] + 1,
                node.end_point[0] + 1,
                SearchQuality.AST,
                max_chars,
            )
            return

    # Fallback.
    _add_chunk(
        chunks,
        file_path,
        "",
        SymbolType.BLOCK,
        _node_text(node, content),
        node.start_point[0] + 1,
        node.end_point[0] + 1,
        SearchQuality.AST,
        max_chars,
    )


def _ts_classify_function(node: Node, name: str, file_path: str) -> SymbolType:
    """Classify a function_declaration as function or component.

    Two-signal approach: PascalCase name + JSX anywhere in body.

    @param node: function_declaration node.
    @param name: Function name.
    @param file_path: File path (for extension check).
    @returns: SymbolType.
    """
    ext = _get_extension(file_path)
    if ext in {".tsx", ".jsx"} and _is_pascal_case(name) and _has_jsx(node):
        return SymbolType.COMPONENT
    return SymbolType.FUNCTION


def _ts_classify_lexical(node: Node, name: str, file_path: str) -> SymbolType:
    """Classify a variable_declarator.

    Detects React components via PascalCase + JSX, including
    React.memo, React.forwardRef, React.lazy wrappers.

    @param node: variable_declarator node.
    @param name: Variable name.
    @param file_path: File path.
    @returns: SymbolType.
    """
    ext = _get_extension(file_path)
    if ext not in {".tsx", ".jsx"}:
        # Check if it contains an arrow function.
        if _contains_node_type(node, "arrow_function"):
            return SymbolType.FUNCTION
        return SymbolType.BLOCK

    # TSX/JSX: check for component patterns.
    if _is_pascal_case(name):
        # Direct JSX in body.
        if _has_jsx(node):
            return SymbolType.COMPONENT
        # React.memo/forwardRef/lazy wrappers.
        if _has_react_wrapper(node):
            return SymbolType.COMPONENT

    if _contains_node_type(node, "arrow_function"):
        return SymbolType.FUNCTION
    return SymbolType.BLOCK


def _ts_get_name(node: Node) -> str:
    """Extract the name from a TS node.

    @param node: tree-sitter Node.
    @returns: Name string (empty if not found).
    """
    # Try 'name' field first (works for most declarations).
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return (name_node.text or b"").decode("utf-8", errors="replace")

    # For variable_declarator, name is the first identifier child.
    for child in node.children:
        if child.type == "identifier":
            return (child.text or b"").decode("utf-8", errors="replace")
        if child.type == "type_identifier":
            return (child.text or b"").decode("utf-8", errors="replace")
        if child.type == "property_identifier":
            return (child.text or b"").decode("utf-8", errors="replace")
    return ""


# ---------------------------------------------------------------------------
# Python chunking
# ---------------------------------------------------------------------------

_PY_TOP_LEVEL = frozenset(
    {
        "function_definition",
        "class_definition",
        "decorated_definition",
        "assignment",
        "expression_statement",
    }
)


def _chunk_python(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk Python via tree-sitter.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    tree, quality = _parse_with_fallback(file_path, content, "python")
    if quality == SearchQuality.TEXT_FALLBACK:
        return _chunk_text_fallback(file_path, content, max_chars)

    root = tree.root_node
    chunks: list[ChunkData] = []

    # Collect preamble: all top-level nodes before the first
    # function/class definition.  This ensures import statements and
    # module-level setup code get their own chunk, preventing
    # mis-attribution of refs to the first function (M-5 fix).
    first_def_line: int | None = None
    for node in root.children:
        if not node.is_named:
            continue
        ntype = node.type
        if ntype == "decorated_definition":
            inner = _py_unwrap_decorated(node)
            if inner is not None and inner.type in {
                "function_definition",
                "class_definition",
            }:
                first_def_line = node.start_point[0]
                break
        elif ntype in {"function_definition", "class_definition"}:
            first_def_line = node.start_point[0]
            break

    if first_def_line is not None and first_def_line > 0:
        preamble = "\n".join(content.split("\n")[:first_def_line]).strip()
        if preamble and len(preamble) > 20:
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.MODULE,
                preamble,
                1,
                first_def_line,
                SearchQuality.AST,
                max_chars,
            )

    for node in root.children:
        if not node.is_named:
            continue

        if node.type == "decorated_definition":
            inner = _py_unwrap_decorated(node)
            if inner is not None:
                _py_process_node(inner, node, file_path, content, max_chars, chunks)
            continue

        if node.type in _PY_TOP_LEVEL:
            _py_process_node(node, node, file_path, content, max_chars, chunks)

    if not chunks:
        return _chunk_text_fallback(file_path, content, max_chars)

    return chunks, SearchQuality.AST


def _py_unwrap_decorated(node: Node) -> Node | None:
    """Get the definition inside a decorated_definition.

    @param node: decorated_definition node.
    @returns: The inner function/class node, or None.
    """
    for child in node.named_children:
        if child.type in {"function_definition", "class_definition"}:
            return child
    return None


def _py_process_node(
    node: Node,
    outer: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Process a single top-level Python node.

    @param node: The definition node (may differ from outer if decorated).
    @param outer: The outermost node (includes decorators).
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    if node.type == "class_definition":
        _py_chunk_class(node, outer, file_path, content, max_chars, chunks)
    elif node.type == "function_definition":
        name = _py_get_name(node)
        _add_chunk(
            chunks,
            file_path,
            name,
            SymbolType.FUNCTION,
            _node_text(outer, content),
            outer.start_point[0] + 1,
            outer.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
    elif node.type in {"assignment", "expression_statement"}:
        # Module-level assignments.
        text = _node_text(node, content)
        if len(text) > 20:  # Skip trivial one-liners.
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.MODULE,
                text,
                node.start_point[0] + 1,
                node.end_point[0] + 1,
                SearchQuality.AST,
                max_chars,
            )


def _py_chunk_class(
    node: Node,
    outer: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Chunk a Python class into shell + methods.

    @param node: class_definition node.
    @param outer: Outermost node (includes decorators).
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    class_name = _py_get_name(node)

    body = node.child_by_field_name("body")
    if body is None:
        _add_chunk(
            chunks,
            file_path,
            class_name,
            SymbolType.CLASS,
            _node_text(outer, content),
            outer.start_point[0] + 1,
            outer.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
        return

    # Shell: class signature + docstring + class-level assignments.
    shell = _build_py_class_shell(node, body, content)
    if shell:
        _add_chunk(
            chunks,
            file_path,
            class_name,
            SymbolType.CLASS_SHELL,
            shell,
            outer.start_point[0] + 1,
            outer.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )

    # Methods.
    for child in body.named_children:
        actual = child
        if child.type == "decorated_definition":
            inner = _py_unwrap_decorated(child)
            if inner is None:
                continue
            actual = inner

        if actual.type == "function_definition":
            method_name = _py_get_name(actual)
            full_name = f"{class_name}.{method_name}"
            _add_chunk(
                chunks,
                file_path,
                full_name,
                SymbolType.METHOD,
                _node_text(child, content),  # Include decorators.
                child.start_point[0] + 1,
                child.end_point[0] + 1,
                SearchQuality.AST,
                max_chars,
            )


def _py_get_name(node: Node) -> str:
    """Extract name from a Python definition node.

    @param node: tree-sitter Node.
    @returns: Name string.
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return (name_node.text or b"").decode("utf-8", errors="replace")
    return ""


def _build_py_class_shell(node: Node, body: Node, content: str) -> str:
    """Build a class shell: signature + docstring + class vars.

    @param node: class_definition node.
    @param body: The block/body child.
    @param content: Full file content.
    @returns: Shell text.
    """
    lines = content.split("\n")
    # Class signature line(s).
    sig_start = node.start_point[0]
    body_start = body.start_point[0]
    shell_parts = lines[sig_start:body_start]

    # Add docstring if present.
    first_stmt = None
    for child in body.named_children:
        if child.is_named:
            first_stmt = child
            break

    if first_stmt is not None and first_stmt.type == "expression_statement":
        expr = first_stmt.named_children[0] if first_stmt.named_children else None
        if expr is not None and expr.type == "string":
            shell_parts.extend(
                lines[first_stmt.start_point[0] : first_stmt.end_point[0] + 1]
            )

    # Class-level assignments (not inside methods).
    for child in body.named_children:
        if child.type in {"assignment", "expression_statement"}:
            # Skip the docstring we already added.
            if child == first_stmt:
                continue
            shell_parts.extend(lines[child.start_point[0] : child.end_point[0] + 1])

    return "\n".join(shell_parts)


# ---------------------------------------------------------------------------
# PHP chunking
# ---------------------------------------------------------------------------

# Type declarations that get the shell + members treatment.
_PHP_TYPE_DECLARATIONS: dict[str, SymbolType] = {
    "class_declaration": SymbolType.CLASS,
    "interface_declaration": SymbolType.INTERFACE,
    "trait_declaration": SymbolType.TRAIT,
    "enum_declaration": SymbolType.ENUM,
}

# Members of a declaration_list that become their own chunk.  Everything
# else (properties, constants, `use` of traits, enum cases) stays in the
# shell so the class-level context travels with the signature.
_PHP_MEMBER_TYPES = frozenset({"method_declaration"})

# Top-level nodes that end the file preamble.
_PHP_TOP_LEVEL_DEFS = frozenset(_PHP_TYPE_DECLARATIONS) | {"function_definition"}

# Inline HTML the PHP grammar hands back verbatim in mixed template files.
_PHP_TEXT_TYPES = frozenset({"text", "text_interpolation"})


def _chunk_php(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk PHP via tree-sitter.

    Mirrors the Python strategy: a preamble chunk for the namespace and
    ``use`` imports, one chunk per top-level function, and one shell plus
    one chunk per method for each class/interface/trait/enum.  Method
    bodies are always emitted whole - a method is only ever split by the
    shared ``_add_chunk`` sub-chunker, never at an arbitrary boundary.

    Files that mix HTML and PHP parse fine: the grammar exposes the
    inline markup as ``text`` / ``text_interpolation`` nodes, which become
    their own block chunks alongside the real declarations.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    tree, quality = _parse_with_fallback(file_path, content, "php")
    if quality == SearchQuality.TEXT_FALLBACK:
        return _chunk_text_fallback(file_path, content, max_chars)

    root = tree.root_node
    chunks: list[ChunkData] = []

    # Preamble: `<?php`, declare(), namespace, and the `use` imports that
    # precede the first definition.  Keeping them in their own chunk stops
    # imports being mis-attributed to whichever symbol happens to be first.
    first_def_line = _php_first_definition_line(root)
    preamble_end = first_def_line or 0
    if preamble_end > 0:
        preamble = "\n".join(content.split("\n")[:preamble_end]).strip()
        if len(preamble) > 20:
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.MODULE,
                preamble,
                1,
                preamble_end,
                SearchQuality.AST,
                max_chars,
            )

    for node, doc in _php_iter_with_docblocks(root):
        # Nodes already covered verbatim by the preamble chunk.
        if node.start_point[0] < preamble_end:
            continue
        _php_process_node(node, doc, file_path, content, max_chars, chunks)

    if not chunks:
        return _chunk_text_fallback(file_path, content, max_chars)

    return chunks, SearchQuality.AST


def _php_first_definition_line(root: Node) -> int | None:
    """Find the 0-indexed line where the first PHP definition starts.

    Includes a leading docblock so the preamble stops before it.

    @param root: program node.
    @returns: 0-indexed line, or None if the file has no definitions.
    """
    pending_doc: Node | None = None
    for node in root.children:
        if not node.is_named:
            continue
        if node.type == "comment":
            pending_doc = node
            continue
        if node.type in _PHP_TOP_LEVEL_DEFS:
            start = pending_doc if pending_doc is not None else node
            return start.start_point[0]
        pending_doc = None
    return None


def _php_iter_with_docblocks(parent: Node) -> Iterator[tuple[Node, Node | None]]:
    """Yield each named child of ``parent`` with its preceding comment.

    PHP attaches documentation as a sibling ``comment`` node rather than
    nesting it (unlike a Python docstring or a decorated definition), so
    the docblock has to be re-associated here or it would land in the
    wrong chunk.

    @param parent: Node whose children to walk.
    @returns: Iterator of (node, docblock-or-None) pairs.
    """
    pending_doc: Node | None = None
    for node in parent.children:
        if not node.is_named:
            continue
        if node.type == "comment":
            pending_doc = node
            continue
        yield node, pending_doc
        pending_doc = None


def _php_span(node: Node, doc: Node | None) -> tuple[int, int]:
    """Return the (start_byte, 0-indexed start line) covering an optional doc.

    @param node: The declaration node.
    @param doc: Preceding docblock comment, if any.
    @returns: (start_byte, start_line).
    """
    if doc is None:
        return node.start_byte, node.start_point[0]
    return doc.start_byte, doc.start_point[0]


def _php_process_node(
    node: Node,
    doc: Node | None,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Process a single top-level PHP node into chunks.

    @param node: The top-level node.
    @param doc: Preceding docblock comment, if any.
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    start_byte, start_line = _php_span(node, doc)
    text = _byte_slice(content, start_byte, node.end_byte)
    end_line = node.end_point[0] + 1

    if node.type in _PHP_TYPE_DECLARATIONS:
        _php_chunk_type_declaration(node, doc, file_path, content, max_chars, chunks)
        return

    if node.type == "function_definition":
        _add_chunk(
            chunks,
            file_path,
            _php_get_name(node),
            SymbolType.FUNCTION,
            text,
            start_line + 1,
            end_line,
            SearchQuality.AST,
            max_chars,
        )
        return

    if node.type in _PHP_TEXT_TYPES or len(text.strip()) > 20:
        # Inline HTML and top-level statements in script-style / mixed
        # template files.  Trivial one-liners are dropped as noise.
        stripped = text.strip()
        if stripped:
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.BLOCK,
                stripped,
                start_line + 1,
                end_line,
                SearchQuality.AST,
                max_chars,
            )


def _php_chunk_type_declaration(
    node: Node,
    doc: Node | None,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Chunk a class/interface/trait/enum into a shell plus its methods.

    @param node: The declaration node.
    @param doc: Preceding docblock comment, if any.
    @param file_path: Repo-relative path.
    @param content: Full file content.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    type_name = _php_get_name(node)
    start_byte, start_line = _php_span(node, doc)
    body = _php_body(node)

    if body is None:
        _add_chunk(
            chunks,
            file_path,
            type_name,
            _PHP_TYPE_DECLARATIONS[node.type],
            _byte_slice(content, start_byte, node.end_byte),
            start_line + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
        return

    shell = _build_php_type_shell(body, content, start_line)
    if shell.strip():
        # A class shell is the established CLASS_SHELL type; interfaces,
        # traits and enums keep their own type so a symbol search can tell
        # them apart from the classes that use them.
        shell_type = (
            SymbolType.CLASS_SHELL
            if node.type == "class_declaration"
            else _PHP_TYPE_DECLARATIONS[node.type]
        )
        _add_chunk(
            chunks,
            file_path,
            type_name,
            shell_type,
            shell,
            start_line + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )

    for member, member_doc in _php_iter_with_docblocks(body):
        if member.type not in _PHP_MEMBER_TYPES:
            continue
        member_start_byte, member_start_line = _php_span(member, member_doc)
        method_name = _php_get_name(member)
        full_name = f"{type_name}.{method_name}" if method_name else type_name
        _add_chunk(
            chunks,
            file_path,
            full_name,
            SymbolType.METHOD,
            _byte_slice(content, member_start_byte, member.end_byte),
            member_start_line + 1,
            member.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )


def _php_body(node: Node) -> Node | None:
    """Find the member list of a PHP type declaration.

    Classes, interfaces and traits expose a ``declaration_list``; enums
    expose an ``enum_declaration_list``.  Neither is reachable through a
    consistent field name across the grammar's rules.

    @param node: The declaration node.
    @returns: The member-list node, or None.
    """
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    for child in node.children:
        if child.type.endswith("declaration_list"):
            return child
    return None


def _build_php_type_shell(body: Node, content: str, start_line: int) -> str:
    """Build a PHP type shell: docblock + signature + non-method members.

    @param body: The member-list node.
    @param content: Full file content.
    @param start_line: 0-indexed line where the declaration (or its
        docblock) starts.
    @returns: Shell text.
    """
    lines = content.split("\n")
    body_start = body.start_point[0]
    shell_parts = lines[start_line : body_start + 1]

    for child in body.named_children:
        if child.type in _PHP_MEMBER_TYPES:
            continue
        shell_parts.extend(lines[child.start_point[0] : child.end_point[0] + 1])

    shell_parts.append("}")
    return "\n".join(shell_parts)


def _php_get_name(node: Node) -> str:
    """Extract the name from a PHP declaration node.

    @param node: tree-sitter Node.
    @returns: Name string (empty if not found).
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return (name_node.text or b"").decode("utf-8", errors="replace")
    for child in node.children:
        if child.type == "name":
            return (child.text or b"").decode("utf-8", errors="replace")
    return ""


# ---------------------------------------------------------------------------
# Vue single-file-component chunking
# ---------------------------------------------------------------------------
#
# tree-sitter-language-pack 0.13.0 does ship a `vue` grammar, and it gives
# exactly the top-level split an SFC needs: `template_element`,
# `script_element` and `style_element`, with the script body exposed as an
# unparsed `raw_text` node.  That last part is what makes the delegation
# work: the script text is handed to the existing TypeScript strategy, so
# `<script setup>` produces the same function/interface/class chunks a `.ts`
# file would, and only the line numbers need rebasing onto the SFC.

_VUE_ELEMENT_TYPES = frozenset({"element", "template_element"})

# How deep to descend into an oversized template before handing the
# remainder to the shared line-window sub-chunker.
_VUE_MAX_DEPTH = 3


def _chunk_vue(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk a Vue SFC by top-level block.

    @param file_path: Repo-relative path.
    @param content: SFC text.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    tree, quality = _parse_with_fallback(file_path, content, "vue")
    if quality == SearchQuality.TEXT_FALLBACK:
        return _chunk_text_fallback(file_path, content, max_chars)

    chunks: list[ChunkData] = []

    for node in tree.root_node.children:
        if not node.is_named:
            continue
        if node.type == "script_element":
            _vue_chunk_script(node, file_path, content, max_chars, chunks)
        elif node.type == "template_element":
            _vue_emit_element(
                node, file_path, content, max_chars, chunks, "template", 0
            )
        elif node.type == "style_element":
            _vue_emit_element(node, file_path, content, max_chars, chunks, "style", 0)
        elif node.type != "comment":
            # Custom blocks (<i18n>, <docs>, …) and stray top-level markup.
            _vue_emit_element(node, file_path, content, max_chars, chunks, "", 0)

    if not chunks:
        return _chunk_text_fallback(file_path, content, max_chars)

    return chunks, SearchQuality.AST


def _vue_chunk_script(
    node: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
) -> None:
    """Chunk a ``<script>`` block with the TypeScript strategy.

    @param node: script_element node.
    @param file_path: Repo-relative path.
    @param content: Full SFC text.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    """
    raw = next((c for c in node.children if c.type == "raw_text"), None)

    if raw is not None:
        source = _node_text(raw, content)
        # `raw_text` starts just past the `>` of the opening tag, so its
        # 0-indexed row is the offset that rebases script-local line 1 onto
        # the SFC.
        sub_chunks, sub_quality = _chunk_ts_source(
            file_path, source, max_chars, "typescript"
        )
        if sub_quality == SearchQuality.AST and sub_chunks:
            chunks.extend(_rebase_chunks(sub_chunks, raw.start_point[0]))
            return

    # Empty, or too malformed for the TS grammar: keep the block whole.
    _vue_emit_element(node, file_path, content, max_chars, chunks, "script", 0)


def _rebase_chunks(chunks: list[ChunkData], line_offset: int) -> list[ChunkData]:
    """Shift chunk line numbers onto the enclosing file.

    @param chunks: Chunks whose lines are relative to an embedded block.
    @param line_offset: Lines preceding the embedded block.
    @returns: New chunks with absolute line numbers.
    """
    return [
        ChunkData(
            file_path=c.file_path,
            symbol_name=c.symbol_name,
            symbol_type=c.symbol_type,
            content=c.content,
            start_line=c.start_line + line_offset,
            end_line=c.end_line + line_offset,
            search_quality=c.search_quality,
            parent_chunk_id=c.parent_chunk_id,
            sub_chunk_index=c.sub_chunk_index,
        )
        for c in chunks
    ]


def _vue_emit_element(
    node: Node,
    file_path: str,
    content: str,
    max_chars: int,
    chunks: list[ChunkData],
    name: str,
    depth: int,
) -> None:
    """Emit a chunk for a markup element, descending if it is oversized.

    Descent splits at child-element boundaries, so a chunk never begins or
    ends in the middle of a tag.

    @param node: Element-ish node.
    @param file_path: Repo-relative path.
    @param content: Full SFC text.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    @param name: Symbol name carried down from the enclosing block.
    @param depth: Current recursion depth.
    """
    text = _node_text(node, content)
    children = [c for c in node.named_children if c.type in _VUE_ELEMENT_TYPES]

    if len(text) <= max_chars or depth >= _VUE_MAX_DEPTH or not children:
        _add_chunk(
            chunks,
            file_path,
            name,
            SymbolType.BLOCK,
            text.strip(),
            node.start_point[0] + 1,
            node.end_point[0] + 1,
            SearchQuality.AST,
            max_chars,
        )
        return

    cursor_byte = node.start_byte
    cursor_row = node.start_point[0]
    for child in children:
        _vue_emit_span(
            content,
            cursor_byte,
            child.start_byte,
            cursor_row,
            file_path,
            max_chars,
            chunks,
            name,
        )
        _vue_emit_element(child, file_path, content, max_chars, chunks, name, depth + 1)
        cursor_byte = child.end_byte
        cursor_row = child.end_point[0]

    _vue_emit_span(
        content,
        cursor_byte,
        node.end_byte,
        cursor_row,
        file_path,
        max_chars,
        chunks,
        name,
    )


def _vue_emit_span(
    content: str,
    start_byte: int,
    end_byte: int,
    start_row: int,
    file_path: str,
    max_chars: int,
    chunks: list[ChunkData],
    name: str,
) -> None:
    """Emit a chunk for the markup between two child elements.

    @param content: Full SFC text.
    @param start_byte: Span start.
    @param end_byte: Span end.
    @param start_row: 0-indexed row containing ``start_byte``.
    @param file_path: Repo-relative path.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    @param name: Symbol name from the enclosing block.
    """
    if end_byte <= start_byte:
        return
    raw = _byte_slice(content, start_byte, end_byte)
    text = raw.strip()
    if not text:
        return
    lead = raw[: len(raw) - len(raw.lstrip())]
    start_line = start_row + 1 + lead.count("\n")
    _add_chunk(
        chunks,
        file_path,
        name,
        SymbolType.BLOCK,
        text,
        start_line,
        start_line + text.count("\n"),
        SearchQuality.AST,
        max_chars,
    )


# ---------------------------------------------------------------------------
# Blade template chunking
# ---------------------------------------------------------------------------
#
# tree-sitter-language-pack 0.13.0 ships no Blade grammar (checked against
# its full 173-language list; it has `php`, `html`, `twig` and `vue`, but
# nothing for Blade).  Neither neighbouring grammar is usable as a stand-in:
#
#   * `php` "succeeds" on any Blade file: everything outside `<?php` tags
#     is a single opaque `text` node, so it reports zero parse errors while
#     yielding one file-sized chunk.  That is worse than a real fallback
#     because the error-density guard cannot detect it.
#   * `html` chokes on Blade's control flow: directives routinely open a tag
#     in one branch and close it in another.  Measured over the 82 Blade
#     templates in a production Laravel app, 22 (27%) exceeded the chunker's
#     10% error-node threshold, several above 95%.
#
# So Blade gets a structural directive scanner instead: Blade's own block
# directives (`@section`/`@endsection`, `@push`, `@component`, `@if`, …) are
# the boundaries an author already writes, and they nest properly.  The
# scanner tracks that nesting, so a `@section` is chunked whole no matter how
# much control flow it contains, and descends into a section only when the
# section itself exceeds `max_chars`.  Quality is REGEX, matching the Bash
# and Markdown strategies.

_BLADE_COMMENT_RE = re.compile(r"\{\{--.*?--\}\}", re.DOTALL)

# `(?<!@)` skips Blade's `@@` escape for a literal at-sign.
_BLADE_DIRECTIVE_RE = re.compile(r"(?<!@)@(\w+)")

# Opener directive (lowercased) -> the directives that close it.
_BLADE_BLOCK_CLOSERS: dict[str, frozenset[str]] = {
    "auth": frozenset({"endauth"}),
    "can": frozenset({"endcan"}),
    "canany": frozenset({"endcanany"}),
    "cannot": frozenset({"endcannot"}),
    "component": frozenset({"endcomponent"}),
    "empty": frozenset({"endempty"}),
    "env": frozenset({"endenv"}),
    "error": frozenset({"enderror"}),
    "for": frozenset({"endfor"}),
    "foreach": frozenset({"endforeach"}),
    "forelse": frozenset({"endforelse"}),
    "fragment": frozenset({"endfragment"}),
    "guest": frozenset({"endguest"}),
    "hassection": frozenset({"endif"}),
    "if": frozenset({"endif"}),
    "isset": frozenset({"endisset"}),
    "once": frozenset({"endonce"}),
    "php": frozenset({"endphp"}),
    "prepend": frozenset({"endprepend"}),
    "prependonce": frozenset({"endprependonce"}),
    "production": frozenset({"endproduction"}),
    "push": frozenset({"endpush"}),
    "pushonce": frozenset({"endpushonce"}),
    "section": frozenset({"endsection", "show", "stop", "append", "overwrite"}),
    "sectionmissing": frozenset({"endif"}),
    "session": frozenset({"endsession"}),
    "slot": frozenset({"endslot"}),
    "switch": frozenset({"endswitch"}),
    "unless": frozenset({"endunless"}),
    "verbatim": frozenset({"endverbatim"}),
    "while": frozenset({"endwhile"}),
}

# Directives whose first argument names the block, and which therefore make
# a good symbol_name for retrieval (`@section('content')` -> section:content).
_BLADE_NAMED_BLOCKS = frozenset(
    {
        "component",
        "fragment",
        "prepend",
        "prependonce",
        "push",
        "pushonce",
        "section",
        "slot",
    }
)

# Directives that take a *second* argument instead of a body, e.g.
# `@section('title', 'Reviews')` or `@php($x = 1)`.  A top-level comma in
# the argument list (or, for @php, any argument at all) means inline.
_BLADE_INLINE_WHEN_TWO_ARGS = frozenset({"section", "slot"})

# How deep to descend into an oversized block before handing the remainder
# to the shared line-window sub-chunker.
_BLADE_MAX_DEPTH = 3


@dataclass(slots=True)
class _BladeBlock:
    """One Blade block directive and everything it encloses.

    @param name: Lowercased directive name (``section``, ``if``, …).
    @param label: Retrieval symbol name, or "" for unnamed control flow.
    @param start: Character offset of the opening ``@``.
    @param body_start: Character offset just past the opening directive.
    @param end: Character offset just past the closing directive.
    @param children: Nested blocks, in source order.
    """

    name: str
    label: str
    start: int
    body_start: int
    end: int
    children: list[_BladeBlock]


def _chunk_blade(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk a Blade template on its block-directive boundaries.

    @param file_path: Repo-relative path.
    @param content: Template text.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    if not content.strip():
        return [], SearchQuality.REGEX

    blocks = _blade_parse_blocks(content)
    line_of = _blade_line_indexer(content)

    chunks: list[ChunkData] = []
    _blade_emit(
        blocks,
        content,
        0,
        len(content),
        file_path,
        max_chars,
        chunks,
        line_of,
        inherited_name="",
        depth=0,
    )

    if not chunks:
        return _chunk_text_fallback(file_path, content, max_chars)

    return chunks, SearchQuality.REGEX


def _blade_line_indexer(content: str) -> Any:
    """Build an O(log n) character-offset → 1-indexed line lookup.

    @param content: Full template text.
    @returns: Callable mapping offset to line number.
    """
    newlines = [i for i, ch in enumerate(content) if ch == "\n"]

    def _line_of(pos: int) -> int:
        return bisect.bisect_left(newlines, pos) + 1

    return _line_of


def _blade_parse_blocks(content: str) -> list[_BladeBlock]:
    """Parse a template into a tree of nested block directives.

    Unbalanced templates are tolerated: a closer with no matching opener is
    ignored, and blocks still open at EOF are closed there.

    @param content: Template text.
    @returns: Top-level blocks, in source order.
    """
    comment_spans = [(m.start(), m.end()) for m in _BLADE_COMMENT_RE.finditer(content)]
    comment_starts = [s for s, _ in comment_spans]

    def _in_comment(pos: int) -> bool:
        idx = bisect.bisect_right(comment_starts, pos) - 1
        if idx < 0:
            return False
        start, end = comment_spans[idx]
        return start <= pos < end

    roots: list[_BladeBlock] = []
    stack: list[_BladeBlock] = []
    pos = 0

    while True:
        match = _BLADE_DIRECTIVE_RE.search(content, pos)
        if match is None:
            break
        pos = match.end()

        if _in_comment(match.start()):
            continue

        raw_name = match.group(1)
        name = raw_name.lower()
        args, after = _blade_scan_args(content, match.end())

        # `@verbatim` suppresses Blade parsing entirely; its body may hold
        # literal directives that must not disturb the nesting.
        if name == "verbatim":
            close = content.find("@endverbatim", after)
            end = len(content) if close < 0 else close + len("@endverbatim")
            block = _BladeBlock("verbatim", "", match.start(), after, end, [])
            (stack[-1].children if stack else roots).append(block)
            pos = end
            continue

        closed = _blade_close(stack, name, after)
        if closed:
            continue

        if not _blade_opens_block(name, args):
            pos = after
            continue

        block = _BladeBlock(
            name,
            _blade_label(name, args),
            match.start(),
            after,
            len(content),
            [],
        )
        (stack[-1].children if stack else roots).append(block)
        stack.append(block)
        pos = after

    return roots


def _blade_close(stack: list[_BladeBlock], name: str, end: int) -> bool:
    """Close the innermost open block that ``name`` terminates.

    Blocks nested inside it are closed at the same offset, which recovers
    the outer structure of a template with a stray unclosed directive.

    @param stack: Open-block stack (mutated).
    @param name: Lowercased directive name.
    @param end: Character offset just past the closing directive.
    @returns: True if ``name`` closed a block.
    """
    for depth in range(len(stack) - 1, -1, -1):
        if name in _BLADE_BLOCK_CLOSERS.get(stack[depth].name, frozenset()):
            for block in stack[depth:]:
                block.end = end
            del stack[depth:]
            return True
    return False


def _blade_opens_block(name: str, args: str | None) -> bool:
    """Decide whether a directive opens a body-bearing block.

    @param name: Lowercased directive name.
    @param args: Argument text, or None when the directive took no parens.
    @returns: True if the directive opens a block.
    """
    if name not in _BLADE_BLOCK_CLOSERS:
        return False
    if name == "php":
        # `@php($x = 1)` is a one-liner; only the bare form takes a body.
        return args is None
    if name == "empty":
        # Bare `@empty` is the `@forelse` separator, not `@empty($x)`.
        return args is not None
    if name in _BLADE_INLINE_WHEN_TWO_ARGS and args is not None:
        return not _blade_has_top_level_comma(args)
    return True


_BLADE_FIRST_ARG_RE = re.compile(r"^\s*['\"]([^'\"]*)['\"]")


def _blade_label(name: str, args: str | None) -> str:
    """Build the retrieval symbol name for a block directive.

    Control-flow blocks deliberately get no name: an index full of chunks
    called "if" would pollute symbol lookup.

    @param name: Lowercased directive name.
    @param args: Argument text, or None.
    @returns: Symbol name, or "" for unnamed blocks.
    """
    if name not in _BLADE_NAMED_BLOCKS or not args:
        return ""
    match = _BLADE_FIRST_ARG_RE.match(args)
    if match is None:
        return ""
    return f"{name}:{match.group(1)}"


def _blade_scan_args(content: str, pos: int) -> tuple[str | None, int]:
    """Read a directive's parenthesised argument list, if present.

    @param content: Template text.
    @param pos: Offset just past the directive name.
    @returns: (argument text or None, offset just past the directive).
    """
    i = pos
    while i < len(content) and content[i] in " \t":
        i += 1
    if i >= len(content) or content[i] != "(":
        return None, pos

    depth = 0
    quote: str | None = None
    j = i
    while j < len(content):
        ch = content[j]
        if quote is not None:
            if ch == "\\":
                j += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return content[i + 1 : j], j + 1
        j += 1

    # Unterminated argument list - treat the directive as argument-less.
    return None, pos


def _blade_has_top_level_comma(args: str) -> bool:
    """Check for a comma outside any nested quotes/brackets.

    @param args: Argument text.
    @returns: True if the directive received more than one argument.
    """
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(args):
        ch = args[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
        i += 1
    return False


def _blade_emit(
    blocks: list[_BladeBlock],
    content: str,
    lo: int,
    hi: int,
    file_path: str,
    max_chars: int,
    chunks: list[ChunkData],
    line_of: Any,
    *,
    inherited_name: str,
    depth: int,
) -> None:
    """Emit chunks for the span ``[lo, hi)``, splitting at ``blocks``.

    @param blocks: Blocks contained directly in this span.
    @param content: Template text.
    @param lo: Span start offset.
    @param hi: Span end offset.
    @param file_path: Repo-relative path.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    @param line_of: Offset → line lookup.
    @param inherited_name: Symbol name carried down from an enclosing
        named block, so markup inside `@section('content')` is still
        attributed to that section.
    @param depth: Current recursion depth.
    """
    cursor = lo
    for block in blocks:
        _blade_emit_markup(
            content,
            cursor,
            block.start,
            file_path,
            max_chars,
            chunks,
            line_of,
            inherited_name,
        )
        cursor = block.end

        text = content[block.start : block.end]
        name = block.label or inherited_name
        if len(text) <= max_chars or depth >= _BLADE_MAX_DEPTH or not block.children:
            _add_chunk(
                chunks,
                file_path,
                name,
                SymbolType.MODULE if block.label else SymbolType.BLOCK,
                text.strip(),
                line_of(block.start),
                line_of(block.end),
                SearchQuality.REGEX,
                max_chars,
            )
            continue

        # Oversized block with structure inside it: descend.  The opening
        # and closing directives fall into the surrounding markup runs, so
        # no source text is dropped.
        _blade_emit(
            block.children,
            content,
            block.start,
            block.end,
            file_path,
            max_chars,
            chunks,
            line_of,
            inherited_name=name,
            depth=depth + 1,
        )

    _blade_emit_markup(
        content, cursor, hi, file_path, max_chars, chunks, line_of, inherited_name
    )


def _blade_emit_markup(
    content: str,
    lo: int,
    hi: int,
    file_path: str,
    max_chars: int,
    chunks: list[ChunkData],
    line_of: Any,
    inherited_name: str,
) -> None:
    """Emit a chunk for a run of markup between block directives.

    @param content: Template text.
    @param lo: Span start offset.
    @param hi: Span end offset.
    @param file_path: Repo-relative path.
    @param max_chars: Sub-chunk threshold.
    @param chunks: Accumulator.
    @param line_of: Offset → line lookup.
    @param inherited_name: Symbol name from the enclosing named block.
    """
    if hi <= lo:
        return
    text = content[lo:hi].strip()
    if not text:
        return
    offset = lo + (len(content[lo:hi]) - len(content[lo:hi].lstrip()))
    _add_chunk(
        chunks,
        file_path,
        inherited_name,
        SymbolType.BLOCK,
        text,
        line_of(offset),
        line_of(offset) + text.count("\n"),
        SearchQuality.REGEX,
        max_chars,
    )


# ---------------------------------------------------------------------------
# Bash regex chunking
# ---------------------------------------------------------------------------

_BASH_FUNC_RE = re.compile(r"^(?:function\s+(\w+)|(\w+)\s*\(\s*\))", re.MULTILINE)


def _chunk_bash_regex(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Chunk Bash files using regex function detection.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    lines = content.split("\n")
    boundaries: list[tuple[str, int]] = []  # (name, line_idx)

    for match in _BASH_FUNC_RE.finditer(content):
        name = match.group(1) or match.group(2)
        line_idx = content[: match.start()].count("\n")
        boundaries.append((name, line_idx))

    if not boundaries:
        return _chunk_text_fallback(file_path, content, max_chars)

    chunks: list[ChunkData] = []

    # Content before first function.
    if boundaries[0][1] > 0:
        preamble = "\n".join(lines[: boundaries[0][1]])
        if preamble.strip():
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.MODULE,
                preamble,
                1,
                boundaries[0][1],
                SearchQuality.REGEX,
                max_chars,
            )

    # Each function.
    for i, (name, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(lines)
        text = "\n".join(lines[start:end]).rstrip()
        if text:
            _add_chunk(
                chunks,
                file_path,
                name,
                SymbolType.FUNCTION,
                text,
                start + 1,
                end,
                SearchQuality.REGEX,
                max_chars,
            )

    return chunks, SearchQuality.REGEX


# ---------------------------------------------------------------------------
# Text fallback
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PDF chunking
# ---------------------------------------------------------------------------


def chunk_pdf(file_path: str, pdf_path: Path) -> tuple[list[ChunkData], SearchQuality]:
    """Extract text from a PDF and produce one chunk per non-empty page.

    @param file_path: Repo-relative path for chunk metadata.
    @param pdf_path: Absolute path to the PDF file on disk.
    @returns: (chunks, quality).
    """
    import fitz

    chunks: list[ChunkData] = []
    doc = fitz.open(str(pdf_path))

    try:
        page_limit = min(len(doc), _PDF_MAX_PAGES)
        if len(doc) > _PDF_MAX_PAGES:
            logger.warning(
                "PDF %s has %d pages; capping extraction at %d",
                file_path,
                len(doc),
                _PDF_MAX_PAGES,
            )

        total_chars = 0
        for page_num in range(page_limit):
            remaining = _PDF_MAX_CHARS - total_chars
            if remaining <= 0:
                logger.warning(
                    "PDF %s exceeded the %d-char extraction cap at page %d; "
                    "remaining pages skipped",
                    file_path,
                    _PDF_MAX_CHARS,
                    page_num + 1,
                )
                break

            page = doc[page_num]
            text = page.get_text().strip()
            if not text:
                continue

            # Truncate a single oversized page so no chunk — and the total
            # extracted text — can exceed the cap, even for a PDF with one
            # enormous text stream.
            if len(text) > remaining:
                text = text[:remaining]
                logger.warning(
                    "PDF %s page %d truncated to the %d-char extraction cap",
                    file_path,
                    page_num + 1,
                    _PDF_MAX_CHARS,
                )

            total_chars += len(text)
            chunks.append(
                ChunkData(
                    file_path=file_path,
                    symbol_name=f"Page {page_num + 1}",
                    symbol_type=SymbolType.MODULE,
                    content=text,
                    start_line=page_num + 1,
                    end_line=page_num + 1,
                    search_quality=SearchQuality.TEXT_FALLBACK,
                )
            )
    finally:
        doc.close()

    return chunks, SearchQuality.TEXT_FALLBACK


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

# Matches ATX headings: # Foo, ## Bar, etc.  Skips lines inside fenced blocks.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)


def _chunk_markdown(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Split markdown on ATX headings, respecting fenced code blocks.

    Each heading starts a new chunk.  The heading text becomes the
    symbol_name.  Content before the first heading is its own chunk.

    @param file_path: Repo-relative path.
    @param content: Markdown text.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    # Find headings that are NOT inside fenced code blocks.
    # Pair fences by matching opener/closer marker type and length
    # (``` only closes ```, not ~~~), with proper state tracking.
    fenced_ranges: list[tuple[int, int]] = []
    open_fence: tuple[str, int, int] | None = (
        None  # (marker_char, start_pos, marker_len)
    )
    for m in _FENCE_RE.finditer(content):
        marker = m.group(1)
        marker_char = marker[0]  # '`' or '~'
        if open_fence is None:
            # Opening a new fence.
            open_fence = (marker_char, m.start(), len(marker))
        elif marker_char == open_fence[0] and len(marker) >= open_fence[2]:
            # Matching closer — same character type, at least as long.
            fenced_ranges.append((open_fence[1], m.end()))
            open_fence = None
        # Mismatched closer (e.g. ~~~ inside ```, or shorter fence) — ignore.

    # Unclosed fence — treat remaining content as fenced (M8).
    if open_fence is not None:
        fenced_ranges.append((open_fence[1], len(content)))

    # Build sorted start positions for O(log n) bisect lookup.
    _fence_starts = [s for s, _e in fenced_ranges]

    def _in_fence(pos: int) -> bool:
        idx = bisect.bisect_right(_fence_starts, pos) - 1
        if idx < 0:
            return False
        s, e = fenced_ranges[idx]
        return s <= pos <= e

    # Collect heading positions.
    sections: list[tuple[str, int]] = []  # (heading_text, char_offset)
    for m in _HEADING_RE.finditer(content):
        if not _in_fence(m.start()):
            sections.append((m.group(2).strip(), m.start()))

    chunks: list[ChunkData] = []

    # Content before first heading.
    # Markdown is heading-split, not AST-parsed. Use REGEX quality
    # so the querier applies the correct scoring multiplier.
    _md_quality = SearchQuality.REGEX

    if sections:
        preamble = content[: sections[0][1]].strip()
        if preamble:
            line_count = preamble.count("\n") + 1
            _add_chunk(
                chunks,
                file_path,
                "",
                SymbolType.BLOCK,
                preamble,
                1,
                line_count,
                _md_quality,
                max_chars,
            )

    # Each section: from this heading to the next (or EOF).
    for i, (heading, start) in enumerate(sections):
        end = sections[i + 1][1] if i + 1 < len(sections) else len(content)
        body = content[start:end].strip()
        if not body:
            continue

        start_line = content[:start].count("\n") + 1
        end_line = start_line + body.count("\n")
        _add_chunk(
            chunks,
            file_path,
            heading,
            SymbolType.MODULE,
            body,
            start_line,
            end_line,
            _md_quality,
            max_chars,
        )

    # File with no headings at all → single chunk.
    if not sections and content.strip():
        _add_chunk(
            chunks,
            file_path,
            "",
            SymbolType.BLOCK,
            content.strip(),
            1,
            content.count("\n") + 1,
            _md_quality,
            max_chars,
        )

    return chunks, _md_quality


# ---------------------------------------------------------------------------
# Prose chunking (sentence-boundary sliding window)
# ---------------------------------------------------------------------------


def _chunk_prose(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Split plain text on sentence boundaries within max_chars.

    Accumulates sentences until adding the next would exceed max_chars,
    then emits a chunk and starts a new one.  Line numbers are derived
    from each sentence's position in the original content.

    @param file_path: Repo-relative path.
    @param content: Plain text content.
    @param max_chars: Target max characters per chunk.
    @returns: (chunks, quality).
    """
    if not content.strip():
        return [], SearchQuality.TEXT_FALLBACK

    # Build (sentence, start_line) pairs using split positions
    # directly (no re-searching the content for substrings).
    sentence_infos: list[tuple[str, int]] = []  # (text, start_line)

    # Use finditer to get exact boundary positions, then slice between them.
    boundaries = [m.start() for m in _SENTENCE_END_RE.finditer(content)]
    boundaries.append(len(content))

    prev = 0
    for boundary in boundaries:
        raw = content[prev:boundary]
        stripped = raw.strip()
        if stripped:
            start_line = content[:prev].count("\n") + 1
            sentence_infos.append((stripped, start_line))
        prev = boundary

    chunks: list[ChunkData] = []
    current: list[str] = []
    current_len = 0
    chunk_start_line = sentence_infos[0][1] if sentence_infos else 1
    chunk_end_line = chunk_start_line

    for sentence, start_line in sentence_infos:
        added_len = len(sentence) + (1 if current else 0)

        if current and current_len + added_len > max_chars:
            # Emit current chunk.
            text = " ".join(current)
            chunks.append(
                ChunkData(
                    file_path=file_path,
                    symbol_name="",
                    symbol_type=SymbolType.BLOCK,
                    content=text,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    search_quality=SearchQuality.TEXT_FALLBACK,
                )
            )
            current = []
            current_len = 0
            chunk_start_line = start_line

        current.append(sentence)
        current_len += added_len
        # End line is the last line of the current sentence.
        chunk_end_line = start_line + sentence.count("\n")

    # Emit remaining.
    if current:
        text = " ".join(current)
        chunks.append(
            ChunkData(
                file_path=file_path,
                symbol_name="",
                symbol_type=SymbolType.BLOCK,
                content=text,
                start_line=chunk_start_line,
                end_line=chunk_end_line,
                search_quality=SearchQuality.TEXT_FALLBACK,
            )
        )

    return chunks, SearchQuality.TEXT_FALLBACK


def _chunk_text_fallback(
    file_path: str, content: str, max_chars: int
) -> tuple[list[ChunkData], SearchQuality]:
    """Split by blank lines for unsupported languages.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param max_chars: Sub-chunk threshold.
    @returns: (chunks, quality).
    """
    # Split on blank lines, keeping the separators so we can count
    # their actual newlines instead of assuming +2 (M4).
    boundary_re = re.compile(r"\n\s*\n")
    matches = list(boundary_re.finditer(content))
    block_starts = [0] + [m.end() for m in matches]
    block_ends = [m.start() for m in matches] + [len(content)]

    chunks: list[ChunkData] = []

    for start_pos, end_pos in zip(block_starts, block_ends, strict=True):
        text = content[start_pos:end_pos].strip()
        if not text:
            continue

        start_line = content[:start_pos].count("\n") + 1
        line_count = text.count("\n") + 1
        _add_chunk(
            chunks,
            file_path,
            "",
            SymbolType.BLOCK,
            text,
            start_line,
            start_line + line_count - 1,
            SearchQuality.TEXT_FALLBACK,
            max_chars,
        )

    if not chunks and content.strip():
        _add_chunk(
            chunks,
            file_path,
            "",
            SymbolType.BLOCK,
            content.strip(),
            1,
            content.count("\n") + 1,
            SearchQuality.TEXT_FALLBACK,
            max_chars,
        )

    return chunks, SearchQuality.TEXT_FALLBACK


# ---------------------------------------------------------------------------
# Sub-chunking
# ---------------------------------------------------------------------------


def _add_chunk(
    chunks: list[ChunkData],
    file_path: str,
    symbol_name: str,
    symbol_type: SymbolType,
    content: str,
    start_line: int,
    end_line: int,
    quality: SearchQuality,
    max_chars: int,
) -> None:
    """Add a chunk, sub-chunking if it exceeds max_chars.

    @param chunks: Accumulator list.
    @param file_path: Repo-relative path.
    @param symbol_name: Symbol name.
    @param symbol_type: Symbol type.
    @param content: Chunk text.
    @param start_line: First line (1-indexed).
    @param end_line: Last line (1-indexed).
    @param quality: Search quality.
    @param max_chars: Character threshold.
    """
    if len(content) <= max_chars:
        chunks.append(
            ChunkData(
                file_path=file_path,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                content=content,
                start_line=start_line,
                end_line=end_line,
                search_quality=quality,
            )
        )
        return

    # Sub-chunk with overlap.
    sub_chunks = _split_into_sub_chunks(content, max_chars, symbol_name)

    # Compute the parent chunk ID (using full content).
    parent = ChunkData(
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        content=content,
        start_line=start_line,
        end_line=end_line,
        search_quality=quality,
    )
    parent_id = parent.chunk_id

    for i, (sub_text, sub_start_offset, sub_end_offset) in enumerate(sub_chunks):
        chunks.append(
            ChunkData(
                file_path=file_path,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                content=sub_text,
                start_line=start_line + sub_start_offset,
                end_line=start_line + sub_end_offset,
                search_quality=quality,
                parent_chunk_id=parent_id,
                sub_chunk_index=i,
            )
        )


def _split_into_sub_chunks(
    content: str,
    max_chars: int,
    _symbol_name: str,  # reserved for future signature-aware splitting
) -> list[tuple[str, int, int]]:
    """Split content into sub-chunks with overlap.

    Prepends the function/class signature to each sub-chunk after the
    first.  Overlap rewind never crosses back into the signature area
    to avoid duplicating signature content.

    @param content: Full chunk text.
    @param max_chars: Character limit per sub-chunk.
    @param symbol_name: Used for signature extraction.
    @returns: List of (text, start_line_offset, end_line_offset).
    """
    lines = content.split("\n")

    # Extract signature (first line or first few lines ending with : or {).
    signature = _extract_signature(lines)
    sig_line_count = signature.count("\n") + 1
    sig_len = len(signature) + 1  # +1 for newline.

    effective_max = max_chars - sig_len
    if effective_max < 500:
        effective_max = 500

    result: list[tuple[str, int, int]] = []
    i = 0
    chunk_idx = 0

    while i < len(lines):
        # Collect lines up to effective_max characters.
        chunk_lines: list[str] = []
        char_count = 0

        start_i = i
        while i < len(lines) and char_count + len(lines[i]) + 1 <= effective_max:
            chunk_lines.append(lines[i])
            char_count += len(lines[i]) + 1
            i += 1

        # If we couldn't fit even one line, force-include it.
        if not chunk_lines and i < len(lines):
            chunk_lines.append(lines[i])
            i += 1

        # Build sub-chunk text.
        if chunk_idx == 0:
            # First sub-chunk includes the original signature.
            text = "\n".join(chunk_lines)
        else:
            text = signature + "\n\n" + "\n".join(chunk_lines)

        end_i = i - 1
        result.append((text, start_i, end_i))
        chunk_idx += 1

        # Apply overlap — rewind by _SUB_CHUNK_OVERLAP lines, but never
        # back into the signature area (which is prepended separately).
        #
        # The clamp must only move the cursor BACKWARD.  Capping the
        # signature guard at the lines actually consumed keeps it from
        # exceeding ``i``; otherwise a signature spanning more lines than
        # the first sub-chunk could jump the cursor forward, silently
        # dropping (or, when it lands past ``len(lines)``, never emitting)
        # the intervening source lines.
        if i < len(lines):
            sig_guard = min(sig_line_count, i)
            lower_bound = max(start_i + 1, sig_guard)
            i = max(lower_bound, min(i, i - _SUB_CHUNK_OVERLAP))

    return result


def _extract_signature(lines: list[str]) -> str:
    """Extract the function/class signature from the first lines.

    Uses parenthesis-depth tracking to distinguish parameter-list
    continuations from body statements.  A line like ``x = {`` inside
    a function body is never included; only lines that are part of the
    declaration (open parens, type annotations, arrow functions) are
    collected.

    @param lines: All lines of the chunk.
    @returns: Signature string (may be multi-line).
    """
    if not lines:
        return ""

    sig_lines: list[str] = [lines[0]]
    first = lines[0].rstrip()

    # Track parenthesis depth to detect parameter lists.
    paren_depth = first.count("(") - first.count(")")

    # If the first line already completes the signature (balanced parens
    # and ends with a body opener), there's nothing more to collect.
    if paren_depth <= 0 and first.endswith((":", "{")):
        return "\n".join(sig_lines)

    for line in lines[1:5]:
        stripped = line.rstrip()
        if not stripped:
            break
        lstripped = stripped.lstrip()
        # Stop at docstrings.
        if lstripped.startswith(('"""', "'''", 'r"""', "r'''")):
            break

        paren_depth += stripped.count("(") - stripped.count(")")

        if paren_depth > 0:
            # Inside parameter list — always include.
            sig_lines.append(line)
        elif stripped.endswith((":", "{", "=>")):
            # Closing line of signature (e.g. `) -> bool:`).
            sig_lines.append(line)
            break
        elif stripped.endswith(")") or stripped.endswith("->"):
            sig_lines.append(line)
            break
        else:
            break

    return "\n".join(sig_lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_parser_cache: dict[str, Any] = {}


def _get_cached_parser(language: str) -> Any:
    """Return a cached tree-sitter parser for the given language (M5).

    Avoids per-file parser allocation overhead on large repos.

    @param language: tree-sitter language key.
    @returns: Parser instance (reused across calls).
    """
    cached = _parser_cache.get(language)
    if cached is not None:
        return cached

    from tree_sitter_language_pack import get_parser

    parser = get_parser(language)
    _parser_cache[language] = parser
    return parser


def _parse_with_fallback(
    _file_path: str, content: str, language: str
) -> tuple[Any, SearchQuality]:
    """Parse with tree-sitter, falling back on high error density.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param language: tree-sitter language key.
    @returns: (tree, quality). Tree may be None if fallback.
    """
    parser = _get_cached_parser(language)
    tree = parser.parse(content.encode("utf-8"))

    # Check error node density.
    total, errors = _count_nodes(tree.root_node)
    if total > 0 and errors / total > 0.10:
        return tree, SearchQuality.TEXT_FALLBACK

    return tree, SearchQuality.AST


def _count_nodes(node: Node) -> tuple[int, int]:
    """Count total and error nodes in a tree.

    @param node: Root node.
    @returns: (total_named, error_count).
    """
    total = 0
    errors = 0
    cursor = node.walk()

    reached_root = False
    while not reached_root:
        if cursor.node.is_named:
            total += 1
            if cursor.node.is_error or cursor.node.type == "ERROR":
                errors += 1

        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracing = False

    return total, errors


def _byte_slice(content: str, start_byte: int, end_byte: int) -> str:
    """Slice ``content`` by UTF-8 byte offsets.

    tree-sitter reports byte offsets; Python strings index by code point.
    The two agree only for pure-ASCII input, so anything else has to go
    through the encoded form.

    @param content: Full file content.
    @param start_byte: Start offset in UTF-8 bytes.
    @param end_byte: End offset in UTF-8 bytes.
    @returns: The corresponding text.
    """
    if content.isascii():
        return content[start_byte:end_byte]
    return _encoded(content)[start_byte:end_byte].decode("utf-8", errors="replace")


# One-entry cache: chunking a file makes many slices against the same
# string object, and re-encoding per node would be quadratic.  Holding the
# string keeps it alive, so an identity hit can never be a stale match.
_encoded_cache: tuple[str, bytes] | None = None


def _encoded(content: str) -> bytes:
    """Return ``content`` UTF-8 encoded, reusing the last encoding.

    @param content: Full file content.
    @returns: UTF-8 bytes.
    """
    global _encoded_cache
    cached = _encoded_cache
    if cached is not None and cached[0] is content:
        return cached[1]
    data = content.encode("utf-8")
    _encoded_cache = (content, data)
    return data


def _node_text(node: Node, content: str) -> str:
    """Extract the text of a node from the source content.

    @param node: tree-sitter Node.
    @param content: Full file content.
    @returns: Text slice corresponding to the node.
    """
    return _byte_slice(content, node.start_byte, node.end_byte)


def _build_class_shell(node: Node, body: Node, content: str) -> str:
    """Build a TS class shell (signature + property declarations).

    @param node: class_declaration node.
    @param body: class_body node.
    @param content: Full file content.
    """
    lines = content.split("\n")
    sig_start = node.start_point[0]
    body_start = body.start_point[0]

    # Class signature.
    shell_parts = lines[sig_start : body_start + 1]  # Include opening brace.

    # Property declarations (not methods).
    for child in body.named_children:
        if child.type not in _TS_METHOD_TYPES:
            shell_parts.extend(lines[child.start_point[0] : child.end_point[0] + 1])

    shell_parts.append("}")
    return "\n".join(shell_parts)


def _has_jsx(node: Node) -> bool:
    """Check if a node contains any JSX elements (recursively).

    @param node: tree-sitter Node.
    @returns: True if JSX found.
    """
    jsx_types = {"jsx_element", "jsx_self_closing_element", "jsx_fragment"}
    return _contains_any_node_type(node, jsx_types)


_REACT_WRAPPER_RE = re.compile(r"(?:^|\.)(memo|forwardRef|lazy)$")


def _iter_nodes(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every descendant via an iterative cursor walk.

    Mirrors ``_count_nodes`` so deeply-nested files cannot raise
    ``RecursionError`` (a resource-exhaustion surface on hostile input).

    @param node: tree-sitter Node to start from.
    @returns: Iterator over the node and all its descendants.
    """
    cursor = node.walk()
    reached_root = False
    while not reached_root:
        yield cursor.node

        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracing = False


def _has_react_wrapper(node: Node) -> bool:
    """Check if a node contains React.memo/forwardRef/lazy calls.

    Uses word-boundary matching to avoid false positives on names
    like 'memoize' or 'lazyLoad'.

    @param node: tree-sitter Node.
    @returns: True if a React wrapper call is found.
    """
    for descendant in _iter_nodes(node):
        if descendant.type != "call_expression":
            continue
        func = descendant.child_by_field_name("function")
        if func is not None:
            text = (func.text or b"").decode("utf-8", errors="replace")
            if _REACT_WRAPPER_RE.search(text):
                return True
    return False


def _contains_node_type(node: Node, node_type: str) -> bool:
    """Check if a node or any descendant has the given type.

    @param node: tree-sitter Node.
    @param node_type: Type string to look for.
    @returns: True if found.
    """
    return any(n.type == node_type for n in _iter_nodes(node))


def _contains_any_node_type(node: Node, node_types: set[str]) -> bool:
    """Check if a node or any descendant has any of the given types.

    @param node: tree-sitter Node.
    @param node_types: Set of type strings.
    @returns: True if any found.
    """
    return any(n.type in node_types for n in _iter_nodes(node))


def _is_pascal_case(name: str) -> bool:
    """Check if a name is PascalCase (starts with uppercase, has lowercase).

    @param name: Identifier name.
    @returns: True if PascalCase.
    """
    if not name or not name[0].isupper():
        return False
    # Must have at least one lowercase letter (not ALL_CAPS).
    return any(c.islower() for c in name)


def _get_extension(file_path: str) -> str:
    """Get the lowercase file extension.

    Uses ``Path.suffix`` so only the basename's extension is considered,
    not dots in parent directory names (M-4 fix).

    @param file_path: File path.
    @returns: Extension including dot (e.g. '.py'), or '' if none.
    """
    return Path(file_path).suffix.lower()


def _is_blade(file_path: str) -> bool:
    """Check whether a path is a Blade template.

    ``Path.suffix`` cannot express a compound extension, so Blade has to
    be matched on the full basename.  Without this, ``.blade.php`` would
    collide with plain ``.php`` and be routed to the PHP grammar.

    @param file_path: File path.
    @returns: True if the file is a ``.blade.php`` template.
    """
    return Path(file_path).name.lower().endswith(_BLADE_SUFFIX)


def _build_line_index(
    chunks: list[ChunkData],
) -> Any:
    """Build a bisect-based lookup from line number to chunk_id.

    Returns a callable: (line: int) -> str | None.

    @param chunks: Chunks sorted by start_line (as produced by chunkers).
    @returns: Callable that maps line → chunk_id or None.
    """
    if not chunks:
        return lambda _line: None

    # Sort by start_line and build parallel arrays for bisect.
    sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
    starts = [c.start_line for c in sorted_chunks]
    fallback_id = sorted_chunks[0].chunk_id

    def _lookup(line: int) -> str | None:
        idx = bisect.bisect_right(starts, line) - 1
        if idx < 0:
            # Line is before any chunk (e.g. import in preamble) —
            # attribute to first chunk.  May mis-attribute refs in
            # files with large preambles before the first definition (L7).
            return fallback_id
        c = sorted_chunks[idx]
        if c.start_line <= line <= c.end_line:
            return c.chunk_id
        return fallback_id

    return _lookup


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

# Python patterns
_PY_IMPORT_FROM_RE = re.compile(
    r"^from\s+([\w.]+)\s+import\s+(.+?)(?:\s+#.*)?$", re.MULTILINE
)
_PY_IMPORT_RE = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^class\s+\w+\(([^)]+)\)", re.MULTILINE)
_PY_DECORATOR_RE = re.compile(r"^@([\w.]+)", re.MULTILINE)


def _extract_python_refs(chunks: list[ChunkData], content: str) -> list[RefData]:
    """Extract cross-references from Python source.

    @param chunks: Chunks produced from this file.
    @param content: Full file content.
    @returns: List of RefData.
    """
    refs: list[RefData] = []

    # Build sorted index for O(log n) line → chunk_id lookup.
    _chunk_for_line = _build_line_index(chunks)

    # from X import Y, Z
    for m in _PY_IMPORT_FROM_RE.finditer(content):
        module = m.group(1)
        names = [n.strip().split(" as ")[0].strip() for n in m.group(2).split(",")]
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            for name in names:
                if name and name != "*":
                    refs.append(RefData(cid, f"{module}.{name}", RefType.IMPORT))

    # import X
    for m in _PY_IMPORT_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            refs.append(RefData(cid, m.group(1), RefType.IMPORT))

    # class Foo(Bar, Baz):
    for m in _PY_CLASS_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            bases = [b.strip() for b in m.group(1).split(",")]
            for base in bases:
                # Strip generic params: Base[T] → Base
                base = base.split("[")[0].strip()
                if base and base not in ("object",):
                    refs.append(RefData(cid, base, RefType.INHERITS))

    # @decorator
    for m in _PY_DECORATOR_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            refs.append(RefData(cid, m.group(1), RefType.DECORATOR))

    return refs


# PHP patterns
_PHP_USE_RE = re.compile(
    r"^\s*use\s+(?:function\s+|const\s+)?([\w\\]+)(?:\s+as\s+\w+)?\s*;", re.MULTILINE
)
_PHP_EXTENDS_RE = re.compile(
    r"\b(?:class|interface)\s+\w+\s+extends\s+([\w\\,\s]+?)\s*(?:\{|implements\b)"
)
_PHP_IMPLEMENTS_RE = re.compile(r"\bimplements\s+([\w\\,\s]+?)\s*\{")


def _extract_php_refs(chunks: list[ChunkData], content: str) -> list[RefData]:
    """Extract cross-references from PHP source.

    ``use`` statements are the import graph a Laravel codebase actually
    navigates by; ``extends`` / ``implements`` give the inheritance edges
    that connect a controller or model to its framework base class.

    @param chunks: Chunks produced from this file.
    @param content: Full file content.
    @returns: List of RefData.
    """
    refs: list[RefData] = []
    _chunk_for_line = _build_line_index(chunks)

    for m in _PHP_USE_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            refs.append(RefData(cid, m.group(1).lstrip("\\"), RefType.IMPORT))

    for pattern in (_PHP_EXTENDS_RE, _PHP_IMPLEMENTS_RE):
        for m in pattern.finditer(content):
            line = content[: m.start()].count("\n") + 1
            cid = _chunk_for_line(line)
            if not cid:
                continue
            for base in m.group(1).split(","):
                name = base.strip().lstrip("\\")
                if name:
                    refs.append(RefData(cid, name, RefType.INHERITS))

    return refs


# TypeScript/JavaScript patterns
_TS_IMPORT_RE = re.compile(
    r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE
)
_TS_IMPORT_DEFAULT_RE = re.compile(
    r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE
)


def _extract_ts_refs(chunks: list[ChunkData], content: str) -> list[RefData]:
    """Extract cross-references from TypeScript/JavaScript source.

    @param chunks: Chunks produced from this file.
    @param content: Full file content.
    @returns: List of RefData.
    """
    refs: list[RefData] = []

    _chunk_for_line = _build_line_index(chunks)

    # import { X, Y } from 'module'
    for m in _TS_IMPORT_RE.finditer(content):
        names = [n.strip().split(" as ")[0].strip() for n in m.group(1).split(",")]
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            for name in names:
                if name:
                    refs.append(RefData(cid, name, RefType.IMPORT))

    # import X from 'module'
    for m in _TS_IMPORT_DEFAULT_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        cid = _chunk_for_line(line)
        if cid:
            refs.append(RefData(cid, m.group(1), RefType.IMPORT))

    return refs
