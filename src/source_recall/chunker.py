"""Tree-sitter chunking for TS/Python, regex for Bash, text fallback."""

from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from source_recall.models import ChunkData, RefData, RefType, SearchQuality, SymbolType

if TYPE_CHECKING:
    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# File extensions mapped to their language key.
_TS_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".jsx"})
_PY_EXTENSIONS = frozenset({".py"})
_BASH_EXTENSIONS = frozenset({".sh", ".bash"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".mdx", ".markdown"})
_TEXT_EXTENSIONS = frozenset({".txt", ".text", ".rst", ".log"})

# Sentence-ending punctuation followed by whitespace.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Overlap lines between consecutive sub-chunks.
_SUB_CHUNK_OVERLAP = 8


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
    ext = _get_extension(file_path)

    if ext in _TS_EXTENSIONS:
        return _chunk_typescript(file_path, content, max_chars)
    if ext in _PY_EXTENSIONS:
        return _chunk_python(file_path, content, max_chars)
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
    if ext in _PY_EXTENSIONS:
        refs = _extract_python_refs(chunks, content)
    elif ext in _TS_EXTENSIONS:
        refs = _extract_ts_refs(chunks, content)

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

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            continue

        chunks.append(
            ChunkData(
                file_path=file_path,
                symbol_name=f"Page {page_num + 1}",
                symbol_type=SymbolType.MODULE,
                content=text,
                start_line=page_num + 1,
                end_line=page_num + 1,
                search_quality=SearchQuality.AST,
            )
        )

    doc.close()
    return chunks, SearchQuality.AST


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
    fenced_ranges: list[tuple[int, int]] = []
    for m in _FENCE_RE.finditer(content):
        if len(fenced_ranges) % 2 == 0:
            fenced_ranges.append((m.start(), -1))
        elif fenced_ranges:
            fenced_ranges[-1] = (fenced_ranges[-1][0], m.end())

    def _in_fence(pos: int) -> bool:
        return any(s <= pos <= e for s, e in fenced_ranges if e != -1)

    # Collect heading positions.
    sections: list[tuple[str, int]] = []  # (heading_text, char_offset)
    for m in _HEADING_RE.finditer(content):
        if not _in_fence(m.start()):
            sections.append((m.group(2).strip(), m.start()))

    chunks: list[ChunkData] = []

    # Content before first heading.
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
                SearchQuality.AST,
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
            SearchQuality.AST,
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
            SearchQuality.AST,
            max_chars,
        )

    return chunks, SearchQuality.AST


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

    # Build (sentence, start_line) pairs by finding each sentence's
    # position in the original text so line numbers are accurate.
    raw_sentences = _SENTENCE_END_RE.split(content)
    sentence_infos: list[tuple[str, int]] = []  # (text, start_line)
    char_pos = 0
    for raw in raw_sentences:
        stripped = raw.strip()
        if not stripped:
            char_pos += len(raw)
            # Account for the whitespace removed by split.
            continue
        # Find this sentence's start in original content.
        idx = content.find(raw.lstrip()[:20], char_pos) if raw.lstrip() else char_pos
        if idx == -1:
            idx = char_pos
        start_line = content[:idx].count("\n") + 1
        sentence_infos.append((stripped, start_line))
        char_pos = idx + len(raw)

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
    blocks = re.split(r"\n\s*\n", content)
    chunks: list[ChunkData] = []
    current_line = 1

    for block in blocks:
        text = block.strip()
        if not text:
            current_line += block.count("\n") + 1
            continue

        line_count = text.count("\n") + 1
        _add_chunk(
            chunks,
            file_path,
            "",
            SymbolType.BLOCK,
            text,
            current_line,
            current_line + line_count - 1,
            SearchQuality.TEXT_FALLBACK,
            max_chars,
        )
        current_line += block.count("\n") + 2  # +2 for the blank line separator.

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

    Prepends the function/class signature to each sub-chunk.

    @param content: Full chunk text.
    @param max_chars: Character limit per sub-chunk.
    @param symbol_name: Used for signature extraction.
    @returns: List of (text, start_line_offset, end_line_offset).
    """
    lines = content.split("\n")

    # Extract signature (first line or first few lines ending with : or {).
    signature = _extract_signature(lines)
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

        # Apply overlap — rewind by _SUB_CHUNK_OVERLAP lines.
        if i < len(lines):
            i = max(start_i + 1, i - _SUB_CHUNK_OVERLAP)

    return result


def _extract_signature(lines: list[str]) -> str:
    """Extract the function/class signature from the first lines.

    Collects the opening declaration (def/class/function) and any
    continuation lines that are part of the parameter list or type
    annotation.  Stops at the body (docstring, statements, etc.).

    @param lines: All lines of the chunk.
    @returns: Signature string (may be multi-line).
    """
    if not lines:
        return ""

    # For Python: lines ending with ':'
    # For TS/JS: lines ending with '{' or '=>'
    sig_lines: list[str] = [lines[0]]
    for line in lines[1:5]:  # Check first 5 lines.
        stripped = line.rstrip()
        if stripped.endswith((":", "{", "=>", "(", ",")):
            sig_lines.append(line)
        elif stripped.endswith(")") or stripped.endswith(") ->"):
            # Closing paren of parameter list.
            sig_lines.append(line)
            break
        else:
            break

    return "\n".join(sig_lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_with_fallback(
    _file_path: str, content: str, language: str
) -> tuple[Any, SearchQuality]:
    """Parse with tree-sitter, falling back on high error density.

    @param file_path: Repo-relative path.
    @param content: File content.
    @param language: tree-sitter language key.
    @returns: (tree, quality). Tree may be None if fallback.
    """
    from tree_sitter_language_pack import get_parser

    parser = get_parser(language)
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


def _node_text(node: Node, content: str) -> str:
    """Extract the text of a node from the source content.

    @param node: tree-sitter Node.
    @param content: Full file content.
    @returns: Text slice corresponding to the node.
    """
    return content[node.start_byte : node.end_byte]


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


def _has_react_wrapper(node: Node) -> bool:
    """Check if a node contains React.memo/forwardRef/lazy calls.

    @param node: tree-sitter Node.
    @returns: True if a React wrapper call is found.
    """
    _REACT_WRAPPERS = {"memo", "forwardRef", "lazy"}

    for child in node.named_children:
        if child.type == "call_expression":
            func = child.child_by_field_name("function")
            if func is not None:
                text = (func.text or b"").decode("utf-8", errors="replace")
                for wrapper in _REACT_WRAPPERS:
                    if wrapper in text:
                        return True
        if _has_react_wrapper(child):
            return True
    return False


def _contains_node_type(node: Node, node_type: str) -> bool:
    """Check if a node or any descendant has the given type.

    @param node: tree-sitter Node.
    @param node_type: Type string to look for.
    @returns: True if found.
    """
    if node.type == node_type:
        return True
    return any(_contains_node_type(child, node_type) for child in node.children)


def _contains_any_node_type(node: Node, node_types: set[str]) -> bool:
    """Check if a node or any descendant has any of the given types.

    @param node: tree-sitter Node.
    @param node_types: Set of type strings.
    @returns: True if any found.
    """
    if node.type in node_types:
        return True
    return any(_contains_any_node_type(child, node_types) for child in node.children)


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

    @param file_path: File path.
    @returns: Extension including dot (e.g. '.py').
    """
    idx = file_path.rfind(".")
    if idx == -1:
        return ""
    return file_path[idx:].lower()


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
