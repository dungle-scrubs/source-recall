"""CLI: sr index, sr ask, sr status, sr list, sr clean, sr config show."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer(
    name="sr",
    help=(
        "source-recall: Code search and retrieval for AI coding tools.\n\n"
        "Indexes repositories into searchable chunks (AST-parsed, FTS5-indexed,\n"
        "optionally vector-embedded). Designed for integration with coding agents\n"
        "and LLM pipelines.\n\n"
        "Quick start:\n\n"
        "  sr index .          # Build index for current repo\n\n"
        "  sr ask 'auth flow'  # Search for relevant code\n\n"
        "  sr status .         # Check index health\n\n"
        "  sr list             # Show all indexed repos\n\n"
        "Agent integration:\n\n"
        "  Most commands accept --json for machine-readable output.\n"
        "  Use 'sr serve' for persistent HTTP access."
    ),
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# sr index
# ---------------------------------------------------------------------------


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the repository root."),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Skip vector embeddings (FTS-only, much faster)."
    ),
    rerank: bool = typer.Option(
        False, "--rerank", help="Enable cross-encoder reranking."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output final summary as JSON to stdout (progress on stderr).",
    ),
) -> None:
    """Build or rebuild the index for a repository.

    Discovers files, parses them with tree-sitter (AST), regex, or text
    fallback, stores chunks in FTS5, and optionally generates vector
    embeddings for semantic search.

    Progress is printed to stderr. With --json, a machine-readable
    summary is printed to stdout on completion.

    Examples:\n
      sr index .                    # Index current repo\n
      sr index ~/dev/myproject      # Index specific repo\n
      sr index . --no-embed         # FTS-only (fast, no model download)\n
      sr index . --json             # JSON summary for agent consumption
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from source_recall import Index

    repo_path = Path(path).resolve()

    progress = Progress(
        TextColumn("[bold blue]Indexing"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[file]}"),
        console=err_console,
        transient=True,
    )
    task_id: object = None

    def on_progress(file_path: str, current: int, total: int) -> None:
        """Progress callback for indexing."""
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("index", total=total, file=file_path)
            progress.start()
        progress.update(task_id, completed=current, file=file_path)  # type: ignore[arg-type]

    try:
        kwargs: dict[str, object] = {}
        if no_embed:
            kwargs["embedder"] = None
        if rerank:
            kwargs["rerank_enabled"] = True
        idx = Index(repo_path, on_progress=on_progress, **kwargs)
        db_path = idx.build()

        if progress.live.is_started:
            progress.stop()
    except KeyboardInterrupt:
        if progress.live.is_started:
            progress.stop()
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(130)  # noqa: B904

    try:
        s = idx.status()

        if json_output:
            data = _status_to_dict(s)
            data["db_path"] = str(db_path)
            print(json.dumps(data, indent=2))
        else:
            err_console.print(
                f"[green]✓[/green] Indexed {s.file_count} files "
                f"({s.chunk_count} chunks) → {db_path}"
            )
            err_console.print(
                f"  AST: {s.ast_files}  "
                f"Regex: {s.regex_files}  "
                f"Text: {s.text_fallback_files}"
            )
            if s.vector_count > 0:
                pct = s.vector_count / s.chunk_count * 100 if s.chunk_count > 0 else 0
                err_console.print(
                    f"  Vectors: {s.vector_count}/{s.chunk_count} "
                    f"({pct:.0f}%) — {s.embed_model} ({s.embed_dimensions}d)"
                )
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


# ---------------------------------------------------------------------------
# sr ask
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="Search query."),
    path: str = typer.Argument(".", help="Path to the repository root."),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
    plain: bool = typer.Option(False, "--plain", help="Plain text, no formatting."),
    files_only: bool = typer.Option(False, "--files", help="File paths only."),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Skip vector search (FTS-only)."
    ),
) -> None:
    """Search the index for relevant code.

    Returns ranked code chunks matching the query using FTS5 full-text
    search and (optionally) vector similarity. Results include file paths,
    line ranges, symbol names, scores, and the matched code.

    Output modes:\n
      (default)   Rich-formatted with syntax highlighting\n
      --json      Structured JSON array for piping to agents/LLMs\n
      --plain     Plain text without ANSI formatting\n
      --files     Unique file paths only (one per line)

    Examples:\n
      sr ask 'authentication middleware'\n
      sr ask 'database connection' --json | jq '.[0].content'\n
      sr ask 'error handling' --files\n
      sr ask 'parse config' ~/dev/myproject -k 20
    """
    from source_recall import Index

    repo_path = Path(path).resolve()

    try:
        idx = Index(repo_path, **({"embedder": None} if no_embed else {}))

        # Auto-refresh if configured.
        if idx.config.auto_refresh:
            try:
                refreshed = idx.refresh()
                if refreshed > 0:
                    err_console.print(
                        f"[dim]Refreshed {refreshed} files[/dim]",
                        highlight=False,
                    )
            except Exception:
                logging.getLogger(__name__).debug("Auto-refresh failed", exc_info=True)

        results = idx.query(question, top_k=top_k)

    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if not results:
        if json_output:
            print("[]")
        else:
            err_console.print("[yellow]No results found.[/yellow]")
        return

    # Output modes.
    if json_output:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    elif files_only:
        seen: set[str] = set()
        for r in results:
            if r.file_path not in seen:
                print(r.file_path)
                seen.add(r.file_path)
    elif plain:
        for r in results:
            print(f"--- {r.file_path}")
            if r.symbol_name:
                print(f"    {r.symbol_name} ({r.symbol_type})")
            print(r.content)
            print()
    else:
        _rich_output(results)


def _rich_output(results: list) -> None:
    """Print results with Rich formatting.

    @param results: List of QueryResult.
    """
    for i, r in enumerate(results):
        # Header.
        console.print()
        score_str = f"[dim]score={r.score:.3f} ({r.match_reason})[/dim]"
        console.print(
            f"[bold cyan]{r.file_path}[/bold cyan]"
            f":{r.start_line}-{r.end_line}  {score_str}"
        )
        if r.symbol_name:
            console.print(
                f"  [green]{r.symbol_name}[/green] [dim]({r.symbol_type})[/dim]"
            )

        # Code block.
        ext = Path(r.file_path).suffix.lstrip(".")
        lexer = _ext_to_lexer(ext)
        syntax = Syntax(
            r.content,
            lexer,
            theme="monokai",
            line_numbers=True,
            start_line=r.start_line,
        )
        console.print(syntax)

        if i < len(results) - 1:
            console.rule(style="dim")


def _ext_to_lexer(ext: str) -> str:
    """Map file extension to Pygments lexer name.

    @param ext: Extension without dot.
    @returns: Lexer name.
    """
    mapping = {
        "py": "python",
        "ts": "typescript",
        "tsx": "tsx",
        "js": "javascript",
        "jsx": "jsx",
        "sh": "bash",
        "bash": "bash",
        "go": "go",
        "rs": "rust",
        "rb": "ruby",
        "java": "java",
        "kt": "kotlin",
        "sql": "sql",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
        "json": "json",
        "md": "markdown",
    }
    return mapping.get(ext, "text")


# ---------------------------------------------------------------------------
# sr status
# ---------------------------------------------------------------------------


@app.command()
def status(
    path: str = typer.Argument(".", help="Path to the repository root."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON for machine consumption.",
    ),
) -> None:
    """Show index status for a repository.

    Reports file count, chunk count, vector coverage, embedding model,
    database size, and last indexed timestamp. Use --json for structured
    output suitable for agent health checks.

    Examples:\n
      sr status .                    # Human-readable status\n
      sr status ~/dev/myproject      # Check specific repo\n
      sr status . --json             # JSON for agent integration\n
      sr status . --json | jq '.vector_coverage_pct'
    """
    from source_recall import Index

    repo_path = Path(path).resolve()

    try:
        idx = Index(repo_path)
        s = idx.status()
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if json_output:
        print(json.dumps(_status_to_dict(s), indent=2))
    else:
        _human_size = _format_bytes(s.db_size_bytes)
        console.print(f"[bold]Repository:[/bold]  {s.repo_path}")
        console.print(f"[bold]Index:[/bold]       {s.db_path}")
        console.print(f"[bold]Size:[/bold]        {_human_size}")
        console.print(f"[bold]Indexed at:[/bold]  {s.indexed_at}")
        console.print(
            f"[bold]Last commit:[/bold] {s.last_commit[:8] if s.last_commit else '(none)'}"
        )
        console.print(
            f"[bold]Files:[/bold]       {s.file_count} "
            f"({s.ast_files} AST, {s.regex_files} regex, "
            f"{s.text_fallback_files} text fallback)"
        )
        console.print(f"[bold]Chunks:[/bold]      {s.chunk_count:,}")
        if s.vector_count > 0:
            pct = s.vector_count / s.chunk_count * 100 if s.chunk_count > 0 else 0
            console.print(
                f"[bold]Vectors:[/bold]     {s.vector_count:,}/{s.chunk_count:,} ({pct:.0f}%)"
            )
            console.print(
                f"[bold]Embed model:[/bold] {s.embed_model} ({s.embed_dimensions}d)"
            )
        else:
            console.print("[bold]Vectors:[/bold]     [dim]none[/dim]")


# ---------------------------------------------------------------------------
# sr list
# ---------------------------------------------------------------------------


@app.command("list")
def list_indexes(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON array for machine consumption.",
    ),
) -> None:
    """List all indexed repositories.

    Scans the source-recall data directory and reports every indexed
    repo with its path, size, chunk count, and vector coverage.
    Use --json for structured output.

    Examples:\n
      sr list                  # Human-readable table\n
      sr list --json           # JSON array for agent integration\n
      sr list --json | jq '.[].repo_path'
    """
    base = Path.home() / ".local" / "share" / "source-recall"
    if not base.exists():
        if json_output:
            print("[]")
        else:
            console.print("[dim]No indexes found.[/dim]")
        return

    from source_recall.store import IndexStore

    entries: list[dict[str, object]] = []

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "index.db"
        if not db_path.exists():
            continue

        try:
            with IndexStore(db_path) as store:
                repo_path = store.get_meta("repo_path") or ""
                indexed_at = store.get_meta("indexed_at") or ""
                last_commit = store.get_meta("last_commit") or ""

                file_count = store.conn.execute(
                    "SELECT COUNT(*) FROM file_hashes"
                ).fetchone()[0]
                chunk_count = store.conn.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]

                # Parse mode breakdown.
                mode_rows = store.conn.execute(
                    "SELECT parse_mode, COUNT(*) FROM file_hashes GROUP BY parse_mode"
                ).fetchall()
                modes = dict(mode_rows)
                ast_files = modes.get("ast", 0)
                regex_files = modes.get("regex", 0)
                text_files = modes.get("text_fallback", 0)

                # Vector count — query via store's apsw connection
                # which loads sqlite-vec extension automatically.
                vec_count = 0
                try:
                    vec_conn = store._get_vec_conn()  # noqa: SLF001
                    vec_count = vec_conn.execute(
                        "SELECT COUNT(*) FROM vec_chunks"
                    ).fetchone()[0]
                except Exception:
                    # sqlite-vec not available or no vec table.
                    pass  # noqa: SIM105

                # Embed model info.
                embed_model = store.get_meta("embed_model") or ""
                embed_dims_str = store.get_meta("embed_dimensions") or "0"
                embed_dims = int(embed_dims_str)

            db_size = db_path.stat().st_size
            exists = Path(repo_path).exists() if repo_path else False
            vec_pct = round(vec_count / chunk_count * 100, 1) if chunk_count > 0 else 0

            entries.append(
                {
                    "repo_path": repo_path,
                    "repo_exists": exists,
                    "db_path": str(db_path),
                    "db_size_bytes": db_size,
                    "indexed_at": indexed_at,
                    "last_commit": last_commit,
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "ast_files": ast_files,
                    "regex_files": regex_files,
                    "text_fallback_files": text_files,
                    "vector_count": vec_count,
                    "vector_coverage_pct": vec_pct,
                    "embed_model": embed_model,
                    "embed_dimensions": embed_dims,
                }
            )
        except Exception:
            continue

    if not entries:
        if json_output:
            print("[]")
        else:
            console.print("[dim]No indexes found.[/dim]")
        return

    if json_output:
        print(json.dumps(entries, indent=2))
    else:
        from rich.table import Table

        table = Table(title="Indexed Repositories", show_lines=False)
        table.add_column("Repository", style="cyan", no_wrap=True)
        table.add_column("Files", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Vectors", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Indexed", style="dim")
        table.add_column("", style="dim")  # exists indicator

        for e in entries:
            repo_name = Path(str(e["repo_path"])).name if e["repo_path"] else "?"
            vec_str = (
                f"{e['vector_count']}/{e['chunk_count']} ({e['vector_coverage_pct']}%)"
                if e["vector_count"]
                else "—"
            )
            size_str = _format_bytes(int(str(e["db_size_bytes"])))
            indexed = str(e["indexed_at"])[:10] if e["indexed_at"] else "?"
            exists_mark = "✓" if e["repo_exists"] else "[red]✗ gone[/red]"

            table.add_row(
                repo_name,
                str(e["file_count"]),
                f"{e['chunk_count']:,}",
                vec_str,
                size_str,
                indexed,
                exists_mark,
            )

        console.print(table)


# ---------------------------------------------------------------------------
# sr clean
# ---------------------------------------------------------------------------


@app.command()
def clean(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be removed."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output removed paths as JSON array."
    ),
) -> None:
    """Remove orphaned indexes (repos that no longer exist).

    Scans all indexes and removes those whose original repo path no
    longer exists on disk. Use --dry-run to preview, --json for
    machine-readable output.

    Examples:\n
      sr clean                 # Remove orphaned indexes\n
      sr clean --dry-run       # Preview what would be removed\n
      sr clean --json          # JSON output for automation
    """
    base = Path.home() / ".local" / "share" / "source-recall"
    if not base.exists():
        if json_output:
            print("[]")
        else:
            console.print("[dim]No indexes found.[/dim]")
        return

    removed_entries: list[dict[str, str]] = []

    from source_recall.store import IndexStore

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "index.db"
        if not db_path.exists():
            continue

        try:
            with IndexStore(db_path) as store:
                stored_path = store.get_meta("repo_path")

            if stored_path is None:
                continue
            repo_path = Path(stored_path)
            if repo_path.exists():
                continue

            if not dry_run and not json_output:
                shutil.rmtree(d)
                console.print(f"[green]Removed:[/green] {d} → {stored_path}")
            elif not json_output:
                console.print(f"[yellow]Would remove:[/yellow] {d} → {stored_path}")
            else:
                if not dry_run:
                    shutil.rmtree(d)

            removed_entries.append(
                {
                    "index_dir": str(d),
                    "repo_path": stored_path,
                    "action": "would_remove" if dry_run else "removed",
                }
            )

        except Exception:
            continue

    if json_output:
        print(json.dumps(removed_entries, indent=2))
    elif not removed_entries:
        console.print("[dim]No orphaned indexes found.[/dim]")
    elif not dry_run:
        console.print(
            f"\n[green]Cleaned {len(removed_entries)} orphaned index(es).[/green]"
        )


# ---------------------------------------------------------------------------
# sr config show
# ---------------------------------------------------------------------------


@app.command("config")
def config_show(
    path: str = typer.Argument(".", help="Path to the repository root."),
) -> None:
    """Print the resolved configuration as TOML.

    Shows all configuration values after merging defaults, project-level
    .source-recall.toml, and environment variables. Useful for debugging
    config resolution.

    Examples:\n
      sr config .              # Show config for current repo\n
      sr config ~/dev/project  # Show config for specific repo
    """
    from source_recall.config import format_config, resolve_config

    repo_path = Path(path).resolve()
    config = resolve_config(repo_path)
    console.print(format_config(config))


# ---------------------------------------------------------------------------
# sr serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    paths: list[str] = typer.Argument(
        None, help="Repository paths to serve (default: current dir)."
    ),
    port: int = typer.Option(7249, "--port", "-p", help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Disable vector search (FTS-only)."
    ),
    rerank: bool = typer.Option(
        False, "--rerank", help="Enable cross-encoder reranking."
    ),
) -> None:
    """Start a persistent query server (HTTP/JSON).

    Loads the embedding model once on startup, then serves queries
    over HTTP. Accepts one or more repo paths. Designed for integration
    with coding agents that need persistent search access.

    Endpoints:\n
      GET  /health   → liveness check (200 OK)\n
      GET  /repos    → list loaded repos with metadata\n
      GET  /status   → index metrics (file/chunk/vector counts)\n
      POST /query    → search: {question, top_k?, repo?}\n
      POST /refresh  → incremental re-index

    Examples:\n
      sr serve                           # Serve current dir on :7249\n
      sr serve ~/dev/project             # Serve one repo\n
      sr serve ~/dev/a ~/dev/b -p 8080   # Serve multiple repos\n
      sr serve . --no-embed              # FTS-only mode
    """
    import uvicorn

    from source_recall.server import create_app

    if not paths:
        paths = ["."]

    repo_paths = [Path(p).resolve() for p in paths]

    if len(repo_paths) == 1:
        err_console.print(
            f"[bold]Starting source-recall server[/bold] for {repo_paths[0]}"
        )
    else:
        err_console.print(
            f"[bold]Starting source-recall server[/bold] for {len(repo_paths)} repos:"
        )
        for rp in repo_paths:
            err_console.print(f"  • {rp.name} → {rp}")

    if no_embed:
        err_console.print("  FTS-only mode (embeddings disabled)")
    else:
        err_console.print("  Loading model (first time may download ~522 MB)...")

    # Build the app with explicit config instead of mutating os.environ.
    if no_embed:
        server_app = create_app(repo_paths, embedder=None)
    else:
        server_app = create_app(repo_paths)

    err_console.print(f"  Listening on [cyan]http://{host}:{port}[/cyan]")
    err_console.print()

    uvicorn.run(server_app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_to_dict(s: object) -> dict[str, object]:
    """Convert an IndexStatus to a JSON-serializable dict.

    Adds computed fields like vector_coverage_pct and human-readable size.

    @param s: IndexStatus instance.
    @returns: Dict with all status fields plus computed helpers.
    """
    vec_pct = (
        round(s.vector_count / s.chunk_count * 100, 1)  # type: ignore[union-attr]
        if s.chunk_count > 0  # type: ignore[union-attr]
        else 0
    )
    return {
        "repo_path": s.repo_path,  # type: ignore[union-attr]
        "db_path": s.db_path,  # type: ignore[union-attr]
        "db_size_bytes": s.db_size_bytes,  # type: ignore[union-attr]
        "db_size_human": _format_bytes(s.db_size_bytes),  # type: ignore[union-attr]
        "indexed_at": s.indexed_at,  # type: ignore[union-attr]
        "last_commit": s.last_commit,  # type: ignore[union-attr]
        "file_count": s.file_count,  # type: ignore[union-attr]
        "chunk_count": s.chunk_count,  # type: ignore[union-attr]
        "ast_files": s.ast_files,  # type: ignore[union-attr]
        "regex_files": s.regex_files,  # type: ignore[union-attr]
        "text_fallback_files": s.text_fallback_files,  # type: ignore[union-attr]
        "vector_count": s.vector_count,  # type: ignore[union-attr]
        "vector_coverage_pct": vec_pct,
        "embed_model": s.embed_model,  # type: ignore[union-attr]
        "embed_dimensions": s.embed_dimensions,  # type: ignore[union-attr]
    }


def _format_bytes(n: int) -> str:
    """Format bytes as human-readable size.

    @param n: Number of bytes.
    @returns: Formatted string (e.g. '12.4 MB').
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
