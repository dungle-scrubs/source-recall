"""CLI: sr index, sr ask, sr status, sr list, sr clean, sr config show, sr daemon."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.syntax import Syntax

from source_recall.__init__ import __version__
from source_recall.store import get_index_base

if TYPE_CHECKING:
    from source_recall.models import IndexStatus


def _version_callback(value: bool) -> None:
    """Print the installed version and exit when --version is passed."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """source-recall: Code search and retrieval for AI coding tools."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit


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


_DEFAULT_DAEMON_URL = "http://127.0.0.1:7249"


def _daemon_url() -> str:
    """Get the daemon base URL.

    @returns: Base URL string.
    """
    import os

    return os.environ.get("SR_DAEMON_URL", _DEFAULT_DAEMON_URL)


def _daemon_headers() -> dict[str, str]:
    """Auth headers for daemon requests.

    Loads the local token the daemon wrote to the config dir and forwards
    it as ``X-SR-Token`` so ``sr ask`` / ``sr refresh`` / ``sr add`` keep
    working transparently. Returns an empty dict when no token exists yet
    (daemon never started) — the request will then get a 401, which the
    callers surface as an error.

    @returns: Header dict (possibly empty).
    """
    from source_recall.daemon_config import load_token

    token = load_token()
    return {"X-SR-Token": token} if token else {}


def _try_daemon_query(
    question: str,
    *,
    top_k: int | None = None,
    repo: str | None = None,
    branch: str | None = None,
) -> object | None:
    """Attempt to query the daemon. Returns response or None if down.

    @param question: Search query.
    @param top_k: Max results.
    @param repo: Repo name filter.
    @param branch: Branch filter.
    @returns: httpx.Response or None if daemon unreachable.
    """
    import httpx

    url = f"{_daemon_url()}/query"
    payload: dict[str, object] = {"question": question}
    if top_k:
        payload["top_k"] = top_k
    if repo:
        payload["repo"] = repo
    if branch:
        payload["branch"] = branch

    try:
        resp = httpx.post(url, json=payload, timeout=10, headers=_daemon_headers())
        # Return any successful response or a daemon-side error that
        # the caller should handle (not silently fall back).
        if resp.status_code == 200:
            return resp
        # Daemon is up but returned an error — bubble it instead of
        # falling back to in-process with wrong repo context.
        if resp.status_code in (400, 404, 503):
            return resp
    except Exception:
        pass
    return None


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
    repo: str = typer.Option(
        None, "--repo", "-r", help="Query a specific repo (daemon mode)."
    ),
) -> None:
    """Search the index for relevant code.

    Tries the daemon first (D-006). Falls back to in-process if the
    daemon is not running.

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
      sr ask 'parse config' ~/dev/myproject -k 20\n
      sr ask 'auth flow' --repo marrow   # daemon multi-repo
    """
    # D-006: Try daemon first.
    daemon_resp = _try_daemon_query(question, top_k=top_k, repo=repo)
    if daemon_resp is not None:
        # Handle daemon-side errors before assuming success.
        status_code = daemon_resp.status_code  # type: ignore[union-attr]
        if status_code != 200:
            data = daemon_resp.json()  # type: ignore[union-attr]
            detail = data.get("detail", f"Daemon error (HTTP {status_code})")
            err_console.print(f"[red]Error:[/red] {detail}")
            raise typer.Exit(1)

        data = daemon_resp.json()  # type: ignore[union-attr]
        results_data = data.get("results", [])

        if not results_data:
            if json_output:
                print("[]")
            else:
                err_console.print("[yellow]No results found.[/yellow]")
            return

        if json_output:
            print(json.dumps(results_data, indent=2))
        elif files_only:
            seen: set[str] = set()
            for r in results_data:
                fp = r["file_path"]
                if fp not in seen:
                    print(fp)
                    seen.add(fp)
        elif plain:
            for r in results_data:
                prefix = f"[{r.get('repo_name', '')}] " if r.get("repo_name") else ""
                print(f"--- {prefix}{r['file_path']}")
                if r.get("symbol_name"):
                    print(f"    {r['symbol_name']} ({r['symbol_type']})")
                print(r["content"])
                print()
        else:
            # Convert to QueryResult-like dicts for _rich_output.
            from source_recall.models import QueryResult

            qr_list = [
                QueryResult(
                    chunk_id=r["chunk_id"],
                    file_path=r["file_path"],
                    symbol_name=r.get("symbol_name", ""),
                    symbol_type=r.get("symbol_type", ""),
                    content=r["content"],
                    score=r["score"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    search_quality=r.get("search_quality", ""),
                    match_reason=r.get("match_reason", ""),
                )
                for r in results_data
            ]
            _rich_output(qr_list)
        return

    # Fallback: in-process query.
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
        seen_local: set[str] = set()
        for r in results:
            if r.file_path not in seen_local:
                print(r.file_path)
                seen_local.add(r.file_path)
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
    base = get_index_base()
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
    base = get_index_base()
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


def _is_loopback_host(host: str) -> bool:
    """Whether ``host`` binds only to the local machine.

    ``0.0.0.0`` / ``::`` (bind-all) and any routable address are treated as
    non-loopback so they trip the ``serve`` exposure guard.

    @param host: Host string from --host.
    @returns: True only for localhost / 127.0.0.0/8 / ::1.
    """
    import ipaddress

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
    insecure: bool = typer.Option(
        False,
        "--insecure",
        help="Allow binding to a non-loopback host (unauthenticated API).",
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

    # The query server is unauthenticated. Binding it to a non-loopback
    # host exposes every repo's source to the network — refuse unless the
    # operator explicitly opts in with --insecure.
    if not _is_loopback_host(host):
        if not insecure:
            err_console.print(
                f"[red]Refusing to bind to non-loopback host '{host}':[/red] "
                "the query server is unauthenticated and would expose your "
                "source to the network. Re-run with [bold]--insecure[/bold] "
                "if you understand the risk (prefer an SSH tunnel instead)."
            )
            raise typer.Exit(1)
        err_console.print(
            f"[yellow]⚠ WARNING:[/yellow] serving on non-loopback host "
            f"'{host}' without authentication (--insecure). Anyone who can "
            "reach this host can read your indexed source."
        )

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
        server_app = create_app(
            repo_paths, embedder=None, rerank_enabled=rerank, host=host
        )
    else:
        server_app = create_app(repo_paths, rerank_enabled=rerank, host=host)

    err_console.print(f"  Listening on [cyan]http://{host}:{port}[/cyan]")
    err_console.print()

    uvicorn.run(server_app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Daemon subcommands
# ---------------------------------------------------------------------------

daemon_app = typer.Typer(
    name="daemon",
    help="Manage the source-recall daemon (multi-repo background server).",
    no_args_is_help=True,
)
app.add_typer(daemon_app, name="daemon")


def uvicorn_run(app: object, **kwargs: object) -> None:
    """Thin wrapper around uvicorn.run for testability.

    @param app: ASGI application.
    @param kwargs: Forwarded to uvicorn.run.
    """
    import uvicorn

    uvicorn.run(app, **kwargs)  # type: ignore[arg-type]


@daemon_app.command("run")
def daemon_run(
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to repos.toml (default: ~/.config/source-recall/repos.toml).",
    ),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Disable vector search (FTS-only)."
    ),
) -> None:
    """Start the daemon in the foreground.

    Loads repos from repos.toml and starts the HTTP server.
    All registered repos are served immediately.

    Examples:\n
      sr daemon run                          # Default config\n
      sr daemon run --config ./repos.toml    # Custom config\n
      sr daemon run --no-embed               # FTS-only mode
    """
    from source_recall.daemon import create_daemon_app
    from source_recall.daemon_config import DaemonConfig

    config_path = Path(config) if config else DaemonConfig.default_config_path()

    try:
        cfg = DaemonConfig.from_toml(config_path)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    err_console.print("[bold]Starting source-recall daemon[/bold]")
    err_console.print(f"  Config: {config_path}")
    err_console.print(f"  Repos:  {len(cfg.repos)}")

    if no_embed:
        err_console.print("  Mode:   FTS-only")
        daemon_fapp = create_daemon_app(cfg, embedder=None)
    else:
        err_console.print("  Mode:   full (loading model...)")
        daemon_fapp = create_daemon_app(cfg)

    err_console.print(f"  Listen:  [cyan]http://{cfg.host}:{cfg.port}[/cyan]")
    err_console.print()

    uvicorn_run(
        daemon_fapp,
        host=cfg.host,
        port=cfg.port,
        log_level="warning",
        timeout_graceful_shutdown=cfg.shutdown_timeout_s,
    )


def _write_and_load_plist(config: object) -> Path:
    """Write plist and bootstrap via launchctl. Mockable in tests.

    @param config: DaemonConfig instance.
    @returns: Path to the plist file.
    """
    from source_recall.launchd import install_plist

    return install_plist(config)  # type: ignore[arg-type]


def _wait_for_health(url: str, timeout: float = 30.0) -> bool:
    """Poll /health until ok or timeout.

    @param url: Daemon base URL.
    @param timeout: Max seconds to wait.
    @returns: True if healthy within timeout.
    """
    import time

    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=2, headers=_daemon_headers())
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@daemon_app.command("start")
def daemon_start(
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to repos.toml.",
    ),
) -> None:
    """Start the daemon via launchd (background).

    Writes a launchd plist and bootstraps the service.
    The daemon auto-restarts on crash or reboot.

    Examples:\n
      sr daemon start\n
      sr daemon start --config ./repos.toml
    """
    from source_recall.daemon_config import DaemonConfig

    config_path = Path(config) if config else DaemonConfig.default_config_path()

    try:
        cfg = DaemonConfig.from_toml(config_path)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    err_console.print("[bold]Starting daemon via launchd...[/bold]")

    try:
        plist_path = _write_and_load_plist(cfg)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] Failed to load plist: {e}")
        raise typer.Exit(1) from e

    url = f"http://{cfg.host}:{cfg.port}"
    err_console.print(f"  Plist: {plist_path}")
    err_console.print(f"  Waiting for health at {url}...")

    if _wait_for_health(url):
        console.print("[green]✓[/green] Daemon started.")
    else:
        err_console.print(
            "[yellow]⚠[/yellow] Daemon launched but health check timed out. "
            "Check logs: sr daemon logs"
        )


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the daemon (unload launchd plist).

    Examples:\n
      sr daemon stop
    """
    from source_recall.launchd import unload_plist

    err_console.print("Stopping daemon...")
    try:
        unload_plist()
        console.print("[green]✓[/green] Daemon stopped.")
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@daemon_app.command("status")
def daemon_status() -> None:
    """Show daemon status and repo summary.

    Probes /health on the running daemon. Falls back to
    launchctl state if daemon is unreachable.

    Examples:\n
      sr daemon status
    """
    try:
        resp = _daemon_get("/health")
        data = resp.json()  # type: ignore[union-attr]
        if data.get("ok"):
            console.print(
                f"[green]●[/green] Daemon running "
                f"(uptime {data['uptime_s']}s, {len(data['repos'])} repos)"
            )
            for name in data["repos"]:
                console.print(f"  • {name}")
        else:
            console.print("[yellow]●[/yellow] Daemon running but no repos ready")
    except ConnectionError:
        from source_recall.launchd import is_loaded

        if is_loaded():
            err_console.print(
                "[yellow]●[/yellow] Plist loaded but daemon not responding"
            )
        else:
            err_console.print("[red]●[/red] Daemon not running")
        raise typer.Exit(1) from None


@daemon_app.command("logs")
def daemon_logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show."),
) -> None:
    """Show daemon log output.

    Examples:\n
      sr daemon logs\n
      sr daemon logs -n 100
    """
    from source_recall.launchd import LOG_PATH

    if not LOG_PATH.exists():
        err_console.print("[dim]No log file found.[/dim]")
        return

    import subprocess

    result = subprocess.run(
        ["tail", f"-{lines}", str(LOG_PATH)],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        console.print(result.stdout, end="")


# ---------------------------------------------------------------------------
# Daemon HTTP client helpers
# ---------------------------------------------------------------------------


def _daemon_get(path: str) -> object:
    """GET request to the daemon.

    @param path: URL path (e.g. '/repos').
    @returns: httpx.Response.
    @raises ConnectionError: If daemon is unreachable.
    """
    import httpx

    try:
        return httpx.get(
            f"{_daemon_url()}{path}", timeout=10, headers=_daemon_headers()
        )
    except httpx.ConnectError as e:
        raise ConnectionError(f"Daemon not running at {_daemon_url()}") from e


def _daemon_post(path: str, **kwargs: object) -> object:
    """POST request to the daemon.

    @param path: URL path.
    @param kwargs: Forwarded to httpx.post.
    @returns: httpx.Response.
    @raises ConnectionError: If daemon is unreachable.
    """
    import httpx

    headers = {**_daemon_headers(), **(kwargs.pop("headers", None) or {})}  # type: ignore[dict-item]
    try:
        return httpx.post(
            f"{_daemon_url()}{path}", timeout=10, headers=headers, **kwargs
        )  # type: ignore[arg-type]
    except httpx.ConnectError as e:
        raise ConnectionError(f"Daemon not running at {_daemon_url()}") from e


def _daemon_delete(path: str) -> object:
    """DELETE request to the daemon.

    @param path: URL path.
    @returns: httpx.Response.
    @raises ConnectionError: If daemon is unreachable.
    """
    import httpx

    try:
        return httpx.delete(
            f"{_daemon_url()}{path}", timeout=10, headers=_daemon_headers()
        )
    except httpx.ConnectError as e:
        raise ConnectionError(f"Daemon not running at {_daemon_url()}") from e


# ---------------------------------------------------------------------------
# sr add / sr remove / sr repos
# ---------------------------------------------------------------------------


@app.command("add")
def add_repo(
    path: str = typer.Argument(..., help="Path to the repository to add."),
    name: str = typer.Option(None, "--name", "-n", help="Custom display name."),
) -> None:
    """Add a repository to the daemon.

    Sends POST /repos to the running daemon. The repo will be
    queued for indexing immediately.

    Examples:\n
      sr add ~/dev/myproject\n
      sr add ~/dev/myproject --name my-proj
    """
    payload: dict[str, str] = {"path": str(Path(path).resolve())}
    if name:
        payload["name"] = name

    try:
        resp = _daemon_post("/repos", json=payload)
    except ConnectionError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if resp.status_code == 201:  # type: ignore[union-attr]
        data = resp.json()  # type: ignore[union-attr]
        console.print(
            f"[green]✓[/green] Added [cyan]{data['name']}[/cyan] ({data['state']})"
        )
    else:
        detail = resp.json().get("detail", "Unknown error")  # type: ignore[union-attr]
        err_console.print(f"[red]Error:[/red] {detail}")
        raise typer.Exit(1)


@app.command("remove")
def remove_repo(
    name: str = typer.Argument(..., help="Name of the repo to remove."),
) -> None:
    """Remove a repository from the daemon.

    Sends DELETE /repos/{name} to the running daemon.

    Examples:\n
      sr remove myproject
    """
    try:
        resp = _daemon_delete(f"/repos/{name}")
    except ConnectionError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if resp.status_code == 200:  # type: ignore[union-attr]
        console.print(f"[green]✓[/green] Removed [cyan]{name}[/cyan]")
    else:
        detail = resp.json().get("detail", "Unknown error")  # type: ignore[union-attr]
        err_console.print(f"[red]Error:[/red] {detail}")
        raise typer.Exit(1)


@app.command("repos")
def list_daemon_repos() -> None:
    """List repos registered in the daemon.

    Queries GET /repos on the running daemon and displays
    each repo with its current state.

    Examples:\n
      sr repos
    """
    try:
        resp = _daemon_get("/repos")
    except ConnectionError:
        err_console.print("[red]Error:[/red] Daemon not running.")
        raise typer.Exit(1) from None

    if resp.status_code != 200:  # type: ignore[union-attr]
        err_console.print(f"[red]Error:[/red] Unexpected status {resp.status_code}")  # type: ignore[union-attr]
        raise typer.Exit(1)

    data = resp.json()  # type: ignore[union-attr]
    repos_list = data.get("repos", [])

    if not repos_list:
        console.print("[dim]No repos registered.[/dim]")
        return

    from rich.table import Table

    table = Table(title="Daemon Repos", show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("State", justify="center")
    table.add_column("Path", style="dim")

    state_icons = {
        "ready": "[green]✓[/green]",
        "indexing": "[yellow]⟳[/yellow]",
        "queued": "[dim]…[/dim]",
        "error": "[red]✗[/red]",
    }

    for r in repos_list:
        icon = state_icons.get(r["state"], r["state"])
        table.add_row(r["name"], icon, r["path"])

    console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_to_dict(s: IndexStatus) -> dict[str, object]:
    """Convert an IndexStatus to a JSON-serializable dict.

    Adds computed fields like vector_coverage_pct and human-readable size.

    @param s: IndexStatus instance.
    @returns: Dict with all status fields plus computed helpers.
    """
    vec_pct = round(s.vector_count / s.chunk_count * 100, 1) if s.chunk_count > 0 else 0
    return {
        "repo_path": s.repo_path,
        "db_path": s.db_path,
        "db_size_bytes": s.db_size_bytes,
        "db_size_human": _format_bytes(s.db_size_bytes),
        "indexed_at": s.indexed_at,
        "last_commit": s.last_commit,
        "file_count": s.file_count,
        "chunk_count": s.chunk_count,
        "ast_files": s.ast_files,
        "regex_files": s.regex_files,
        "text_fallback_files": s.text_fallback_files,
        "vector_count": s.vector_count,
        "vector_coverage_pct": vec_pct,
        "embed_model": s.embed_model,
        "embed_dimensions": s.embed_dimensions,
    }


def _format_bytes(n: int | float) -> str:
    """Format bytes as human-readable size.

    @param n: Number of bytes (int from stat, float during conversion).
    @returns: Formatted string (e.g. '12.4 MB').
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
