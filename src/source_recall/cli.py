"""CLI: sr index, sr ask, sr status, sr clean, sr config show."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer(
    name="sr",
    help="source-recall: Code search and retrieval for AI coding tools.",
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
) -> None:
    """Build or rebuild the index for a repository."""
    from source_recall import Index

    repo_path = Path(path).resolve()

    def on_progress(file_path: str, current: int, total: int) -> None:
        """Progress callback for indexing."""
        err_console.print(
            f"  [{current}/{total}] {file_path}",
            highlight=False,
            end="\r",
        )

    try:
        idx = Index(repo_path, on_progress=on_progress)
        db_path = idx.build()
        status = idx.status()

        err_console.print()  # Clear progress line.
        err_console.print(
            f"[green]✓[/green] Indexed {status.file_count} files "
            f"({status.chunk_count} chunks) → {db_path}"
        )
        err_console.print(
            f"  AST: {status.ast_files}  "
            f"Regex: {status.regex_files}  "
            f"Text: {status.text_fallback_files}"
        )
        if status.vector_count > 0:
            pct = (
                status.vector_count / status.chunk_count * 100
                if status.chunk_count > 0
                else 0
            )
            err_console.print(
                f"  Vectors: {status.vector_count}/{status.chunk_count} "
                f"({pct:.0f}%) — {status.embed_model} ({status.embed_dimensions}d)"
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
) -> None:
    """Search the index for relevant code.

    Returns ranked code chunks matching the query. Use --json for
    structured output suitable for piping to other tools or LLMs.
    """
    from source_recall import Index

    repo_path = Path(path).resolve()

    try:
        idx = Index(repo_path)

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
                pass  # Refresh failure is non-fatal for queries.

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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show index status for a repository."""
    from source_recall import Index

    repo_path = Path(path).resolve()

    try:
        idx = Index(repo_path)
        s = idx.status()
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if json_output:
        data = {
            "repo_path": s.repo_path,
            "db_path": s.db_path,
            "db_size_bytes": s.db_size_bytes,
            "indexed_at": s.indexed_at,
            "last_commit": s.last_commit,
            "file_count": s.file_count,
            "chunk_count": s.chunk_count,
            "ast_files": s.ast_files,
            "regex_files": s.regex_files,
            "text_fallback_files": s.text_fallback_files,
            "vector_count": s.vector_count,
            "embed_model": s.embed_model,
            "embed_dimensions": s.embed_dimensions,
        }
        print(json.dumps(data, indent=2))
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
# sr clean
# ---------------------------------------------------------------------------


@app.command()
def clean(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be removed."
    ),
) -> None:
    """Remove orphaned indexes (repos that no longer exist)."""
    base = Path.home() / ".local" / "share" / "source-recall"
    if not base.exists():
        console.print("[dim]No indexes found.[/dim]")
        return

    removed = 0
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "index.db"
        if not db_path.exists():
            continue

        # Read repo_path from meta.
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'repo_path'"
            ).fetchone()
            conn.close()

            if row is None:
                continue
            repo_path = Path(row[0])
            if repo_path.exists():
                continue

            if dry_run:
                console.print(f"[yellow]Would remove:[/yellow] {d} → {row[0]}")
            else:
                shutil.rmtree(d)
                console.print(f"[green]Removed:[/green] {d} → {row[0]}")
            removed += 1

        except Exception:
            continue

    if removed == 0:
        console.print("[dim]No orphaned indexes found.[/dim]")
    elif not dry_run:
        console.print(f"\n[green]Cleaned {removed} orphaned index(es).[/green]")


# ---------------------------------------------------------------------------
# sr config show
# ---------------------------------------------------------------------------


@app.command("config")
def config_show(
    path: str = typer.Argument(".", help="Path to the repository root."),
) -> None:
    """Print the resolved configuration as TOML."""
    from source_recall.config import format_config, resolve_config

    repo_path = Path(path).resolve()
    config = resolve_config(repo_path)
    console.print(format_config(config))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
