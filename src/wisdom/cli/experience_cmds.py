"""Experience subcommands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from wisdom.cli.formatters import console, experience_panel, experience_table
from wisdom.models.common import ExperienceResult, ExperienceType

exp_app = typer.Typer(help="Manage experiences (raw interactions)")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@exp_app.command()
def add(
    description: str = typer.Argument(..., help="Description of the experience"),
    domain: str = typer.Option("", "--domain", "-d", help="Domain (e.g., python, devops)"),
    subdomain: str = typer.Option("", "--subdomain", help="Subdomain"),
    task_type: str = typer.Option("", "--task-type", "-t", help="Task type"),
    input_text: str = typer.Option("", "--input", "-i", help="Input text/context"),
    output_text: str = typer.Option("", "--output", "-o", help="Output/result text"),
    result: str = typer.Option("success", "--result", "-r", help="Result: success/partial/failure/error"),
    quality: float = typer.Option(0.5, "--quality", "-q", help="Quality score 0-1"),
    exp_type: str = typer.Option("task", "--type", help="Type: task/conversation/debugging/review/learning/other"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
):
    """Add a new experience."""
    system = _get_system()
    try:
        exp = system.experiences.add(
            description=description,
            domain=domain,
            subdomain=subdomain,
            task_type=task_type,
            input_text=input_text,
            output_text=output_text,
            result=ExperienceResult(result),
            quality_score=quality,
            exp_type=ExperienceType(exp_type),
            tags=tags.split(",") if tags else [],
        )
        console.print(f"[green]Added experience:[/] {exp.id}")
    finally:
        system.close()


@exp_app.command("list")
def list_cmd(
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    limit: int = typer.Option(20, "--limit", "-n"),
    offset: int = typer.Option(0, "--offset"),
):
    """List experiences."""
    system = _get_system()
    try:
        exps = system.experiences.list(domain=domain, limit=limit, offset=offset)
        if not exps:
            console.print("[dim]No experiences found.[/]")
            return
        console.print(experience_table(exps))
    finally:
        system.close()


@exp_app.command()
def show(id: str = typer.Argument(..., help="Experience ID")):
    """Show details of an experience."""
    system = _get_system()
    try:
        exp = system.experiences.get(id)
        if not exp:
            console.print(f"[red]Experience not found:[/] {id}")
            raise typer.Exit(1)
        console.print(experience_panel(exp))
    finally:
        system.close()


@exp_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
):
    """Semantic search over experiences."""
    system = _get_system()
    try:
        results = system.experiences.search(query=query, top_k=top_k, domain=domain)
        if not results:
            console.print("[dim]No results found.[/]")
            return
        exps = [r["experience"] for r in results]
        console.print(experience_table(exps, title=f"Search: '{query}'"))
        for r in results:
            console.print(f"  [dim]{r['experience'].id}: similarity={r['similarity']:.3f}[/]")
    finally:
        system.close()


@exp_app.command()
def delete(id: str = typer.Argument(..., help="Experience ID to delete")):
    """Delete an experience."""
    system = _get_system()
    try:
        if system.experiences.delete(id):
            console.print(f"[green]Deleted experience:[/] {id}")
        else:
            console.print(f"[red]Experience not found:[/] {id}")
    finally:
        system.close()


@exp_app.command()
def stats():
    """Show experience statistics."""
    system = _get_system()
    try:
        s = system.experiences.stats()
        console.print(f"[bold]Total:[/] {s['total']}")
        console.print(f"[bold]Processed:[/] {s['processed']}")
        console.print(f"[bold]Unprocessed:[/] {s['unprocessed']}")
        if s['domains']:
            console.print("[bold]By domain:[/]")
            for d, c in s['domains'].items():
                console.print(f"  {d}: {c}")
    finally:
        system.close()
