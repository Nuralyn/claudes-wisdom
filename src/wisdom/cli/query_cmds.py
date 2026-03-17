"""Query subcommands — unified search and composition."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from wisdom.cli.formatters import console, search_results_table

query_app = typer.Typer(help="Query across all wisdom layers")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@query_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    layers: Optional[str] = typer.Option(None, "--layers", "-l", help="Comma-separated: wisdom,knowledge"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum effective confidence"),
    include_deprecated: bool = typer.Option(False, "--include-deprecated"),
):
    """Unified search across wisdom and knowledge layers."""
    system = _get_system()
    try:
        layer_list = layers.split(",") if layers else None
        results = system.retrieval.search(
            query=query,
            domain=domain,
            top_k=top_k,
            layers=layer_list,
            min_confidence=min_confidence,
            include_deprecated=include_deprecated,
        )
        if not results:
            console.print("[dim]No results found.[/]")
            return
        console.print(search_results_table(
            [r.to_dict() for r in results],
            title=f"Search: '{query}'",
        ))
    finally:
        system.close()


@query_app.command("for-task")
def for_task(
    task: str = typer.Argument(..., help="Task description"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Search for wisdom relevant to a specific task."""
    system = _get_system()
    try:
        results = system.retrieval.search_for_task(task, domain=domain, top_k=top_k)
        if not results:
            console.print("[dim]No applicable wisdom found for this task.[/]")
            return
        console.print(search_results_table(
            [r.to_dict() for r in results],
            title=f"Wisdom for: '{task[:50]}'",
        ))

        # Also show composed guidance
        composition = system.retrieval.compose_wisdom(task, domain=domain, top_k=top_k)
        if composition.get("composition"):
            console.print(Panel(
                composition["composition"],
                title="Composed Guidance",
                border_style="green",
            ))
    finally:
        system.close()


@query_app.command()
def conflicts(
    wisdom_id: Optional[str] = typer.Option(None, "--id", help="Check conflicts for specific wisdom ID"),
):
    """Find conflicting wisdom entries."""
    system = _get_system()
    try:
        results = system.retrieval.find_contradictions(wisdom_id)
        if not results:
            console.print("[dim]No conflicts found.[/]")
            return
        console.print(f"[yellow]Found {len(results)} conflict(s):[/]")
        for r in results:
            if "a_id" in r:
                console.print(f"\n  [cyan]{r['a_id']}:[/] {r.get('a_statement', '')[:60]}")
                console.print(f"  [red]conflicts with[/]")
                console.print(f"  [cyan]{r['b_id']}:[/] {r.get('b_statement', '')[:60]}")
            else:
                console.print(f"\n  [cyan]{r.get('wisdom_id', '')}[/] conflicts with [cyan]{r.get('conflicting_id', '')}[/]")
                console.print(f"    {r.get('conflicting_statement', '')[:60]}")
    finally:
        system.close()
