"""Knowledge subcommands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from wisdom.cli.formatters import console, knowledge_panel, knowledge_table

know_app = typer.Typer(help="Manage knowledge (extracted patterns and rules)")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@know_app.command()
def extract(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain to extract from"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max experiences to process"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for extraction"),
):
    """Extract knowledge from unprocessed experiences."""
    system = _get_system()
    try:
        experiences = system.experiences.get_unprocessed(domain=domain, limit=limit)
        if not experiences:
            console.print("[dim]No unprocessed experiences found.[/]")
            return

        console.print(f"Processing {len(experiences)} experiences...")

        if use_llm:
            system.init_providers()
            if not system.providers.has_provider:
                console.print("[red]No LLM provider available. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or install Ollama.[/]")
                raise typer.Exit(1)
            from wisdom.llm.extraction import extract_knowledge
            provider = system.providers.get()
            knowledge_list = extract_knowledge(provider, experiences, domain=domain or "")
            # Save extracted knowledge
            for k in knowledge_list:
                system.knowledge.add(k)
            system.sqlite.mark_processed([e.id for e in experiences])
        else:
            knowledge_list = system.knowledge.extract_from_experiences(experiences, domain=domain or "")

        if knowledge_list:
            console.print(f"[green]Extracted {len(knowledge_list)} knowledge entries[/]")
            console.print(knowledge_table(knowledge_list))
        else:
            console.print("[dim]No knowledge patterns found.[/]")
    finally:
        system.close()


@know_app.command("list")
def list_cmd(
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    limit: int = typer.Option(20, "--limit", "-n"),
    offset: int = typer.Option(0, "--offset"),
):
    """List knowledge entries."""
    system = _get_system()
    try:
        entries = system.knowledge.list(domain=domain, limit=limit, offset=offset)
        if not entries:
            console.print("[dim]No knowledge entries found.[/]")
            return
        console.print(knowledge_table(entries))
    finally:
        system.close()


@know_app.command()
def show(id: str = typer.Argument(..., help="Knowledge ID")):
    """Show details of a knowledge entry."""
    system = _get_system()
    try:
        k = system.knowledge.get(id)
        if not k:
            console.print(f"[red]Knowledge not found:[/] {id}")
            raise typer.Exit(1)
        console.print(knowledge_panel(k))
    finally:
        system.close()


@know_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
):
    """Semantic search over knowledge."""
    system = _get_system()
    try:
        results = system.knowledge.search(query=query, top_k=top_k, domain=domain)
        if not results:
            console.print("[dim]No results found.[/]")
            return
        entries = [r["knowledge"] for r in results]
        console.print(knowledge_table(entries, title=f"Search: '{query}'"))
        for r in results:
            console.print(f"  [dim]{r['knowledge'].id}: similarity={r['similarity']:.3f}[/]")
    finally:
        system.close()


@know_app.command()
def validate(
    id: str = typer.Argument(..., help="Knowledge ID"),
    valid: bool = typer.Option(True, "--valid/--invalid", help="Mark as valid or challenged"),
    details: str = typer.Option("", "--details", help="Validation details"),
):
    """Validate or challenge a knowledge entry."""
    system = _get_system()
    try:
        k = system.knowledge.validate(id, is_valid=valid, details=details)
        if not k:
            console.print(f"[red]Knowledge not found:[/] {id}")
            raise typer.Exit(1)
        status = "validated" if valid else "challenged"
        console.print(f"[green]Knowledge {id} marked as {status}[/]")
    finally:
        system.close()


@know_app.command()
def delete(id: str = typer.Argument(..., help="Knowledge ID to delete")):
    """Delete a knowledge entry."""
    system = _get_system()
    try:
        if system.knowledge.delete(id):
            console.print(f"[green]Deleted knowledge:[/] {id}")
        else:
            console.print(f"[red]Knowledge not found:[/] {id}")
    finally:
        system.close()
