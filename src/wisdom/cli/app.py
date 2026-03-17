"""Main CLI application — typer app with subcommand registration."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from wisdom.cli.analytics_cmds import analytics_app
from wisdom.cli.experience_cmds import exp_app
from wisdom.cli.formatters import console
from wisdom.cli.io_cmds import io_app
from wisdom.cli.knowledge_cmds import know_app
from wisdom.cli.query_cmds import query_app
from wisdom.cli.wisdom_cmds import wis_app

app = typer.Typer(
    name="wisdom",
    help="CLI Wisdom System — accumulate, retain, and apply wisdom through the DIKW hierarchy.",
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(exp_app, name="exp", help="Manage experiences")
app.add_typer(know_app, name="know", help="Manage knowledge")
app.add_typer(wis_app, name="wis", help="Manage wisdom")
app.add_typer(query_app, name="query", help="Query across layers")
app.add_typer(analytics_app, name="analytics", help="Analytics and health")
app.add_typer(io_app, name="io", help="Import/export data")


@app.command()
def init(
    seed: Optional[str] = typer.Option(
        None, "--seed", "-s",
        help="Seed pack to load: software_engineering, debugging, communication, meta, or 'all'",
    ),
):
    """Initialize the wisdom system data directory and optionally load seeds."""
    from wisdom import WisdomSystem

    system = WisdomSystem()
    try:
        console.print(f"[green]Data directory:[/] {system.config.data_dir}")
        console.print(f"[green]SQLite:[/] {system.config.sqlite_path}")
        console.print(f"[green]ChromaDB:[/] {system.config.chroma_path}")

        # Warmup embedding model
        console.print("Warming up embedding model...")
        system.warmup()
        console.print("[green]Embedding model ready.[/]")

        # Load seeds if requested
        if seed:
            from wisdom.seeds import load_all_seeds, load_seed_pack

            if seed == "all":
                entries = load_all_seeds()
            else:
                entries = load_seed_pack(seed)

            if not entries:
                console.print(f"[yellow]No entries found in seed pack '{seed}'.[/]")
            else:
                loaded = 0
                for w in entries:
                    system.wisdom.add(
                        statement=w.statement,
                        reasoning=w.reasoning,
                        wisdom_type=w.type,
                        domains=w.applicable_domains,
                        applicability_conditions=w.applicability_conditions,
                        inapplicability_conditions=w.inapplicability_conditions,
                        trade_offs=w.trade_offs,
                        implications=w.implications,
                        counterexamples=w.counterexamples,
                        confidence=w.confidence,
                        creation_method=w.creation_method,
                        tags=w.tags,
                    )
                    loaded += 1
                console.print(f"[green]Loaded {loaded} wisdom entries from seed pack '{seed}'.[/]")

        stats = system.stats()
        console.print(f"\n[bold]Current state:[/]")
        console.print(f"  Experiences: {stats['experiences']}")
        console.print(f"  Knowledge: {stats['knowledge']}")
        console.print(f"  Wisdom: {stats['wisdom']}")
        console.print(f"\n[green]Wisdom system initialized successfully.[/]")
    finally:
        system.close()


@app.command()
def maintenance():
    """Run auto-triggered maintenance (extraction, synthesis, deprecation)."""
    from wisdom import WisdomSystem

    system = WisdomSystem()
    try:
        console.print("Running maintenance...")
        summary = system.run_maintenance()

        if summary["extracted"]:
            for r in summary["extracted"]:
                console.print(f"  [green]Extracted:[/] {r['knowledge_created']} knowledge from {r['experiences_processed']} experiences in {r['domain']}")
        if summary["synthesized"]:
            for r in summary["synthesized"]:
                console.print(f"  [green]Synthesized:[/] {r['wisdom_created']} wisdom from {r['knowledge_processed']} knowledge in {r['domain']}")
        if summary["deprecated"]:
            console.print(f"  [red]Deprecated:[/] {len(summary['deprecated'])} wisdom entries")
        if summary["validated"]:
            console.print(f"  [yellow]Needs validation:[/] {len(summary['validated'])} wisdom entries")

        if not any(summary.values()):
            console.print("[green]No maintenance needed.[/]")
    finally:
        system.close()


if __name__ == "__main__":
    app()
