"""Import/export subcommands."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from wisdom.cli.formatters import console

io_app = typer.Typer(help="Import and export wisdom data")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@io_app.command("export")
def export_cmd(
    output: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("wisdom-pack", "--format", "-f", help="Format: wisdom-pack or claude-md"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Min confidence for export"),
):
    """Export wisdom data."""
    system = _get_system()
    try:
        if format == "claude-md":
            from wisdom.llm.injection import generate_claude_md
            wisdom_entries = system.wisdom.list(domain=domain, limit=10000)
            content = generate_claude_md(wisdom_entries, domain=domain, min_confidence=min_confidence)
            Path(output).write_text(content, encoding="utf-8")
            console.print(f"[green]Exported CLAUDE.md to:[/] {output}")
            return

        # wisdom-pack format
        experiences = system.experiences.list(domain=domain, limit=100000)
        knowledge = system.knowledge.list(domain=domain, limit=100000)
        wisdom_entries = system.wisdom.list(domain=domain, limit=100000)

        # Filter by confidence if specified
        if min_confidence > 0:
            wisdom_entries = [w for w in wisdom_entries if w.confidence.overall >= min_confidence]
            knowledge = [k for k in knowledge if k.confidence.overall >= min_confidence]

        # Gather all domains
        domains = set()
        for e in experiences:
            if e.domain:
                domains.add(e.domain)
        for k in knowledge:
            if k.domain:
                domains.add(k.domain)
        for w in wisdom_entries:
            domains.update(w.applicable_domains)

        pack = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "experiences": [e.model_dump(mode="json") for e in experiences],
            "knowledge": [k.model_dump(mode="json") for k in knowledge],
            "wisdom": [w.model_dump(mode="json") for w in wisdom_entries],
            "metadata": {
                "source_system": "wisdom-system-v0.1",
                "total_entries": len(experiences) + len(knowledge) + len(wisdom_entries),
                "domains": sorted(domains),
            },
        }

        Path(output).write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Exported wisdom pack to:[/] {output}")
        console.print(f"  Experiences: {len(experiences)}")
        console.print(f"  Knowledge: {len(knowledge)}")
        console.print(f"  Wisdom: {len(wisdom_entries)}")
    finally:
        system.close()


@io_app.command("import")
def import_cmd(
    input_file: str = typer.Argument(..., help="Input file path"),
    mode: str = typer.Option("merge", "--mode", "-m", help="Import mode: merge or replace"),
):
    """Import wisdom data from a pack file."""
    system = _get_system()
    try:
        path = Path(input_file)
        if not path.exists():
            console.print(f"[red]File not found:[/] {input_file}")
            raise typer.Exit(1)

        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != "1.0":
            console.print(f"[yellow]Warning: Unknown pack version {data.get('version')}[/]")

        if mode == "replace":
            # Clear everything first
            console.print("[yellow]Replacing all data...[/]")
            for exp in system.experiences.list(limit=100000):
                system.experiences.delete(exp.id)
            for k in system.knowledge.list(limit=100000):
                system.knowledge.delete(k.id)
            for w in system.wisdom.list(limit=100000):
                system.wisdom.delete(w.id)

        # Import experiences
        from wisdom.models.experience import Experience
        from wisdom.models.knowledge import Knowledge
        from wisdom.models.wisdom import Wisdom

        exp_count = 0
        for exp_data in data.get("experiences", []):
            exp = Experience(**exp_data)
            existing = system.experiences.get(exp.id)
            if mode == "merge" and existing:
                continue  # Skip duplicates
            system.sqlite.save_experience(exp)
            system.vector.add("experience", exp.id, exp.embedding_text, {"domain": exp.domain})
            exp_count += 1

        know_count = 0
        for k_data in data.get("knowledge", []):
            k = Knowledge(**k_data)
            existing = system.knowledge.get(k.id)
            if mode == "merge" and existing:
                continue
            system.sqlite.save_knowledge(k)
            system.vector.add("knowledge", k.id, k.embedding_text, {"domain": k.domain})
            know_count += 1

        wis_count = 0
        for w_data in data.get("wisdom", []):
            w = Wisdom(**w_data)
            existing = system.wisdom.get(w.id)
            if mode == "merge" and existing:
                continue
            system.sqlite.save_wisdom(w)
            system.vector.add("wisdom", w.id, w.embedding_text, {"domains": ",".join(w.applicable_domains)})
            wis_count += 1

        console.print(f"[green]Import complete ({mode} mode):[/]")
        console.print(f"  Experiences: {exp_count}")
        console.print(f"  Knowledge: {know_count}")
        console.print(f"  Wisdom: {wis_count}")
    finally:
        system.close()


@io_app.command("claude-md")
def claude_md(
    output: str = typer.Option("./CLAUDE.md", "--output", "-o", help="Output path"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    min_confidence: float = typer.Option(0.5, "--min-confidence"),
):
    """Generate a CLAUDE.md file from high-confidence wisdom."""
    from wisdom.llm.injection import generate_claude_md

    system = _get_system()
    try:
        wisdom_entries = system.wisdom.list(domain=domain, limit=10000)
        content = generate_claude_md(wisdom_entries, domain=domain, min_confidence=min_confidence)
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Generated CLAUDE.md:[/] {output}")

        # Count included entries
        included = [
            w for w in wisdom_entries
            if w.confidence.overall >= min_confidence and w.lifecycle.value != "deprecated"
        ]
        console.print(f"  Included {len(included)} wisdom entries (confidence >= {min_confidence})")
    finally:
        system.close()
