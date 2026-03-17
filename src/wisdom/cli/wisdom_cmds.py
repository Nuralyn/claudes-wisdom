"""Wisdom subcommands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from wisdom.cli.formatters import console, wisdom_panel, wisdom_table
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    RelationshipType,
    TradeOff,
    WisdomType,
)

wis_app = typer.Typer(help="Manage wisdom (higher-order principles)")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@wis_app.command()
def add(
    statement: str = typer.Argument(..., help="Wisdom statement"),
    reasoning: str = typer.Option("", "--reasoning", "-r", help="Reasoning behind this wisdom"),
    wisdom_type: str = typer.Option("principle", "--type", "-t", help="Type: principle/heuristic/judgment_rule/meta_pattern/trade_off"),
    domains: Optional[str] = typer.Option(None, "--domains", "-d", help="Comma-separated domains"),
    conditions: Optional[str] = typer.Option(None, "--conditions", help="Comma-separated applicability conditions"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
):
    """Add wisdom directly (human expert path)."""
    system = _get_system()
    try:
        w = system.wisdom.add(
            statement=statement,
            reasoning=reasoning,
            wisdom_type=WisdomType(wisdom_type),
            domains=domains.split(",") if domains else [],
            applicability_conditions=conditions.split(",") if conditions else [],
            creation_method=CreationMethod.HUMAN_INPUT,
            tags=tags.split(",") if tags else [],
        )
        console.print(f"[green]Added wisdom:[/] {w.id}")
        console.print(f"  Statement: {w.statement[:80]}")
        console.print(f"  Lifecycle: {w.lifecycle.value}")
        console.print(f"  Confidence: {w.confidence.overall:.2f}")
    finally:
        system.close()


@wis_app.command()
def synthesize(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain to synthesize from"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max knowledge entries to process"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for synthesis"),
):
    """Synthesize wisdom from unsynthesized knowledge."""
    system = _get_system()
    try:
        knowledge_entries = system.knowledge.get_unsynthesized(domain=domain, limit=limit)
        if not knowledge_entries:
            console.print("[dim]No unsynthesized knowledge found.[/]")
            return

        console.print(f"Processing {len(knowledge_entries)} knowledge entries...")

        if use_llm:
            system.init_providers()
            if not system.providers.has_provider:
                console.print("[red]No LLM provider available.[/]")
                raise typer.Exit(1)
            from wisdom.llm.synthesis import synthesize_wisdom
            provider = system.providers.get()
            existing = system.wisdom.list(domain=domain)
            wisdom_list, contradictions = synthesize_wisdom(
                provider, knowledge_entries, existing_wisdom=existing, domain=domain or "",
            )
            for w in wisdom_list:
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
                    creation_method=CreationMethod.PIPELINE,
                    tags=w.tags,
                    source_knowledge_ids=w.source_knowledge_ids,
                )
            system.sqlite.mark_synthesized([k.id for k in knowledge_entries])
            if contradictions:
                console.print(f"\n[yellow]Detected {len(contradictions)} contradiction(s):[/]")
                for c in contradictions:
                    console.print(f"  - {c.get('description', 'Unknown conflict')}")
        else:
            wisdom_list = system.wisdom.synthesize_from_knowledge(knowledge_entries, domain=domain or "")

        if wisdom_list:
            console.print(f"[green]Synthesized {len(wisdom_list)} wisdom entries[/]")
            console.print(wisdom_table(wisdom_list))
        else:
            console.print("[dim]No wisdom synthesized.[/]")
    finally:
        system.close()


@wis_app.command("list")
def list_cmd(
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    lifecycle: Optional[str] = typer.Option(None, "--lifecycle", "-l"),
    limit: int = typer.Option(20, "--limit", "-n"),
    offset: int = typer.Option(0, "--offset"),
):
    """List wisdom entries."""
    system = _get_system()
    try:
        entries = system.wisdom.list(domain=domain, lifecycle=lifecycle, limit=limit, offset=offset)
        if not entries:
            console.print("[dim]No wisdom entries found.[/]")
            return
        console.print(wisdom_table(entries))
    finally:
        system.close()


@wis_app.command()
def show(id: str = typer.Argument(..., help="Wisdom ID")):
    """Show full details of a wisdom entry."""
    system = _get_system()
    try:
        w = system.wisdom.get(id)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)
        console.print(wisdom_panel(w))

        # Show confidence history
        history = system.evolution.get_confidence_history(id)
        if history:
            console.print(f"\n[bold]Recent Confidence History:[/]")
            for h in history[:10]:
                console.print(
                    f"  {h['timestamp'][:19]} | {h['old_confidence']:.3f} -> {h['new_confidence']:.3f} | {h['reason']}"
                )
    finally:
        system.close()


@wis_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
):
    """Semantic search over wisdom."""
    system = _get_system()
    try:
        results = system.wisdom.search(query=query, top_k=top_k, domain=domain)
        if not results:
            console.print("[dim]No results found.[/]")
            return
        entries = [r["wisdom"] for r in results]
        console.print(wisdom_table(entries, title=f"Search: '{query}'"))
        for r in results:
            console.print(f"  [dim]{r['wisdom'].id}: similarity={r['similarity']:.3f}[/]")
    finally:
        system.close()


@wis_app.command()
def reinforce(
    id: str = typer.Argument(..., help="Wisdom ID"),
    helpful: bool = typer.Option(True, "--helpful/--not-helpful", help="Was this wisdom helpful?"),
    feedback: str = typer.Option("", "--feedback", "-f", help="Optional feedback text"),
    context: str = typer.Option("", "--context", "-c", help="Task context for the application"),
):
    """Reinforce a wisdom entry with feedback."""
    system = _get_system()
    try:
        w = system.evolution.reinforce(id, was_helpful=helpful, feedback=feedback, task_context=context)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)
        status = "positive" if helpful else "negative"
        console.print(f"[green]Reinforced ({status}):[/] {id}")
        console.print(f"  Confidence: {w.confidence.overall:.3f}")
        console.print(f"  Lifecycle: {w.lifecycle.value}")
        console.print(f"  Applications: {w.application_count}")
    finally:
        system.close()


@wis_app.command()
def challenge(
    id: str = typer.Argument(..., help="Wisdom ID"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for challenging"),
):
    """Challenge a wisdom entry."""
    system = _get_system()
    try:
        w = system.wisdom.challenge(id, reason=reason)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)
        console.print(f"[yellow]Challenged:[/] {id}")
        console.print(f"  Lifecycle: {w.lifecycle.value}")
    finally:
        system.close()


@wis_app.command()
def deprecate(
    id: str = typer.Argument(..., help="Wisdom ID"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for deprecation"),
):
    """Deprecate a wisdom entry."""
    system = _get_system()
    try:
        w = system.wisdom.deprecate(id, reason=reason)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)
        console.print(f"[red]Deprecated:[/] {id}")
        console.print(f"  Reason: {reason}")
    finally:
        system.close()


@wis_app.command()
def relate(
    source: str = typer.Argument(..., help="Source wisdom ID"),
    target: str = typer.Argument(..., help="Target wisdom ID"),
    rel_type: str = typer.Option("complements", "--type", "-t",
        help="Relationship: generalizes/specializes/complements/conflicts/supports/derived_from"),
    strength: float = typer.Option(0.5, "--strength", "-s", help="Relationship strength 0-1"),
):
    """Create a relationship between two wisdom entries."""
    system = _get_system()
    try:
        rel = system.wisdom.relate(source, target, RelationshipType(rel_type), strength)
        if not rel:
            console.print("[red]One or both wisdom entries not found.[/]")
            raise typer.Exit(1)
        console.print(f"[green]Related:[/] {source} -[{rel_type}]-> {target} (strength: {strength})")
    finally:
        system.close()


@wis_app.command()
def transfer(
    id: str = typer.Argument(..., help="Wisdom ID to transfer"),
    to_domain: str = typer.Option(..., "--to", help="Target domain"),
):
    """Transfer wisdom to a new domain."""
    system = _get_system()
    try:
        w = system.wisdom.transfer(id, to_domain)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)
        console.print(f"[green]Transferred to '{to_domain}':[/] {w.id}")
        console.print(f"  Statement: {w.statement[:80]}")
        console.print(f"  Confidence: {w.confidence.overall:.2f}")
    finally:
        system.close()


@wis_app.command("validate")
def validate_cmd(
    id: str = typer.Argument(..., help="Wisdom ID"),
    source: str = typer.Option("external", "--source", "-s",
        help="Validation source: self_report/peer/external/adversarial"),
    verdict: str = typer.Option("confirmed", "--verdict", "-v",
        help="Verdict: confirmed/confirmed_with_caveats/challenged/refuted"),
    evidence: str = typer.Option("", "--evidence", "-e", help="Supporting evidence"),
    validator: str = typer.Option("", "--validator", help="Who is validating"),
):
    """Record an external validation event for a wisdom entry."""
    system = _get_system()
    try:
        result = system.validation.validate(
            id, source=source, verdict=verdict, evidence=evidence, validator=validator,
        )
        if "error" in result:
            console.print(f"[red]{result['error']}[/]")
            raise typer.Exit(1)
        console.print(f"[green]Validated:[/] {id}")
        console.print(f"  Source: {source}, Verdict: {verdict}")
        console.print(f"  Validation Score: {result['validation_score']:.2f}")
        console.print(f"  Confidence: {result['confidence']:.3f}")
    finally:
        system.close()


@wis_app.command("validation-summary")
def validation_summary(id: str = typer.Argument(..., help="Wisdom ID")):
    """Show validation history and score for a wisdom entry."""
    system = _get_system()
    try:
        summary = system.validation.validation_summary(id)
        console.print(f"[bold]Validation Summary for {id}[/]")
        console.print(f"  Total events: {summary['total_events']}")
        console.print(f"  Validation score: {summary['validation_score']:.2f}")
        console.print(f"  Is validated: {'Yes' if summary['is_validated'] else '[red]No[/]'}")
        if summary['by_source']:
            console.print("  By source:")
            for src, count in summary['by_source'].items():
                console.print(f"    {src}: {count}")
        if summary['events']:
            console.print("\n  [bold]Recent events:[/]")
            for e in summary['events'][:10]:
                console.print(f"    {e['timestamp'][:19]} | {e['source']} | {e['verdict']} | {e.get('evidence', '')[:40]}")
    finally:
        system.close()


@wis_app.command("devil-advocate")
def devil_advocate(id: str = typer.Argument(..., help="Wisdom ID to challenge")):
    """Run the adversarial challenge battery against a wisdom entry."""
    system = _get_system()
    try:
        w = system.wisdom.get(id)
        if not w:
            console.print(f"[red]Wisdom not found:[/] {id}")
            raise typer.Exit(1)

        console.print(f"[bold]Running adversarial challenge for:[/] {w.statement[:60]}...")
        report = system.adversarial.challenge(w)

        status_color = "green" if report.passed else "red"
        console.print(f"\n[{status_color}]{'PASSED' if report.passed else 'FAILED'}[/]: {report.summary}")

        if report.findings:
            console.print(f"\n[bold]Findings ({len(report.findings)}):[/]")
            for f in report.findings:
                sev_color = {"critical": "red", "warning": "yellow", "info": "dim"}.get(f.severity, "white")
                console.print(f"  [{sev_color}][{f.severity.upper()}][/] [{f.category}] {f.description}")
                if f.evidence:
                    console.print(f"    Evidence: {f.evidence[:100]}")

        if report.passed:
            # Record as adversarial validation
            system.validation.validate(
                id, source="adversarial", verdict="confirmed",
                evidence=report.summary, validator="adversarial_engine",
            )
            console.print(f"\n[green]Adversarial validation recorded.[/]")
    finally:
        system.close()


@wis_app.command("provenance")
def provenance(id: str = typer.Argument(..., help="Wisdom ID")):
    """Trace the full provenance chain of a wisdom entry."""
    system = _get_system()
    try:
        prov = system.propagation.trace_provenance(id)
        if "error" in prov:
            console.print(f"[red]{prov['error']}[/]")
            raise typer.Exit(1)

        w = prov["wisdom"]
        console.print(f"[bold]Provenance: {w['id']}[/]")
        console.print(f"  Statement: {w['statement']}")
        console.print(f"  Confidence: {w['confidence']:.3f} | Lifecycle: {w['lifecycle']}")
        console.print(f"  Created via: {w['creation_method']}")

        if prov["source_knowledge"]:
            console.print(f"\n[bold]Source Knowledge ({len(prov['source_knowledge'])}):[/]")
            for k in prov["source_knowledge"]:
                console.print(f"  {k['id']}: {k['statement']} (confidence: {k['confidence']:.2f})")
                if k["experiences"]:
                    for e in k["experiences"][:3]:
                        tag = "[red][CONTAMINATED][/]" if e["contaminated"] else ""
                        console.print(f"    <- {e['id']}: {e['description']} [{e['result']}] {tag}")

        if prov["applications"]:
            console.print(f"\n[bold]Applications ({len(prov['applications'])}):[/]")
            contaminated = sum(1 for a in prov["applications"] if a["contaminated"])
            console.print(f"  Total: {len(prov['applications'])}, Contaminated: {contaminated}")

        if prov["contamination_history"]:
            console.print(f"\n[bold]Contamination History ({len(prov['contamination_history'])}):[/]")
            for c in prov["contamination_history"][:5]:
                console.print(f"  {c['reason'][:60]}")
    finally:
        system.close()


@wis_app.command("cascade-failure")
def cascade_failure(
    id: str = typer.Argument(..., help="Wisdom ID that failed"),
    severity: float = typer.Option(1.0, "--severity", "-s", help="Failure severity 0-1"),
):
    """Cascade failure consequences through the provenance graph."""
    system = _get_system()
    try:
        result = system.propagation.cascade_failure(id, severity=severity)
        console.print(f"[red]Failure cascade from wisdom {id}:[/]")
        console.print(f"  Total affected: {result.total_affected}")
        if result.affected_wisdom:
            console.print(f"  Wisdom penalized: {len(result.affected_wisdom)}")
            for aw in result.affected_wisdom:
                console.print(f"    {aw['id']}: penalty={aw['penalty']:.4f}, new_conf={aw['new_confidence']:.3f}")
        if result.affected_knowledge:
            console.print(f"  Knowledge penalized: {len(result.affected_knowledge)}")
        if result.contaminated_experiences:
            console.print(f"  Experiences contaminated: {result.contaminated_experiences}")
    finally:
        system.close()
