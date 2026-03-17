"""Analytics subcommands — stats, health, gaps, audit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from wisdom.cli.formatters import console, gap_analysis_table, stats_panel

analytics_app = typer.Typer(help="Analytics and system health")


def _get_system():
    from wisdom import WisdomSystem
    return WisdomSystem()


@analytics_app.command()
def summary():
    """Show system-wide statistics."""
    system = _get_system()
    try:
        s = system.stats()
        console.print(stats_panel(s))
    finally:
        system.close()


@analytics_app.command()
def domains():
    """List all domains with counts."""
    system = _get_system()
    try:
        all_domains = system.sqlite.get_all_domains()
        if not all_domains:
            console.print("[dim]No domains found.[/]")
            return

        table = Table(title="Domains", show_lines=True)
        table.add_column("Domain", style="blue")
        table.add_column("Experiences", justify="right")
        table.add_column("Knowledge", justify="right")
        table.add_column("Wisdom", justify="right")

        for d in all_domains:
            exp_c = system.sqlite.count_experiences(domain=d)
            know_c = system.sqlite.count_knowledge(domain=d)
            wis_c = system.sqlite.count_wisdom(domain=d)
            table.add_row(d, str(exp_c), str(know_c), str(wis_c))

        console.print(table)
    finally:
        system.close()


@analytics_app.command()
def confidence():
    """Show confidence distribution across wisdom entries."""
    system = _get_system()
    try:
        all_wisdom = system.wisdom.list(limit=10000)
        if not all_wisdom:
            console.print("[dim]No wisdom entries.[/]")
            return

        # Bucket by confidence range
        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for w in all_wisdom:
            c = w.confidence.overall
            if c < 0.2:
                buckets["0.0-0.2"] += 1
            elif c < 0.4:
                buckets["0.2-0.4"] += 1
            elif c < 0.6:
                buckets["0.4-0.6"] += 1
            elif c < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        table = Table(title="Confidence Distribution")
        table.add_column("Range")
        table.add_column("Count", justify="right")
        table.add_column("Bar")

        max_count = max(buckets.values()) if buckets.values() else 1
        colors = ["red", "red", "yellow", "green", "green"]
        for (rng, cnt), color in zip(buckets.items(), colors):
            bar_len = int(30 * cnt / max_count) if max_count else 0
            bar = f"[{color}]{'█' * bar_len}[/]"
            table.add_row(rng, str(cnt), bar)

        console.print(table)

        # Lifecycle distribution
        lifecycle_counts = {}
        for w in all_wisdom:
            lc = w.lifecycle.value
            lifecycle_counts[lc] = lifecycle_counts.get(lc, 0) + 1

        console.print("\n[bold]Lifecycle Distribution:[/]")
        for lc, cnt in sorted(lifecycle_counts.items()):
            console.print(f"  {lc}: {cnt}")
    finally:
        system.close()


@analytics_app.command()
def health():
    """Overall system health check."""
    system = _get_system()
    try:
        stats = system.stats()
        issues = []

        # Check for unprocessed experiences
        unprocessed = system.sqlite.count_experiences(unprocessed=True)
        if unprocessed > 20:
            issues.append(f"[yellow]{unprocessed} unprocessed experiences — consider running extraction[/]")

        # Check for unsynthesized knowledge
        unsynthesized = system.sqlite.count_knowledge(unsynthesized=True)
        if unsynthesized > 10:
            issues.append(f"[yellow]{unsynthesized} unsynthesized knowledge entries — consider running synthesis[/]")

        # Check for challenged wisdom
        challenged = system.wisdom.list(lifecycle="challenged")
        if challenged:
            issues.append(f"[red]{len(challenged)} challenged wisdom entries need review[/]")

        # Check for stale domains
        stale = system.gaps.find_stale_domains()
        if stale:
            issues.append(f"[yellow]{len(stale)} stale domain(s): {', '.join(s['domain'] for s in stale)}[/]")

        console.print(stats_panel(stats))

        if issues:
            console.print("\n[bold]Health Issues:[/]")
            for issue in issues:
                console.print(f"  - {issue}")
        else:
            console.print("\n[green]System health: OK[/]")

        # Show triggers
        triggers = system.triggers.check_all()
        if triggers.has_actions:
            console.print("\n[bold]Pending Triggers:[/]")
            if triggers.should_extract:
                console.print(f"  - Extract: {', '.join(d or '(all)' for d in triggers.should_extract)}")
            if triggers.should_synthesize:
                console.print(f"  - Synthesize: {', '.join(d or '(all)' for d in triggers.should_synthesize)}")
            if triggers.should_validate:
                console.print(f"  - Validate: {len(triggers.should_validate)} wisdom entries")
    finally:
        system.close()


@analytics_app.command()
def gaps(domain: Optional[str] = typer.Option(None, "--domain", "-d")):
    """Show wisdom gaps and learning priorities."""
    system = _get_system()
    try:
        gap_data = system.gaps.summary(domain=domain)

        if gap_data["gaps"]:
            console.print(gap_analysis_table(gap_data["gaps"]))
        else:
            console.print("[green]No significant wisdom gaps found.[/]")

        if gap_data["low_coverage_tasks"]:
            console.print("\n[bold]Low Coverage Task Types:[/]")
            for t in gap_data["low_coverage_tasks"]:
                console.print(f"  - {t['task_type']}: {t['experience_count']} experiences, {t['wisdom_count']} wisdom")

        if gap_data["stale_domains"]:
            console.print("\n[bold]Stale Domains:[/]")
            for s in gap_data["stale_domains"]:
                console.print(f"  - {s['domain']}: {s['active_wisdom_count']} entries, avg age {s['avg_age_days']:.0f} days")

        if gap_data["extraction_suggestions"]:
            console.print("\n[bold]Extraction Priorities:[/]")
            for s in gap_data["extraction_suggestions"]:
                console.print(f"  - {s['domain']}: {s['unprocessed_experiences']} unprocessed (priority: {s['priority_score']:.1f})")
    finally:
        system.close()


@analytics_app.command()
def audit(since: str = typer.Option("7d", "--since", help="Time period: 1d, 7d, 30d")):
    """Show recent system events (confidence changes, lifecycle transitions)."""
    system = _get_system()
    try:
        # Parse since (e.g. "7d", "24h", "30d")
        days = 7  # default
        try:
            if since.endswith("d"):
                days = int(since[:-1])
            elif since.endswith("h"):
                days = int(since[:-1]) / 24
            else:
                days = int(since)
        except (ValueError, IndexError):
            console.print(f"[yellow]Invalid --since value '{since}', using 7d[/]")
            days = 7
        since_dt = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        events = system.sqlite.get_recent_events(since=since_dt)
        if not events:
            console.print(f"[dim]No events in the last {since}.[/]")
            return

        table = Table(title=f"System Events (last {since})", show_lines=True)
        table.add_column("Timestamp", style="dim", width=20)
        table.add_column("Type", style="magenta")
        table.add_column("Entity ID", style="cyan", width=18)
        table.add_column("Change")
        table.add_column("Reason")

        for e in events:
            change = f"{e['old_confidence']:.3f} -> {e['new_confidence']:.3f}"
            table.add_row(
                str(e["timestamp"])[:19],
                e["entity_type"],
                e["entity_id"],
                change,
                e["reason"][:40],
            )

        console.print(table)
    finally:
        system.close()


@analytics_app.command()
def meta():
    """Meta-learning insights — how the system fails and what it should distrust."""
    system = _get_system()
    try:
        data = system.meta_learning.summary()

        # Failure profiles
        profiles = data["failure_profiles"]
        if profiles:
            table = Table(title="Failure Profiles (by type & creation method)", show_lines=True)
            table.add_column("Category", style="cyan")
            table.add_column("Total", justify="right")
            table.add_column("Deprecated", justify="right")
            table.add_column("Failure Rate", justify="right")
            table.add_column("Contamination", justify="right")

            for p in profiles:
                rate_style = "red" if p.failure_rate > 0.3 else "yellow" if p.failure_rate > 0.1 else "green"
                table.add_row(
                    p.category,
                    str(p.total_count),
                    str(p.deprecated_count),
                    f"[{rate_style}]{p.failure_rate:.0%}[/]",
                    str(p.contamination_events),
                )
            console.print(table)
        else:
            console.print("[dim]No failure profiles yet — no deprecated wisdom.[/]")

        # Risky domains
        domains = data["risky_domains"]
        if domains:
            console.print()
            table = Table(title="Domain Risk Assessment", show_lines=True)
            table.add_column("Domain", style="blue")
            table.add_column("Active", justify="right")
            table.add_column("Deprecated", justify="right")
            table.add_column("Contamination", justify="right")
            table.add_column("Risk", justify="right")

            for d in domains:
                risk_style = "red" if d["risk_score"] > 0.5 else "yellow" if d["risk_score"] > 0.2 else "green"
                table.add_row(
                    d["domain"],
                    str(d["active_wisdom"]),
                    str(d["deprecated_count"]),
                    str(d["contamination_events"]),
                    f"[{risk_style}]{d['risk_score']:.2f}[/]",
                )
            console.print(table)

        # Super-spreaders
        spreaders = data["super_spreaders"]
        if spreaders:
            console.print()
            console.print("[bold]Top Contamination Sources (super-spreaders):[/]")
            for s in spreaders:
                w = system.sqlite.get_wisdom(s.source_wisdom_id)
                label = w.statement[:60] if w else s.source_wisdom_id
                console.print(
                    f"  - [cyan]{label}[/]: {s.total_affected} affected "
                    f"(avg penalty: {s.avg_penalty:.3f})"
                )

        # Confidence trajectory
        traj = data["trajectory"]
        if traj["total_events"] > 0:
            console.print()
            direction_style = {
                "improving": "green",
                "declining": "red",
                "stable": "yellow",
            }.get(traj["net_direction"], "dim")
            console.print(
                f"[bold]Confidence Trajectory:[/] [{direction_style}]{traj['net_direction']}[/] "
                f"(avg delta: {traj['avg_delta']:+.4f}, "
                f"{traj['positive_changes']} up / {traj['negative_changes']} down)"
            )
            if traj["top_decrease_reasons"]:
                console.print("  Top decrease reasons:")
                for r in traj["top_decrease_reasons"][:3]:
                    console.print(f"    - {r['reason']}: {r['count']}x (avg drop: {r['avg_drop']:.3f})")
        else:
            console.print("\n[dim]No confidence history yet.[/]")

    finally:
        system.close()


@analytics_app.command()
def coverage(domain: Optional[str] = typer.Option(None, "--domain", "-d")):
    """Analyze semantic coverage — what wisdom fails to mention."""
    system = _get_system()
    try:
        if domain:
            result = system.coverage.find_domain_blind_spots(domain)
            if result.get("status") == "no_wisdom":
                console.print(f"[dim]No active wisdom in domain '{domain}'[/]")
                return

            console.print(f"[bold]Coverage Analysis: {domain}[/]")
            console.print(f"  Domain coverage: {result.get('domain_coverage', 0):.1%}")
            console.print(f"  Experiences: {result.get('experience_count', 0)}")
            console.print(f"  Active wisdom: {result.get('wisdom_count', 0)}")

            blind_spots = result.get("blind_spots", [])
            if blind_spots:
                console.print(f"\n[bold red]Blind Spots ({len(blind_spots)}):[/]")
                for bs in blind_spots[:15]:
                    console.print(f"  - '{bs['concept']}' appears in {bs['frequency']} experiences ({bs['coverage_ratio']:.0%})")
            else:
                console.print("\n[green]No significant blind spots found.[/]")
        else:
            suspicious = system.coverage.find_suspicious_wisdom()
            if not suspicious:
                console.print("[green]No suspiciously low-coverage wisdom found.[/]")
                return

            console.print(f"[bold]Suspicious Wisdom ({len(suspicious)}):[/]")
            for s in suspicious:
                console.print(f"\n  [cyan]{s['wisdom_id']}[/]: {s['statement']}")
                console.print(f"  Domain: {s['domain']} | Coverage: {s['coverage_score']:.1%}")
                console.print(f"  Missing: {', '.join(s['top_missing'])}")
    finally:
        system.close()
