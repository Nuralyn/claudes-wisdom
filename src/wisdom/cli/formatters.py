"""Rich output formatters for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom

console = Console()

# ── Color Scheme ────────────────────────────────────────────────────────────

LIFECYCLE_COLORS = {
    "emerging": "yellow",
    "established": "green",
    "challenged": "red",
    "deprecated": "dim",
}

RESULT_COLORS = {
    "success": "green",
    "partial": "yellow",
    "failure": "red",
    "error": "red bold",
}

CONFIDENCE_THRESHOLDS = [
    (0.8, "green"),
    (0.5, "yellow"),
    (0.3, "red"),
    (0.0, "dim red"),
]


def confidence_color(score: float) -> str:
    for threshold, color in CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            return color
    return "dim"


def format_confidence(score: float) -> Text:
    color = confidence_color(score)
    return Text(f"{score:.2f}", style=color)


# ── Experience Formatting ───────────────────────────────────────────────────


def experience_table(experiences: list[Experience], title: str = "Experiences") -> Table:
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="cyan", width=18)
    table.add_column("Domain", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Result")
    table.add_column("Description", max_width=50)
    table.add_column("Timestamp", style="dim")

    for exp in experiences:
        result_color = RESULT_COLORS.get(exp.result.value, "white")
        table.add_row(
            exp.id,
            exp.domain or "-",
            exp.type.value,
            Text(exp.result.value, style=result_color),
            exp.description[:50] + ("..." if len(exp.description) > 50 else ""),
            exp.timestamp[:19],
        )
    return table


def experience_panel(exp: Experience) -> Panel:
    lines = [
        f"[cyan]ID:[/] {exp.id}",
        f"[cyan]Timestamp:[/] {exp.timestamp}",
        f"[cyan]Type:[/] {exp.type.value}",
        f"[cyan]Domain:[/] {exp.domain or '-'}",
        f"[cyan]Subdomain:[/] {exp.subdomain or '-'}",
        f"[cyan]Task Type:[/] {exp.task_type or '-'}",
        f"[cyan]Result:[/] [{RESULT_COLORS.get(exp.result.value, 'white')}]{exp.result.value}[/]",
        f"[cyan]Quality:[/] {exp.quality_score:.2f}",
        f"[cyan]Processed:[/] {'Yes' if exp.processed else 'No'}",
        "",
        f"[bold]Description:[/] {exp.description}",
    ]
    if exp.input_text:
        lines.append(f"\n[bold]Input:[/]\n{exp.input_text[:500]}")
    if exp.output_text:
        lines.append(f"\n[bold]Output:[/]\n{exp.output_text[:500]}")
    if exp.tags:
        lines.append(f"\n[cyan]Tags:[/] {', '.join(exp.tags)}")
    if exp.metadata:
        lines.append(f"[cyan]Metadata:[/] {exp.metadata}")

    return Panel("\n".join(lines), title=f"Experience: {exp.id}", border_style="cyan")


# ── Knowledge Formatting ───────────────────────────────────────────────────


def knowledge_table(entries: list[Knowledge], title: str = "Knowledge") -> Table:
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="cyan", width=18)
    table.add_column("Type", style="magenta")
    table.add_column("Domain", style="blue")
    table.add_column("Confidence")
    table.add_column("Status")
    table.add_column("Statement", max_width=50)

    for k in entries:
        table.add_row(
            k.id,
            k.type.value,
            k.domain or "-",
            format_confidence(k.confidence.overall),
            k.validation_status.value,
            k.statement[:50] + ("..." if len(k.statement) > 50 else ""),
        )
    return table


def knowledge_panel(k: Knowledge) -> Panel:
    lines = [
        f"[cyan]ID:[/] {k.id}",
        f"[cyan]Created:[/] {k.created_at[:19]}",
        f"[cyan]Updated:[/] {k.updated_at[:19]}",
        f"[cyan]Type:[/] {k.type.value}",
        f"[cyan]Domain:[/] {k.domain or '-'}",
        f"[cyan]Specificity:[/] {k.specificity:.2f}",
        f"[cyan]Validation:[/] {k.validation_status.value}",
        f"[cyan]Supporting:[/] {k.supporting_count} | [cyan]Contradicting:[/] {k.contradicting_count}",
        "",
        f"[bold]Statement:[/] {k.statement}",
    ]
    if k.explanation:
        lines.append(f"\n[bold]Explanation:[/] {k.explanation}")
    if k.preconditions:
        lines.append(f"\n[cyan]Preconditions:[/] {'; '.join(k.preconditions)}")
    if k.postconditions:
        lines.append(f"[cyan]Postconditions:[/] {'; '.join(k.postconditions)}")

    conf = k.confidence
    lines.append(f"\n[bold]Confidence:[/]")
    lines.append(f"  Overall: {conf.overall:.2f} | Theoretical: {conf.theoretical:.2f} | Empirical: {conf.empirical:.2f} | Observational: {conf.observational:.2f}")

    if k.tags:
        lines.append(f"\n[cyan]Tags:[/] {', '.join(k.tags)}")
    if k.source_experience_ids:
        lines.append(f"[cyan]Source Experiences:[/] {', '.join(k.source_experience_ids[:5])}")

    return Panel("\n".join(lines), title=f"Knowledge: {k.id}", border_style="yellow")


# ── Wisdom Formatting ──────────────────────────────────────────────────────


def wisdom_table(entries: list[Wisdom], title: str = "Wisdom") -> Table:
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="cyan", width=18)
    table.add_column("Type", style="magenta")
    table.add_column("Lifecycle")
    table.add_column("Confidence")
    table.add_column("Apps", justify="right")
    table.add_column("Statement", max_width=50)

    for w in entries:
        lc_color = LIFECYCLE_COLORS.get(w.lifecycle.value, "white")
        table.add_row(
            w.id,
            w.type.value,
            Text(w.lifecycle.value, style=lc_color),
            format_confidence(w.confidence.overall),
            str(w.application_count),
            w.statement[:50] + ("..." if len(w.statement) > 50 else ""),
        )
    return table


def wisdom_panel(w: Wisdom) -> Panel:
    lc_color = LIFECYCLE_COLORS.get(w.lifecycle.value, "white")
    lines = [
        f"[cyan]ID:[/] {w.id}",
        f"[cyan]Created:[/] {w.created_at[:19]}",
        f"[cyan]Updated:[/] {w.updated_at[:19]}",
        f"[cyan]Type:[/] {w.type.value}",
        f"[cyan]Lifecycle:[/] [{lc_color}]{w.lifecycle.value}[/]",
        f"[cyan]Version:[/] {w.version}",
        f"[cyan]Creation Method:[/] {w.creation_method.value}",
        f"[cyan]Applications:[/] {w.application_count} (success: {w.success_count}, failure: {w.failure_count})",
        "",
        f"[bold]Statement:[/] {w.statement}",
    ]
    if w.reasoning:
        lines.append(f"\n[bold]Reasoning:[/] {w.reasoning}")
    if w.implications:
        lines.append(f"\n[bold]Implications:[/]")
        for imp in w.implications:
            lines.append(f"  - {imp}")
    if w.counterexamples:
        lines.append(f"\n[bold]Counterexamples:[/]")
        for ce in w.counterexamples:
            lines.append(f"  - {ce}")
    if w.applicable_domains:
        lines.append(f"\n[cyan]Domains:[/] {', '.join(w.applicable_domains)}")
    if w.applicability_conditions:
        lines.append(f"[cyan]Applies When:[/]")
        for c in w.applicability_conditions:
            lines.append(f"  - {c}")
    if w.inapplicability_conditions:
        lines.append(f"[cyan]Does NOT Apply When:[/]")
        for c in w.inapplicability_conditions:
            lines.append(f"  - {c}")
    if w.trade_offs:
        lines.append(f"\n[bold]Trade-offs:[/]")
        for t in w.trade_offs:
            lines.append(f"  [{t.dimension}] {t.benefit} ({t.benefit_magnitude:.1f}) vs {t.cost} ({t.cost_magnitude:.1f})")

    conf = w.confidence
    lines.append(f"\n[bold]Confidence:[/]")
    lines.append(f"  Overall: {conf.overall:.2f} | Theoretical: {conf.theoretical:.2f} | Empirical: {conf.empirical:.2f} | Observational: {conf.observational:.2f}")

    if w.deprecation_reason:
        lines.append(f"\n[red]Deprecation Reason:[/] {w.deprecation_reason}")
    if w.tags:
        lines.append(f"\n[cyan]Tags:[/] {', '.join(w.tags)}")

    return Panel("\n".join(lines), title=f"Wisdom: {w.id}", border_style="green")


# ── Search Result Formatting ───────────────────────────────────────────────


def search_results_table(results: list[dict], title: str = "Search Results") -> Table:
    """Format ScoredResult.to_dict() items."""
    table = Table(title=title, show_lines=True)
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Layer", style="magenta")
    table.add_column("ID", style="cyan", width=18)
    table.add_column("Score")
    table.add_column("Confidence")
    table.add_column("Statement", max_width=50)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.get("layer", "?"),
            r.get("id", "?"),
            f"{r.get('final_score', 0):.3f}",
            format_confidence(r.get("effective_confidence", 0)),
            (r.get("statement", "")[:50] + "...") if len(r.get("statement", "")) > 50 else r.get("statement", ""),
        )
    return table


# ── Stats Formatting ───────────────────────────────────────────────────────


def stats_panel(stats: dict) -> Panel:
    lines = [
        f"[bold]Experiences:[/] {stats.get('experiences', 0)}",
        f"[bold]Knowledge:[/] {stats.get('knowledge', 0)}",
        f"[bold]Wisdom:[/] {stats.get('wisdom', 0)}",
        f"[bold]Relationships:[/] {stats.get('relationships', 0)}",
    ]
    domains = stats.get("domains", [])
    if domains:
        lines.append(f"\n[bold]Domains:[/] {', '.join(domains)}")
    return Panel("\n".join(lines), title="Wisdom System Statistics", border_style="blue")


def gap_analysis_table(gaps: list[dict], title: str = "Wisdom Gaps") -> Table:
    table = Table(title=title, show_lines=True)
    table.add_column("Domain", style="blue")
    table.add_column("Experiences", justify="right")
    table.add_column("Knowledge", justify="right")
    table.add_column("Wisdom", justify="right")
    table.add_column("Severity")
    table.add_column("Suggestion", max_width=40)

    for g in gaps:
        sev_color = "red" if g["severity"] == "high" else "yellow"
        table.add_row(
            g["domain"],
            str(g["experiences"]),
            str(g["knowledge"]),
            str(g["wisdom"]),
            Text(g["severity"], style=sev_color),
            g["suggestion"],
        )
    return table
