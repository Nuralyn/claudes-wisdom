"""Wisdom prompt injection — format wisdom for LLM augmentation."""

from __future__ import annotations

from wisdom.engine.retrieval import ScoredResult
from wisdom.models.wisdom import Wisdom


def format_wisdom_injection(
    results: list[ScoredResult],
    conflicts: list[dict] | None = None,
) -> str:
    """Format retrieved wisdom into an XML injection block for LLM prompts.

    This is the format used when augmenting LLM prompts with relevant wisdom.
    """
    if not results:
        return ""

    lines = [
        "<relevant_wisdom>",
        "You have access to accumulated wisdom. Apply where appropriate.",
        "",
    ]

    for i, r in enumerate(results, 1):
        entity = r.entity
        conf = round(r.effective_confidence, 2)
        domains = ""
        applies_when = ""
        trade_off_text = ""

        if isinstance(entity, Wisdom):
            if entity.applicable_domains:
                domains = ", ".join(entity.applicable_domains)
            if entity.applicability_conditions:
                applies_when = "; ".join(entity.applicability_conditions)
            if entity.trade_offs:
                parts = []
                for t in entity.trade_offs:
                    parts.append(f"{t.benefit} vs {t.cost}")
                trade_off_text = "; ".join(parts)

        type_label = entity.type.value.upper().replace("_", " ") if hasattr(entity, "type") else "ENTRY"
        domain_str = f", domain: {domains}" if domains else ""

        lines.append(f"[{i}] (confidence: {conf}{domain_str})")
        lines.append(f"{type_label}: {entity.statement}")
        if applies_when:
            lines.append(f"APPLIES WHEN: {applies_when}")
        if trade_off_text:
            lines.append(f"TRADE-OFF: {trade_off_text}")
        lines.append("")

    # Add conflict/complementary notes
    if conflicts:
        for c in conflicts:
            lines.append(f"NOTE: Entries [{c.get('a', '?')}] and [{c.get('b', '?')}] may conflict.")
    elif len(results) > 1:
        indices = ", ".join(str(i) for i in range(1, len(results) + 1))
        lines.append(f"NOTE: Entries [{indices}] are complementary, not contradictory.")

    lines.append("</relevant_wisdom>")
    return "\n".join(lines)


def generate_claude_md(
    wisdom_entries: list[Wisdom],
    domain: str | None = None,
    min_confidence: float = 0.5,
) -> str:
    """Generate a CLAUDE.md file content from high-confidence wisdom.

    This bridges the wisdom system with Claude Code's native memory.
    """
    filtered = [
        w for w in wisdom_entries
        if w.confidence.overall >= min_confidence
        and w.lifecycle.value not in ("deprecated",)
    ]

    if domain:
        filtered = [
            w for w in filtered
            if domain in w.applicable_domains or not w.applicable_domains
        ]

    # Sort by confidence descending
    filtered.sort(key=lambda w: w.confidence.overall, reverse=True)

    lines = [
        "# Project Wisdom",
        "",
        f"Auto-generated from the Wisdom System. Domain: {domain or 'all'}.",
        f"Entries: {len(filtered)} (confidence >= {min_confidence})",
        "",
        "## Principles & Guidelines",
        "",
    ]

    for w in filtered:
        conf = round(w.confidence.overall, 2)
        lines.append(f"### {w.statement}")
        lines.append(f"*Confidence: {conf} | Type: {w.type.value} | Lifecycle: {w.lifecycle.value}*")
        lines.append("")
        if w.reasoning:
            lines.append(f"**Why:** {w.reasoning}")
            lines.append("")
        if w.applicability_conditions:
            lines.append(f"**Applies when:** {'; '.join(w.applicability_conditions)}")
            lines.append("")
        if w.inapplicability_conditions:
            lines.append(f"**Does NOT apply when:** {'; '.join(w.inapplicability_conditions)}")
            lines.append("")
        if w.trade_offs:
            lines.append("**Trade-offs:**")
            for t in w.trade_offs:
                lines.append(f"- {t.dimension}: {t.benefit} (gain) vs {t.cost} (cost)")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
