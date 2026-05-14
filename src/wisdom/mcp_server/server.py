"""FastMCP server with tools and resources for the Wisdom System."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from wisdom import WisdomSystem
from wisdom.config import WisdomConfig
from wisdom.logging_config import get_logger

logger = get_logger("mcp.server")


@dataclass
class AppState:
    system: WisdomSystem


@asynccontextmanager
async def lifespan(server: FastMCP):
    logger.info("Initializing Wisdom System for MCP server...")
    config = WisdomConfig()
    system = WisdomSystem(config)
    system.init_providers()
    logger.info("Wisdom System ready. Providers: %s", system.providers.list_available())
    try:
        yield AppState(system=system)
    finally:
        system.close()
        logger.info("Wisdom System shut down.")


mcp = FastMCP("wisdom-system", lifespan=lifespan)


def _system(ctx: Context) -> WisdomSystem:
    state: AppState = ctx.request_context.lifespan_context
    return state.system


# ── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
def search_wisdom(
    ctx: Context,
    query: str,
    domain: str = "",
    top_k: int = 5,
) -> str:
    """Search for relevant wisdom by semantic similarity.

    Args:
        query: Natural language search query
        domain: Optional domain filter (e.g., 'python', 'devops')
        top_k: Number of results to return (default 5)
    """
    system = _system(ctx)
    results = system.retrieval.search(
        query=query,
        domain=domain or None,
        top_k=top_k,
        layers=["wisdom"],
    )
    if not results:
        return "No relevant wisdom found."

    lines = []
    for i, r in enumerate(results, 1):
        w = r.entity
        lines.append(f"[{i}] (confidence: {r.effective_confidence:.2f}, score: {r.final_score:.3f})")
        lines.append(f"  {w.statement}")
        if hasattr(w, "applicable_domains") and w.applicable_domains:
            lines.append(f"  Domains: {', '.join(w.applicable_domains)}")
        if hasattr(w, "applicability_conditions") and w.applicability_conditions:
            lines.append(f"  Applies when: {'; '.join(w.applicability_conditions)}")
        lines.append(f"  ID: {w.id}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_wisdom(ctx: Context, id: str) -> str:
    """Get full details of a wisdom entry by ID."""
    system = _system(ctx)
    w = system.wisdom.get(id)
    if not w:
        return f"Wisdom not found: {id}"

    lines = [
        f"ID: {w.id}",
        f"Type: {w.type.value}",
        f"Statement: {w.statement}",
        f"Reasoning: {w.reasoning}",
        f"Confidence: {w.confidence.overall:.2f} (theoretical: {w.confidence.theoretical:.2f}, empirical: {w.confidence.empirical:.2f})",
        f"Lifecycle: {w.lifecycle.value}",
        f"Applications: {w.application_count} (success: {w.success_count}, failure: {w.failure_count})",
        f"Domains: {', '.join(w.applicable_domains)}",
    ]
    if w.applicability_conditions:
        lines.append(f"Applies when: {'; '.join(w.applicability_conditions)}")
    if w.inapplicability_conditions:
        lines.append(f"Does NOT apply when: {'; '.join(w.inapplicability_conditions)}")
    if w.trade_offs:
        lines.append("Trade-offs:")
        for t in w.trade_offs:
            lines.append(f"  - {t.dimension}: {t.benefit} vs {t.cost}")
    if w.implications:
        lines.append(f"Implications: {'; '.join(w.implications)}")
    return "\n".join(lines)


@mcp.tool()
def add_experience(
    ctx: Context,
    description: str,
    domain: str = "",
    input_text: str = "",
    output_text: str = "",
    result: str = "success",
    tags: str = "",
) -> str:
    """Record a new experience for the wisdom system to learn from.

    Args:
        description: What happened in this experience
        domain: Domain area (e.g., 'python', 'debugging', 'devops')
        input_text: The input/context for this experience
        output_text: The output/result of this experience
        result: Outcome — 'success', 'partial', 'failure', or 'error'
        tags: Comma-separated tags
    """
    from wisdom.models.common import ExperienceResult

    system = _system(ctx)
    exp = system.experiences.add(
        description=description,
        domain=domain,
        input_text=input_text,
        output_text=output_text,
        result=ExperienceResult(result),
        tags=tags.split(",") if tags else [],
    )
    return f"Added experience {exp.id} in domain '{domain}'"


@mcp.tool()
def add_wisdom(
    ctx: Context,
    statement: str,
    reasoning: str = "",
    domain: str = "",
    wisdom_type: str = "principle",
    applicability: str = "",
    tags: str = "",
) -> str:
    """Add wisdom directly (expert input path).

    Args:
        statement: The wisdom statement/principle
        reasoning: Why this wisdom holds
        domain: Applicable domain
        wisdom_type: Type — 'principle', 'heuristic', 'judgment_rule', 'meta_pattern', 'trade_off'
        applicability: When this applies (comma-separated conditions)
        tags: Comma-separated tags
    """
    from wisdom.models.common import WisdomType

    system = _system(ctx)
    w = system.wisdom.add(
        statement=statement,
        reasoning=reasoning,
        wisdom_type=WisdomType(wisdom_type),
        domains=[domain] if domain else [],
        applicability_conditions=applicability.split(",") if applicability else [],
        tags=tags.split(",") if tags else [],
    )
    return f"Added wisdom {w.id}: {statement[:80]}"


@mcp.tool()
def extract_knowledge(
    ctx: Context,
    domain: str = "",
    use_llm: bool = False,
) -> str:
    """Extract knowledge from unprocessed experiences.

    Args:
        domain: Optional domain to filter experiences
        use_llm: Whether to use LLM for extraction (requires configured provider)
    """
    system = _system(ctx)
    experiences = system.experiences.get_unprocessed(domain=domain or None)

    if not experiences:
        return "No unprocessed experiences found."

    if use_llm and system.providers.has_provider:
        from wisdom.llm.extraction import extract_knowledge as llm_extract
        provider = system.providers.get()
        knowledge = llm_extract(provider, experiences, domain=domain)
        for k in knowledge:
            system.knowledge.add(k)
        system.sqlite.mark_processed([e.id for e in experiences])
    else:
        knowledge = system.knowledge.extract_from_experiences(experiences, domain=domain)

    return f"Extracted {len(knowledge)} knowledge entries from {len(experiences)} experiences"


@mcp.tool()
def synthesize_wisdom(
    ctx: Context,
    domain: str = "",
    use_llm: bool = False,
) -> str:
    """Synthesize wisdom from unsynthesized knowledge.

    Args:
        domain: Optional domain filter
        use_llm: Whether to use LLM for synthesis
    """
    system = _system(ctx)
    knowledge_entries = system.knowledge.get_unsynthesized(domain=domain or None)

    if not knowledge_entries:
        return "No unsynthesized knowledge found."

    if use_llm and system.providers.has_provider:
        from wisdom.llm.synthesis import synthesize_wisdom as llm_synth
        provider = system.providers.get()
        existing = system.wisdom.list(domain=domain or None)
        wisdom_list, contradictions = llm_synth(provider, knowledge_entries, existing, domain=domain)
        for w in wisdom_list:
            system.wisdom.add(
                statement=w.statement, reasoning=w.reasoning,
                wisdom_type=w.type, domains=w.applicable_domains,
                applicability_conditions=w.applicability_conditions,
                inapplicability_conditions=w.inapplicability_conditions,
                trade_offs=w.trade_offs, implications=w.implications,
                counterexamples=w.counterexamples, confidence=w.confidence,
                tags=w.tags, source_knowledge_ids=w.source_knowledge_ids,
            )
        system.sqlite.mark_synthesized([k.id for k in knowledge_entries])
        result_msg = f"Synthesized {len(wisdom_list)} wisdom entries"
        if contradictions:
            result_msg += f", found {len(contradictions)} contradictions"
        return result_msg
    else:
        wisdom_list = system.wisdom.synthesize_from_knowledge(knowledge_entries, domain=domain)
        return f"Synthesized {len(wisdom_list)} wisdom entries from {len(knowledge_entries)} knowledge entries"


@mcp.tool()
def reinforce_wisdom(
    ctx: Context,
    wisdom_id: str,
    was_helpful: bool,
    feedback: str = "",
) -> str:
    """Provide feedback on applied wisdom to adjust confidence.

    Args:
        wisdom_id: ID of the wisdom entry
        was_helpful: Whether the wisdom was helpful when applied
        feedback: Optional feedback text
    """
    system = _system(ctx)
    w = system.evolution.reinforce(wisdom_id, was_helpful=was_helpful, feedback=feedback)
    if not w:
        return f"Wisdom not found: {wisdom_id}"
    status = "positive" if was_helpful else "negative"
    return f"Reinforced ({status}): confidence={w.confidence.overall:.3f}, lifecycle={w.lifecycle.value}, applications={w.application_count}"


@mcp.tool()
def get_domain_summary(ctx: Context, domain: str) -> str:
    """Get statistics and health info for a specific domain."""
    system = _system(ctx)
    exp_count = system.sqlite.count_experiences(domain=domain)
    know_count = system.sqlite.count_knowledge(domain=domain)
    wis_count = system.sqlite.count_wisdom(domain=domain)
    unprocessed = system.sqlite.count_experiences(domain=domain, unprocessed=True)
    unsynthesized = system.sqlite.count_knowledge(domain=domain, unsynthesized=True)

    lines = [
        f"Domain: {domain}",
        f"Experiences: {exp_count} ({unprocessed} unprocessed)",
        f"Knowledge: {know_count} ({unsynthesized} unsynthesized)",
        f"Wisdom: {wis_count}",
    ]

    gaps = system.gaps.find_wisdom_gaps(domain=domain)
    if gaps:
        for g in gaps:
            lines.append(f"Gap: {g['suggestion']} (severity: {g['severity']})")

    return "\n".join(lines)


@mcp.tool()
def find_contradictions(ctx: Context, wisdom_id: str = "") -> str:
    """Find conflicting wisdom entries.

    Args:
        wisdom_id: Optional — check conflicts for this specific entry. If empty, find all conflicts.
    """
    system = _system(ctx)
    results = system.retrieval.find_contradictions(wisdom_id or None)
    if not results:
        return "No contradictions found."

    lines = []
    for r in results:
        if "a_id" in r:
            lines.append(f"{r['a_id']}: {r.get('a_statement', '')[:60]}")
            lines.append(f"  conflicts with")
            lines.append(f"{r['b_id']}: {r.get('b_statement', '')[:60]}")
            lines.append("")
        else:
            lines.append(f"{r.get('wisdom_id', '')} conflicts with {r.get('conflicting_id', '')}")
            lines.append(f"  {r.get('conflicting_statement', '')[:60]}")
            lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_wisdom_gaps(ctx: Context, domain: str = "") -> str:
    """Identify blind spots — domains with many experiences but little wisdom."""
    system = _system(ctx)
    gaps = system.gaps.summary(domain=domain or None)

    lines = []
    if gaps["gaps"]:
        lines.append("Wisdom Gaps:")
        for g in gaps["gaps"]:
            lines.append(f"  {g['domain']}: {g['experiences']} exp, {g['wisdom']} wisdom ({g['severity']})")
            lines.append(f"    Suggestion: {g['suggestion']}")

    if gaps["extraction_suggestions"]:
        lines.append("\nExtraction Priorities:")
        for s in gaps["extraction_suggestions"]:
            lines.append(f"  {s['domain']}: {s['unprocessed_experiences']} unprocessed (priority: {s['priority_score']:.1f})")

    return "\n".join(lines) if lines else "No significant gaps found."


@mcp.tool()
def run_maintenance(ctx: Context) -> str:
    """Run automatic maintenance — extraction, synthesis, deprecation sweeps.

    Includes post-deprecation meta-learning analysis that identifies WHY
    entries were deprecated, enabling systemic pattern detection.
    """
    system = _system(ctx)
    summary = system.run_maintenance()

    lines = []
    if summary["extracted"]:
        for r in summary["extracted"]:
            lines.append(f"Extracted: {r['knowledge_created']} knowledge from {r['experiences_processed']} exp in {r['domain']}")
    if summary["synthesized"]:
        for r in summary["synthesized"]:
            lines.append(f"Synthesized: {r['wisdom_created']} wisdom from {r['knowledge_processed']} knowledge in {r['domain']}")
    if summary["deprecated"]:
        lines.append(f"Deprecated: {len(summary['deprecated'])} wisdom entries")
    if summary.get("deprecation_analysis"):
        lines.append("Deprecation analysis:")
        for a in summary["deprecation_analysis"]:
            lines.append(f"  {a['wisdom_id']}: risk={a['risk_score']:.2f} ({a['risk_level']})")
    if summary["validated"]:
        lines.append(f"Needs validation: {len(summary['validated'])} entries")

    return "\n".join(lines) if lines else "No maintenance needed."


@mcp.tool()
def validate_wisdom(
    ctx: Context,
    wisdom_id: str,
    source: str = "external",
    verdict: str = "confirmed",
    evidence: str = "",
    validator: str = "",
) -> str:
    """Record an external validation event for a wisdom entry.

    Args:
        wisdom_id: ID of the wisdom entry to validate
        source: Who is validating — 'self_report', 'peer', 'external', 'adversarial'
        verdict: The judgment — 'confirmed', 'confirmed_with_caveats', 'challenged', 'refuted'
        evidence: Supporting evidence or reasoning
        validator: Identifier for the validator
    """
    system = _system(ctx)
    result = system.validation.validate(
        wisdom_id, source=source, verdict=verdict,
        evidence=evidence, validator=validator,
    )
    if "error" in result:
        return result["error"]
    return (
        f"Validated {wisdom_id}: {source}/{verdict}\n"
        f"Validation score: {result['validation_score']:.2f}\n"
        f"Confidence: {result['confidence']:.3f}"
    )


@mcp.tool()
def challenge_wisdom(ctx: Context, wisdom_id: str) -> str:
    """Run the adversarial devil's advocate challenge against a wisdom entry.

    Checks for counterexamples, vagueness, contradictions, and blind spots.
    Uses meta-learning risk profiles to adjust challenge intensity — high-risk
    entries get scrutinized harder.
    """
    system = _system(ctx)
    w = system.wisdom.get(wisdom_id)
    if not w:
        return f"Wisdom not found: {wisdom_id}"

    # Use meta-learning to compute risk-adjusted challenge thresholds
    risk_profile = system.meta_learning.risk_profile_for_adversarial(wisdom_id)
    report = system.adversarial.challenge(w, risk_profile=risk_profile)

    lines = [f"{'PASSED' if report.passed else 'FAILED'}: {report.summary}"]
    if risk_profile:
        lines.append(f"Risk level: {risk_profile.get('risk_level', 'standard')} (thresholds adjusted)")
    for f in report.findings:
        lines.append(f"[{f.severity.upper()}] [{f.category}] {f.description}")
        if f.evidence:
            lines.append(f"  Evidence: {f.evidence[:150]}")

    if report.passed:
        system.validation.validate(
            wisdom_id, "adversarial", "confirmed",
            evidence=report.summary, validator="adversarial_engine",
        )
        lines.append("\nAdversarial validation recorded.")

    return "\n".join(lines)


@mcp.tool()
def cascade_failure(ctx: Context, wisdom_id: str, severity: float = 1.0) -> str:
    """Cascade failure consequences when wisdom is found to be wrong.

    Args:
        wisdom_id: The failed wisdom entry
        severity: How bad (0.0=minor, 1.0=completely wrong)
    """
    system = _system(ctx)
    result = system.propagation.cascade_failure(wisdom_id, severity=severity)
    lines = [f"Cascade from {wisdom_id}: {result.total_affected} entities affected"]
    if result.affected_wisdom:
        lines.append(f"Wisdom penalized: {len(result.affected_wisdom)}")
    if result.affected_knowledge:
        lines.append(f"Knowledge penalized: {len(result.affected_knowledge)}")
    if result.contaminated_experiences:
        lines.append(f"Experiences contaminated: {result.contaminated_experiences}")
    return "\n".join(lines)


@mcp.tool()
def analyze_coverage(ctx: Context, domain: str) -> str:
    """Analyze semantic coverage — what wisdom fails to mention for a domain.

    Runs two types of analysis:
    1. Token-level: frequent concepts in experiences missing from wisdom text
    2. Embedding-level: experiences semantically distant from all wisdom entries
    """
    system = _system(ctx)
    result = system.coverage.find_domain_blind_spots(domain)
    if result.get("status") == "no_wisdom":
        return f"No active wisdom in domain '{domain}'"

    lines = [
        f"Domain: {domain}",
        f"Coverage: {result.get('domain_coverage', 0):.1%}",
        f"Experiences: {result.get('experience_count', 0)}",
        f"Active wisdom: {result.get('wisdom_count', 0)}",
    ]
    blind_spots = result.get("blind_spots", [])
    if blind_spots:
        lines.append(f"\nToken-level blind spots ({len(blind_spots)}):")
        for bs in blind_spots[:10]:
            lines.append(f"  '{bs['concept']}' — {bs['frequency']} experiences ({bs['coverage_ratio']:.0%})")
    else:
        lines.append("\nNo token-level blind spots.")

    # Semantic (embedding) gap analysis
    semantic = system.coverage.find_semantic_gaps(domain)
    if semantic.get("status") == "analyzed" and semantic.get("uncovered_count", 0) > 0:
        lines.append(f"\nSemantic gaps ({semantic['uncovered_count']}/{semantic['experience_count']} experiences uncovered):")
        for gap in semantic.get("most_distant", [])[:8]:
            lines.append(f"  [{gap['best_wisdom_similarity']:.2f}] {gap['description']}")
    elif semantic.get("status") == "analyzed":
        lines.append("\nNo semantic gaps — all experiences have nearby wisdom.")

    return "\n".join(lines)


@mcp.tool()
def get_risk_score(ctx: Context, wisdom_id: str) -> str:
    """Compute a risk score for a wisdom entry based on historical failure patterns.

    Analyzes type risk, creation method risk, domain contamination, validation
    status, and application history. Returns a risk level and recommended
    challenge intensity.

    Args:
        wisdom_id: The wisdom entry to assess
    """
    system = _system(ctx)
    w = system.wisdom.get(wisdom_id)
    if not w:
        return f"Wisdom not found: {wisdom_id}"

    risk = system.meta_learning.compute_risk_score(wisdom_id)
    lines = [
        f"Wisdom: {w.statement[:60]}",
        f"Risk score: {risk.base_risk:.2f} ({risk.recommended_challenge_level})",
    ]
    if risk.risk_factors:
        lines.append("Risk factors:")
        for factor in risk.risk_factors:
            lines.append(f"  - {factor.get('name', '?')}: {factor.get('value', 0):.2f} ({factor.get('reason', '')})")
    return "\n".join(lines)


@mcp.tool()
def get_meta_learning_summary(ctx: Context) -> str:
    """Get a comprehensive meta-learning analysis of the system's failure patterns.

    Shows failure profiles by type/method, domain risk assessments,
    contamination super-spreaders, confidence trajectory, learning velocity
    by category, and most volatile entries.
    """
    system = _system(ctx)
    s = system.meta_learning.summary()

    lines = []
    # Failure profiles
    profiles = s.get("failure_profiles", [])
    if profiles:
        lines.append("Failure Profiles:")
        for p in profiles:
            lines.append(f"  {p.category}: {p.failure_rate:.0%} failure ({p.deprecated_count}/{p.total_count})")
    else:
        lines.append("No failure profiles yet (need deprecation history).")

    # Domain risks
    risky_domains = s.get("risky_domains", [])
    if risky_domains:
        lines.append("\nDomain Risk Assessment:")
        for d in risky_domains:
            lines.append(f"  {d['domain']}: risk={d['risk_score']:.2f}")

    # Contamination super-spreaders
    patterns = s.get("super_spreaders", [])
    if patterns:
        lines.append("\nContamination Super-spreaders:")
        for p in patterns:
            lines.append(f"  {p.source_wisdom_id}: affected {p.total_affected} entries (avg penalty: {p.avg_penalty:.3f})")

    # Confidence trajectory
    traj = s.get("trajectory", {})
    if traj.get("total_events"):
        lines.append(f"\nConfidence Trajectory: {traj['total_events']} events, avg delta: {traj.get('avg_delta', 0):.3f}")
        if traj.get("top_decrease_reasons"):
            lines.append("  Top decrease reasons:")
            for item in traj["top_decrease_reasons"]:
                lines.append(f"    - {item['reason']}: {item['count']}x")

    # Learning velocity
    velocity = s.get("velocity", [])
    if velocity:
        lines.append("\nLearning Velocity (confidence delta/day):")
        for vp in velocity[:8]:
            sign = "+" if vp.avg_velocity >= 0 else ""
            lines.append(f"  {vp.category}: {sign}{vp.avg_velocity:.4f}/day ({vp.entry_count} entries, {vp.avg_events_per_day:.2f} events/day)")

    # Confidence volatility
    volatility = s.get("volatility", [])
    if volatility:
        lines.append("\nMost Volatile Entries:")
        for ve in volatility:
            lines.append(f"  [{ve.volatility_score:.2f}] {ve.statement_preview}")
            lines.append(f"    {ve.total_events} events, {ve.direction_changes} reversals, max swing: {ve.max_swing:.3f}")

    return "\n".join(lines) if lines else "No meta-learning data available yet."


# ── Resources ───────────────────────────────────────────────────────────────
# Resources in FastMCP don't receive context; they must be self-contained.
# We use a helper that creates a temporary system instance for resource reads.


def _resource_system() -> WisdomSystem:
    """Create a short-lived system for resource reads."""
    return WisdomSystem()


@mcp.resource("wisdom://domains")
def domains_resource() -> str:
    """List all domains in the wisdom system."""
    system = None
    try:
        system = _resource_system()
        domains = system.sqlite.get_all_domains()
        return "\n".join(domains) if domains else "No domains yet."
    except Exception as e:
        logger.error("Failed to list domains: %s", e)
        return f"Error: {e}"
    finally:
        if system:
            system.close()


@mcp.resource("wisdom://stats")
def stats_resource() -> str:
    """System-wide statistics."""
    system = None
    try:
        system = _resource_system()
        stats = system.stats()
        lines = [
            f"Experiences: {stats['experiences']}",
            f"Knowledge: {stats['knowledge']}",
            f"Wisdom: {stats['wisdom']}",
            f"Relationships: {stats['relationships']}",
            f"Domains: {', '.join(stats['domains'])}",
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.error("Failed to get stats: %s", e)
        return f"Error: {e}"
    finally:
        if system:
            system.close()


@mcp.resource("wisdom://recent")
def recent_resource() -> str:
    """Recent system events."""
    system = None
    try:
        system = _resource_system()
        events = system.sqlite.get_recent_events(limit=20)
        if not events:
            return "No recent events."
        lines = []
        for e in events:
            lines.append(f"{e['timestamp'][:19]} | {e['entity_type']} {e['entity_id'][:12]} | {e['reason']}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Failed to get recent events: %s", e)
        return f"Error: {e}"
    finally:
        if system:
            system.close()
