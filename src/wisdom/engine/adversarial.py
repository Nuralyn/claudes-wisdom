"""Adversarial engine — devil's advocate for wisdom entries.

Every piece of wisdom must survive active challenge before promotion.
This engine doesn't just check scores — it actively tries to break
principles by searching for counterevidence, testing for vagueness,
and probing boundary conditions.

A ChallengeReport is the output: it contains specific, actionable
findings that must be addressed before the system promotes wisdom.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from wisdom.engine.coverage import _extract_concepts
from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.adversarial")


@dataclass
class Finding:
    """A single adversarial finding."""
    category: str  # 'counterexample', 'vagueness', 'contradiction', 'blind_spot', 'untested'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    evidence: str = ""


@dataclass
class ChallengeReport:
    """The result of an adversarial challenge."""
    wisdom_id: str
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def to_dict(self) -> dict:
        return {
            "wisdom_id": self.wisdom_id,
            "passed": self.passed,
            "critical_findings": self.critical_count,
            "warning_findings": self.warning_count,
            "total_findings": len(self.findings),
            "summary": self.summary,
            "findings": [
                {"category": f.category, "severity": f.severity,
                 "description": f.description, "evidence": f.evidence}
                for f in self.findings
            ],
        }


class AdversarialEngine:
    """Actively challenges wisdom before promotion.

    Challenge types:
    1. Counterexample search: Find failure experiences semantically close to the wisdom
    2. Vagueness detection: Check if applicability conditions actually discriminate
    3. Contradiction scan: Find existing wisdom that conflicts
    4. Blind spot detection: What does this wisdom fail to mention that it should?
    5. Untested conditions: Are there domains/conditions where this was never applied?
    """

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore):
        self.sqlite = sqlite
        self.vector = vector

    def challenge(self, w: Wisdom, risk_profile: dict | None = None) -> ChallengeReport:
        """Run the full adversarial challenge battery against a wisdom entry.

        Returns a ChallengeReport. Wisdom should not be promoted if the report
        contains critical findings.

        Args:
            w: The wisdom entry to challenge.
            risk_profile: Optional dict from MetaLearningEngine with threshold
                overrides. Keys: 'counterexample_threshold', 'blind_spot_frequency',
                'risk_level'. If None, uses default thresholds.
                The adversarial engine has no dependency on MetaLearningEngine —
                this is just a dict of numbers.
        """
        findings: list[Finding] = []

        findings.extend(self._search_counterexamples(w, risk_profile))
        findings.extend(self._check_vagueness(w))
        findings.extend(self._scan_contradictions(w))
        findings.extend(self._detect_blind_spots(w, risk_profile))
        findings.extend(self._check_untested(w))

        passed = all(f.severity != "critical" for f in findings)

        summary_parts = []
        if not findings:
            summary_parts.append("No adversarial findings. Wisdom appears robust.")
        else:
            if (cc := sum(1 for f in findings if f.severity == "critical")):
                summary_parts.append(f"{cc} critical issue(s) must be addressed")
            if (wc := sum(1 for f in findings if f.severity == "warning")):
                summary_parts.append(f"{wc} warning(s) to consider")
            if (ic := sum(1 for f in findings if f.severity == "info")):
                summary_parts.append(f"{ic} note(s)")

        report = ChallengeReport(
            wisdom_id=w.id,
            passed=passed,
            findings=findings,
            summary=". ".join(summary_parts),
        )

        logger.info(
            "Challenge report for %s: %s (%d findings, %d critical)",
            w.id, "PASSED" if passed else "FAILED",
            len(findings), report.critical_count,
        )
        return report

    def _search_counterexamples(
        self, w: Wisdom, risk_profile: dict | None = None,
    ) -> list[Finding]:
        """Search for failure experiences that are semantically close to this wisdom.

        If failures cluster near the wisdom's domain of applicability,
        the wisdom may be wrong or at least incomplete.
        """
        findings = []

        # Risk-adjusted thresholds: high-risk profiles get searched harder
        strong_threshold = 0.7
        if risk_profile and "counterexample_threshold" in risk_profile:
            strong_threshold = risk_profile["counterexample_threshold"]
        weak_threshold = strong_threshold - 0.2

        # Search experiences semantically similar to the wisdom statement
        results = self.vector.search(
            layer="experience",
            query=w.statement,
            top_k=20,
        )

        failure_matches = []
        for r in results:
            exp = self.sqlite.get_experience(r["id"])
            if exp and exp.result.value in ("failure", "error") and r["similarity"] > weak_threshold:
                failure_matches.append((exp, r["similarity"]))

        if not failure_matches:
            return findings

        # High similarity failures are potential counterexamples
        strong_counter = [(e, s) for e, s in failure_matches if s > strong_threshold]
        weak_counter = [(e, s) for e, s in failure_matches if weak_threshold < s <= strong_threshold]

        if strong_counter:
            descriptions = "; ".join(
                f"'{e.description[:60]}' (similarity={s:.2f})"
                for e, s in strong_counter[:3]
            )
            findings.append(Finding(
                category="counterexample",
                severity="critical",
                description=(
                    f"Found {len(strong_counter)} failure(s) highly similar to this wisdom, "
                    f"suggesting it may be wrong or incomplete in these cases"
                ),
                evidence=descriptions,
            ))

        if weak_counter:
            findings.append(Finding(
                category="counterexample",
                severity="warning",
                description=(
                    f"Found {len(weak_counter)} failure(s) moderately similar to this wisdom. "
                    f"Consider whether the wisdom's applicability conditions exclude these cases"
                ),
                evidence="; ".join(
                    f"'{e.description[:50]}'" for e, _ in weak_counter[:3]
                ),
            ))

        return findings

    def _check_vagueness(self, w: Wisdom) -> list[Finding]:
        """Check if the wisdom is too vague to be useful.

        A vague wisdom has:
        - No applicability conditions (applies everywhere = applies nowhere)
        - Very short statement with no specifics
        - No trade-offs (everything has trade-offs)
        """
        findings = []

        if not w.applicability_conditions and not w.inapplicability_conditions:
            findings.append(Finding(
                category="vagueness",
                severity="critical",
                description=(
                    "No applicability conditions specified. Wisdom without boundaries "
                    "is opinion — when does this apply and when doesn't it?"
                ),
            ))

        if len(w.statement.split()) < 5:
            findings.append(Finding(
                category="vagueness",
                severity="warning",
                description="Statement is very short. Is it specific enough to be actionable?",
            ))

        if not w.trade_offs and w.type.value in ("principle", "heuristic", "judgment_rule"):
            findings.append(Finding(
                category="vagueness",
                severity="warning",
                description=(
                    "No trade-offs documented. Every principle has costs. "
                    "What do you give up by following this wisdom?"
                ),
            ))

        if not w.reasoning:
            findings.append(Finding(
                category="vagueness",
                severity="warning",
                description="No reasoning provided. Why should anyone trust this?",
            ))

        # Check for weasel words that indicate vagueness
        weasel_indicators = [
            "sometimes", "maybe", "might", "could", "possibly",
            "in some cases", "it depends", "generally",
        ]
        statement_lower = w.statement.lower()
        found_weasels = [ww for ww in weasel_indicators if ww in statement_lower]
        if found_weasels:
            findings.append(Finding(
                category="vagueness",
                severity="info",
                description=(
                    f"Statement contains hedging language ({', '.join(found_weasels)}). "
                    f"Consider making the conditions explicit instead of hedging"
                ),
            ))

        return findings

    def _scan_contradictions(self, w: Wisdom) -> list[Finding]:
        """Search for existing wisdom that may contradict this entry."""
        findings = []

        # Check explicit conflict relationships
        conflicts = self.sqlite.find_conflicts(w.id)
        for rel in conflicts:
            other_id = rel.target_id if rel.source_id == w.id else rel.source_id
            other = self.sqlite.get_wisdom(other_id)
            if other and other.lifecycle != LifecycleState.DEPRECATED:
                findings.append(Finding(
                    category="contradiction",
                    severity="critical" if rel.strength > 0.7 else "warning",
                    description=f"Explicit conflict with wisdom {other_id}",
                    evidence=f"Conflicting statement: '{other.statement[:80]}'",
                ))

        # Semantic search for potentially conflicting wisdom
        similar = self.vector.search(
            layer="wisdom",
            query=w.statement,
            top_k=10,
        )
        for r in similar:
            if r["id"] == w.id:
                continue
            other = self.sqlite.get_wisdom(r["id"])
            if not other or other.lifecycle == LifecycleState.DEPRECATED:
                continue
            # High similarity could mean support OR contradiction
            # Check if the statements seem opposed
            if r["similarity"] > 0.7 and self._statements_may_conflict(w.statement, other.statement):
                already_flagged = any(
                    f.evidence and r["id"] in f.evidence for f in findings
                )
                if not already_flagged:
                    findings.append(Finding(
                        category="contradiction",
                        severity="warning",
                        description=(
                            f"Highly similar wisdom {r['id']} may conflict "
                            f"(similarity={r['similarity']:.2f})"
                        ),
                        evidence=f"'{other.statement[:80]}'",
                    ))

        return findings

    def _detect_blind_spots(
        self, w: Wisdom, risk_profile: dict | None = None,
    ) -> list[Finding]:
        """Check what this wisdom fails to mention that it should.

        Search for experiences in the wisdom's domain, extract key concepts
        that appear frequently but are absent from the wisdom's text.

        Uses the shared _extract_concepts from coverage.py for consistent
        tokenization, suffix normalization, and bigram extraction.
        """
        findings = []

        if not w.applicable_domains:
            return findings  # Can't check blind spots without a domain

        frequency_threshold = 0.3
        if risk_profile and "blind_spot_frequency" in risk_profile:
            frequency_threshold = risk_profile["blind_spot_frequency"]

        wisdom_concepts = _extract_concepts(
            f"{w.statement} {w.reasoning} "
            f"{' '.join(w.implications)} "
            f"{' '.join(w.applicability_conditions)}"
        )

        for domain in w.applicable_domains:
            domain_exps = self.sqlite.list_experiences(domain=domain, limit=100)
            if len(domain_exps) < 5:
                continue

            exp_concepts: Counter[str] = Counter()
            for exp in domain_exps:
                text = f"{exp.description} {exp.input_text}"
                exp_concepts.update(_extract_concepts(text))

            missing = []
            for concept, count in exp_concepts.most_common(30):
                if count >= len(domain_exps) * frequency_threshold and concept not in wisdom_concepts:
                    missing.append((concept, count))

            if missing:
                top_missing = missing[:5]
                concepts_str = ", ".join(f"'{c}' ({n}x)" for c, n in top_missing)
                findings.append(Finding(
                    category="blind_spot",
                    severity="warning" if len(missing) <= 3 else "critical",
                    description=(
                        f"In domain '{domain}', {len(missing)} frequent concept(s) from experiences "
                        f"are absent from this wisdom"
                    ),
                    evidence=f"Missing concepts: {concepts_str}",
                ))

        return findings

    def _check_untested(self, w: Wisdom) -> list[Finding]:
        """Check if wisdom has been applied in too narrow a context."""
        findings = []

        if w.application_count == 0:
            findings.append(Finding(
                category="untested",
                severity="info",
                description="This wisdom has never been applied. Confidence is purely theoretical.",
            ))
        elif w.application_count < 3:
            findings.append(Finding(
                category="untested",
                severity="info",
                description=(
                    f"Only {w.application_count} application(s). "
                    f"Too few to establish reliability."
                ),
            ))

        # Check if applications are all in the same narrow context
        if w.application_count > 0:
            app_exps = self.sqlite.list_experiences_for_wisdom(w.id)
            if app_exps:
                domains_tested = set(e.domain for e in app_exps if e.domain)
                if len(domains_tested) <= 1 and len(w.applicable_domains) > 1:
                    findings.append(Finding(
                        category="untested",
                        severity="warning",
                        description=(
                            f"Wisdom claims applicability in {len(w.applicable_domains)} domains "
                            f"but has only been tested in {domains_tested or {'(none)'}}"
                        ),
                    ))

        return findings

    def _statements_may_conflict(self, s1: str, s2: str) -> bool:
        """Heuristic check for whether two statements might conflict.

        Looks for negation patterns and opposing sentiment markers.
        Uses normalized concept extraction for subject-matter overlap
        so morphological variants (indexing/indexed) count as matches.
        This is a rough heuristic — LLM-powered analysis would be better.
        """
        s1_lower, s2_lower = s1.lower(), s2.lower()

        negation_words = {"not", "never", "avoid", "don't", "doesn't", "shouldn't", "without"}
        s1_negated = any(w in s1_lower.split() for w in negation_words)
        s2_negated = any(w in s2_lower.split() for w in negation_words)

        # Normalized concept overlap (morphological variants match)
        s1_concepts = _extract_concepts(s1)
        s2_concepts = _extract_concepts(s2)
        concept_overlap = len(s1_concepts & s2_concepts)

        if s1_negated != s2_negated and concept_overlap >= 2:
            return True

        prefer_words = {"prefer", "always", "best", "should", "must", "use"}
        avoid_words = {"avoid", "never", "worst", "shouldn't", "don't", "skip"}
        s1_prefers = any(w in s1_lower.split() for w in prefer_words)
        s2_avoids = any(w in s2_lower.split() for w in avoid_words)
        s1_avoids = any(w in s1_lower.split() for w in avoid_words)
        s2_prefers = any(w in s2_lower.split() for w in prefer_words)

        if (s1_prefers and s2_avoids) or (s1_avoids and s2_prefers):
            if concept_overlap >= 2:
                return True

        return False
