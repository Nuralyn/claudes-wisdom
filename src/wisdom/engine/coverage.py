"""Coverage engine — semantic absence detection.

Gap analysis v1 counts ratios: "you have 50 experiences but only 2 wisdom entries."
That finds quantity gaps. This engine finds quality gaps — what wisdom *fails to mention*.

A database design principle that never references indexing isn't just incomplete.
It's suspicious. Absence is signal.
"""

from __future__ import annotations

import re
from collections import Counter

from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.coverage")

STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "have", "been",
    "was", "were", "are", "not", "but", "what", "when", "how", "which",
    "their", "there", "them", "then", "than", "into", "also", "just",
    "more", "some", "could", "would", "should", "about", "after", "before",
    "only", "other", "very", "will", "each", "make", "like", "over",
    "using", "used", "does", "being", "most", "such", "need", "want",
    "through", "where", "while", "because", "since", "every", "same",
    "your", "they", "it's", "can't", "don't", "it'll", "i've",
})

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")


def _normalize(word: str) -> str:
    """Lightweight suffix normalization for concept matching.

    Groups morphological variants so 'indexing', 'indexed', and 'indexes'
    all resolve to 'index'. Conservative — prefers under-stemming to
    mangling. No external dependencies.
    """
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"  # queries → query, strategies → strategy
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        # Doubled consonant before -ing: running → run, not runn
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeious":
            return stem[:-1]
        return stem  # indexing → index
    if word.endswith("ed") and len(word) > 5:
        stem = word[:-2]
        # Doubled consonant before -ed: stopped → stop
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeious":
            return stem[:-1]
        return stem  # indexed → index
    if word.endswith("s") and len(word) > 4:
        if word.endswith("ss") or word.endswith("us"):
            return word  # access, status — not plurals
        return word[:-1]  # connections → connection
    return word


def _extract_concepts(text: str, min_length: int = 4) -> set[str]:
    """Extract meaningful concepts from text — unigrams and bigrams.

    Returns normalized unigrams (suffix-stripped) plus bigrams for
    multi-word concepts like 'connection_pooling' or 'race_condition'.
    Both sides of a comparison must go through this function for
    matching to work correctly.
    """
    tokens = _TOKEN_RE.findall(text.lower())

    # Build list of (normalized_token, is_valid) for bigram adjacency tracking
    normalized: list[str | None] = []
    concepts: set[str] = set()

    for tok in tokens:
        if len(tok) >= min_length and tok not in STOPWORDS:
            norm = _normalize(tok)
            if len(norm) >= min_length:
                concepts.add(norm)
                normalized.append(norm)
            else:
                normalized.append(None)
        else:
            normalized.append(None)

    # Bigrams: adjacent meaningful words form compound concepts
    for i in range(len(normalized) - 1):
        a, b = normalized[i], normalized[i + 1]
        if a is not None and b is not None and a != b:
            concepts.add(f"{a}_{b}")

    return concepts


class CoverageEngine:
    """Finds what wisdom fails to mention — detects semantic blind spots."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore):
        self.sqlite = sqlite
        self.vector = vector

    def analyze_wisdom_coverage(self, w: Wisdom) -> dict:
        """Analyze what a single wisdom entry covers and what it misses.

        For each domain the wisdom claims to cover, checks:
        1. What concepts appear frequently in experiences for that domain
        2. Which of those concepts are absent from the wisdom's text
        3. How suspicious each absence is (based on frequency)
        """
        wisdom_concepts = _extract_concepts(
            f"{w.statement} {w.reasoning} "
            f"{' '.join(w.implications)} "
            f"{' '.join(w.applicability_conditions)} "
            f"{' '.join(w.inapplicability_conditions)} "
            f"{' '.join(t.dimension + ' ' + t.benefit + ' ' + t.cost for t in w.trade_offs)}"
        )

        domain_reports = {}
        for domain in w.applicable_domains:
            domain_reports[domain] = self._check_domain_coverage(
                wisdom_concepts, domain
            )

        return {
            "wisdom_id": w.id,
            "wisdom_concepts": sorted(wisdom_concepts),
            "domain_reports": domain_reports,
        }

    def _check_domain_coverage(
        self, wisdom_concepts: set[str], domain: str
    ) -> dict:
        """Check coverage against a specific domain's experience base."""
        experiences = self.sqlite.list_experiences(domain=domain, limit=500)
        knowledge = self.sqlite.list_knowledge(domain=domain, limit=500)

        if len(experiences) < 3:
            return {
                "status": "insufficient_data",
                "experience_count": len(experiences),
            }

        # Count concept frequency across experiences (unique per experience)
        concept_freq: Counter[str] = Counter()
        for exp in experiences:
            text = f"{exp.description} {exp.input_text} {exp.output_text}"
            concepts = _extract_concepts(text)
            concept_freq.update(concepts)

        # Also count from knowledge
        for k in knowledge:
            text = f"{k.statement} {k.explanation}"
            concepts = _extract_concepts(text)
            concept_freq.update(concepts)

        # Find concepts that appear in many experiences but not in the wisdom
        n = len(experiences)
        threshold = max(3, n * 0.2)  # At least 20% of experiences
        missing = []
        for concept, count in concept_freq.most_common(50):
            if count >= threshold and concept not in wisdom_concepts:
                missing.append({
                    "concept": concept,
                    "frequency": count,
                    "coverage_ratio": round(count / n, 2),
                })

        # Concepts the wisdom mentions that don't appear in experiences
        # (might indicate the wisdom is about the wrong things)
        unused = []
        for concept in wisdom_concepts:
            if concept_freq[concept] == 0 and len(concept) > 3:
                unused.append(concept)

        return {
            "status": "analyzed",
            "experience_count": n,
            "knowledge_count": len(knowledge),
            "missing_concepts": missing[:15],
            "unused_concepts": sorted(unused)[:10],
            "coverage_score": self._compute_coverage_score(
                wisdom_concepts, concept_freq, threshold
            ),
        }

    def _compute_coverage_score(
        self,
        wisdom_concepts: set[str],
        concept_freq: Counter[str],
        threshold: float,
    ) -> float:
        """Compute a 0-1 coverage score.

        1.0 = wisdom mentions all frequently-occurring concepts
        0.0 = wisdom mentions none of them
        """
        frequent = {c for c, count in concept_freq.items() if count >= threshold}
        if not frequent:
            return 1.0  # Nothing to cover
        covered = wisdom_concepts & frequent
        return len(covered) / len(frequent)

    def find_domain_blind_spots(self, domain: str) -> dict:
        """Find blind spots across ALL wisdom for a domain.

        This is different from per-wisdom analysis: it looks at
        what the entire wisdom corpus for a domain misses.
        """
        # Collect all wisdom concepts for this domain
        all_wisdom = self.sqlite.list_wisdom(domain=domain, limit=10000)
        active_wisdom = [
            w for w in all_wisdom
            if w.lifecycle != LifecycleState.DEPRECATED
        ]

        if not active_wisdom:
            return {
                "domain": domain,
                "status": "no_wisdom",
                "suggestion": "No active wisdom in this domain",
            }

        all_wisdom_concepts: set[str] = set()
        for w in active_wisdom:
            all_wisdom_concepts.update(_extract_concepts(
                f"{w.statement} {w.reasoning} "
                f"{' '.join(w.implications)} "
                f"{' '.join(w.applicability_conditions)}"
            ))

        # Get experience concepts
        experiences = self.sqlite.list_experiences(domain=domain, limit=500)
        concept_freq: Counter[str] = Counter()
        for exp in experiences:
            text = f"{exp.description} {exp.input_text} {exp.output_text}"
            concept_freq.update(_extract_concepts(text))

        n = len(experiences)
        threshold = max(3, n * 0.15)

        blind_spots = []
        for concept, count in concept_freq.most_common(100):
            if count >= threshold and concept not in all_wisdom_concepts:
                blind_spots.append({
                    "concept": concept,
                    "frequency": count,
                    "coverage_ratio": round(count / n, 2) if n > 0 else 0,
                })

        # Compute domain-level coverage
        frequent = {c for c, count in concept_freq.items() if count >= threshold}
        coverage = len(all_wisdom_concepts & frequent) / len(frequent) if frequent else 1.0

        return {
            "domain": domain,
            "status": "analyzed",
            "experience_count": n,
            "wisdom_count": len(active_wisdom),
            "domain_coverage": round(coverage, 3),
            "blind_spots": blind_spots[:20],
            "wisdom_concepts_count": len(all_wisdom_concepts),
            "experience_concepts_count": len(frequent),
        }

    def find_semantic_gaps(
        self, domain: str, similarity_threshold: float = 0.5
    ) -> dict:
        """Find experiences semantically distant from all wisdom entries.

        Uses vector similarity (embeddings) to catch gaps that word-level
        analysis misses — e.g., experiences about 'memoization' when wisdom
        only covers 'caching'.

        An experience is 'uncovered' when its best semantic match to any
        wisdom entry falls below the similarity threshold.
        """
        experiences = self.sqlite.list_experiences(domain=domain, limit=200)
        if len(experiences) < 3:
            return {
                "domain": domain,
                "status": "insufficient_data",
                "experience_count": len(experiences),
            }

        if self.vector.count("wisdom") == 0:
            return {
                "domain": domain,
                "status": "no_wisdom_embeddings",
                "experience_count": len(experiences),
            }

        uncovered = []
        for exp in experiences:
            text = f"{exp.description} {exp.input_text}"
            results = self.vector.search(layer="wisdom", query=text, top_k=1)
            best_sim = results[0]["similarity"] if results else 0.0
            if best_sim < similarity_threshold:
                uncovered.append({
                    "experience_id": exp.id,
                    "description": exp.description[:80],
                    "best_wisdom_similarity": round(best_sim, 3),
                })

        uncovered.sort(key=lambda x: x["best_wisdom_similarity"])

        return {
            "domain": domain,
            "status": "analyzed",
            "experience_count": len(experiences),
            "uncovered_count": len(uncovered),
            "uncovered_ratio": round(
                len(uncovered) / len(experiences), 3
            ) if experiences else 0,
            "most_distant": uncovered[:15],
        }

    def find_suspicious_wisdom(self, domain: str | None = None) -> list[dict]:
        """Find wisdom entries with suspiciously low coverage.

        These are wisdom entries that claim to cover a domain but
        miss many of the key concepts in that domain's experiences.
        """
        if domain:
            all_wisdom = self.sqlite.list_wisdom(domain=domain, limit=10000)
        else:
            all_wisdom = self.sqlite.list_wisdom(limit=10000)

        active = [w for w in all_wisdom if w.lifecycle != LifecycleState.DEPRECATED]
        suspicious = []

        for w in active:
            analysis = self.analyze_wisdom_coverage(w)
            for dom, report in analysis["domain_reports"].items():
                if report.get("status") != "analyzed":
                    continue
                score = report.get("coverage_score", 1.0)
                missing_count = len(report.get("missing_concepts", []))
                if score < 0.3 and missing_count >= 3:
                    suspicious.append({
                        "wisdom_id": w.id,
                        "statement": w.statement[:60],
                        "domain": dom,
                        "coverage_score": score,
                        "missing_count": missing_count,
                        "top_missing": [
                            m["concept"] for m in report["missing_concepts"][:5]
                        ],
                    })

        suspicious.sort(key=lambda s: s["coverage_score"])
        return suspicious
