"""Knowledge engine — extract and manage knowledge from experiences."""

from __future__ import annotations

import hashlib
from collections import Counter

from wisdom.logging_config import get_logger
from wisdom.models.common import ConfidenceScore, KnowledgeType, ValidationStatus
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.knowledge")


class KnowledgeEngine:
    """Extract, validate, and manage knowledge."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore):
        self.sqlite = sqlite
        self.vector = vector

    def add(self, knowledge: Knowledge) -> Knowledge:
        """Add a knowledge entry directly."""
        self.sqlite.save_knowledge(knowledge)
        self.vector.add(
            layer="knowledge",
            id=knowledge.id,
            text=knowledge.embedding_text,
            metadata={"domain": knowledge.domain, "type": knowledge.type.value},
        )
        logger.info("Added knowledge %s: %s", knowledge.id, knowledge.statement[:60])
        return knowledge

    def extract_from_experiences(
        self, experiences: list[Experience], domain: str = ""
    ) -> list[Knowledge]:
        """Fallback extraction without LLM — keyword frequency analysis.

        This produces basic knowledge entries with low confidence that need review.
        For LLM-powered extraction, use wisdom.llm.extraction.extract_knowledge().
        """
        if not experiences:
            return []

        # Analyze patterns across experiences
        word_freq: Counter[str] = Counter()
        result_patterns: dict[str, list[str]] = {"success": [], "failure": []}
        domains_seen: set[str] = set()

        for exp in experiences:
            text = f"{exp.description} {exp.input_text} {exp.output_text}".lower()
            # Extract meaningful words (length > 3, not common stopwords)
            stopwords = {
                "the", "and", "for", "that", "this", "with", "from", "have", "been",
                "was", "were", "are", "not", "but", "what", "when", "how", "which",
                "their", "there", "them", "then", "than", "into", "also", "just",
                "more", "some", "could", "would", "should", "about", "after", "before",
                "only", "other", "very", "will", "each", "make", "like", "over",
            }
            words = [w for w in text.split() if len(w) > 3 and w not in stopwords]
            word_freq.update(words)

            if exp.result.value in ("success", "failure"):
                result_patterns[exp.result.value].append(exp.description)

            if exp.domain:
                domains_seen.add(exp.domain)

        target_domain = domain or (domains_seen.pop() if len(domains_seen) == 1 else "")
        extracted: list[Knowledge] = []

        # Pattern 1: Frequently co-occurring terms suggest patterns
        top_terms = [term for term, count in word_freq.most_common(10) if count >= 2]
        if top_terms:
            statement = f"Common concepts in this domain: {', '.join(top_terms[:5])}"
            k = Knowledge(
                type=KnowledgeType.PATTERN,
                statement=statement,
                explanation=f"Identified from frequency analysis of {len(experiences)} experiences.",
                domain=target_domain,
                specificity=0.7,
                confidence=ConfidenceScore(empirical=0.3, theoretical=0.2, observational=0.3),
                supporting_count=len(experiences),
                source_experience_ids=[e.id for e in experiences],
                tags=["auto_extracted", "needs_review"],
            )
            extracted.append(k)

        # Pattern 2: Success/failure patterns
        success_count = len(result_patterns["success"])
        failure_count = len(result_patterns["failure"])
        total = success_count + failure_count
        if total >= 3:
            rate = success_count / total if total else 0
            statement = f"Success rate in this context is approximately {rate:.0%} ({success_count}/{total})"
            k = Knowledge(
                type=KnowledgeType.FACT,
                statement=statement,
                explanation=f"Based on {total} outcomes across {len(experiences)} experiences.",
                domain=target_domain,
                specificity=0.6,
                confidence=ConfidenceScore(
                    empirical=min(0.6, total / 15),
                    theoretical=0.2,
                    observational=min(0.5, total / 20),
                ),
                supporting_count=success_count,
                contradicting_count=failure_count,
                source_experience_ids=[e.id for e in experiences],
                tags=["auto_extracted", "needs_review"],
            )
            extracted.append(k)

        # Pattern 3: If there are clear failure patterns, extract them
        if failure_count >= 2:
            # Find common words in failure descriptions
            fail_words: Counter[str] = Counter()
            for desc in result_patterns["failure"]:
                words = [w for w in desc.lower().split() if len(w) > 3 and w not in stopwords]
                fail_words.update(words)
            common_fail_terms = [t for t, c in fail_words.most_common(5) if c >= 2]
            if common_fail_terms:
                statement = f"Failures commonly involve: {', '.join(common_fail_terms)}"
                k = Knowledge(
                    type=KnowledgeType.HEURISTIC,
                    statement=statement,
                    explanation=f"Pattern from {failure_count} failure cases.",
                    domain=target_domain,
                    specificity=0.6,
                    confidence=ConfidenceScore(empirical=0.4, theoretical=0.2, observational=0.3),
                    supporting_count=failure_count,
                    source_experience_ids=[e.id for e in experiences],
                    tags=["auto_extracted", "needs_review", "failure_pattern"],
                )
                extracted.append(k)

        # Save and index all extracted knowledge
        for k in extracted:
            self.add(k)

        # Mark experiences as processed
        self.sqlite.mark_processed([e.id for e in experiences])
        logger.info(
            "Extracted %d knowledge entries from %d experiences",
            len(extracted), len(experiences),
        )
        return extracted

    def get(self, id: str) -> Knowledge | None:
        return self.sqlite.get_knowledge(id)

    def list(self, domain: str | None = None, limit: int = 100, offset: int = 0) -> list[Knowledge]:
        return self.sqlite.list_knowledge(domain=domain, limit=limit, offset=offset)

    def search(self, query: str, top_k: int = 10, domain: str | None = None) -> list[dict]:
        where = {"domain": domain} if domain else None
        results = self.vector.search(layer="knowledge", query=query, top_k=top_k, where=where)
        enriched = []
        for r in results:
            k = self.sqlite.get_knowledge(r["id"])
            if k:
                enriched.append({"knowledge": k, "similarity": r["similarity"]})
        return enriched

    def validate(self, id: str, is_valid: bool, details: str = "") -> Knowledge | None:
        """Mark knowledge as validated or challenged."""
        k = self.sqlite.get_knowledge(id)
        if not k:
            return None
        old_status = k.validation_status
        k.validation_status = ValidationStatus.VALIDATED if is_valid else ValidationStatus.CHALLENGED
        k.touch()
        self.sqlite.save_knowledge(k)
        logger.info(
            "Knowledge %s: %s -> %s (%s)",
            id, old_status.value, k.validation_status.value, details,
        )
        return k

    def delete(self, id: str) -> bool:
        self.vector.delete(layer="knowledge", id=id)
        return self.sqlite.delete_knowledge(id)

    def count(self, domain: str | None = None, unsynthesized: bool = False) -> int:
        return self.sqlite.count_knowledge(domain=domain, unsynthesized=unsynthesized)

    def get_unsynthesized(self, domain: str | None = None, limit: int = 50) -> list[Knowledge]:
        return self.sqlite.get_unsynthesized(domain=domain, limit=limit)
