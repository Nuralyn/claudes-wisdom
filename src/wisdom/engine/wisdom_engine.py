"""Wisdom engine — synthesize and manage wisdom with lifecycle tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from wisdom.engine.lifecycle import LifecycleManager
from wisdom.logging_config import get_logger
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    LifecycleState,
    Relationship,
    RelationshipType,
    TradeOff,
    WisdomType,
)
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.wisdom")


class WisdomEngine:
    """Synthesize, lifecycle-manage, and operate on wisdom entries."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore, lifecycle: LifecycleManager):
        self.sqlite = sqlite
        self.vector = vector
        self.lifecycle = lifecycle

    def add(
        self,
        statement: str,
        reasoning: str = "",
        wisdom_type: WisdomType = WisdomType.PRINCIPLE,
        domains: list[str] | None = None,
        applicability_conditions: list[str] | None = None,
        inapplicability_conditions: list[str] | None = None,
        trade_offs: list[TradeOff] | None = None,
        implications: list[str] | None = None,
        counterexamples: list[str] | None = None,
        confidence: ConfidenceScore | None = None,
        creation_method: CreationMethod = CreationMethod.HUMAN_INPUT,
        tags: list[str] | None = None,
        source_knowledge_ids: list[str] | None = None,
    ) -> Wisdom:
        """Add a wisdom entry directly (human expert path or pipeline)."""
        w = Wisdom(
            type=wisdom_type,
            statement=statement,
            reasoning=reasoning,
            applicable_domains=domains or [],
            applicability_conditions=applicability_conditions or [],
            inapplicability_conditions=inapplicability_conditions or [],
            trade_offs=trade_offs or [],
            implications=implications or [],
            counterexamples=counterexamples or [],
            confidence=confidence or ConfidenceScore(empirical=0.5, theoretical=0.5, observational=0.5),
            lifecycle=LifecycleState.EMERGING,
            creation_method=creation_method,
            tags=tags or [],
            source_knowledge_ids=source_knowledge_ids or [],
        )
        self.sqlite.save_wisdom(w)
        self.vector.add(
            layer="wisdom",
            id=w.id,
            text=w.embedding_text,
            metadata={
                "domains": ",".join(w.applicable_domains),
                "type": w.type.value,
                "lifecycle": w.lifecycle.value,
            },
        )
        logger.info("Added wisdom %s: %s", w.id, w.statement[:60])
        return w

    def synthesize_from_knowledge(
        self, knowledge_entries: list[Knowledge], domain: str = ""
    ) -> list[Wisdom]:
        """Fallback synthesis without LLM — group and merge knowledge.

        For LLM-powered synthesis, use wisdom.llm.synthesis.synthesize_wisdom().
        """
        if not knowledge_entries:
            return []

        # Group knowledge by domain
        by_domain: dict[str, list[Knowledge]] = {}
        for k in knowledge_entries:
            d = k.domain or domain or "general"
            by_domain.setdefault(d, []).append(k)

        synthesized: list[Wisdom] = []

        for dom, entries in by_domain.items():
            if len(entries) < 2:
                # Single knowledge entry — promote to wisdom directly
                k = entries[0]
                w = self.add(
                    statement=k.statement,
                    reasoning=k.explanation or "Promoted from single knowledge entry.",
                    wisdom_type=WisdomType.HEURISTIC,
                    domains=[dom],
                    confidence=ConfidenceScore(
                        theoretical=k.confidence.theoretical * 0.6,
                        empirical=k.confidence.empirical * 0.6,
                        observational=k.confidence.observational * 0.6,
                    ),
                    creation_method=CreationMethod.PIPELINE,
                    tags=["auto_synthesized"],
                    source_knowledge_ids=[k.id],
                )
                synthesized.append(w)
                continue

            # Multiple entries — merge into higher-order principle
            statements = [k.statement for k in entries]
            avg_empirical = sum(k.confidence.empirical for k in entries) / len(entries)
            combined_statement = f"Across {len(entries)} observations in {dom}: " + "; ".join(
                s[:80] for s in statements[:5]
            )
            if len(statements) > 5:
                combined_statement += f" (and {len(statements) - 5} more)"

            explanations = [k.explanation for k in entries if k.explanation]
            combined_reasoning = " | ".join(explanations[:3]) if explanations else "Synthesized from multiple knowledge entries."

            w = self.add(
                statement=combined_statement,
                reasoning=combined_reasoning,
                wisdom_type=WisdomType.META_PATTERN,
                domains=[dom],
                confidence=ConfidenceScore(
                    theoretical=avg_empirical * 0.4,
                    empirical=avg_empirical * 0.5,
                    observational=avg_empirical * 0.5,
                ),
                creation_method=CreationMethod.PIPELINE,
                tags=["auto_synthesized", "needs_review"],
                source_knowledge_ids=[k.id for k in entries],
            )
            synthesized.append(w)

        # Mark knowledge as synthesized
        self.sqlite.mark_synthesized([k.id for k in knowledge_entries])
        logger.info(
            "Synthesized %d wisdom entries from %d knowledge entries",
            len(synthesized), len(knowledge_entries),
        )
        return synthesized

    def get(self, id: str) -> Wisdom | None:
        return self.sqlite.get_wisdom(id)

    def list(
        self,
        domain: str | None = None,
        lifecycle: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Wisdom]:
        return self.sqlite.list_wisdom(domain=domain, lifecycle=lifecycle, limit=limit, offset=offset)

    def search(self, query: str, top_k: int = 10, domain: str | None = None) -> list[dict]:
        where = None
        if domain:
            where = {"domains": {"$contains": domain}}
        results = self.vector.search(layer="wisdom", query=query, top_k=top_k, where=where)
        enriched = []
        for r in results:
            w = self.sqlite.get_wisdom(r["id"])
            if w:
                enriched.append({"wisdom": w, "similarity": r["similarity"]})
        return enriched

    def delete(self, id: str) -> bool:
        self.vector.delete(layer="wisdom", id=id)
        return self.sqlite.delete_wisdom(id)

    def count(self, domain: str | None = None) -> int:
        return self.sqlite.count_wisdom(domain=domain)

    def deprecate(self, id: str, reason: str) -> Wisdom | None:
        """Deprecate a wisdom entry. Delegates to LifecycleManager."""
        w = self.sqlite.get_wisdom(id)
        if not w:
            return None
        self.lifecycle.force_deprecate(w, reason)
        return w

    def challenge(self, id: str, reason: str = "") -> Wisdom | None:
        """Challenge a wisdom entry. Delegates to LifecycleManager."""
        w = self.sqlite.get_wisdom(id)
        if not w:
            return None
        self.lifecycle.force_challenge(w, reason)
        return w

    def relate(
        self,
        source_id: str,
        target_id: str,
        relationship: RelationshipType,
        strength: float = 0.5,
    ) -> Relationship | None:
        """Create a relationship between two wisdom entries."""
        source = self.sqlite.get_wisdom(source_id)
        target = self.sqlite.get_wisdom(target_id)
        if not source or not target:
            return None
        rel = Relationship(
            source_id=source_id,
            source_type="wisdom",
            target_id=target_id,
            target_type="wisdom",
            relationship=relationship,
            strength=strength,
        )
        self.sqlite.save_relationship(rel)
        logger.info("Related wisdom %s -[%s]-> %s", source_id, relationship.value, target_id)
        return rel

    def transfer(self, id: str, to_domain: str) -> Wisdom | None:
        """Transfer wisdom to a new domain (creates a copy in EMERGING state)."""
        original = self.sqlite.get_wisdom(id)
        if not original:
            return None

        new_w = self.add(
            statement=original.statement,
            reasoning=original.reasoning + f" [Transferred from {', '.join(original.applicable_domains) or 'unknown'}]",
            wisdom_type=original.type,
            domains=[to_domain],
            applicability_conditions=original.applicability_conditions,
            inapplicability_conditions=original.inapplicability_conditions,
            trade_offs=original.trade_offs,
            implications=original.implications,
            counterexamples=original.counterexamples,
            confidence=ConfidenceScore(
                theoretical=original.confidence.theoretical * 0.5,
                empirical=0.2,
                observational=original.confidence.observational * 0.5,
            ),
            creation_method=CreationMethod.PIPELINE,
            tags=["transferred", f"from:{original.id}"],
            source_knowledge_ids=original.source_knowledge_ids,
        )

        # Create a relationship between original and transfer
        self.sqlite.save_relationship(Relationship(
            source_id=original.id,
            source_type="wisdom",
            target_id=new_w.id,
            target_type="wisdom",
            relationship=RelationshipType.GENERALIZES,
            strength=0.7,
        ))

        logger.info(
            "Transferred wisdom %s to domain '%s' as %s", id, to_domain, new_w.id,
        )
        return new_w

    def check_lifecycle_transitions(self, w: Wisdom) -> Wisdom:
        """Delegate lifecycle evaluation to LifecycleManager."""
        self.lifecycle.evaluate(w)
        return w
