"""Propagation engine — consequence cascading when wisdom fails.

When wisdom is found to be wrong, everything downstream is suspect:
- Knowledge that was used to synthesize it may be flawed
- Other wisdom derived from the same knowledge is contaminated
- Experiences where this wisdom was applied are now unreliable data points

This is not bookkeeping. This is the system's immune response.
Bad wisdom that propagated to 50 downstream decisions should create
50 problems, not a 0.08 confidence decrement.
"""

from __future__ import annotations

from wisdom.config import WisdomConfig
from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState, Relationship, RelationshipType
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.propagation")


class ContaminationResult:
    """Summary of a contamination cascade."""

    def __init__(self):
        self.source_wisdom_id: str = ""
        self.affected_wisdom: list[dict] = []
        self.affected_knowledge: list[dict] = []
        self.contaminated_experiences: int = 0
        self.relationship_affected: list[dict] = []
        self.total_penalty_events: int = 0

    @property
    def total_affected(self) -> int:
        return (
            len(self.affected_wisdom)
            + len(self.affected_knowledge)
            + self.contaminated_experiences
            + len(self.relationship_affected)
        )

    def to_dict(self) -> dict:
        return {
            "source_wisdom_id": self.source_wisdom_id,
            "affected_wisdom": self.affected_wisdom,
            "affected_knowledge": self.affected_knowledge,
            "contaminated_experiences": self.contaminated_experiences,
            "relationship_affected": self.relationship_affected,
            "total_affected": self.total_affected,
            "total_penalty_events": self.total_penalty_events,
        }


class PropagationEngine:
    """Cascades consequences when wisdom is found to be wrong."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore, config: WisdomConfig):
        self.sqlite = sqlite
        self.vector = vector
        self.config = config

    def _relationship_penalty(
        self, rel: Relationship, failed_id: str, severity: float
    ) -> float:
        """Compute the penalty for a related wisdom entry when wisdom fails.

        The penalty depends on the relationship type and directionality.
        These are intentionally lighter than provenance-based penalties —
        relationships are softer connections than shared knowledge.
        """
        is_source = rel.source_id == failed_id
        rtype = rel.relationship

        if rtype == RelationshipType.SUPPORTS:
            if is_source:
                # Failed wisdom supported the other → other loses backing
                return rel.strength * severity * 0.03
            # Other supported failed wisdom → other is fine
            return 0.0

        if rtype == RelationshipType.DERIVED_FROM:
            if is_source:
                # Failed wisdom was derived from other → other might be flawed
                return rel.strength * severity * 0.02
            # Other was derived from failed wisdom → suspect
            return rel.strength * severity * 0.05

        if rtype == RelationshipType.GENERALIZES:
            if is_source:
                # Failed general rule; specific cases might still hold
                return 0.0
            # Other generalizes failed wisdom → very mild
            return rel.strength * severity * 0.01

        if rtype == RelationshipType.SPECIALIZES:
            if is_source:
                # Failed specific case weakens the general version
                return rel.strength * severity * 0.02
            # Other specializes failed wisdom → specific might still hold
            return 0.0

        if rtype == RelationshipType.COMPLEMENTS:
            # Symmetric: complement loses context
            return rel.strength * severity * 0.02

        if rtype == RelationshipType.CONFLICTS:
            # Conflict with failed wisdom is a positive signal for the other
            return 0.0

        return 0.0

    def cascade_failure(self, wisdom_id: str, severity: float = 1.0) -> ContaminationResult:
        """Cascade the consequences of a wisdom failure through the provenance graph.

        Args:
            wisdom_id: The failed/deprecated wisdom entry
            severity: How bad the failure is (0.0 = minor, 1.0 = completely wrong)

        Returns:
            ContaminationResult describing what was affected
        """
        result = ContaminationResult()
        result.source_wisdom_id = wisdom_id

        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            return result

        # 1. Find sibling wisdom — other wisdom derived from the same knowledge
        if w.source_knowledge_ids:
            siblings = self.sqlite.find_wisdom_sharing_knowledge(w.source_knowledge_ids)
            for sibling in siblings:
                if sibling.id == wisdom_id:
                    continue
                if sibling.lifecycle == LifecycleState.DEPRECATED:
                    continue

                # Penalty proportional to knowledge overlap
                sibling_knowledge = set(sibling.source_knowledge_ids)
                failed_knowledge = set(w.source_knowledge_ids)
                overlap = len(sibling_knowledge & failed_knowledge)
                total = len(sibling_knowledge | failed_knowledge)
                overlap_ratio = overlap / total if total > 0 else 0

                if overlap_ratio < 0.1:
                    continue  # Negligible overlap

                penalty = severity * overlap_ratio * 0.1
                old_conf = sibling.confidence.overall
                # Contamination is empirical evidence — the shared knowledge was flawed
                sibling.confidence.apply_delta("empirical", -penalty)
                sibling.touch()
                self.sqlite.update_wisdom(sibling)

                self.sqlite.log_confidence_change(
                    "wisdom", sibling.id, old_conf, sibling.confidence.overall,
                    "contamination_cascade",
                    f"Sibling of failed wisdom {wisdom_id} (overlap: {overlap_ratio:.2f})",
                )
                self.sqlite.log_contamination(
                    source_wisdom_id=wisdom_id,
                    affected_entity_id=sibling.id,
                    affected_entity_type="wisdom",
                    penalty_applied=penalty,
                    reason=f"Knowledge overlap {overlap_ratio:.2f} with failed wisdom",
                )

                result.affected_wisdom.append({
                    "id": sibling.id,
                    "statement": sibling.statement[:60],
                    "overlap_ratio": round(overlap_ratio, 3),
                    "penalty": round(penalty, 4),
                    "new_confidence": round(sibling.confidence.overall, 3),
                })
                result.total_penalty_events += 1

                logger.info(
                    "Contamination: wisdom %s penalized %.4f (overlap %.2f with %s)",
                    sibling.id, penalty, overlap_ratio, wisdom_id,
                )

        # 2. Flag contaminated knowledge entries
        for kid in w.source_knowledge_ids:
            k = self.sqlite.get_knowledge(kid)
            if not k:
                continue

            penalty = severity * 0.05
            old_conf = k.confidence.overall
            # Source knowledge is empirically tainted by the failed wisdom
            k.confidence.apply_delta("empirical", -penalty)
            k.touch()
            self.sqlite.save_knowledge(k)

            self.sqlite.log_confidence_change(
                "knowledge", k.id, old_conf, k.confidence.overall,
                "contamination_cascade",
                f"Source knowledge of failed wisdom {wisdom_id}",
            )
            self.sqlite.log_contamination(
                source_wisdom_id=wisdom_id,
                affected_entity_id=k.id,
                affected_entity_type="knowledge",
                penalty_applied=penalty,
                reason=f"Source knowledge of failed wisdom",
            )

            result.affected_knowledge.append({
                "id": k.id,
                "statement": k.statement[:60],
                "penalty": round(penalty, 4),
                "new_confidence": round(k.confidence.overall, 3),
            })
            result.total_penalty_events += 1

        # 3. Mark application experiences as contaminated
        contaminated_exps = self.sqlite.list_experiences_for_wisdom(wisdom_id)
        for exp in contaminated_exps:
            exp.metadata["contaminated"] = "true"
            exp.metadata["contamination_source"] = wisdom_id
            exp.quality_score = max(0.0, exp.quality_score * (1.0 - severity * 0.5))
            self.sqlite.save_experience(exp)

            self.sqlite.log_contamination(
                source_wisdom_id=wisdom_id,
                affected_entity_id=exp.id,
                affected_entity_type="experience",
                penalty_applied=severity * 0.5,
                reason="Application of failed wisdom",
            )

        result.contaminated_experiences = len(contaminated_exps)
        result.total_penalty_events += len(contaminated_exps)

        # 4. Cascade through explicit relationships (lighter than provenance)
        already_penalized = {d["id"] for d in result.affected_wisdom}
        rels = self.sqlite.get_relationships(wisdom_id, "wisdom")
        for rel in rels:
            other_id = (
                rel.target_id if rel.source_id == wisdom_id else rel.source_id
            )
            if other_id in already_penalized or other_id == wisdom_id:
                continue

            penalty = self._relationship_penalty(rel, wisdom_id, severity)
            if penalty <= 0:
                continue

            other = self.sqlite.get_wisdom(other_id)
            if not other or other.lifecycle == LifecycleState.DEPRECATED:
                continue

            old_conf = other.confidence.overall
            other.confidence.apply_delta("empirical", -penalty)
            other.touch()
            self.sqlite.update_wisdom(other)

            self.sqlite.log_confidence_change(
                "wisdom", other.id, old_conf, other.confidence.overall,
                "relationship_cascade",
                f"{rel.relationship.value} relationship with failed wisdom {wisdom_id}",
            )
            self.sqlite.log_contamination(
                source_wisdom_id=wisdom_id,
                affected_entity_id=other.id,
                affected_entity_type="wisdom",
                penalty_applied=penalty,
                reason=f"{rel.relationship.value} relationship cascade",
            )

            result.relationship_affected.append({
                "id": other.id,
                "statement": other.statement[:60],
                "relationship": rel.relationship.value,
                "strength": rel.strength,
                "penalty": round(penalty, 4),
                "new_confidence": round(other.confidence.overall, 3),
            })
            result.total_penalty_events += 1

            logger.info(
                "Relationship cascade: wisdom %s penalized %.4f "
                "(%s relationship with %s)",
                other.id, penalty, rel.relationship.value, wisdom_id,
            )

        logger.info(
            "Contamination cascade from wisdom %s: "
            "%d wisdom (provenance), %d knowledge, %d experiences, "
            "%d wisdom (relationships) affected",
            wisdom_id, len(result.affected_wisdom),
            len(result.affected_knowledge), result.contaminated_experiences,
            len(result.relationship_affected),
        )
        return result

    def trace_provenance(self, wisdom_id: str) -> dict:
        """Trace the full provenance chain of a wisdom entry.

        Returns a tree showing: wisdom <- knowledge <- experiences
        """
        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            return {"error": f"Wisdom not found: {wisdom_id}"}

        knowledge_chain = []
        for kid in w.source_knowledge_ids:
            k = self.sqlite.get_knowledge(kid)
            if k:
                exp_chain = []
                for eid in k.source_experience_ids:
                    exp = self.sqlite.get_experience(eid)
                    if exp:
                        exp_chain.append({
                            "id": exp.id,
                            "description": exp.description[:60],
                            "result": exp.result.value,
                            "domain": exp.domain,
                            "contaminated": exp.metadata.get("contaminated", "false") == "true",
                        })
                knowledge_chain.append({
                    "id": k.id,
                    "statement": k.statement[:60],
                    "confidence": k.confidence.overall,
                    "experiences": exp_chain,
                })

        # Find application experiences
        linked_exps = self.sqlite.list_experiences_for_wisdom(wisdom_id)
        applications = [
            {
                "id": e.id,
                "result": e.result.value,
                "domain": e.domain,
                "contaminated": e.metadata.get("contaminated", "false") == "true",
            }
            for e in linked_exps
        ]

        # Find contamination history
        contamination = self.sqlite.get_contamination_history(wisdom_id)

        # Find explicit relationships
        rels = self.sqlite.get_relationships(wisdom_id, "wisdom")
        related = []
        for rel in rels:
            other_id = (
                rel.target_id if rel.source_id == wisdom_id else rel.source_id
            )
            other = self.sqlite.get_wisdom(other_id)
            if other:
                related.append({
                    "id": other.id,
                    "statement": other.statement[:60],
                    "relationship": rel.relationship.value,
                    "strength": rel.strength,
                    "direction": "outgoing" if rel.source_id == wisdom_id else "incoming",
                })

        return {
            "wisdom": {
                "id": w.id,
                "statement": w.statement[:80],
                "confidence": w.confidence.overall,
                "lifecycle": w.lifecycle.value,
                "creation_method": w.creation_method.value,
            },
            "source_knowledge": knowledge_chain,
            "applications": applications,
            "relationships": related,
            "contamination_history": contamination[:20],
        }
