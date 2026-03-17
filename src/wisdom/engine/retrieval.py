"""Unified multi-layer retrieval engine with scoring and temporal decay.

The retrieval scoring combines four factors:
    semantic_similarity — how well the query matches the content
    effective_confidence — confidence adjusted for temporal decay AND validation
    applicability — domain match
    recency — freshness

Unvalidated wisdom is discounted. This is not a bug — it's the system
being structurally skeptical about its own outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from wisdom.config import WisdomConfig
from wisdom.engine.evolution import _compute_temporal_decay
from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.retrieval")


class ScoredResult:
    """A search result with multi-factor scoring."""

    def __init__(
        self,
        entity: Wisdom | Knowledge,
        layer: str,
        semantic_similarity: float,
        raw_confidence: float,
        applicability: float,
        recency: float,
        effective_confidence: float,
        final_score: float,
    ):
        self.entity = entity
        self.layer = layer
        self.semantic_similarity = semantic_similarity
        self.raw_confidence = raw_confidence
        self.applicability = applicability
        self.recency = recency
        self.effective_confidence = effective_confidence
        self.final_score = final_score

    def to_dict(self) -> dict:
        return {
            "id": self.entity.id,
            "layer": self.layer,
            "statement": self.entity.statement,
            "final_score": round(self.final_score, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "effective_confidence": round(self.effective_confidence, 4),
            "applicability": round(self.applicability, 4),
            "recency": round(self.recency, 4),
            "entity": self.entity,
        }


class RetrievalEngine:
    """Unified search across knowledge and wisdom layers with multi-factor scoring."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore, config: WisdomConfig):
        self.sqlite = sqlite
        self.vector = vector
        self.config = config

    def _compute_recency(self, entity: Wisdom | Knowledge) -> float:
        """Recency score: 1.0 / (1.0 + days_since_update / 365.0)"""
        days = entity.age_days
        return 1.0 / (1.0 + days / 365.0)

    def _compute_applicability(self, entity: Wisdom | Knowledge, domain: str | None) -> float:
        """Applicability score based on domain match."""
        if not domain:
            return 0.5  # Neutral when no domain specified

        if isinstance(entity, Wisdom):
            if domain in entity.applicable_domains:
                return 1.0
            for d in entity.applicable_domains:
                if d.startswith(domain + "/") or domain.startswith(d + "/"):
                    return 0.5
            if not entity.applicable_domains:
                return 0.3
            return 0.0
        elif isinstance(entity, Knowledge):
            if entity.domain == domain:
                return 1.0
            if entity.domain and (
                entity.domain.startswith(domain + "/") or domain.startswith(entity.domain + "/")
            ):
                return 0.5
            if entity.specificity < 0.3:
                return 0.3
            return 0.0
        return 0.0

    def _compute_effective_confidence(self, entity: Wisdom | Knowledge) -> float:
        """Compute confidence with temporal decay AND validation discount.

        Unvalidated wisdom is discounted by 40%. This makes the system
        structurally skeptical about its own unverified outputs.
        """
        raw = entity.confidence.overall
        decay_rate = self.config.confidence.decay_rate_per_month

        # Apply temporal decay
        effective = _compute_temporal_decay(raw, entity.age_days, decay_rate)

        # Apply validation discount for wisdom entries
        if isinstance(entity, Wisdom):
            events = self.sqlite.get_validation_events(entity.id)
            has_external = any(
                e["source"] in ("external", "peer", "adversarial")
                and e["verdict"] in ("confirmed", "confirmed_with_caveats")
                for e in events
            )
            if not has_external:
                # Unvalidated — discount
                effective *= 0.6

        return max(0.0, min(1.0, effective))

    def _score_result(
        self,
        entity: Wisdom | Knowledge,
        layer: str,
        semantic_similarity: float,
        domain: str | None,
    ) -> ScoredResult:
        """Compute the multi-factor score for a result."""
        w = self.config.retrieval
        raw_confidence = entity.confidence.overall
        effective_confidence = self._compute_effective_confidence(entity)
        applicability = self._compute_applicability(entity, domain)
        recency = self._compute_recency(entity)

        final_score = (
            w.semantic * semantic_similarity
            + w.confidence * effective_confidence
            + w.applicability * applicability
            + w.recency * recency
        )

        return ScoredResult(
            entity=entity,
            layer=layer,
            semantic_similarity=semantic_similarity,
            raw_confidence=raw_confidence,
            applicability=applicability,
            recency=recency,
            effective_confidence=effective_confidence,
            final_score=final_score,
        )

    def search(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 10,
        layers: list[str] | None = None,
        min_confidence: float = 0.0,
        include_deprecated: bool = False,
    ) -> list[ScoredResult]:
        """Search across wisdom and knowledge layers with unified scoring."""
        search_layers = layers or ["wisdom", "knowledge"]
        results: list[ScoredResult] = []

        if "wisdom" in search_layers:
            vector_results = self.vector.search(
                layer="wisdom", query=query, top_k=top_k * 2
            )
            for vr in vector_results:
                w = self.sqlite.get_wisdom(vr["id"])
                if not w:
                    continue
                if not include_deprecated and w.lifecycle == LifecycleState.DEPRECATED:
                    continue
                scored = self._score_result(w, "wisdom", vr["similarity"], domain)
                if scored.effective_confidence >= min_confidence:
                    results.append(scored)

        if "knowledge" in search_layers:
            vector_results = self.vector.search(
                layer="knowledge", query=query, top_k=top_k * 2
            )
            for vr in vector_results:
                k = self.sqlite.get_knowledge(vr["id"])
                if not k:
                    continue
                scored = self._score_result(k, "knowledge", vr["similarity"], domain)
                if scored.effective_confidence >= min_confidence:
                    results.append(scored)

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def search_for_task(
        self, task_description: str, domain: str | None = None, top_k: int = 5
    ) -> list[ScoredResult]:
        """Convenience method: search specifically for task-relevant wisdom."""
        return self.search(
            query=task_description,
            domain=domain,
            top_k=top_k,
            layers=["wisdom"],
            min_confidence=0.1,
        )

    def find_contradictions(self, wisdom_id: str | None = None) -> list[dict]:
        """Find conflicting wisdom entries."""
        if wisdom_id:
            conflicts = self.sqlite.find_conflicts(wisdom_id)
            results = []
            for rel in conflicts:
                other_id = rel.target_id if rel.source_id == wisdom_id else rel.source_id
                other = self.sqlite.get_wisdom(other_id)
                if other:
                    results.append({
                        "wisdom_id": wisdom_id,
                        "conflicting_id": other_id,
                        "conflicting_statement": other.statement,
                        "relationship": rel,
                    })
            return results
        else:
            all_wisdom = self.sqlite.list_wisdom(limit=1000)
            conflicts = []
            seen: set[tuple[str, str]] = set()
            for w in all_wisdom:
                if w.lifecycle == LifecycleState.DEPRECATED:
                    continue
                rels = self.sqlite.find_conflicts(w.id)
                for rel in rels:
                    other_id = rel.target_id if rel.source_id == w.id else rel.source_id
                    pair = tuple(sorted([w.id, other_id]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    other = self.sqlite.get_wisdom(other_id)
                    if other and other.lifecycle != LifecycleState.DEPRECATED:
                        conflicts.append({
                            "a_id": w.id,
                            "a_statement": w.statement,
                            "b_id": other.id,
                            "b_statement": other.statement,
                            "strength": rel.strength,
                        })
            return conflicts

    def compose_wisdom(
        self, query: str, domain: str | None = None, top_k: int = 5
    ) -> dict:
        """Compose multiple wisdom entries into structured guidance."""
        results = self.search(
            query=query, domain=domain, top_k=top_k, layers=["wisdom"]
        )
        if not results:
            return {"entries": [], "composition": "No relevant wisdom found.", "conflicts": []}

        entries = []
        conflict_pairs = []

        for i, r in enumerate(results):
            entry = {
                "index": i + 1,
                "statement": r.entity.statement,
                "confidence": round(r.effective_confidence, 2),
                "domains": getattr(r.entity, "applicable_domains", []),
                "type": r.entity.type.value if hasattr(r.entity, "type") else "unknown",
                "score": round(r.final_score, 3),
            }
            if isinstance(r.entity, Wisdom) and r.entity.trade_offs:
                entry["trade_offs"] = [
                    f"{t.benefit} vs {t.cost}" for t in r.entity.trade_offs
                ]
            if isinstance(r.entity, Wisdom) and r.entity.applicability_conditions:
                entry["applies_when"] = r.entity.applicability_conditions
            entries.append(entry)

        # Check for conflicts between returned entries
        for i, r1 in enumerate(results):
            for j, r2 in enumerate(results):
                if j <= i:
                    continue
                rels = self.sqlite.find_conflicts(r1.entity.id)
                for rel in rels:
                    other_id = rel.target_id if rel.source_id == r1.entity.id else rel.source_id
                    if other_id == r2.entity.id:
                        conflict_pairs.append({
                            "a": i + 1,
                            "b": j + 1,
                            "note": f"Entries [{i+1}] and [{j+1}] may conflict",
                        })

        # Build composition text
        lines = []
        for e in entries:
            line = f"[{e['index']}] (confidence: {e['confidence']}, type: {e['type']})"
            lines.append(line)
            lines.append(f"  {e['statement']}")
            if "applies_when" in e:
                lines.append(f"  APPLIES WHEN: {'; '.join(e['applies_when'])}")
            if "trade_offs" in e:
                lines.append(f"  TRADE-OFFS: {'; '.join(e['trade_offs'])}")
            lines.append("")

        if conflict_pairs:
            for cp in conflict_pairs:
                lines.append(f"NOTE: {cp['note']}")
        elif len(entries) > 1:
            lines.append(
                f"NOTE: Entries [{', '.join(str(e['index']) for e in entries)}] appear complementary."
            )

        return {
            "entries": entries,
            "composition": "\n".join(lines),
            "conflicts": conflict_pairs,
        }
