"""Validation engine — external verification framework.

Wisdom should not trust itself. This engine tracks external validation
events, computes validation-adjusted confidence, and gates lifecycle
promotions on evidence from outside the system.

Validation sources (weakest to strongest):
    self_report:  User says "this helped" — weakest signal
    peer:         Another wisdom entry or knowledge entry supports it
    external:     Human expert or external system confirms it
    adversarial:  Wisdom survived active devil's advocate challenge
"""

from __future__ import annotations

from wisdom.logging_config import get_logger
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore

logger = get_logger("engine.validation")

# How much each validation source contributes to validation confidence
SOURCE_WEIGHTS = {
    "self_report": 0.1,
    "peer": 0.3,
    "external": 0.6,
    "adversarial": 0.8,
}

# How each verdict modifies the weight
VERDICT_MULTIPLIERS = {
    "confirmed": 1.0,
    "confirmed_with_caveats": 0.6,
    "challenged": -0.5,
    "refuted": -1.0,
}

# Discount factor applied to unvalidated wisdom in retrieval
UNVALIDATED_DISCOUNT = 0.6


class ValidationEngine:
    """Manages external verification of wisdom entries."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def validate(
        self,
        wisdom_id: str,
        source: str,
        verdict: str,
        evidence: str = "",
        validator: str = "",
    ) -> dict:
        """Record a validation event and compute its impact.

        Args:
            wisdom_id: The wisdom entry being validated
            source: Who/what is validating — 'self_report', 'peer', 'external', 'adversarial'
            verdict: The judgment — 'confirmed', 'confirmed_with_caveats', 'challenged', 'refuted'
            evidence: Supporting evidence or reasoning
            validator: Identifier for the validator (human name, system ID, etc.)

        Returns:
            Dict with validation_score and summary
        """
        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            return {"error": f"Wisdom not found: {wisdom_id}"}

        if source not in SOURCE_WEIGHTS:
            return {"error": f"Invalid source: {source}. Use: {list(SOURCE_WEIGHTS.keys())}"}
        if verdict not in VERDICT_MULTIPLIERS:
            return {"error": f"Invalid verdict: {verdict}. Use: {list(VERDICT_MULTIPLIERS.keys())}"}

        # Record the event
        self.sqlite.save_validation_event(
            wisdom_id=wisdom_id,
            source=source,
            verdict=verdict,
            evidence=evidence,
            validator=validator,
        )

        # Apply confidence impact for non-self-report validations
        # Validation is theoretical evidence — external/adversarial/peer verification
        # confirms or refutes the logical/structural basis of the wisdom.
        if source != "self_report":
            impact = SOURCE_WEIGHTS[source] * VERDICT_MULTIPLIERS[verdict]
            old_confidence = w.confidence.overall

            if impact > 0:
                # Positive validation: boost theoretical confidence
                delta = impact * 0.1 * (1.0 - w.confidence.theoretical)
                w.confidence.apply_delta("theoretical", delta)
            else:
                # Negative validation: penalize theoretical confidence
                delta = abs(impact) * 0.15
                w.confidence.apply_delta("theoretical", -delta)

            w.touch()
            self.sqlite.update_wisdom(w)
            self.sqlite.log_confidence_change(
                "wisdom", wisdom_id, old_confidence, w.confidence.overall,
                f"validation_{source}_{verdict}", evidence[:200],
            )
            logger.info(
                "Validated wisdom %s: %s/%s, confidence %.3f -> %.3f",
                wisdom_id, source, verdict, old_confidence, w.confidence.overall,
            )

        # Compute overall validation score
        score = self.compute_validation_score(wisdom_id)

        return {
            "wisdom_id": wisdom_id,
            "source": source,
            "verdict": verdict,
            "validation_score": score,
            "confidence": w.confidence.overall,
        }

    def compute_validation_score(self, wisdom_id: str) -> float:
        """Compute a validation score from all validation events.

        Returns a score from -1.0 (strongly refuted) to 1.0 (strongly confirmed).
        0.0 means no meaningful validation has occurred.
        """
        events = self.sqlite.get_validation_events(wisdom_id)
        if not events:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for e in events:
            source = e["source"]
            verdict = e["verdict"]
            weight = SOURCE_WEIGHTS.get(source, 0.1)
            multiplier = VERDICT_MULTIPLIERS.get(verdict, 0.0)
            weighted_sum += weight * multiplier
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def is_validated(self, wisdom_id: str) -> bool:
        """Check if wisdom has been meaningfully validated (not just self-report)."""
        events = self.sqlite.get_validation_events(wisdom_id)
        return any(
            e["source"] in ("external", "peer", "adversarial")
            and e["verdict"] in ("confirmed", "confirmed_with_caveats")
            for e in events
        )

    def effective_confidence(self, w: Wisdom) -> float:
        """Compute confidence adjusted for validation state.

        Unvalidated wisdom gets discounted by UNVALIDATED_DISCOUNT.
        Positively validated wisdom gets a small boost.
        Negatively validated wisdom gets penalized.
        """
        validation_score = self.compute_validation_score(w.id)

        if validation_score > 0.3:
            # Positively validated — slight boost
            return min(1.0, w.confidence.overall * (1.0 + validation_score * 0.1))
        elif validation_score < -0.3:
            # Negatively validated — penalty proportional to refutation strength
            return max(0.0, w.confidence.overall * (1.0 + validation_score * 0.3))
        elif self.is_validated(w.id):
            # Validated but neutral — no adjustment
            return w.confidence.overall
        else:
            # Unvalidated — discount
            return w.confidence.overall * UNVALIDATED_DISCOUNT

    def validation_summary(self, wisdom_id: str) -> dict:
        """Get a complete validation summary for a wisdom entry."""
        events = self.sqlite.get_validation_events(wisdom_id)
        by_source: dict[str, list[dict]] = {}
        for e in events:
            by_source.setdefault(e["source"], []).append(e)

        return {
            "wisdom_id": wisdom_id,
            "total_events": len(events),
            "validation_score": self.compute_validation_score(wisdom_id),
            "is_validated": self.is_validated(wisdom_id),
            "by_source": {s: len(evts) for s, evts in by_source.items()},
            "events": events[:20],
        }
