"""Evolution engine — reinforcement, confidence adjustment, and feedback loops.

This engine manages the core feedback loop. When wisdom is applied and
the user reports back, consequences flow in both directions:
- Good feedback: confidence increases (with diminishing returns)
- Bad feedback: confidence decreases (harder), downstream is contaminated
- Auto-experience: every application becomes data for future learning

Lifecycle transitions are delegated to LifecycleManager — the single
source of truth for the state machine.
"""

from __future__ import annotations

from wisdom.config import WisdomConfig
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.logging_config import get_logger
from wisdom.models.common import (
    ExperienceResult,
    ExperienceType,
    LifecycleState,
)
from wisdom.models.experience import Experience
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.evolution")


def _compute_temporal_decay(confidence: float, age_days: float, decay_rate: float) -> float:
    """Apply temporal confidence decay. Single source of this formula."""
    months = age_days / 30.44
    return max(0.0, min(1.0, confidence * ((1.0 - decay_rate) ** months)))


class EvolutionEngine:
    """Manages the reinforcement loop: feedback, confidence adjustment, lifecycle transitions."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        vector: VectorStore,
        config: WisdomConfig,
        lifecycle: LifecycleManager,
    ):
        self.sqlite = sqlite
        self.vector = vector
        self.config = config
        self.lifecycle = lifecycle

    def reinforce(
        self,
        wisdom_id: str,
        was_helpful: bool,
        feedback: str = "",
        task_context: str = "",
    ) -> Wisdom | None:
        """Apply reinforcement feedback to a wisdom entry.

        This is the core feedback loop:
        1. Adjust confidence (asymmetric Bayesian)
        2. Auto-create an experience from the application
        3. Delegate lifecycle evaluation to LifecycleManager
        """
        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            logger.warning("Reinforce: wisdom %s not found", wisdom_id)
            return None

        old_confidence = w.confidence.overall

        # Step 1: Adjust confidence (asymmetric — failures weigh more)
        # Reinforcement is empirical evidence — route through the empirical dimension.
        # apply_delta scales so the effective overall change matches delta.
        if was_helpful:
            delta = self.config.confidence.success_factor * (1.0 - old_confidence)
            w.success_count += 1
        else:
            delta = self.config.confidence.failure_delta
            w.failure_count += 1

        w.confidence.apply_delta("empirical", delta)
        new_confidence = w.confidence.overall
        w.application_count += 1
        w.touch()

        # Log the confidence change
        reason = "reinforcement_positive" if was_helpful else "reinforcement_negative"
        self.sqlite.log_confidence_change(
            "wisdom", wisdom_id, old_confidence, new_confidence, reason, feedback,
        )

        # Also record as a self-report validation event
        self.sqlite.save_validation_event(
            wisdom_id=wisdom_id,
            source="self_report",
            verdict="confirmed" if was_helpful else "challenged",
            evidence=feedback or task_context,
            validator="reinforcement_loop",
        )

        # Step 2: Auto-create experience from the application
        exp = Experience(
            type=ExperienceType.WISDOM_APPLICATION,
            domain=w.applicable_domains[0] if w.applicable_domains else "",
            description=f"Applied wisdom: {w.statement[:100]}",
            input_text=task_context or f"Task requiring wisdom {wisdom_id}",
            output_text=f"Wisdom applied: {w.statement}",
            result=ExperienceResult.SUCCESS if was_helpful else ExperienceResult.FAILURE,
            quality_score=new_confidence,
            tags=["wisdom_application", f"wisdom:{wisdom_id}"],
            metadata={
                "applied_wisdom_id": wisdom_id,
                "original_confidence": str(old_confidence),
                "new_confidence": str(new_confidence),
                "was_helpful": str(was_helpful),
            },
        )
        self.sqlite.save_experience(exp)
        self.vector.add(
            layer="experience",
            id=exp.id,
            text=exp.embedding_text,
            metadata={
                "domain": exp.domain,
                "type": exp.type.value,
                "result": exp.result.value,
            },
        )

        # Step 3: Delegate lifecycle evaluation to the single authority
        # LifecycleManager saves if a transition occurs; we save here for the no-transition case
        transition = self.lifecycle.evaluate(w, old_confidence=old_confidence)
        if not transition.transitioned:
            self.sqlite.update_wisdom(w)

        # Update vector store metadata
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

        logger.info(
            "Reinforced wisdom %s: confidence %.3f -> %.3f (%s), lifecycle: %s",
            wisdom_id, old_confidence, new_confidence,
            "positive" if was_helpful else "negative",
            w.lifecycle.value,
        )
        return w

    def apply_contradiction(self, wisdom_id: str, details: str = "") -> Wisdom | None:
        """Apply a contradiction penalty to wisdom."""
        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            return None

        old_confidence = w.confidence.overall
        delta = self.config.confidence.contradiction_delta

        # Contradictions are empirical evidence that something is wrong
        w.confidence.apply_delta("empirical", delta)
        new_confidence = w.confidence.overall
        w.touch()

        self.sqlite.log_confidence_change(
            "wisdom", wisdom_id, old_confidence, new_confidence,
            "contradiction", details,
        )

        # Evaluate lifecycle after contradiction; save only if no transition
        transition = self.lifecycle.evaluate(w, old_confidence=old_confidence)
        if not transition.transitioned:
            self.sqlite.update_wisdom(w)

        logger.info(
            "Contradiction applied to wisdom %s: confidence %.3f -> %.3f",
            wisdom_id, old_confidence, new_confidence,
        )
        return w

    def auto_deprecate_sweep(self) -> list[str]:
        """Sweep all wisdom and auto-deprecate entries below threshold.

        Returns list of deprecated wisdom IDs.
        """
        threshold = self.config.thresholds.deprecated_confidence
        all_wisdom = self.sqlite.list_wisdom(limit=10000)
        deprecated = []

        for w in all_wisdom:
            if w.lifecycle == LifecycleState.DEPRECATED:
                continue

            # Check effective confidence with temporal decay
            effective = _compute_temporal_decay(
                w.confidence.overall,
                w.age_days,
                self.config.confidence.decay_rate_per_month,
            )

            if effective < threshold and w.lifecycle == LifecycleState.CHALLENGED:
                result = self.lifecycle.force_deprecate(
                    w,
                    f"Auto-deprecated: effective confidence {effective:.3f} "
                    f"below threshold {threshold}",
                )
                if result.transitioned:
                    deprecated.append(w.id)

        return deprecated

    def get_confidence_history(self, wisdom_id: str) -> list[dict]:
        """Get the confidence change history for a wisdom entry."""
        return self.sqlite.get_confidence_history("wisdom", wisdom_id)
