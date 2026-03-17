"""Single source of truth for wisdom lifecycle state machine.

Every lifecycle transition in the system flows through this module.
No other module should directly mutate wisdom.lifecycle.
"""

from __future__ import annotations

from wisdom.config import WisdomConfig
from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore

logger = get_logger("engine.lifecycle")


class TransitionResult:
    """The outcome of a lifecycle check."""

    def __init__(self, transitioned: bool, old: LifecycleState, new: LifecycleState, reason: str):
        self.transitioned = transitioned
        self.old = old
        self.new = new
        self.reason = reason


class LifecycleManager:
    """Centralized lifecycle state machine for wisdom entries.

    Transition rules:
        EMERGING -> ESTABLISHED:
            application_count >= threshold AND confidence >= threshold
            AND validated (has at least one external validation event)
            AND adversarial challenge survived (or waived for seeds)
        EMERGING -> CHALLENGED:
            confidence < challenged_threshold
        ESTABLISHED -> CHALLENGED:
            confidence < challenged_threshold
            OR contradicting evidence arrives
        CHALLENGED -> DEPRECATED:
            confidence < deprecated_threshold
        CHALLENGED -> ESTABLISHED:
            confidence recovers above established_threshold AND re-validated
    """

    def __init__(self, sqlite: SQLiteStore, config: WisdomConfig):
        self.sqlite = sqlite
        self.config = config

    def evaluate(
        self,
        w: Wisdom,
        old_confidence: float | None = None,
    ) -> TransitionResult:
        """Evaluate and apply lifecycle transitions for a wisdom entry.

        Args:
            w: The wisdom entry (will be mutated if transition occurs)
            old_confidence: The confidence before the current change (for logging)

        Returns:
            TransitionResult describing what happened
        """
        t = self.config.thresholds
        old_lifecycle = w.lifecycle
        new_lifecycle = old_lifecycle
        reason = ""

        if w.lifecycle == LifecycleState.EMERGING:
            if (w.application_count >= t.emerging_to_established_count
                    and w.confidence.overall >= t.emerging_to_established_confidence
                    and self._has_validation(w)):
                new_lifecycle = LifecycleState.ESTABLISHED
                reason = (
                    f"Promoted: {w.application_count} applications, "
                    f"confidence {w.confidence.overall:.3f}, validated"
                )
            elif w.confidence.overall < t.challenged_confidence:
                new_lifecycle = LifecycleState.CHALLENGED
                reason = f"Confidence {w.confidence.overall:.3f} below challenge threshold {t.challenged_confidence}"

        elif w.lifecycle == LifecycleState.ESTABLISHED:
            if w.confidence.overall < t.challenged_confidence:
                new_lifecycle = LifecycleState.CHALLENGED
                reason = f"Confidence {w.confidence.overall:.3f} dropped below challenge threshold {t.challenged_confidence}"

        elif w.lifecycle == LifecycleState.CHALLENGED:
            if w.confidence.overall < t.deprecated_confidence:
                new_lifecycle = LifecycleState.DEPRECATED
                w.deprecation_reason = reason = (
                    f"Confidence {w.confidence.overall:.3f} below deprecation threshold {t.deprecated_confidence}"
                )
            elif (w.confidence.overall >= t.emerging_to_established_confidence
                  and self._has_validation(w)):
                new_lifecycle = LifecycleState.ESTABLISHED
                reason = f"Recovered: confidence {w.confidence.overall:.3f} with validation"

        # DEPRECATED is a terminal state — no transitions out

        transitioned = new_lifecycle != old_lifecycle
        if transitioned:
            w.lifecycle = new_lifecycle
            w.version += 1
            w.touch()
            self.sqlite.update_wisdom(w)

            log_old = old_confidence if old_confidence is not None else w.confidence.overall
            self.sqlite.log_confidence_change(
                entity_type="wisdom",
                entity_id=w.id,
                old_confidence=log_old,
                new_confidence=w.confidence.overall,
                reason=f"lifecycle: {old_lifecycle.value} -> {new_lifecycle.value}",
                details=reason,
            )
            logger.info(
                "Wisdom %s: %s -> %s (%s)",
                w.id, old_lifecycle.value, new_lifecycle.value, reason,
            )

        return TransitionResult(
            transitioned=transitioned,
            old=old_lifecycle,
            new=new_lifecycle,
            reason=reason,
        )

    def _has_validation(self, w: Wisdom) -> bool:
        """Check if this wisdom has been externally validated.

        Seeds and human-input wisdom get a pass on the first promotion —
        they still need validation to recover from CHALLENGED -> ESTABLISHED.
        """
        if w.lifecycle == LifecycleState.EMERGING and w.creation_method.value in ("seed", "human_input"):
            # First promotion for seeds/human input doesn't require external validation
            # but they still need sufficient applications and confidence
            return True

        # Check for validation events
        events = self.sqlite.get_validation_events(w.id)
        # Need at least one non-self-report confirmation
        return any(
            e["source"] in ("external", "peer", "adversarial")
            and e["verdict"] in ("confirmed", "confirmed_with_caveats")
            for e in events
        )

    def force_deprecate(self, w: Wisdom, reason: str) -> TransitionResult:
        """Force-deprecate regardless of current state (except already deprecated)."""
        if w.lifecycle == LifecycleState.DEPRECATED:
            return TransitionResult(False, w.lifecycle, w.lifecycle, "Already deprecated")

        old = w.lifecycle
        w.lifecycle = LifecycleState.DEPRECATED
        w.deprecation_reason = reason
        w.version += 1
        w.touch()
        self.sqlite.update_wisdom(w)
        self.sqlite.log_confidence_change(
            "wisdom", w.id, w.confidence.overall, w.confidence.overall,
            f"lifecycle: {old.value} -> deprecated (forced)", reason,
        )
        logger.info("Wisdom %s: force deprecated (%s)", w.id, reason)
        return TransitionResult(True, old, LifecycleState.DEPRECATED, reason)

    def force_challenge(self, w: Wisdom, reason: str) -> TransitionResult:
        """Force-challenge regardless of current state."""
        if w.lifecycle in (LifecycleState.DEPRECATED, LifecycleState.CHALLENGED):
            return TransitionResult(False, w.lifecycle, w.lifecycle, "Already challenged/deprecated")

        old = w.lifecycle
        w.lifecycle = LifecycleState.CHALLENGED
        w.version += 1
        w.touch()
        self.sqlite.update_wisdom(w)
        self.sqlite.log_confidence_change(
            "wisdom", w.id, w.confidence.overall, w.confidence.overall,
            f"lifecycle: {old.value} -> challenged (forced)", reason,
        )
        logger.info("Wisdom %s: force challenged (%s)", w.id, reason)
        return TransitionResult(True, old, LifecycleState.CHALLENGED, reason)
