"""Auto-trigger rules for pipeline automation."""

from __future__ import annotations

from wisdom.config import WisdomConfig
from wisdom.logging_config import get_logger
from wisdom.storage.sqlite_store import SQLiteStore

logger = get_logger("engine.triggers")


class TriggerResult:
    """Result of checking triggers."""

    def __init__(self):
        self.should_extract: list[str] = []  # domains needing extraction
        self.should_synthesize: list[str] = []  # domains needing synthesis
        self.should_validate: list[str] = []  # wisdom IDs needing validation
        self.should_deprecate: bool = False

    @property
    def has_actions(self) -> bool:
        return bool(
            self.should_extract
            or self.should_synthesize
            or self.should_validate
            or self.should_deprecate
        )


class TriggerEngine:
    """Check and fire auto-triggers for pipeline automation."""

    def __init__(self, sqlite: SQLiteStore, config: WisdomConfig):
        self.sqlite = sqlite
        self.config = config

    def check_all(self) -> TriggerResult:
        """Check all trigger conditions and return what needs to happen."""
        result = TriggerResult()

        # Check extraction triggers by domain
        domains = self.sqlite.get_all_domains()
        all_domains = set(domains)

        # Also check domain-less experiences
        unprocessed_no_domain = self.sqlite.count_experiences(unprocessed=True)
        if unprocessed_no_domain >= self.config.thresholds.auto_extract_experiences:
            result.should_extract.append("")

        for domain in all_domains:
            unprocessed = self.sqlite.count_experiences(domain=domain, unprocessed=True)
            if unprocessed >= self.config.thresholds.auto_extract_experiences:
                result.should_extract.append(domain)
                logger.info(
                    "Trigger: extract knowledge for domain '%s' (%d unprocessed experiences)",
                    domain, unprocessed,
                )

        # Check synthesis triggers by domain
        for domain in all_domains:
            unsynthesized = self.sqlite.count_knowledge(domain=domain, unsynthesized=True)
            if unsynthesized >= self.config.thresholds.auto_synthesize_knowledge:
                result.should_synthesize.append(domain)
                logger.info(
                    "Trigger: synthesize wisdom for domain '%s' (%d unsynthesized knowledge)",
                    domain, unsynthesized,
                )

        # Also check domain-less knowledge
        unsynthesized_no_domain = self.sqlite.count_knowledge(unsynthesized=True)
        if unsynthesized_no_domain >= self.config.thresholds.auto_synthesize_knowledge:
            if "" not in result.should_synthesize:
                result.should_synthesize.append("")

        # Check validation triggers
        all_wisdom = self.sqlite.list_wisdom(limit=10000)
        for w in all_wisdom:
            if w.lifecycle.value == "deprecated":
                continue
            if w.application_count > 0 and w.negative_feedback_ratio > self.config.thresholds.negative_feedback_ratio:
                result.should_validate.append(w.id)
                logger.info(
                    "Trigger: validate wisdom %s (negative ratio: %.2f)",
                    w.id, w.negative_feedback_ratio,
                )

        # Always allow deprecation sweep
        result.should_deprecate = True

        return result

    def run_maintenance(
        self,
        experience_engine,
        knowledge_engine,
        wisdom_engine,
        evolution_engine,
    ) -> dict:
        """Run all triggered maintenance actions.

        Returns a summary of what was done.
        """
        triggers = self.check_all()
        summary: dict = {
            "extracted": [],
            "synthesized": [],
            "validated": [],
            "deprecated": [],
        }

        if not triggers.has_actions:
            logger.info("No maintenance actions needed")
            return summary

        # Run extractions
        for domain in triggers.should_extract:
            experiences = experience_engine.get_unprocessed(domain=domain or None)
            if experiences:
                knowledge = knowledge_engine.extract_from_experiences(
                    experiences, domain=domain
                )
                summary["extracted"].append({
                    "domain": domain or "(all)",
                    "experiences_processed": len(experiences),
                    "knowledge_created": len(knowledge),
                })

        # Run syntheses
        for domain in triggers.should_synthesize:
            knowledge_entries = knowledge_engine.get_unsynthesized(domain=domain or None)
            if knowledge_entries:
                wisdom = wisdom_engine.synthesize_from_knowledge(
                    knowledge_entries, domain=domain
                )
                summary["synthesized"].append({
                    "domain": domain or "(all)",
                    "knowledge_processed": len(knowledge_entries),
                    "wisdom_created": len(wisdom),
                })

        # Run deprecation sweep
        if triggers.should_deprecate:
            deprecated_ids = evolution_engine.auto_deprecate_sweep()
            summary["deprecated"] = deprecated_ids

        # Note validated wisdom IDs (actual validation requires LLM or human)
        summary["validated"] = triggers.should_validate

        logger.info("Maintenance complete: %s", summary)
        return summary
