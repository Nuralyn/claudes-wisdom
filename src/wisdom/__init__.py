"""Wisdom System — accumulate, retain, and apply wisdom through the DIKW hierarchy."""

from __future__ import annotations

from pathlib import Path

from wisdom.config import WisdomConfig
from wisdom.engine.adversarial import AdversarialEngine
from wisdom.engine.coverage import CoverageEngine
from wisdom.engine.evolution import EvolutionEngine
from wisdom.engine.experience_engine import ExperienceEngine
from wisdom.engine.gap_analysis import GapAnalysisEngine
from wisdom.engine.meta_learning import MetaLearningEngine
from wisdom.engine.knowledge_engine import KnowledgeEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.engine.propagation import PropagationEngine
from wisdom.engine.retrieval import RetrievalEngine
from wisdom.engine.triggers import TriggerEngine
from wisdom.engine.validation import ValidationEngine
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.llm.provider import ProviderRegistry
from wisdom.logging_config import setup_logging
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

__version__ = "0.1.0"


class WisdomSystem:
    """Composition root — wires everything together.

    The dependency graph is intentional:
    - LifecycleManager is the single authority for state transitions
    - Both WisdomEngine and EvolutionEngine delegate to it
    - ValidationEngine gates promotions
    - PropagationEngine cascades consequences
    - AdversarialEngine challenges before promotion
    - CoverageEngine detects semantic blind spots
    """

    def __init__(self, config: WisdomConfig | None = None):
        self.config = config or WisdomConfig()
        self.config.ensure_dirs()

        # Logging
        self.logger = setup_logging(self.config.log_level)

        # Storage
        self.sqlite = SQLiteStore(self.config.sqlite_path)
        self.vector = VectorStore(self.config.chroma_path)

        # Lifecycle — single source of truth for state machine
        self._lifecycle = LifecycleManager(self.sqlite, self.config)

        # Core engines
        self.experiences = ExperienceEngine(self.sqlite, self.vector)
        self.knowledge = KnowledgeEngine(self.sqlite, self.vector)
        self.wisdom = WisdomEngine(self.sqlite, self.vector, self._lifecycle)
        self.retrieval = RetrievalEngine(self.sqlite, self.vector, self.config)
        self.evolution = EvolutionEngine(self.sqlite, self.vector, self.config, self._lifecycle)

        # Trust & verification engines
        self.validation = ValidationEngine(self.sqlite)
        self.adversarial = AdversarialEngine(self.sqlite, self.vector)
        self.propagation = PropagationEngine(self.sqlite, self.vector, self.config)
        self.coverage = CoverageEngine(self.sqlite, self.vector)

        # Analysis engines
        self.triggers = TriggerEngine(self.sqlite, self.config)
        self.gaps = GapAnalysisEngine(self.sqlite)
        self.meta_learning = MetaLearningEngine(self.sqlite, self.config)

        # LLM providers (lazy)
        self.providers = ProviderRegistry()

    def init_providers(self) -> None:
        """Initialize LLM providers from config."""
        self.providers.auto_register(self.config.llm)

    def warmup(self) -> None:
        """Pre-warm embedding model."""
        self.vector.warmup()

    def run_maintenance(self) -> dict:
        """Run all auto-triggered maintenance."""
        return self.triggers.run_maintenance(
            experience_engine=self.experiences,
            knowledge_engine=self.knowledge,
            wisdom_engine=self.wisdom,
            evolution_engine=self.evolution,
        )

    def stats(self) -> dict:
        """Get system-wide statistics."""
        return self.sqlite.get_stats()

    def close(self) -> None:
        """Clean up resources."""
        self.sqlite.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
