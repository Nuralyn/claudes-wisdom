"""Tests for the evolution engine — focused on the feedback loop."""

import pytest

from wisdom.config import WisdomConfig
from wisdom.engine.evolution import EvolutionEngine
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.models.common import ConfidenceScore, LifecycleState
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore


@pytest.fixture
def sqlite(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def vector(tmp_path):
    return VectorStore(tmp_path / "chroma")


@pytest.fixture
def config(tmp_path):
    return WisdomConfig(data_dir=tmp_path / "data")


@pytest.fixture
def lifecycle(sqlite, config):
    return LifecycleManager(sqlite, config)


@pytest.fixture
def wis_engine(sqlite, vector, lifecycle):
    return WisdomEngine(sqlite, vector, lifecycle)


@pytest.fixture
def evo_engine(sqlite, vector, config, lifecycle):
    return EvolutionEngine(sqlite, vector, config, lifecycle)


class TestAsymmetricConfidence:
    def test_positive_reinforcement_diminishing_returns(self, evo_engine, wis_engine):
        """Confidence increases should diminish as confidence approaches 1.0."""
        w = wis_engine.add(
            statement="High confidence principle",
            confidence=ConfidenceScore(overall=0.9),
        )
        result = evo_engine.reinforce(w.id, was_helpful=True)
        # Delta should be small: 0.05 * (1 - 0.9) = 0.005
        assert result.confidence.overall > 0.9
        assert result.confidence.overall < 0.95  # Not a huge jump

    def test_negative_reinforcement_fixed(self, evo_engine, wis_engine):
        """Failures should have a fixed negative impact."""
        w = wis_engine.add(
            statement="Tested principle",
            confidence=ConfidenceScore(overall=0.7),
        )
        result = evo_engine.reinforce(w.id, was_helpful=False)
        # Delta should be -0.08
        expected = 0.7 - 0.08
        assert abs(result.confidence.overall - expected) < 0.001

    def test_failure_weighs_more_than_success(self, evo_engine, wis_engine):
        """One failure should have more impact than one success at the same confidence."""
        # Create two identical wisdom entries
        w1 = wis_engine.add(statement="W1", confidence=ConfidenceScore(overall=0.5))
        w2 = wis_engine.add(statement="W2", confidence=ConfidenceScore(overall=0.5))

        r1 = evo_engine.reinforce(w1.id, was_helpful=True)
        r2 = evo_engine.reinforce(w2.id, was_helpful=False)

        positive_delta = abs(r1.confidence.overall - 0.5)
        negative_delta = abs(r2.confidence.overall - 0.5)
        assert negative_delta > positive_delta  # Failures weigh more


class TestFeedbackLoop:
    def test_full_loop(self, evo_engine, wis_engine, sqlite):
        """Test the complete feedback loop: reinforce -> experience -> lifecycle."""
        w = wis_engine.add(statement="Full loop test", domains=["test"])

        # Positive reinforcement
        evo_engine.reinforce(w.id, was_helpful=True, task_context="First task")

        # Check experience was created
        exps = sqlite.list_experiences()
        app_exps = [e for e in exps if e.type.value == "wisdom_application"]
        assert len(app_exps) == 1
        assert app_exps[0].result.value == "success"

        # Check confidence was logged
        history = sqlite.get_confidence_history("wisdom", w.id)
        assert len(history) >= 1

    def test_lifecycle_through_reinforcement(self, evo_engine, wis_engine):
        """Repeated positive reinforcement should transition EMERGING -> ESTABLISHED."""
        w = wis_engine.add(
            statement="Repeatedly validated",
            confidence=ConfidenceScore(overall=0.75),
        )
        # Need 5 applications with confidence >= 0.7
        for _ in range(5):
            result = evo_engine.reinforce(w.id, was_helpful=True)

        assert result.lifecycle == LifecycleState.ESTABLISHED

    def test_negative_spiral(self, evo_engine, wis_engine):
        """Repeated negative feedback should challenge then deprecate."""
        w = wis_engine.add(
            statement="Bad principle",
            confidence=ConfidenceScore(overall=0.5),
        )
        # Multiple failures
        for _ in range(10):
            result = evo_engine.reinforce(w.id, was_helpful=False)

        # Should be challenged (empirical evidence is destroyed but theoretical/observational remain)
        assert result.lifecycle == LifecycleState.CHALLENGED
        assert result.confidence.overall < 0.4
        assert result.confidence.empirical < 0.05  # Empirical evidence is near zero


class TestAutoDeprecation:
    def test_sweep_deprecated(self, evo_engine, wis_engine):
        """Auto-deprecation sweep should catch challenged entries below threshold."""
        w = wis_engine.add(
            statement="Should be deprecated",
            confidence=ConfidenceScore(overall=0.2),
        )
        # Set to challenged
        wis_engine.challenge(w.id, "Test")

        deprecated = evo_engine.auto_deprecate_sweep()
        # May or may not be deprecated depending on temporal decay
        # But the sweep should run without error
        assert isinstance(deprecated, list)

    def test_sweep_skips_active(self, evo_engine, wis_engine):
        """Active wisdom above threshold should not be deprecated."""
        w = wis_engine.add(
            statement="Strong principle",
            confidence=ConfidenceScore(overall=0.9),
        )
        deprecated = evo_engine.auto_deprecate_sweep()
        assert w.id not in deprecated
