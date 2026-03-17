"""Tests for engine layer."""

import pytest

from wisdom.config import WisdomConfig
from wisdom.engine.evolution import EvolutionEngine
from wisdom.engine.experience_engine import ExperienceEngine
from wisdom.engine.gap_analysis import GapAnalysisEngine
from wisdom.engine.knowledge_engine import KnowledgeEngine
from wisdom.engine.triggers import TriggerEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    ExperienceResult,
    LifecycleState,
    WisdomType,
)
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
def exp_engine(sqlite, vector):
    return ExperienceEngine(sqlite, vector)


@pytest.fixture
def know_engine(sqlite, vector):
    return KnowledgeEngine(sqlite, vector)


@pytest.fixture
def wis_engine(sqlite, vector, lifecycle):
    return WisdomEngine(sqlite, vector, lifecycle)


@pytest.fixture
def evo_engine(sqlite, vector, config, lifecycle):
    return EvolutionEngine(sqlite, vector, config, lifecycle)


class TestExperienceEngine:
    def test_add_and_get(self, exp_engine):
        exp = exp_engine.add(description="Test task", domain="python")
        assert exp.id
        loaded = exp_engine.get(exp.id)
        assert loaded.description == "Test task"

    def test_search(self, exp_engine):
        exp_engine.add(description="Python debugging session", domain="python")
        exp_engine.add(description="Rust performance tuning", domain="rust")
        results = exp_engine.search("debugging python", top_k=5)
        assert len(results) >= 1
        # Python debugging should rank higher
        assert results[0]["experience"].domain == "python"

    def test_stats(self, exp_engine):
        exp_engine.add(description="A", domain="python")
        exp_engine.add(description="B", domain="python")
        exp_engine.add(description="C", domain="rust")
        s = exp_engine.stats()
        assert s["total"] == 3
        assert s["unprocessed"] == 3

    def test_delete(self, exp_engine):
        exp = exp_engine.add(description="To delete")
        assert exp_engine.delete(exp.id) is True
        assert exp_engine.get(exp.id) is None


class TestKnowledgeEngine:
    def test_extract_from_experiences(self, know_engine, exp_engine):
        # Add several experiences with patterns
        for i in range(5):
            exp_engine.add(
                description=f"Testing Python code iteration {i}",
                domain="python",
                result=ExperienceResult.SUCCESS if i < 4 else ExperienceResult.FAILURE,
            )
        experiences = exp_engine.get_unprocessed(domain="python")
        knowledge = know_engine.extract_from_experiences(experiences, domain="python")
        assert len(knowledge) >= 1
        # Check experiences were marked as processed
        assert exp_engine.count(unprocessed=True) == 0

    def test_validate(self, know_engine):
        from wisdom.models.knowledge import Knowledge
        k = Knowledge(statement="Test knowledge", domain="python")
        know_engine.add(k)
        validated = know_engine.validate(k.id, is_valid=True, details="Looks correct")
        assert validated.validation_status.value == "validated"

    def test_search(self, know_engine):
        from wisdom.models.knowledge import Knowledge
        know_engine.add(Knowledge(statement="Python is dynamically typed", domain="python"))
        know_engine.add(Knowledge(statement="Rust has ownership model", domain="rust"))
        results = know_engine.search("dynamic typing")
        assert len(results) >= 1


class TestWisdomEngine:
    def test_add(self, wis_engine):
        w = wis_engine.add(
            statement="Always test your code",
            reasoning="Untested code has unknown bugs",
            wisdom_type=WisdomType.PRINCIPLE,
            domains=["testing"],
        )
        assert w.id
        assert w.lifecycle == LifecycleState.EMERGING
        assert w.creation_method == CreationMethod.HUMAN_INPUT

    def test_synthesize_from_knowledge(self, wis_engine, know_engine):
        from wisdom.models.knowledge import Knowledge
        entries = [
            Knowledge(statement="Tests catch bugs early", domain="testing"),
            Knowledge(statement="Integration tests are more reliable than unit tests", domain="testing"),
            Knowledge(statement="Code coverage above 80% has diminishing returns", domain="testing"),
        ]
        for k in entries:
            know_engine.add(k)
        wisdom = wis_engine.synthesize_from_knowledge(entries, domain="testing")
        assert len(wisdom) >= 1
        # Knowledge should be marked as synthesized
        unsynthesized = know_engine.get_unsynthesized()
        assert len(unsynthesized) == 0

    def test_deprecate(self, wis_engine):
        w = wis_engine.add(statement="Old principle", domains=["test"])
        deprecated = wis_engine.deprecate(w.id, "No longer valid")
        assert deprecated.lifecycle == LifecycleState.DEPRECATED
        assert deprecated.deprecation_reason == "No longer valid"

    def test_challenge(self, wis_engine):
        w = wis_engine.add(statement="Questionable principle", domains=["test"])
        challenged = wis_engine.challenge(w.id, "Contradicted by evidence")
        assert challenged.lifecycle == LifecycleState.CHALLENGED

    def test_transfer(self, wis_engine):
        w = wis_engine.add(
            statement="Transferable principle",
            domains=["python"],
        )
        transferred = wis_engine.transfer(w.id, "rust")
        assert transferred is not None
        assert "rust" in transferred.applicable_domains
        assert transferred.lifecycle == LifecycleState.EMERGING
        # Lower confidence than original
        assert transferred.confidence.overall <= w.confidence.overall

    def test_lifecycle_transitions(self, wis_engine):
        w = wis_engine.add(statement="Evolving principle", domains=["test"])
        # Manually set values to trigger transition
        w_loaded = wis_engine.get(w.id)
        w_loaded.application_count = 6
        w_loaded.confidence.empirical = 0.8
        w_loaded.confidence.theoretical = 0.8
        w_loaded.confidence.observational = 0.8
        wis_engine.sqlite.update_wisdom(w_loaded)
        w_loaded = wis_engine.get(w.id)
        result = wis_engine.check_lifecycle_transitions(w_loaded)
        assert result.lifecycle == LifecycleState.ESTABLISHED


class TestEvolutionEngine:
    def test_reinforce_positive(self, evo_engine, wis_engine):
        w = wis_engine.add(statement="Test principle", domains=["test"])
        old_conf = w.confidence.overall
        result = evo_engine.reinforce(w.id, was_helpful=True, feedback="Worked great")
        assert result is not None
        assert result.confidence.overall > old_conf
        assert result.application_count == 1
        assert result.success_count == 1

    def test_reinforce_negative(self, evo_engine, wis_engine):
        w = wis_engine.add(statement="Bad principle", domains=["test"])
        old_conf = w.confidence.overall
        result = evo_engine.reinforce(w.id, was_helpful=False, feedback="Did not work")
        assert result.confidence.overall < old_conf
        assert result.failure_count == 1

    def test_reinforce_creates_experience(self, evo_engine, wis_engine, sqlite):
        w = wis_engine.add(statement="Applied principle", domains=["test"])
        evo_engine.reinforce(w.id, was_helpful=True, task_context="Debugging a Python app")
        # Should have created a wisdom_application experience
        exps = sqlite.list_experiences()
        app_exps = [e for e in exps if e.type.value == "wisdom_application"]
        assert len(app_exps) == 1
        assert w.id in app_exps[0].metadata.get("applied_wisdom_id", "")

    def test_reinforce_lifecycle_transition(self, evo_engine, wis_engine):
        w = wis_engine.add(statement="Evolving", domains=["test"])
        # Reinforce enough to establish
        w_loaded = wis_engine.get(w.id)
        w_loaded.application_count = 4  # Will be 5 after next reinforce
        w_loaded.confidence.empirical = 0.75
        w_loaded.confidence.theoretical = 0.75
        w_loaded.confidence.observational = 0.75  # Above threshold
        wis_engine.sqlite.update_wisdom(w_loaded)
        result = evo_engine.reinforce(w.id, was_helpful=True)
        assert result.lifecycle == LifecycleState.ESTABLISHED

    def test_apply_contradiction(self, evo_engine, wis_engine):
        w = wis_engine.add(statement="Contradicted", domains=["test"])
        old_conf = w.confidence.overall
        result = evo_engine.apply_contradiction(w.id, "New evidence contradicts this")
        assert result.confidence.overall < old_conf

    def test_confidence_history(self, evo_engine, wis_engine):
        w = wis_engine.add(statement="Tracked", domains=["test"])
        evo_engine.reinforce(w.id, was_helpful=True)
        evo_engine.reinforce(w.id, was_helpful=False)
        history = evo_engine.get_confidence_history(w.id)
        assert len(history) >= 2


class TestTriggerEngine:
    def test_check_extraction_trigger(self, sqlite, config):
        trigger = TriggerEngine(sqlite, config)
        config.thresholds.auto_extract_experiences = 3
        # Add 3 unprocessed experiences
        for i in range(3):
            from wisdom.models.experience import Experience
            sqlite.save_experience(Experience(description=f"Exp {i}", domain="python"))
        result = trigger.check_all()
        assert "python" in result.should_extract

    def test_check_synthesis_trigger(self, sqlite, config):
        trigger = TriggerEngine(sqlite, config)
        config.thresholds.auto_synthesize_knowledge = 2
        from wisdom.models.knowledge import Knowledge
        for i in range(2):
            sqlite.save_knowledge(Knowledge(statement=f"K {i}", domain="python"))
        result = trigger.check_all()
        assert "python" in result.should_synthesize


class TestGapAnalysis:
    def test_find_gaps(self, sqlite):
        gaps = GapAnalysisEngine(sqlite)
        # Domain with experiences but no wisdom
        for i in range(5):
            from wisdom.models.experience import Experience
            sqlite.save_experience(Experience(description=f"Exp {i}", domain="python"))
        result = gaps.find_wisdom_gaps(domain="python")
        assert len(result) >= 1
        assert result[0]["domain"] == "python"

    def test_summary(self, sqlite):
        gaps = GapAnalysisEngine(sqlite)
        result = gaps.summary()
        assert "gaps" in result
        assert "low_coverage_tasks" in result
        assert "stale_domains" in result
        assert "extraction_suggestions" in result
