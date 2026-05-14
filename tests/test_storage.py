"""Tests for storage layer."""

import json

import pytest

from wisdom.models.common import (
    ConfidenceScore,
    ExperienceResult,
    ExperienceType,
    KnowledgeType,
    Relationship,
    RelationshipType,
    ValidationStatus,
    WisdomType,
)
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom
from wisdom.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


class TestExperienceStorage:
    def test_save_and_get(self, store):
        exp = Experience(description="Test exp", domain="python")
        store.save_experience(exp)
        loaded = store.get_experience(exp.id)
        assert loaded is not None
        assert loaded.description == "Test exp"
        assert loaded.domain == "python"

    def test_list(self, store):
        store.save_experience(Experience(description="A", domain="python"))
        store.save_experience(Experience(description="B", domain="rust"))
        store.save_experience(Experience(description="C", domain="python"))

        all_exps = store.list_experiences()
        assert len(all_exps) == 3

        python_exps = store.list_experiences(domain="python")
        assert len(python_exps) == 2

    def test_delete(self, store):
        exp = Experience(description="To delete")
        store.save_experience(exp)
        assert store.delete_experience(exp.id) is True
        assert store.get_experience(exp.id) is None
        assert store.delete_experience("nonexistent") is False

    def test_count(self, store):
        store.save_experience(Experience(description="A", domain="x"))
        store.save_experience(Experience(description="B", domain="x"))
        store.save_experience(Experience(description="C", domain="y"))
        assert store.count_experiences() == 3
        assert store.count_experiences(domain="x") == 2
        assert store.count_experiences(unprocessed=True) == 3

    def test_mark_processed(self, store):
        exp1 = Experience(description="A")
        exp2 = Experience(description="B")
        store.save_experience(exp1)
        store.save_experience(exp2)
        store.mark_processed([exp1.id])
        assert store.count_experiences(unprocessed=True) == 1
        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0].id == exp2.id

    def test_tags_and_metadata(self, store):
        exp = Experience(
            description="Tagged",
            tags=["a", "b"],
            metadata={"key": "val"},
        )
        store.save_experience(exp)
        loaded = store.get_experience(exp.id)
        assert loaded.tags == ["a", "b"]
        assert loaded.metadata == {"key": "val"}


    def test_list_experiences_for_wisdom(self, store):
        """list_experiences_for_wisdom returns only linked application experiences."""
        wid = "wis_target_123"
        # Create two application experiences linked to our wisdom
        for i in range(2):
            exp = Experience(
                description=f"Applied wisdom {i}",
                type=ExperienceType.WISDOM_APPLICATION,
                metadata={"applied_wisdom_id": wid},
            )
            store.save_experience(exp)
        # Create an unrelated experience
        unrelated = Experience(
            description="Unrelated task",
            type=ExperienceType.TASK,
        )
        store.save_experience(unrelated)
        # Create an application experience for a DIFFERENT wisdom
        other = Experience(
            description="Other application",
            type=ExperienceType.WISDOM_APPLICATION,
            metadata={"applied_wisdom_id": "wis_other_456"},
        )
        store.save_experience(other)

        results = store.list_experiences_for_wisdom(wid)
        assert len(results) == 2
        assert all(
            e.metadata.get("applied_wisdom_id") == wid for e in results
        )

    def test_list_experiences_for_wisdom_empty(self, store):
        """Returns empty list when no experiences link to the given wisdom."""
        results = store.list_experiences_for_wisdom("nonexistent_id")
        assert results == []


class TestKnowledgeStorage:
    def test_save_and_get(self, store):
        k = Knowledge(statement="Pattern A", domain="python")
        store.save_knowledge(k)
        loaded = store.get_knowledge(k.id)
        assert loaded is not None
        assert loaded.statement == "Pattern A"

    def test_confidence_roundtrip(self, store):
        k = Knowledge(
            statement="X",
            confidence=ConfidenceScore(theoretical=0.9, empirical=0.7, observational=0.6),
        )
        store.save_knowledge(k)
        loaded = store.get_knowledge(k.id)
        # overall is computed: 0.4*0.7 + 0.3*0.9 + 0.3*0.6 = 0.73
        assert abs(loaded.confidence.overall - 0.73) < 0.001
        assert abs(loaded.confidence.theoretical - 0.9) < 0.001

    def test_mark_synthesized(self, store):
        k1 = Knowledge(statement="A")
        k2 = Knowledge(statement="B")
        store.save_knowledge(k1)
        store.save_knowledge(k2)
        store.mark_synthesized([k1.id])
        unsynthesized = store.get_unsynthesized()
        assert len(unsynthesized) == 1
        assert unsynthesized[0].id == k2.id

    def test_delete(self, store):
        k = Knowledge(statement="To delete")
        store.save_knowledge(k)
        assert store.delete_knowledge(k.id) is True
        assert store.get_knowledge(k.id) is None


class TestWisdomStorage:
    def test_save_and_get(self, store):
        w = Wisdom(statement="Principle X", applicable_domains=["python"])
        store.save_wisdom(w)
        loaded = store.get_wisdom(w.id)
        assert loaded is not None
        assert loaded.statement == "Principle X"
        assert loaded.applicable_domains == ["python"]

    def test_trade_offs_roundtrip(self, store):
        from wisdom.models.common import TradeOff
        w = Wisdom(
            statement="X",
            trade_offs=[
                TradeOff(dimension="speed", benefit="fast", benefit_magnitude=0.8, cost="unsafe", cost_magnitude=0.3),
            ],
        )
        store.save_wisdom(w)
        loaded = store.get_wisdom(w.id)
        assert len(loaded.trade_offs) == 1
        assert loaded.trade_offs[0].dimension == "speed"
        assert loaded.trade_offs[0].benefit_magnitude == 0.8

    def test_list_by_lifecycle(self, store):
        w1 = Wisdom(statement="A", lifecycle="emerging")
        w2 = Wisdom(statement="B", lifecycle="established")
        store.save_wisdom(w1)
        store.save_wisdom(w2)
        emerging = store.list_wisdom(lifecycle="emerging")
        assert len(emerging) == 1
        assert emerging[0].statement == "A"

    def test_list_by_domain(self, store):
        w1 = Wisdom(statement="A", applicable_domains=["python"])
        w2 = Wisdom(statement="B", applicable_domains=["rust"])
        store.save_wisdom(w1)
        store.save_wisdom(w2)
        python_wis = store.list_wisdom(domain="python")
        assert len(python_wis) == 1
        assert python_wis[0].applicable_domains == ["python"]

    def test_update(self, store):
        w = Wisdom(statement="Original")
        store.save_wisdom(w)
        w.statement = "Updated"
        w.application_count = 5
        store.update_wisdom(w)
        loaded = store.get_wisdom(w.id)
        assert loaded.statement == "Updated"
        assert loaded.application_count == 5


class TestRelationshipStorage:
    def test_save_and_get(self, store):
        rel = Relationship(
            source_id="a", source_type="wisdom",
            target_id="b", target_type="wisdom",
            relationship=RelationshipType.COMPLEMENTS,
            strength=0.7,
        )
        store.save_relationship(rel)
        rels = store.get_relationships("a")
        assert len(rels) == 1
        assert rels[0].relationship == RelationshipType.COMPLEMENTS
        assert rels[0].strength == 0.7

    def test_find_conflicts(self, store):
        rel = Relationship(
            source_id="x", source_type="wisdom",
            target_id="y", target_type="wisdom",
            relationship=RelationshipType.CONFLICTS,
        )
        store.save_relationship(rel)
        conflicts = store.find_conflicts("x")
        assert len(conflicts) == 1
        assert conflicts[0].target_id == "y"


class TestConfidenceLog:
    def test_log_and_retrieve(self, store):
        store.log_confidence_change("wisdom", "w1", 0.5, 0.55, "positive_reinforcement", "good")
        store.log_confidence_change("wisdom", "w1", 0.55, 0.48, "negative_reinforcement", "bad")
        history = store.get_confidence_history("wisdom", "w1")
        assert len(history) == 2
        assert history[0]["new_confidence"] == 0.48  # Most recent first

    def test_recent_events(self, store):
        store.log_confidence_change("wisdom", "w1", 0.5, 0.6, "test", "")
        events = store.get_recent_events()
        assert len(events) == 1


class TestBatchQueries:
    """Tests for batch confidence history and creation date queries."""

    def test_get_wisdom_confidence_histories(self, store):
        w1 = Wisdom(statement="W1", applicable_domains=["test"])
        w2 = Wisdom(statement="W2", applicable_domains=["test"])
        store.save_wisdom(w1)
        store.save_wisdom(w2)

        store.log_confidence_change("wisdom", w1.id, 0.5, 0.6, "up")
        store.log_confidence_change("wisdom", w1.id, 0.6, 0.55, "down")
        store.log_confidence_change("wisdom", w2.id, 0.5, 0.7, "up")

        histories = store.get_wisdom_confidence_histories()
        assert w1.id in histories
        assert w2.id in histories
        assert len(histories[w1.id]) == 2
        assert len(histories[w2.id]) == 1
        # Chronological order within each group
        assert histories[w1.id][0]["new_confidence"] == 0.6
        assert histories[w1.id][1]["new_confidence"] == 0.55

    def test_get_wisdom_confidence_histories_empty(self, store):
        histories = store.get_wisdom_confidence_histories()
        assert histories == {}

    def test_get_wisdom_creation_dates(self, store):
        w1 = Wisdom(statement="W1", applicable_domains=["test"])
        w2 = Wisdom(statement="W2", applicable_domains=["test"])
        store.save_wisdom(w1)
        store.save_wisdom(w2)

        dates = store.get_wisdom_creation_dates()
        assert w1.id in dates
        assert w2.id in dates
        assert isinstance(dates[w1.id], str)

    def test_get_wisdom_creation_dates_empty(self, store):
        dates = store.get_wisdom_creation_dates()
        assert dates == {}


class TestStats:
    def test_get_stats(self, store):
        store.save_experience(Experience(description="A", domain="python"))
        store.save_knowledge(Knowledge(statement="B", domain="python"))
        store.save_wisdom(Wisdom(statement="C", applicable_domains=["python"]))
        stats = store.get_stats()
        assert stats["experiences"] == 1
        assert stats["knowledge"] == 1
        assert stats["wisdom"] == 1
        assert "python" in stats["domains"]

    def test_get_all_domains(self, store):
        store.save_experience(Experience(description="A", domain="python"))
        store.save_experience(Experience(description="B", domain="rust"))
        store.save_wisdom(Wisdom(statement="C", applicable_domains=["go"]))
        domains = store.get_all_domains()
        assert set(domains) == {"python", "rust", "go"}
