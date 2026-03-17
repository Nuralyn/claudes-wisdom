"""Tests for data models."""

from wisdom.models.common import (
    ConfidenceScore,
    ExperienceResult,
    ExperienceType,
    KnowledgeType,
    LifecycleState,
    RelationshipType,
    TradeOff,
    WisdomType,
)
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom


class TestConfidenceScore:
    def test_defaults(self):
        cs = ConfidenceScore()
        assert cs.overall == 0.5
        assert cs.theoretical == 0.5
        assert cs.empirical == 0.5
        assert cs.observational == 0.5

    def test_weighted_score(self):
        cs = ConfidenceScore(overall=0.8, theoretical=1.0, empirical=0.6, observational=0.4)
        # 0.4 * 0.6 + 0.3 * 1.0 + 0.3 * 0.4 = 0.24 + 0.3 + 0.12 = 0.66
        assert abs(cs.weighted_score() - 0.66) < 0.001

    def test_clamping(self):
        # Sub-dimensions are clamped to [0.0, 1.0]
        cs = ConfidenceScore(theoretical=1.0, empirical=0.0, observational=0.0)
        assert cs.theoretical == 1.0
        assert cs.empirical == 0.0
        # Overall is computed: 0.4*0.0 + 0.3*1.0 + 0.3*0.0 = 0.3
        assert abs(cs.overall - 0.3) < 0.001

    def test_overall_is_computed(self):
        """overall is derived from sub-dimensions, not independently stored."""
        cs = ConfidenceScore(empirical=0.8, theoretical=0.6, observational=0.4)
        expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.4  # 0.62
        assert abs(cs.overall - expected) < 0.001

    def test_legacy_overall_sets_all_subs(self):
        """Passing only overall= distributes it uniformly across sub-dimensions."""
        cs = ConfidenceScore(overall=0.9)
        assert cs.empirical == 0.9
        assert cs.theoretical == 0.9
        assert cs.observational == 0.9
        assert abs(cs.overall - 0.9) < 0.001

    def test_apply_delta(self):
        """apply_delta scales the sub-dimension change so overall changes by ~delta."""
        cs = ConfidenceScore(empirical=0.5, theoretical=0.5, observational=0.5)
        old_overall = cs.overall  # 0.5
        cs.apply_delta("empirical", 0.05)
        # Overall should have changed by approximately 0.05
        assert abs((cs.overall - old_overall) - 0.05) < 0.01


class TestTradeOff:
    def test_creation(self):
        t = TradeOff(
            dimension="speed vs safety",
            benefit="faster execution",
            benefit_magnitude=0.7,
            cost="less validation",
            cost_magnitude=0.4,
        )
        assert t.dimension == "speed vs safety"
        assert t.benefit_magnitude == 0.7


class TestExperience:
    def test_creation(self):
        exp = Experience(description="Test experience")
        assert exp.id  # Auto-generated
        assert exp.timestamp  # Auto-generated
        assert exp.type == ExperienceType.TASK
        assert exp.result == ExperienceResult.SUCCESS
        assert exp.description == "Test experience"

    def test_embedding_text(self):
        exp = Experience(
            description="Did something",
            input_text="input here",
            output_text="output here",
        )
        text = exp.embedding_text
        assert "Did something" in text
        assert "input here" in text
        assert "output here" in text

    def test_serialization(self):
        exp = Experience(description="Test", tags=["a", "b"], metadata={"key": "val"})
        data = exp.model_dump()
        restored = Experience(**data)
        assert restored.description == "Test"
        assert restored.tags == ["a", "b"]
        assert restored.metadata == {"key": "val"}


class TestKnowledge:
    def test_creation(self):
        k = Knowledge(statement="Patterns repeat")
        assert k.id
        assert k.type == KnowledgeType.PATTERN
        assert k.statement == "Patterns repeat"

    def test_embedding_text(self):
        k = Knowledge(statement="Fact A", explanation="Because of B")
        assert "Fact A" in k.embedding_text
        assert "Because of B" in k.embedding_text

    def test_touch(self):
        k = Knowledge(statement="X")
        old_updated = k.updated_at
        k.touch()
        assert k.updated_at >= old_updated


class TestWisdom:
    def test_creation(self):
        w = Wisdom(statement="Test wisdom")
        assert w.id
        assert w.type == WisdomType.PRINCIPLE
        assert w.lifecycle == LifecycleState.EMERGING

    def test_success_rate(self):
        w = Wisdom(statement="X", application_count=10, success_count=7, failure_count=3)
        assert abs(w.success_rate - 0.7) < 0.001

    def test_success_rate_zero_apps(self):
        w = Wisdom(statement="X")
        assert w.success_rate == 0.0

    def test_negative_feedback_ratio(self):
        w = Wisdom(statement="X", application_count=10, failure_count=4)
        assert abs(w.negative_feedback_ratio - 0.4) < 0.001

    def test_embedding_text(self):
        w = Wisdom(
            statement="Main principle",
            reasoning="Because reasons",
            implications=["Imp1", "Imp2"],
        )
        text = w.embedding_text
        assert "Main principle" in text
        assert "Because reasons" in text
        assert "Imp1" in text

    def test_trade_offs(self):
        w = Wisdom(
            statement="X",
            trade_offs=[
                TradeOff(
                    dimension="speed",
                    benefit="faster",
                    benefit_magnitude=0.8,
                    cost="less safe",
                    cost_magnitude=0.3,
                ),
            ],
        )
        assert len(w.trade_offs) == 1
        assert w.trade_offs[0].dimension == "speed"

    def test_serialization_roundtrip(self):
        w = Wisdom(
            statement="Test",
            trade_offs=[TradeOff(dimension="d", benefit="b", cost="c")],
            applicable_domains=["python"],
            tags=["test"],
        )
        data = w.model_dump()
        restored = Wisdom(**data)
        assert restored.statement == "Test"
        assert len(restored.trade_offs) == 1
        assert restored.applicable_domains == ["python"]
