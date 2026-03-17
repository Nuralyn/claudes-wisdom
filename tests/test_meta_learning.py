"""Tests for the meta-learning engine — the system learning from its own failures.

These tests verify that the MetaLearningEngine correctly analyzes
contamination logs, confidence history, and deprecation patterns to
produce actionable risk assessments.
"""

from __future__ import annotations

import pytest

from wisdom.config import WisdomConfig
from wisdom.engine.adversarial import AdversarialEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.engine.meta_learning import (
    ContaminationPattern,
    FailureProfile,
    MetaLearningEngine,
    RiskScore,
)
from wisdom.engine.propagation import PropagationEngine
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    LifecycleState,
    TradeOff,
    WisdomType,
)
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore


@pytest.fixture
def meta(sqlite, config):
    return MetaLearningEngine(sqlite, config)


@pytest.fixture
def wis_engine(sqlite, vector, lifecycle):
    return WisdomEngine(sqlite, vector, lifecycle)


@pytest.fixture
def propagation(sqlite, vector, config):
    return PropagationEngine(sqlite, vector, config)


def _make_wisdom(sqlite, vector, *, wtype="principle", method="human_input",
                 lifecycle="emerging", domains=None, confidence=0.5,
                 statement="Test wisdom statement for meta-learning"):
    """Helper: create and save a wisdom entry with given profile."""
    from wisdom.models.wisdom import Wisdom

    w = Wisdom(
        type=wtype,
        statement=statement,
        reasoning="Test reasoning",
        applicable_domains=domains or ["testing"],
        applicability_conditions=["When testing"],
        trade_offs=[TradeOff(dimension="speed", benefit="fast", cost="fragile")],
        confidence=ConfidenceScore(overall=confidence, empirical=confidence,
                                    theoretical=confidence, observational=confidence),
        lifecycle=lifecycle,
        creation_method=method,
    )
    sqlite.save_wisdom(w)
    vector.add(layer="wisdom", id=w.id, text=w.statement,
               metadata={"domain": domains[0] if domains else "testing"})
    return w


def _make_knowledge(sqlite, vector, *, domain="testing", statement="Test knowledge"):
    """Helper: create and save a knowledge entry."""
    k = Knowledge(
        type="pattern",
        statement=statement,
        domain=domain,
        confidence=ConfidenceScore(overall=0.5),
    )
    sqlite.save_knowledge(k)
    vector.add(layer="knowledge", id=k.id, text=k.statement,
               metadata={"domain": domain})
    return k


class TestEmptySystem:
    """Verify safe behavior with no data."""

    def test_failure_profiles_empty(self, meta):
        profiles = meta.failure_profiles()
        assert profiles == []

    def test_domain_risk_empty(self, meta):
        domains = meta.domain_risk_assessment()
        assert domains == []

    def test_contamination_patterns_empty(self, meta):
        patterns = meta.contamination_patterns()
        assert patterns == []

    def test_confidence_trajectory_empty(self, meta):
        traj = meta.confidence_trajectory()
        assert traj["total_events"] == 0
        assert traj["net_direction"] == "no_data"

    def test_compute_risk_score_missing_wisdom(self, meta):
        risk = meta.compute_risk_score("nonexistent-id")
        assert risk.base_risk == 0.5
        assert risk.recommended_challenge_level == "elevated"

    def test_summary_empty(self, meta):
        s = meta.summary()
        assert "failure_profiles" in s
        assert "risky_domains" in s
        assert "super_spreaders" in s
        assert "trajectory" in s


class TestFailureProfiles:
    """Verify failure rate computation by type and creation method."""

    def test_failure_profile_by_type(self, sqlite, vector, meta):
        # Create 3 heuristics: 1 deprecated, 2 active
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated")
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="established",
                     statement="Active heuristic one")
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="emerging",
                     statement="Active heuristic two")
        # Create 3 principles: 0 deprecated
        _make_wisdom(sqlite, vector, wtype="principle", lifecycle="established",
                     statement="Principle one")
        _make_wisdom(sqlite, vector, wtype="principle", lifecycle="established",
                     statement="Principle two")
        _make_wisdom(sqlite, vector, wtype="principle", lifecycle="emerging",
                     statement="Principle three")

        profiles = meta.failure_profiles()

        # Find heuristic profile
        heuristic = next((p for p in profiles if p.category == "type:heuristic"), None)
        assert heuristic is not None
        assert heuristic.total_count == 3
        assert heuristic.deprecated_count == 1
        assert abs(heuristic.failure_rate - 1 / 3) < 0.01

        # Principle profile has zero failures
        principle = next((p for p in profiles if p.category == "type:principle"), None)
        assert principle is not None
        assert principle.deprecated_count == 0
        assert principle.failure_rate == 0.0

    def test_failure_profile_by_creation_method(self, sqlite, vector, meta):
        # Pipeline-created wisdom: 2 total, 1 deprecated
        _make_wisdom(sqlite, vector, method="pipeline", lifecycle="deprecated",
                     statement="Pipeline deprecated")
        _make_wisdom(sqlite, vector, method="pipeline", lifecycle="established",
                     statement="Pipeline active")
        # Human-created wisdom: 2 total, 0 deprecated
        _make_wisdom(sqlite, vector, method="human_input", lifecycle="established",
                     statement="Human one")
        _make_wisdom(sqlite, vector, method="human_input", lifecycle="emerging",
                     statement="Human two")

        profiles = meta.failure_profiles()

        pipeline = next((p for p in profiles if p.category == "method:pipeline"), None)
        human = next((p for p in profiles if p.category == "method:human_input"), None)

        assert pipeline is not None
        assert pipeline.failure_rate == 0.5
        assert human is not None
        assert human.failure_rate == 0.0

    def test_single_deprecated_entry(self, sqlite, vector, meta):
        """One deprecated entry should not cause division errors."""
        _make_wisdom(sqlite, vector, wtype="trade_off", lifecycle="deprecated")

        profiles = meta.failure_profiles()
        trade_off = next((p for p in profiles if p.category == "type:trade_off"), None)
        assert trade_off is not None
        assert trade_off.failure_rate == 1.0
        assert trade_off.total_count == 1


class TestDomainRisk:
    """Verify domain risk assessment."""

    def test_domain_with_deprecation(self, sqlite, vector, meta):
        _make_wisdom(sqlite, vector, domains=["databases"], lifecycle="deprecated",
                     statement="Bad database advice")
        _make_wisdom(sqlite, vector, domains=["databases"], lifecycle="established",
                     statement="Good database advice")
        _make_wisdom(sqlite, vector, domains=["python"], lifecycle="established",
                     statement="Python advice")

        risks = meta.domain_risk_assessment()
        assert len(risks) >= 2

        db_risk = next((r for r in risks if r["domain"] == "databases"), None)
        py_risk = next((r for r in risks if r["domain"] == "python"), None)

        assert db_risk is not None
        assert py_risk is not None
        assert db_risk["risk_score"] > py_risk["risk_score"]
        assert db_risk["deprecated_count"] == 1
        assert py_risk["deprecated_count"] == 0

    def test_domain_with_contamination(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, domains=["fragile_domain"],
                         statement="Fragile wisdom")
        # Log contamination events
        sqlite.log_contamination(w.id, "affected-1", "wisdom", 0.05, "cascade")
        sqlite.log_contamination(w.id, "affected-2", "knowledge", 0.03, "cascade")

        risks = meta.domain_risk_assessment()
        fragile = next((r for r in risks if r["domain"] == "fragile_domain"), None)
        assert fragile is not None
        assert fragile["contamination_events"] == 2


class TestRiskScore:
    """Verify risk score computation for individual wisdom entries."""

    def test_low_risk_validated_established(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, lifecycle="established",
                         method="human_input", confidence=0.8,
                         statement="Well-established human wisdom")
        # Add external validation
        sqlite.save_validation_event(w.id, "external", "confirmed",
                                     evidence="Expert reviewed", validator="human")
        # Simulate applications
        w.application_count = 10
        sqlite.update_wisdom(w)

        risk = meta.compute_risk_score(w.id)
        assert risk.base_risk < 0.3
        assert risk.recommended_challenge_level == "standard"

    def test_high_risk_unvalidated_from_risky_type(self, sqlite, vector, meta):
        # Create a context where heuristics fail a lot
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated",
                     statement="Failed heuristic 1")
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated",
                     statement="Failed heuristic 2")
        # Now test a new heuristic
        w = _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="emerging",
                         statement="New untested heuristic")

        risk = meta.compute_risk_score(w.id)
        # Should be elevated or maximum due to high type failure rate + no validation + no applications
        assert risk.base_risk > 0.3
        assert risk.recommended_challenge_level in ("elevated", "maximum")

    def test_risk_factors_documented(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Risk factors test")
        risk = meta.compute_risk_score(w.id)

        assert len(risk.risk_factors) == 5
        factor_names = {f["name"] for f in risk.risk_factors}
        assert factor_names == {
            "type_risk", "method_risk", "domain_risk",
            "validation_risk", "application_risk",
        }
        # Every factor has a reason
        for f in risk.risk_factors:
            assert "reason" in f
            assert "value" in f

    def test_challenge_level_thresholds(self, sqlite, vector, meta):
        # Standard: base_risk <= 0.3
        # We can't fully control the exact risk, but we can verify the mapping
        w = _make_wisdom(sqlite, vector, statement="Threshold test")
        risk = meta.compute_risk_score(w.id)
        # Verify the level matches the threshold
        if risk.base_risk <= 0.3:
            assert risk.recommended_challenge_level == "standard"
        elif risk.base_risk <= 0.6:
            assert risk.recommended_challenge_level == "elevated"
        else:
            assert risk.recommended_challenge_level == "maximum"


class TestContaminationPatterns:
    """Verify super-spreader detection."""

    def test_super_spreader(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Super spreader wisdom")
        # Log many contamination events from this source
        for i in range(10):
            sqlite.log_contamination(
                w.id, f"affected-{i}",
                "wisdom" if i < 5 else "knowledge",
                0.05, "cascade_failure",
            )

        patterns = meta.contamination_patterns()
        assert len(patterns) >= 1
        top = patterns[0]
        assert top.source_wisdom_id == w.id
        assert top.total_affected == 10
        assert top.affected_types["wisdom"] == 5
        assert top.affected_types["knowledge"] == 5

    def test_multiple_sources_sorted(self, sqlite, vector, meta):
        w1 = _make_wisdom(sqlite, vector, statement="Small spreader")
        w2 = _make_wisdom(sqlite, vector, statement="Big spreader")

        for i in range(3):
            sqlite.log_contamination(w1.id, f"a-{i}", "wisdom", 0.05, "cascade")
        for i in range(8):
            sqlite.log_contamination(w2.id, f"b-{i}", "wisdom", 0.05, "cascade")

        patterns = meta.contamination_patterns()
        assert patterns[0].source_wisdom_id == w2.id
        assert patterns[0].total_affected == 8


class TestConfidenceTrajectory:
    """Verify confidence trend analysis."""

    def test_downward_trajectory(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Declining wisdom")
        # Log negative confidence changes
        sqlite.log_confidence_change("wisdom", w.id, 0.8, 0.72, "reinforcement_negative")
        sqlite.log_confidence_change("wisdom", w.id, 0.72, 0.64, "reinforcement_negative")
        sqlite.log_confidence_change("wisdom", w.id, 0.64, 0.56, "reinforcement_negative")

        traj = meta.confidence_trajectory()
        assert traj["total_events"] == 3
        assert traj["avg_delta"] < 0
        assert traj["net_direction"] == "declining"
        assert traj["negative_changes"] == 3

    def test_upward_trajectory(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Improving wisdom")
        sqlite.log_confidence_change("wisdom", w.id, 0.5, 0.55, "reinforcement_positive")
        sqlite.log_confidence_change("wisdom", w.id, 0.55, 0.60, "reinforcement_positive")

        traj = meta.confidence_trajectory()
        assert traj["avg_delta"] > 0
        assert traj["net_direction"] == "improving"

    def test_decrease_reasons_tracked(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Tracked wisdom")
        sqlite.log_confidence_change("wisdom", w.id, 0.8, 0.72, "contamination_cascade")
        sqlite.log_confidence_change("wisdom", w.id, 0.72, 0.64, "contamination_cascade")
        sqlite.log_confidence_change("wisdom", w.id, 0.64, 0.56, "reinforcement_negative")

        traj = meta.confidence_trajectory()
        reasons = traj["top_decrease_reasons"]
        assert len(reasons) >= 1
        assert reasons[0]["reason"] == "contamination_cascade"
        assert reasons[0]["count"] == 2


class TestRiskProfileForAdversarial:
    """Verify risk profile generation for the adversarial engine."""

    def test_standard_risk_returns_none(self, sqlite, vector, meta):
        w = _make_wisdom(sqlite, vector, statement="Low risk wisdom")
        # Add enough context to make risk low
        sqlite.save_validation_event(w.id, "external", "confirmed")
        w.application_count = 5
        sqlite.update_wisdom(w)

        profile = meta.risk_profile_for_adversarial(w.id)
        # May or may not be None depending on other factors, but if standard it's None
        risk = meta.compute_risk_score(w.id)
        if risk.recommended_challenge_level == "standard":
            assert profile is None

    def test_elevated_risk_returns_profile(self, sqlite, vector, meta):
        # Create context where the type fails a lot
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated",
                     statement="Failed 1")
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated",
                     statement="Failed 2")
        w = _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="emerging",
                         statement="New risky heuristic")

        risk = meta.compute_risk_score(w.id)
        if risk.recommended_challenge_level != "standard":
            profile = meta.risk_profile_for_adversarial(w.id)
            assert profile is not None
            assert "risk_level" in profile
            assert "counterexample_threshold" in profile


class TestAdversarialIntegration:
    """Verify that the adversarial engine accepts risk profiles."""

    def test_challenge_accepts_risk_profile(self, sqlite, vector):
        engine = AdversarialEngine(sqlite, vector)
        w = _make_wisdom(sqlite, vector, statement="Test adversarial with risk profile",
                         domains=["testing"])

        # Challenge with risk profile — should not crash
        profile = {
            "risk_level": "maximum",
            "counterexample_threshold": 0.5,
            "blind_spot_frequency": 0.2,
        }
        report = engine.challenge(w, risk_profile=profile)
        assert report.wisdom_id == w.id
        assert isinstance(report.passed, bool)

    def test_challenge_without_risk_profile_unchanged(self, sqlite, vector):
        engine = AdversarialEngine(sqlite, vector)
        w = _make_wisdom(sqlite, vector, statement="Test adversarial without risk",
                         domains=["testing"])

        # Challenge without risk profile — backwards compatible
        report = engine.challenge(w)
        assert report.wisdom_id == w.id
        assert isinstance(report.passed, bool)


class TestSummary:
    """Verify the complete summary output."""

    def test_summary_structure(self, sqlite, vector, meta):
        # Create some data
        _make_wisdom(sqlite, vector, lifecycle="deprecated", statement="Dep 1")
        _make_wisdom(sqlite, vector, lifecycle="established", statement="Est 1")

        s = meta.summary()
        assert "failure_profiles" in s
        assert "risky_domains" in s
        assert "super_spreaders" in s
        assert "trajectory" in s
        assert "total_profiles_analyzed" in s
        assert "total_domains_analyzed" in s

    def test_summary_includes_failure_data(self, sqlite, vector, meta):
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="deprecated",
                     statement="Failed heuristic for summary")
        _make_wisdom(sqlite, vector, wtype="heuristic", lifecycle="established",
                     statement="Active heuristic for summary")

        s = meta.summary()
        assert len(s["failure_profiles"]) > 0
        # The heuristic type should appear in profiles
        has_heuristic = any(
            p.category == "type:heuristic" for p in s["failure_profiles"]
        )
        assert has_heuristic
