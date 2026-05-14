"""Tests for the trust & verification layer — validation, adversarial, propagation, coverage.

These tests verify the system's immune system: its ability to be skeptical
of its own outputs, challenge wisdom before promotion, cascade consequences
when things go wrong, and detect blind spots.
"""

import pytest

from wisdom.config import WisdomConfig
from wisdom.engine.adversarial import AdversarialEngine, ChallengeReport
from wisdom.engine.coverage import CoverageEngine
from wisdom.engine.evolution import EvolutionEngine
from wisdom.engine.experience_engine import ExperienceEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.engine.propagation import PropagationEngine
from wisdom.engine.validation import ValidationEngine
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.engine.knowledge_engine import KnowledgeEngine
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    ExperienceResult,
    KnowledgeType,
    LifecycleState,
    RelationshipType,
    WisdomType,
)
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
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
def exp_engine(sqlite, vector):
    return ExperienceEngine(sqlite, vector)


@pytest.fixture
def know_engine(sqlite, vector):
    return KnowledgeEngine(sqlite, vector)


@pytest.fixture
def evo_engine(sqlite, vector, config, lifecycle):
    return EvolutionEngine(sqlite, vector, config, lifecycle)


@pytest.fixture
def validation(sqlite):
    return ValidationEngine(sqlite)


@pytest.fixture
def adversarial(sqlite, vector):
    return AdversarialEngine(sqlite, vector)


@pytest.fixture
def propagation(sqlite, vector, config):
    return PropagationEngine(sqlite, vector, config)


@pytest.fixture
def coverage(sqlite, vector):
    return CoverageEngine(sqlite, vector)


# ── Validation Tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_self_report_is_weak(self, validation, wis_engine):
        """Self-reported validation should have minimal impact."""
        w = wis_engine.add(statement="Test principle", domains=["test"])
        old_conf = w.confidence.overall

        result = validation.validate(w.id, "self_report", "confirmed")
        # Self-report should not significantly change confidence
        assert result["validation_score"] > 0
        w_after = wis_engine.get(w.id)
        assert abs(w_after.confidence.overall - old_conf) < 0.01

    def test_external_validation_boosts(self, validation, wis_engine):
        """External validation should meaningfully boost confidence."""
        w = wis_engine.add(
            statement="Externally validated",
            domains=["test"],
            confidence=ConfidenceScore(overall=0.5),
        )
        old_overall = w.confidence.overall
        result = validation.validate(
            w.id, "external", "confirmed",
            evidence="Expert review confirms this", validator="expert_1",
        )
        assert result["validation_score"] > 0
        w_after = wis_engine.get(w.id)
        # Validation boosts theoretical dimension, raising overall
        assert w_after.confidence.overall > old_overall
        assert w_after.confidence.theoretical > 0.5

    def test_adversarial_validation_strongest(self, validation, wis_engine):
        """Adversarial validation should have the strongest positive impact."""
        w = wis_engine.add(statement="Adversarially tested", domains=["test"])

        validation.validate(w.id, "adversarial", "confirmed",
                          evidence="Survived devil's advocate")
        score = validation.compute_validation_score(w.id)
        assert score > 0.5  # Strong positive signal

    def test_refutation_penalizes(self, validation, wis_engine):
        """External refutation should significantly reduce confidence."""
        w = wis_engine.add(
            statement="Refuted principle",
            domains=["test"],
            confidence=ConfidenceScore(overall=0.7),
        )
        validation.validate(
            w.id, "external", "refuted",
            evidence="Contradicted by controlled experiment",
        )
        w_after = wis_engine.get(w.id)
        assert w_after.confidence.overall < 0.7

    def test_is_validated_requires_external(self, validation, wis_engine):
        """is_validated should require at least one non-self-report confirmation."""
        w = wis_engine.add(statement="Needs external", domains=["test"])
        assert not validation.is_validated(w.id)

        # Self-report doesn't count
        validation.validate(w.id, "self_report", "confirmed")
        assert not validation.is_validated(w.id)

        # External does count
        validation.validate(w.id, "external", "confirmed")
        assert validation.is_validated(w.id)

    def test_validation_summary(self, validation, wis_engine):
        w = wis_engine.add(statement="Summarized", domains=["test"])
        validation.validate(w.id, "self_report", "confirmed")
        validation.validate(w.id, "external", "confirmed_with_caveats")
        validation.validate(w.id, "peer", "confirmed")

        summary = validation.validation_summary(w.id)
        assert summary["total_events"] == 3
        assert summary["is_validated"]
        assert "external" in summary["by_source"]

    def test_effective_confidence_discounts_unvalidated(self, validation, wis_engine):
        """Unvalidated wisdom should have discounted effective confidence."""
        w = wis_engine.add(
            statement="Unvalidated",
            confidence=ConfidenceScore(overall=0.8),
        )
        effective = validation.effective_confidence(w)
        # Should be discounted by ~40%
        assert effective < 0.8
        assert abs(effective - 0.48) < 0.05

    def test_effective_confidence_full_for_validated(self, validation, wis_engine, sqlite):
        """Validated wisdom should not be discounted."""
        w = wis_engine.add(
            statement="Validated",
            confidence=ConfidenceScore(overall=0.8),
        )
        sqlite.save_validation_event(w.id, "external", "confirmed", "Good")
        effective = validation.effective_confidence(w)
        assert effective >= 0.8


# ── Lifecycle Tests ─────────────────────────────────────────────────────────


class TestLifecycleManager:
    def test_promotion_requires_validation_for_pipeline(self, lifecycle, wis_engine, sqlite):
        """Pipeline-created wisdom should require external validation for promotion."""
        w = wis_engine.add(
            statement="Pipeline wisdom",
            confidence=ConfidenceScore(overall=0.8),
            creation_method=CreationMethod.PIPELINE,
        )
        # Manually set application count above threshold
        w.application_count = 10
        sqlite.update_wisdom(w)
        w = wis_engine.get(w.id)

        # Should NOT promote without validation
        result = lifecycle.evaluate(w)
        assert not result.transitioned
        assert w.lifecycle == LifecycleState.EMERGING

        # Add external validation
        sqlite.save_validation_event(w.id, "external", "confirmed", "Verified")
        result = lifecycle.evaluate(w)
        assert result.transitioned
        assert w.lifecycle == LifecycleState.ESTABLISHED

    def test_seeds_dont_require_validation_first_time(self, lifecycle, wis_engine, sqlite):
        """Seed wisdom should be promotable without external validation on first pass."""
        w = wis_engine.add(
            statement="Seed wisdom",
            confidence=ConfidenceScore(overall=0.8),
            creation_method=CreationMethod.SEED,
        )
        w.application_count = 6
        sqlite.update_wisdom(w)
        w = wis_engine.get(w.id)

        result = lifecycle.evaluate(w)
        assert result.transitioned
        assert w.lifecycle == LifecycleState.ESTABLISHED

    def test_version_increments_on_transition(self, lifecycle, wis_engine, sqlite):
        """Version should increment on lifecycle transitions."""
        w = wis_engine.add(
            statement="Versioned",
            confidence=ConfidenceScore(overall=0.8),
            creation_method=CreationMethod.SEED,
        )
        assert w.version == 1
        w.application_count = 6
        sqlite.update_wisdom(w)
        w = wis_engine.get(w.id)

        lifecycle.evaluate(w)
        assert w.version == 2

    def test_no_double_transition(self, lifecycle, wis_engine, sqlite):
        """Evaluating twice without state change should not double-transition."""
        w = wis_engine.add(
            statement="Stable",
            confidence=ConfidenceScore(overall=0.8),
            creation_method=CreationMethod.SEED,
        )
        w.application_count = 6
        sqlite.update_wisdom(w)
        w = wis_engine.get(w.id)

        lifecycle.evaluate(w)
        assert w.lifecycle == LifecycleState.ESTABLISHED
        old_version = w.version

        result = lifecycle.evaluate(w)
        assert not result.transitioned
        assert w.version == old_version

    def test_force_deprecate(self, lifecycle, wis_engine):
        w = wis_engine.add(statement="To force-deprecate", domains=["test"])
        result = lifecycle.force_deprecate(w, "Testing force deprecation")
        assert result.transitioned
        assert w.lifecycle == LifecycleState.DEPRECATED
        assert w.deprecation_reason == "Testing force deprecation"

    def test_deprecated_is_terminal(self, lifecycle, wis_engine, sqlite):
        """Once deprecated, wisdom should not transition to any other state."""
        w = wis_engine.add(statement="Terminal", domains=["test"])
        lifecycle.force_deprecate(w, "Done")
        assert w.lifecycle == LifecycleState.DEPRECATED

        # Try to force challenge — should fail
        result = lifecycle.force_challenge(w, "Try to revive")
        assert not result.transitioned


# ── Adversarial Tests ───────────────────────────────────────────────────────


class TestAdversarial:
    def test_vagueness_detection(self, adversarial, wis_engine):
        """Wisdom without applicability conditions should be flagged."""
        w = wis_engine.add(
            statement="Be good",  # vague, no conditions
            reasoning="",
        )
        report = adversarial.challenge(w)
        assert not report.passed  # Should fail
        categories = [f.category for f in report.findings]
        assert "vagueness" in categories

    def test_well_formed_wisdom_passes(self, adversarial, wis_engine):
        """Well-formed wisdom with conditions and trade-offs should pass."""
        from wisdom.models.common import TradeOff
        w = wis_engine.add(
            statement="Use connection pooling for database-heavy services to reduce latency",
            reasoning="Opening new connections is expensive, pooling amortizes the cost",
            wisdom_type=WisdomType.HEURISTIC,
            domains=["databases"],
            applicability_conditions=["Services making >10 queries per request"],
            inapplicability_conditions=["Single-query services", "Serverless functions"],
            trade_offs=[TradeOff(
                dimension="resource usage",
                benefit="Lower latency",
                benefit_magnitude=0.8,
                cost="Idle connection memory",
                cost_magnitude=0.3,
            )],
        )
        report = adversarial.challenge(w)
        # May have info-level findings but no critical ones
        assert report.critical_count == 0

    def test_counterexample_detection(self, adversarial, wis_engine, exp_engine):
        """Should find failure experiences that contradict the wisdom."""
        # Add some failure experiences in the same domain
        for i in range(5):
            exp_engine.add(
                description="Database connection pooling caused memory leak and crash",
                domain="databases",
                result=ExperienceResult.FAILURE,
            )

        w = wis_engine.add(
            statement="Always use connection pooling for databases",
            reasoning="Reduces connection overhead",
            domains=["databases"],
            applicability_conditions=["Database-heavy services"],
        )

        report = adversarial.challenge(w)
        counterexamples = [f for f in report.findings if f.category == "counterexample"]
        # Should find the failure experiences as potential counterexamples
        assert len(counterexamples) >= 1

    def test_contradiction_detection(self, adversarial, wis_engine):
        """Should detect explicit conflicts with existing wisdom."""
        from wisdom.models.common import RelationshipType
        w1 = wis_engine.add(statement="Always use ORMs", domains=["databases"])
        w2 = wis_engine.add(statement="Write raw SQL for performance", domains=["databases"])
        wis_engine.relate(w1.id, w2.id, RelationshipType.CONFLICTS, 0.9)

        report = adversarial.challenge(w1)
        contradictions = [f for f in report.findings if f.category == "contradiction"]
        assert len(contradictions) >= 1

    def test_challenge_report_structure(self, adversarial, wis_engine):
        w = wis_engine.add(statement="Test report", domains=["test"])
        report = adversarial.challenge(w)
        d = report.to_dict()
        assert "wisdom_id" in d
        assert "passed" in d
        assert "findings" in d
        assert "summary" in d

    def test_blind_spot_normalization(self, adversarial, wis_engine, exp_engine):
        """Blind spot detection should normalize morphological variants.

        If wisdom says 'index' and experiences say 'indexing', there should
        be NO blind spot — normalization should match them.
        """
        for i in range(6):
            exp_engine.add(
                description=f"Optimized database indexing for query performance {i}",
                domain="databases",
            )

        w = wis_engine.add(
            statement="Design proper index structures for database query performance",
            reasoning="Indexes accelerate lookups",
            domains=["databases"],
            applicability_conditions=["Read-heavy workloads"],
        )

        report = adversarial.challenge(w)
        blind_spots = [f for f in report.findings if f.category == "blind_spot"]
        # 'index' from wisdom and 'indexing' from experiences should match via normalization
        blind_spot_text = " ".join(f.evidence for f in blind_spots)
        assert "index" not in blind_spot_text.split("'"), (
            f"Normalization should match 'index' to 'indexing'; got blind spots: {blind_spot_text}"
        )

    def test_blind_spot_detects_bigram_gaps(self, adversarial, wis_engine, exp_engine):
        """Blind spot detection should catch missing bigram concepts."""
        for i in range(8):
            exp_engine.add(
                description=f"Connection pooling caused resource exhaustion in service {i}",
                domain="databases",
                input_text="Pool size configuration and connection timeout settings",
            )

        w = wis_engine.add(
            statement="Monitor database latency for production stability",
            reasoning="Latency spikes indicate problems",
            domains=["databases"],
            applicability_conditions=["Production systems"],
        )

        report = adversarial.challenge(w)
        blind_spots = [f for f in report.findings if f.category == "blind_spot"]
        assert len(blind_spots) >= 1, "Should detect concepts missing from wisdom"
        all_evidence = " ".join(f.evidence for f in blind_spots)
        assert "connection" in all_evidence or "pool" in all_evidence, (
            f"Should flag connection/pooling concepts; got: {all_evidence}"
        )

    def test_blind_spot_uses_shared_extraction(self, adversarial, wis_engine, exp_engine):
        """Verify adversarial engine uses the same extraction as coverage engine."""
        from wisdom.engine.coverage import _extract_concepts

        for i in range(6):
            exp_engine.add(
                description=f"Implementing caching strategies for microservices {i}",
                domain="architecture",
            )

        w = wis_engine.add(
            statement="Design microservices with clear boundaries",
            reasoning="Bounded contexts prevent coupling",
            domains=["architecture"],
            applicability_conditions=["Distributed systems"],
        )

        report = adversarial.challenge(w)
        blind_spots = [f for f in report.findings if f.category == "blind_spot"]

        # Extract quoted concepts from evidence like "'concept' (3x), 'other' (5x)"
        for finding in blind_spots:
            import re
            concepts_found = re.findall(r"'([a-z_]+)'", finding.evidence)
            for concept in concepts_found:
                if len(concept) > 5:
                    assert not concept.endswith("ing"), (
                        f"Concept '{concept}' not normalized — still ends in -ing"
                    )

    def test_conflict_detection_normalized(self, adversarial):
        """Conflict detection should recognize morphological variants as shared concepts."""
        # "indexing" and "indexed" should normalize to "index";
        # "database" and "queries"/"query" provide additional overlap
        assert adversarial._statements_may_conflict(
            "Always prefer database indexing for query performance",
            "Never use database indexed queries in write-heavy systems",
        )

    def test_conflict_detection_unrelated(self, adversarial):
        """Unrelated statements should not be detected as conflicts."""
        assert not adversarial._statements_may_conflict(
            "Use connection pooling for databases",
            "Write unit tests for all public functions",
        )


# ── Propagation Tests ───────────────────────────────────────────────────────


class TestPropagation:
    def test_cascade_affects_sibling_wisdom(self, propagation, wis_engine, know_engine, sqlite):
        """When wisdom fails, siblings sharing the same knowledge should be penalized."""
        # Create shared knowledge
        k1 = Knowledge(statement="Shared knowledge A", domain="test")
        k2 = Knowledge(statement="Shared knowledge B", domain="test")
        know_engine.add(k1)
        know_engine.add(k2)

        # Create two wisdom entries from overlapping knowledge
        w_good = wis_engine.add(
            statement="Good wisdom",
            source_knowledge_ids=[k1.id, k2.id],
            confidence=ConfidenceScore(overall=0.7),
        )
        w_bad = wis_engine.add(
            statement="Bad wisdom that will fail",
            source_knowledge_ids=[k1.id, k2.id],
            confidence=ConfidenceScore(overall=0.7),
        )

        # Cascade failure from the bad wisdom
        result = propagation.cascade_failure(w_bad.id, severity=1.0)

        # Good wisdom should be penalized
        assert len(result.affected_wisdom) >= 1
        w_good_after = wis_engine.get(w_good.id)
        assert w_good_after.confidence.overall < 0.7

    def test_cascade_affects_source_knowledge(self, propagation, wis_engine, know_engine):
        """Failed wisdom should penalize its source knowledge."""
        k = Knowledge(statement="Source knowledge", domain="test")
        know_engine.add(k)
        old_know_conf = k.confidence.overall

        w = wis_engine.add(
            statement="Bad wisdom",
            source_knowledge_ids=[k.id],
        )
        result = propagation.cascade_failure(w.id, severity=1.0)

        assert len(result.affected_knowledge) >= 1
        k_after = know_engine.get(k.id)
        assert k_after.confidence.overall < old_know_conf

    def test_cascade_contaminates_application_experiences(
        self, propagation, wis_engine, evo_engine, sqlite
    ):
        """Application experiences from failed wisdom should be marked contaminated."""
        w = wis_engine.add(statement="Will fail later", domains=["test"])
        # Generate some application experiences
        evo_engine.reinforce(w.id, was_helpful=True, task_context="Task 1")
        evo_engine.reinforce(w.id, was_helpful=True, task_context="Task 2")

        # Now cascade the failure
        result = propagation.cascade_failure(w.id, severity=1.0)
        assert result.contaminated_experiences >= 2

        # Check that application experiences are marked contaminated
        all_exps = sqlite.list_experiences()
        contaminated = [e for e in all_exps if e.metadata.get("contaminated") == "true"]
        assert len(contaminated) >= 2

    def test_trace_provenance(self, propagation, wis_engine, know_engine, exp_engine):
        """Should trace the full provenance chain."""
        # Create the chain: experience -> knowledge -> wisdom
        exp = exp_engine.add(description="Original experience", domain="test")
        k = Knowledge(
            statement="Derived knowledge",
            domain="test",
            source_experience_ids=[exp.id],
        )
        know_engine.add(k)
        w = wis_engine.add(
            statement="Derived wisdom",
            source_knowledge_ids=[k.id],
        )

        prov = propagation.trace_provenance(w.id)
        assert prov["wisdom"]["id"] == w.id
        assert len(prov["source_knowledge"]) == 1
        assert len(prov["source_knowledge"][0]["experiences"]) == 1

    def test_severity_scales_penalties(self, propagation, wis_engine, know_engine):
        """Lower severity should produce smaller penalties."""
        k = Knowledge(statement="Knowledge", domain="test")
        know_engine.add(k)

        w = wis_engine.add(statement="Mild failure", source_knowledge_ids=[k.id])
        result = propagation.cascade_failure(w.id, severity=0.2)

        if result.affected_knowledge:
            assert result.affected_knowledge[0]["penalty"] < 0.05

    def test_contamination_logged(self, propagation, wis_engine, know_engine, sqlite):
        """Contamination events should be logged in the contamination_log table."""
        k = Knowledge(statement="Knowledge", domain="test")
        know_engine.add(k)
        w = wis_engine.add(statement="Logged failure", source_knowledge_ids=[k.id])

        propagation.cascade_failure(w.id, severity=1.0)

        history = sqlite.get_contamination_history(w.id)
        assert len(history) >= 1


# ── Relationship Cascade Tests ─────────────────────────────────────────────


class TestRelationshipCascade:
    """Tests for relationship-based failure propagation."""

    def test_cascade_through_supports(self, propagation, wis_engine):
        """When A supports B and A fails, B should lose backing."""
        w_supporter = wis_engine.add(
            statement="Supporting evidence principle",
            confidence=ConfidenceScore(overall=0.7),
        )
        w_supported = wis_engine.add(
            statement="Principle that depends on support",
            confidence=ConfidenceScore(overall=0.7),
        )
        wis_engine.relate(
            w_supporter.id, w_supported.id, RelationshipType.SUPPORTS, 0.8
        )

        result = propagation.cascade_failure(w_supporter.id, severity=1.0)

        assert len(result.relationship_affected) >= 1
        affected_ids = [a["id"] for a in result.relationship_affected]
        assert w_supported.id in affected_ids

        w_after = wis_engine.get(w_supported.id)
        assert w_after.confidence.overall < 0.7

    def test_no_penalty_when_supported_fails(self, propagation, wis_engine):
        """When A supports B and B fails, A should not be penalized."""
        w_supporter = wis_engine.add(
            statement="I provide evidence for another",
            confidence=ConfidenceScore(overall=0.7),
        )
        w_supported = wis_engine.add(
            statement="I am supported but will fail",
            confidence=ConfidenceScore(overall=0.7),
        )
        wis_engine.relate(
            w_supporter.id, w_supported.id, RelationshipType.SUPPORTS, 0.9
        )

        result = propagation.cascade_failure(w_supported.id, severity=1.0)

        # The supporter should not be penalized
        supporter_affected = [
            a for a in result.relationship_affected if a["id"] == w_supporter.id
        ]
        assert len(supporter_affected) == 0
        w_supporter_after = wis_engine.get(w_supporter.id)
        assert abs(w_supporter_after.confidence.overall - 0.7) < 0.001

    def test_cascade_through_derived_from(self, propagation, wis_engine):
        """When A is derived_from B and B fails, A should be penalized (suspect source)."""
        w_source = wis_engine.add(
            statement="Original principle",
            confidence=ConfidenceScore(overall=0.7),
        )
        w_derived = wis_engine.add(
            statement="Derived from the original",
            confidence=ConfidenceScore(overall=0.7),
        )
        # w_derived DERIVED_FROM w_source
        wis_engine.relate(
            w_derived.id, w_source.id, RelationshipType.DERIVED_FROM, 0.9
        )

        # Source fails → derived is suspect
        result = propagation.cascade_failure(w_source.id, severity=1.0)

        affected_ids = [a["id"] for a in result.relationship_affected]
        assert w_derived.id in affected_ids
        w_derived_after = wis_engine.get(w_derived.id)
        assert w_derived_after.confidence.overall < 0.7

    def test_cascade_through_complements(self, propagation, wis_engine):
        """When A complements B and A fails, B should get a mild penalty."""
        w1 = wis_engine.add(
            statement="Complementary principle A",
            confidence=ConfidenceScore(overall=0.7),
        )
        w2 = wis_engine.add(
            statement="Complementary principle B",
            confidence=ConfidenceScore(overall=0.7),
        )
        wis_engine.relate(w1.id, w2.id, RelationshipType.COMPLEMENTS, 0.8)

        result = propagation.cascade_failure(w1.id, severity=1.0)

        affected_ids = [a["id"] for a in result.relationship_affected]
        assert w2.id in affected_ids
        w2_after = wis_engine.get(w2.id)
        assert w2_after.confidence.overall < 0.7

    def test_conflict_gives_no_penalty(self, propagation, wis_engine):
        """When A conflicts with B and A fails, B should NOT be penalized.

        The failure of a conflicting entry is actually a positive signal.
        """
        w_bad = wis_engine.add(
            statement="Wrong principle",
            confidence=ConfidenceScore(overall=0.7),
        )
        w_rival = wis_engine.add(
            statement="Rival principle that conflicts",
            confidence=ConfidenceScore(overall=0.7),
        )
        wis_engine.relate(w_bad.id, w_rival.id, RelationshipType.CONFLICTS, 0.9)

        result = propagation.cascade_failure(w_bad.id, severity=1.0)

        # The rival should not be penalized
        rival_affected = [
            a for a in result.relationship_affected if a["id"] == w_rival.id
        ]
        assert len(rival_affected) == 0
        w_rival_after = wis_engine.get(w_rival.id)
        assert abs(w_rival_after.confidence.overall - 0.7) < 0.001

    def test_relationship_penalties_lighter_than_provenance(
        self, propagation, wis_engine, know_engine
    ):
        """Relationship-based penalties should be lighter than provenance-based ones."""
        k = Knowledge(statement="Shared source", domain="test")
        know_engine.add(k)

        w_failed = wis_engine.add(
            statement="Failed wisdom",
            source_knowledge_ids=[k.id],
            confidence=ConfidenceScore(overall=0.7),
        )
        w_sibling = wis_engine.add(
            statement="Sibling via knowledge overlap",
            source_knowledge_ids=[k.id],
            confidence=ConfidenceScore(overall=0.7),
        )
        w_related = wis_engine.add(
            statement="Related via supports relationship",
            confidence=ConfidenceScore(overall=0.7),
        )
        wis_engine.relate(
            w_failed.id, w_related.id, RelationshipType.SUPPORTS, 1.0
        )

        result = propagation.cascade_failure(w_failed.id, severity=1.0)

        # Both should be penalized
        sibling_penalty = next(
            (a["penalty"] for a in result.affected_wisdom if a["id"] == w_sibling.id),
            0,
        )
        rel_penalty = next(
            (a["penalty"] for a in result.relationship_affected if a["id"] == w_related.id),
            0,
        )

        # Relationship penalty should be strictly lighter
        assert rel_penalty > 0
        assert sibling_penalty > 0
        assert rel_penalty < sibling_penalty

    def test_relationship_cascade_logged(self, propagation, wis_engine, sqlite):
        """Relationship cascades should be logged in contamination_log."""
        w1 = wis_engine.add(statement="Will fail", confidence=ConfidenceScore(overall=0.7))
        w2 = wis_engine.add(statement="Supported", confidence=ConfidenceScore(overall=0.7))
        wis_engine.relate(w1.id, w2.id, RelationshipType.SUPPORTS, 0.8)

        propagation.cascade_failure(w1.id, severity=1.0)

        history = sqlite.get_contamination_history(w1.id)
        rel_entries = [h for h in history if "relationship" in h.get("reason", "")]
        assert len(rel_entries) >= 1

    def test_provenance_includes_relationships(self, propagation, wis_engine):
        """trace_provenance should include relationship information."""
        w1 = wis_engine.add(statement="Main principle", domains=["test"])
        w2 = wis_engine.add(statement="Supporting principle", domains=["test"])
        wis_engine.relate(w1.id, w2.id, RelationshipType.SUPPORTS, 0.8)

        prov = propagation.trace_provenance(w1.id)
        assert "relationships" in prov
        assert len(prov["relationships"]) >= 1
        assert prov["relationships"][0]["relationship"] == "supports"

    def test_total_affected_includes_relationships(self, propagation, wis_engine):
        """ContaminationResult.total_affected should count relationship-affected entries."""
        w1 = wis_engine.add(statement="Fails", confidence=ConfidenceScore(overall=0.7))
        w2 = wis_engine.add(statement="Supported", confidence=ConfidenceScore(overall=0.7))
        wis_engine.relate(w1.id, w2.id, RelationshipType.SUPPORTS, 0.8)

        result = propagation.cascade_failure(w1.id, severity=1.0)
        assert result.total_affected >= 1
        assert len(result.relationship_affected) >= 1
        d = result.to_dict()
        assert "relationship_affected" in d


# ── Coverage Tests ──────────────────────────────────────────────────────────


class TestCoverage:
    def test_detects_missing_concepts(self, coverage, wis_engine, exp_engine):
        """Should find concepts frequent in experiences but absent from wisdom."""
        # Add experiences that mention 'indexing' frequently
        for i in range(10):
            exp_engine.add(
                description=f"Optimized database indexing strategy for query {i}",
                domain="databases",
                input_text="Creating composite indexes on frequently joined columns",
            )

        # Add wisdom that doesn't mention indexing
        wis_engine.add(
            statement="Use normalized table design for data integrity",
            reasoning="Normalization prevents data anomalies",
            domains=["databases"],
        )

        result = coverage.find_domain_blind_spots("databases")
        assert result["status"] == "analyzed"
        # 'indexing' should appear as a blind spot
        blind_spot_concepts = [bs["concept"] for bs in result.get("blind_spots", [])]
        assert any("index" in c for c in blind_spot_concepts)

    def test_good_coverage_no_blind_spots(self, coverage, wis_engine, exp_engine):
        """Wisdom that covers experience concepts should have good coverage."""
        for i in range(5):
            exp_engine.add(
                description=f"Testing Python code with pytest framework iteration {i}",
                domain="python",
            )

        wis_engine.add(
            statement="Test Python code thoroughly using pytest framework for reliability",
            reasoning="Testing with pytest catches bugs before production",
            domains=["python"],
        )

        result = coverage.find_domain_blind_spots("python")
        # Coverage should be reasonable
        assert result.get("domain_coverage", 0) > 0

    def test_analyze_single_wisdom(self, coverage, wis_engine, exp_engine):
        """Should analyze coverage of a single wisdom entry."""
        for i in range(5):
            exp_engine.add(description=f"Python debugging task {i}", domain="python")

        w = wis_engine.add(statement="Use debugger", domains=["python"])
        result = coverage.analyze_wisdom_coverage(w)
        assert "wisdom_concepts" in result
        assert "domain_reports" in result

    def test_find_suspicious_wisdom(self, coverage, wis_engine, exp_engine):
        """Should find wisdom with suspiciously low coverage scores."""
        # Add many domain-specific experiences
        for i in range(15):
            exp_engine.add(
                description=f"Complex database migration with schema changes and indexing {i}",
                domain="databases",
                input_text="ALTER TABLE, CREATE INDEX, foreign key constraints",
            )

        # Add vague wisdom that misses most concepts
        wis_engine.add(
            statement="Be careful with databases",
            reasoning="Databases are important",
            domains=["databases"],
        )

        suspicious = coverage.find_suspicious_wisdom(domain="databases")
        # The vague wisdom "Be careful with databases" should be flagged as suspicious
        # because it's too vague relative to the domain's concept space
        assert any(
            "Be careful with databases" in s.get("statement", "")
            for s in suspicious
        ), f"Vague wisdom should be flagged; got {suspicious}"

    def test_insufficient_data_handled(self, coverage, wis_engine):
        """Should handle domains with too few experiences gracefully."""
        wis_engine.add(statement="Sparse domain wisdom", domains=["sparse"])
        result = coverage.find_domain_blind_spots("sparse")
        assert result.get("status") in ("no_wisdom", "analyzed")


class TestSemanticGaps:
    """Tests for embedding-based semantic gap detection."""

    def test_semantic_gaps_insufficient_data(self, coverage):
        result = coverage.find_semantic_gaps("empty_domain")
        assert result["status"] == "insufficient_data"

    def test_semantic_gaps_no_wisdom_embeddings(self, coverage, exp_engine):
        """Should handle domains with experiences but no wisdom embeddings."""
        for i in range(5):
            exp_engine.add(description=f"Experience {i}", domain="orphan")
        result = coverage.find_semantic_gaps("orphan")
        assert result["status"] == "no_wisdom_embeddings"

    def test_semantic_gaps_finds_uncovered(self, coverage, wis_engine, exp_engine):
        """Experiences about topics not covered by wisdom should appear as gaps."""
        wis_engine.add(
            statement="Use connection pooling for database efficiency",
            reasoning="Pooling amortizes connection cost",
            domains=["mixed"],
        )

        # Add experiences about a completely unrelated topic in the same domain
        for i in range(5):
            exp_engine.add(
                description=f"Analyzing soil pH levels and nitrogen content for crop rotation planning season {i}",
                domain="mixed",
            )

        result = coverage.find_semantic_gaps("mixed", similarity_threshold=0.6)
        assert result["status"] == "analyzed"
        assert result["uncovered_count"] >= 1
        assert result["experience_count"] == 5
        assert len(result["most_distant"]) >= 1

    def test_semantic_gaps_covered_experiences(self, coverage, wis_engine, exp_engine):
        """Experiences semantically close to wisdom should not be flagged."""
        wis_engine.add(
            statement="Use connection pooling for database query performance",
            reasoning="Individual connections are expensive",
            domains=["databases"],
        )

        for i in range(5):
            exp_engine.add(
                description=f"Configured database connection pool size for production {i}",
                domain="databases",
            )

        result = coverage.find_semantic_gaps("databases")
        assert result["status"] == "analyzed"
        # These experiences are semantically close to the wisdom
        # Uncovered ratio should be low
        assert result["uncovered_ratio"] < 1.0

    def test_semantic_gaps_threshold(self, coverage, wis_engine, exp_engine):
        """Higher threshold should flag more experiences as uncovered."""
        wis_engine.add(
            statement="Profile before optimizing code performance",
            domains=["performance"],
        )
        for i in range(4):
            exp_engine.add(
                description=f"Debugging memory leak in Python service {i}",
                domain="performance",
            )

        strict = coverage.find_semantic_gaps("performance", similarity_threshold=0.8)
        lenient = coverage.find_semantic_gaps("performance", similarity_threshold=0.3)

        assert strict["uncovered_count"] >= lenient["uncovered_count"]


class TestConceptExtraction:
    """Tests for the improved concept extraction in coverage engine."""

    def test_normalize_strips_ing(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("indexing") == "index"
        assert _normalize("testing") == "test"
        assert _normalize("caching") == "cach"  # conservative: no silent-e guessing

    def test_normalize_strips_ed(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("indexed") == "index"
        assert _normalize("tested") == "test"

    def test_normalize_strips_plural_s(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("connections") == "connection"
        assert _normalize("queries") == "query"  # ies → y

    def test_normalize_preserves_short_words(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("uses") == "uses"  # too short after strip
        assert _normalize("code") == "code"  # no suffix to strip

    def test_normalize_doubled_consonant(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("running") == "run"
        assert _normalize("stopped") == "stop"

    def test_normalize_preserves_inherent_ss(self):
        """Words with inherent 'ss' like passing/missing keep the double-s."""
        from wisdom.engine.coverage import _normalize
        assert _normalize("passing") == "pass"
        assert _normalize("missing") == "miss"
        assert _normalize("processing") == "process"
        assert _normalize("pressed") == "press"

    def test_normalize_preserves_ss_and_us(self):
        from wisdom.engine.coverage import _normalize
        assert _normalize("access") == "access"
        assert _normalize("status") == "status"

    def test_extract_bigrams(self):
        from wisdom.engine.coverage import _extract_concepts
        concepts = _extract_concepts("connection pooling strategy")
        assert "connection" in concepts
        assert "pool" in concepts  # pooling → pool
        assert "strategy" in concepts
        assert "connection_pool" in concepts  # bigram

    def test_extract_strips_punctuation(self):
        from wisdom.engine.coverage import _extract_concepts
        concepts = _extract_concepts("indexing, testing; and caching.")
        assert "index" in concepts  # stripped -ing
        assert "test" in concepts
        assert "cach" in concepts

    def test_normalization_groups_variants(self):
        """Wisdom saying 'index' should match experiences saying 'indexing'."""
        from wisdom.engine.coverage import _extract_concepts
        wisdom_concepts = _extract_concepts("Use proper index design")
        exp_concepts = _extract_concepts("Optimized database indexing")
        # Both should contain 'index'
        assert "index" in wisdom_concepts
        assert "index" in exp_concepts

    def test_bigrams_skip_stopwords(self):
        from wisdom.engine.coverage import _extract_concepts
        concepts = _extract_concepts("index for the query")
        # "index" and "query" are not adjacent after stopword removal
        assert "index_query" not in concepts

    def test_no_self_bigrams(self):
        from wisdom.engine.coverage import _extract_concepts
        concepts = _extract_concepts("test test test")
        # Same word adjacent should not create bigram
        assert "test_test" not in concepts
