"""Tests for the MCP server tools (unit tests without running server)."""

import pytest

from wisdom import WisdomSystem
from wisdom.config import WisdomConfig
from wisdom.models.common import ConfidenceScore


@pytest.fixture
def system(tmp_path):
    config = WisdomConfig(data_dir=tmp_path / "mcp_test")
    ws = WisdomSystem(config)
    yield ws
    ws.close()


class TestMCPToolLogic:
    """Test the logic that MCP tools use, without MCP protocol overhead."""

    def test_search_wisdom_empty(self, system):
        results = system.retrieval.search("anything")
        assert results == []

    def test_add_experience_and_search(self, system):
        exp = system.experiences.add(
            description="Debugging Python memory leaks",
            domain="python",
        )
        assert exp.id
        results = system.experiences.search("memory leak")
        assert len(results) >= 1

    def test_add_wisdom_and_search(self, system):
        w = system.wisdom.add(
            statement="Profile before optimizing",
            reasoning="Premature optimization wastes time on non-bottlenecks",
            domains=["performance"],
        )
        results = system.retrieval.search("how to optimize code performance")
        assert len(results) >= 1

    def test_reinforce_and_check(self, system):
        w = system.wisdom.add(
            statement="Test reinforce via MCP",
            domains=["test"],
        )
        result = system.evolution.reinforce(w.id, was_helpful=True, feedback="Helpful")
        assert result.application_count == 1
        assert result.success_count == 1

    def test_domain_summary(self, system):
        system.experiences.add(description="Python task 1", domain="python")
        system.experiences.add(description="Python task 2", domain="python")
        system.wisdom.add(statement="Python principle", domains=["python"])

        exp_count = system.sqlite.count_experiences(domain="python")
        wis_count = system.sqlite.count_wisdom(domain="python")
        assert exp_count == 2
        assert wis_count == 1

    def test_find_contradictions(self, system):
        from wisdom.models.common import RelationshipType
        w1 = system.wisdom.add(statement="Always use ORM", domains=["databases"])
        w2 = system.wisdom.add(statement="Write raw SQL for performance", domains=["databases"])
        system.wisdom.relate(w1.id, w2.id, RelationshipType.CONFLICTS, strength=0.8)

        conflicts = system.retrieval.find_contradictions(w1.id)
        assert len(conflicts) == 1

    def test_gap_analysis(self, system):
        for i in range(5):
            system.experiences.add(description=f"Task {i}", domain="python")
        gaps = system.gaps.find_wisdom_gaps(domain="python")
        assert len(gaps) >= 1
        assert gaps[0]["domain"] == "python"

    def test_maintenance(self, system):
        summary = system.run_maintenance()
        assert isinstance(summary, dict)
        assert "extracted" in summary

    def test_stats(self, system):
        stats = system.stats()
        assert "experiences" in stats
        assert "knowledge" in stats
        assert "wisdom" in stats

    def test_compose_wisdom(self, system):
        system.wisdom.add(
            statement="Use type hints",
            reasoning="Catches bugs at lint time",
            domains=["python"],
        )
        system.wisdom.add(
            statement="Write docstrings for public APIs",
            reasoning="Improves discoverability",
            domains=["python"],
        )
        composition = system.retrieval.compose_wisdom("writing good Python code")
        assert "entries" in composition
        assert "composition" in composition
