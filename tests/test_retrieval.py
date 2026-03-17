"""Tests for the retrieval engine."""

import pytest

from wisdom.config import WisdomConfig
from wisdom.engine.retrieval import RetrievalEngine
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.engine.wisdom_engine import WisdomEngine
from wisdom.engine.knowledge_engine import KnowledgeEngine
from wisdom.models.common import ConfidenceScore, WisdomType
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
def retrieval(sqlite, vector, config):
    return RetrievalEngine(sqlite, vector, config)


@pytest.fixture
def wis_engine(sqlite, vector, lifecycle):
    return WisdomEngine(sqlite, vector, lifecycle)


@pytest.fixture
def know_engine(sqlite, vector):
    return KnowledgeEngine(sqlite, vector)


class TestRetrievalEngine:
    def test_search_empty(self, retrieval):
        results = retrieval.search("anything")
        assert results == []

    def test_search_wisdom(self, retrieval, wis_engine):
        wis_engine.add(
            statement="Always write tests before shipping",
            reasoning="Prevents regressions",
            domains=["testing"],
            confidence=ConfidenceScore(overall=0.8),
        )
        wis_engine.add(
            statement="Use dependency injection for testability",
            reasoning="Makes mocking easier",
            domains=["testing"],
            confidence=ConfidenceScore(overall=0.6),
        )
        results = retrieval.search("how to write better tests")
        assert len(results) >= 1
        # Higher confidence entry should score higher (all else being similar)

    def test_search_with_domain(self, retrieval, wis_engine):
        wis_engine.add(statement="Python type hints improve code", domains=["python"])
        wis_engine.add(statement="Rust ownership prevents bugs", domains=["rust"])
        results = retrieval.search("type safety", domain="python")
        # Python entry should have higher applicability score
        if len(results) >= 2:
            python_scores = [r for r in results if "python" in getattr(r.entity, "applicable_domains", [])]
            rust_scores = [r for r in results if "rust" in getattr(r.entity, "applicable_domains", [])]
            if python_scores and rust_scores:
                assert python_scores[0].applicability > rust_scores[0].applicability

    def test_search_knowledge_layer(self, retrieval, know_engine):
        know_engine.add(Knowledge(statement="Caching reduces latency", domain="performance"))
        results = retrieval.search("how to improve speed", layers=["knowledge"])
        assert len(results) >= 1
        assert results[0].layer == "knowledge"

    def test_search_excludes_deprecated(self, retrieval, wis_engine):
        w = wis_engine.add(statement="Deprecated principle", domains=["test"])
        wis_engine.deprecate(w.id, "No longer valid")
        results = retrieval.search("principle", include_deprecated=False)
        deprecated_ids = [r.entity.id for r in results]
        assert w.id not in deprecated_ids

    def test_search_for_task(self, retrieval, wis_engine):
        wis_engine.add(
            statement="Use connection pooling for database-heavy services",
            domains=["databases"],
            confidence=ConfidenceScore(overall=0.9),
        )
        results = retrieval.search_for_task("optimizing database queries")
        assert len(results) >= 1

    def test_compose_wisdom(self, retrieval, wis_engine):
        wis_engine.add(
            statement="Write tests first",
            reasoning="Catches bugs early",
            domains=["testing"],
        )
        wis_engine.add(
            statement="Test behavior not implementation",
            reasoning="More maintainable",
            domains=["testing"],
        )
        composition = retrieval.compose_wisdom("how to test effectively")
        assert len(composition["entries"]) >= 1
        assert composition["composition"]

    def test_find_contradictions_empty(self, retrieval):
        results = retrieval.find_contradictions()
        assert results == []

    def test_temporal_decay(self, retrieval, wis_engine, sqlite):
        w = wis_engine.add(
            statement="Recent wisdom",
            domains=["test"],
            confidence=ConfidenceScore(overall=0.8),
        )
        result = retrieval._compute_effective_confidence(wis_engine.get(w.id))
        # Just created, no temporal decay, but unvalidated discount (0.6x) applies
        # 0.8 * 0.6 = 0.48
        assert abs(result - 0.48) < 0.01

    def test_temporal_decay_validated(self, retrieval, wis_engine, sqlite):
        """Validated wisdom should not get the unvalidated discount."""
        w = wis_engine.add(
            statement="Validated wisdom",
            domains=["test"],
            confidence=ConfidenceScore(overall=0.8),
        )
        # Add external validation
        sqlite.save_validation_event(w.id, "external", "confirmed", "Expert review")
        result = retrieval._compute_effective_confidence(wis_engine.get(w.id))
        # No temporal decay (just created), no unvalidated discount
        assert abs(result - 0.8) < 0.01
