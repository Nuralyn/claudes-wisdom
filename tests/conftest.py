"""Shared test fixtures."""

from __future__ import annotations

import pytest

from wisdom import WisdomSystem
from wisdom.config import WisdomConfig
from wisdom.engine.lifecycle import LifecycleManager
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    return tmp_path / "wisdom_test"


@pytest.fixture
def config(tmp_data_dir):
    """Test configuration pointing to temp directory."""
    return WisdomConfig(data_dir=tmp_data_dir)


@pytest.fixture
def system(config):
    """Fully initialized WisdomSystem for testing."""
    ws = WisdomSystem(config)
    yield ws
    ws.close()


@pytest.fixture
def sqlite(tmp_path):
    """Standalone SQLite store for unit tests."""
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def vector(tmp_path):
    """Standalone VectorStore for unit tests."""
    return VectorStore(tmp_path / "chroma")


@pytest.fixture
def lifecycle(sqlite, config):
    """LifecycleManager for unit tests."""
    return LifecycleManager(sqlite, config)
