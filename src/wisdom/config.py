"""Configuration for the Wisdom System."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_data_dir() -> Path:
    env = os.environ.get("WISDOM_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".wisdom"


class RetrievalWeights(BaseModel):
    semantic: float = 0.4
    confidence: float = 0.3
    applicability: float = 0.2
    recency: float = 0.1


class ConfidenceConfig(BaseModel):
    success_factor: float = 0.05
    failure_delta: float = -0.08
    contradiction_delta: float = -0.15
    decay_rate_per_month: float = 0.02


class ThresholdConfig(BaseModel):
    auto_extract_experiences: int = 10
    auto_synthesize_knowledge: int = 5
    negative_feedback_ratio: float = 0.3
    emerging_to_established_count: int = 5
    emerging_to_established_confidence: float = 0.7
    challenged_confidence: float = 0.5
    deprecated_confidence: float = 0.3


class LLMConfig(BaseModel):
    default_provider: str = "anthropic"
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"
    ollama_model: str = "llama3.2"
    temperature: float = 0.3
    max_tokens: int = 4096


class WisdomConfig(BaseModel):
    data_dir: Path = Field(default_factory=_default_data_dir)
    sqlite_filename: str = "wisdom.db"
    chroma_dirname: str = "chroma_data"
    retrieval: RetrievalWeights = Field(default_factory=RetrievalWeights)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    log_level: str = Field(default_factory=lambda: os.environ.get("WISDOM_LOG_LEVEL", "INFO"))

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / self.chroma_dirname

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
