"""Wisdom model — higher-order principles with lifecycle management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    LifecycleState,
    Relationship,
    TradeOff,
    WisdomType,
)


class Wisdom(BaseModel):
    """Distilled wisdom — principles with confidence tracking and trade-off awareness."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: WisdomType = WisdomType.PRINCIPLE
    statement: str
    reasoning: str = ""
    implications: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)
    applicable_domains: list[str] = Field(default_factory=list)
    applicability_conditions: list[str] = Field(default_factory=list)
    inapplicability_conditions: list[str] = Field(default_factory=list)
    trade_offs: list[TradeOff] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)
    lifecycle: LifecycleState = LifecycleState.EMERGING
    version: int = 1
    source_knowledge_ids: list[str] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    application_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    deprecation_reason: str = ""
    creation_method: CreationMethod = CreationMethod.PIPELINE
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        parts = [self.statement]
        if self.reasoning:
            parts.append(self.reasoning)
        if self.implications:
            parts.append(" ".join(self.implications))
        return " ".join(parts)

    @property
    def age_days(self) -> float:
        ts = datetime.fromisoformat(self.updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - ts).total_seconds() / 86400.0

    @property
    def success_rate(self) -> float:
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count

    @property
    def negative_feedback_ratio(self) -> float:
        if self.application_count == 0:
            return 0.0
        return self.failure_count / self.application_count

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
