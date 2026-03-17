"""Knowledge model — extracted patterns and validated rules."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from wisdom.models.common import ConfidenceScore, KnowledgeType, ValidationStatus


class Knowledge(BaseModel):
    """Extracted knowledge — patterns, rules, and principles from experiences."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: KnowledgeType = KnowledgeType.PATTERN
    statement: str
    explanation: str = ""
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    domain: str = ""
    subdomain: str = ""
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)
    supporting_count: int = 0
    contradicting_count: int = 0
    source_experience_ids: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    synthesized: bool = False

    @property
    def embedding_text(self) -> str:
        parts = [self.statement]
        if self.explanation:
            parts.append(self.explanation)
        return " ".join(parts)

    @property
    def age_days(self) -> float:
        ts = datetime.fromisoformat(self.updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - ts).total_seconds() / 86400.0

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
