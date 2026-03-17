"""Experience model — raw interactions and observations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from wisdom.models.common import ExperienceResult, ExperienceType


class Experience(BaseModel):
    """A single recorded experience — the raw data layer of DIKW."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: ExperienceType = ExperienceType.TASK
    domain: str = ""
    subdomain: str = ""
    task_type: str = ""
    description: str
    input_text: str = ""
    output_text: str = ""
    result: ExperienceResult = ExperienceResult.SUCCESS
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    processed: bool = False

    @property
    def embedding_text(self) -> str:
        """Text used for vector embedding."""
        parts = [self.description]
        if self.input_text:
            parts.append(self.input_text)
        if self.output_text:
            parts.append(self.output_text)
        return " ".join(parts)

    @property
    def age_days(self) -> float:
        ts = datetime.fromisoformat(self.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - ts).total_seconds() / 86400.0
