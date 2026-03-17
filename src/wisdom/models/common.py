"""Shared enums, value objects, and types used across all model layers."""

from __future__ import annotations

from enum import Enum

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────


class ExperienceType(str, Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    DEBUGGING = "debugging"
    REVIEW = "review"
    LEARNING = "learning"
    WISDOM_APPLICATION = "wisdom_application"
    OTHER = "other"


class ExperienceResult(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"


class KnowledgeType(str, Enum):
    FACT = "fact"
    PATTERN = "pattern"
    RULE = "rule"
    PRINCIPLE = "principle"
    HEURISTIC = "heuristic"


class ValidationStatus(str, Enum):
    UNVALIDATED = "unvalidated"
    VALIDATED = "validated"
    CHALLENGED = "challenged"


class WisdomType(str, Enum):
    PRINCIPLE = "principle"
    HEURISTIC = "heuristic"
    JUDGMENT_RULE = "judgment_rule"
    META_PATTERN = "meta_pattern"
    TRADE_OFF = "trade_off"


class LifecycleState(str, Enum):
    EMERGING = "emerging"
    ESTABLISHED = "established"
    CHALLENGED = "challenged"
    DEPRECATED = "deprecated"


class RelationshipType(str, Enum):
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    COMPLEMENTS = "complements"
    CONFLICTS = "conflicts"
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"


class CreationMethod(str, Enum):
    PIPELINE = "pipeline"
    HUMAN_INPUT = "human_input"
    SEED = "seed"


# ── Value Objects ──────────────────────────────────────────────────────────


class ConfidenceScore(BaseModel):
    """Multi-dimensional confidence tracking.

    Confidence is tracked across three evidence dimensions:
    - empirical: field evidence from applications (reinforcement feedback)
    - theoretical: logical/structural validation (adversarial/external/peer)
    - observational: weak signals (self-report, user observations)

    `overall` is computed from these sub-dimensions via weighted_score().
    It is never stored independently — this follows the system's principle
    of computing derived values at read time.
    """

    model_config = ConfigDict(extra="ignore")

    # Weights for each sub-dimension in the overall computation
    _WEIGHTS: ClassVar[dict[str, float]] = {
        "empirical": 0.4,
        "theoretical": 0.3,
        "observational": 0.3,
    }

    theoretical: float = Field(default=0.5, ge=0.0, le=1.0)
    empirical: float = Field(default=0.5, ge=0.0, le=1.0)
    observational: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_overall(cls, data: dict) -> dict:
        """Backward-compat: if only overall= is given, set all sub-dimensions to that value.

        When the old API passed ConfidenceScore(overall=0.9), the intent was
        "start at 0.9 across the board." We preserve that by distributing the
        overall value uniformly across sub-dimensions.
        """
        if not isinstance(data, dict):
            return data
        if "overall" in data:
            overall_val = data.pop("overall")
            # Only use overall as a uniform initializer if NO sub-dimensions are specified
            has_subs = any(k in data for k in ("empirical", "theoretical", "observational"))
            if not has_subs:
                data["empirical"] = overall_val
                data["theoretical"] = overall_val
                data["observational"] = overall_val
        return data

    @computed_field
    @property
    def overall(self) -> float:
        """Derived from sub-dimensions. Computed at read time, not stored."""
        return max(0.0, min(1.0, self.weighted_score()))

    def weighted_score(self) -> float:
        """Return a single weighted confidence value."""
        return 0.4 * self.empirical + 0.3 * self.theoretical + 0.3 * self.observational

    def apply_delta(self, dimension: str, desired_overall_delta: float) -> None:
        """Apply a delta scaled so the effective overall change approximates desired_overall_delta.

        The sub-dimension is updated by delta/weight, so after recomputation
        overall changes by approximately desired_overall_delta (exact when
        other dimensions don't change and no clamping occurs).
        """
        weight = self._WEIGHTS[dimension]
        sub_delta = desired_overall_delta / weight
        current = getattr(self, dimension)
        new_val = max(0.0, min(1.0, current + sub_delta))
        setattr(self, dimension, new_val)


class TradeOff(BaseModel):
    """Represents a trade-off dimension in a wisdom entry."""

    dimension: str
    benefit: str
    benefit_magnitude: float = Field(default=0.5, ge=0.0, le=1.0)
    cost: str
    cost_magnitude: float = Field(default=0.5, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A typed relationship between two entities."""

    id: str = ""
    source_id: str
    source_type: str  # 'experience', 'knowledge', 'wisdom'
    target_id: str
    target_type: str
    relationship: RelationshipType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: str = ""


class DomainSpec(BaseModel):
    """Domain and subdomain specification."""

    domain: str
    subdomain: str = ""

    @property
    def full(self) -> str:
        if self.subdomain:
            return f"{self.domain}/{self.subdomain}"
        return self.domain
