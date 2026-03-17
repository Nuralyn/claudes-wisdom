"""Seed packs for bootstrapping a new Wisdom System."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml

from wisdom.logging_config import get_logger
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    LifecycleState,
    TradeOff,
    WisdomType,
)
from wisdom.models.wisdom import Wisdom

logger = get_logger("seeds")

AVAILABLE_PACKS = [
    "software_engineering",
    "debugging",
    "communication",
    "meta",
]


def _load_yaml(name: str) -> list[dict]:
    """Load a seed YAML file from the seeds package."""
    seed_dir = Path(__file__).parent
    path = seed_dir / f"{name}.yaml"
    if not path.exists():
        logger.warning("Seed pack not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("entries", []) if data else []


def load_seed_pack(name: str) -> list[Wisdom]:
    """Load a named seed pack and return Wisdom entries."""
    if name not in AVAILABLE_PACKS:
        logger.warning("Unknown seed pack: %s (available: %s)", name, AVAILABLE_PACKS)
        return []

    entries = _load_yaml(name)
    wisdom_list = []

    for entry in entries:
        trade_offs = []
        for t in entry.get("trade_offs", []):
            trade_offs.append(TradeOff(
                dimension=t.get("dimension", ""),
                benefit=t.get("benefit", ""),
                benefit_magnitude=t.get("benefit_magnitude", 0.5),
                cost=t.get("cost", ""),
                cost_magnitude=t.get("cost_magnitude", 0.5),
            ))

        w = Wisdom(
            type=WisdomType(entry.get("type", "principle")),
            statement=entry["statement"],
            reasoning=entry.get("reasoning", ""),
            implications=entry.get("implications", []),
            counterexamples=entry.get("counterexamples", []),
            applicable_domains=entry.get("domains", []),
            applicability_conditions=entry.get("applicability_conditions", []),
            inapplicability_conditions=entry.get("inapplicability_conditions", []),
            trade_offs=trade_offs,
            confidence=ConfidenceScore(
                empirical=0.5, theoretical=0.5, observational=0.5,
            ),
            lifecycle=LifecycleState.EMERGING,
            creation_method=CreationMethod.SEED,
            tags=["seed", name],
        )
        wisdom_list.append(w)

    logger.info("Loaded %d entries from seed pack '%s'", len(wisdom_list), name)
    return wisdom_list


def load_all_seeds() -> list[Wisdom]:
    """Load all available seed packs."""
    all_wisdom = []
    for pack in AVAILABLE_PACKS:
        all_wisdom.extend(load_seed_pack(pack))
    return all_wisdom
