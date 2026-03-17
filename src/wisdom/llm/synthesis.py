"""LLM-powered wisdom synthesis pipeline: Knowledge[] -> Wisdom[]."""

from __future__ import annotations

import json

from wisdom.exceptions import SynthesisError
from wisdom.llm.prompts import (
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM,
    format_knowledge_for_prompt,
    format_wisdom_for_prompt,
)
from wisdom.llm.provider import LLMProvider
from wisdom.logging_config import get_logger
from wisdom.models.common import (
    ConfidenceScore,
    CreationMethod,
    LifecycleState,
    TradeOff,
    WisdomType,
)
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom

logger = get_logger("llm.synthesis")


def synthesize_wisdom(
    provider: LLMProvider,
    knowledge_entries: list[Knowledge],
    existing_wisdom: list[Wisdom] | None = None,
    domain: str = "",
) -> tuple[list[Wisdom], list[dict]]:
    """Use an LLM to synthesize wisdom from knowledge entries.

    Args:
        provider: The LLM provider to use
        knowledge_entries: Knowledge to synthesize from
        existing_wisdom: Existing wisdom for dedup and conflict detection
        domain: Domain hint

    Returns:
        Tuple of (new wisdom entries, detected contradictions)
    """
    if not knowledge_entries:
        return [], []

    knowledge_block = format_knowledge_for_prompt(knowledge_entries)
    existing_block = format_wisdom_for_prompt(existing_wisdom or []) if existing_wisdom else "(None yet)"

    prompt = SYNTHESIS_PROMPT.format(
        knowledge_block=knowledge_block,
        existing_wisdom_block=existing_block,
    )

    logger.info(
        "Synthesizing wisdom from %d knowledge entries via %s",
        len(knowledge_entries), provider.name,
    )

    try:
        response = provider.generate(prompt=prompt, system=SYNTHESIS_SYSTEM)
    except Exception as e:
        raise SynthesisError(f"LLM generation failed: {e}") from e

    # Parse JSON response
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SynthesisError(f"Failed to parse LLM response: {e}\nResponse: {response[:500]}") from e

    entries = data.get("wisdom_entries", [])
    contradictions = data.get("contradictions", [])
    wisdom_list: list[Wisdom] = []

    for entry in entries:
        try:
            conf = entry.get("confidence", {})
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
                applicable_domains=entry.get("domains", [domain] if domain else []),
                applicability_conditions=entry.get("applicability_conditions", []),
                inapplicability_conditions=entry.get("inapplicability_conditions", []),
                trade_offs=trade_offs,
                confidence=ConfidenceScore(
                    theoretical=conf.get("theoretical", 0.5),
                    empirical=conf.get("empirical", 0.5),
                    observational=conf.get("observational", 0.5),
                ),
                lifecycle=LifecycleState.EMERGING,
                source_knowledge_ids=entry.get("source_knowledge_ids", [k.id for k in knowledge_entries]),
                creation_method=CreationMethod.PIPELINE,
                tags=["llm_synthesized"],
                metadata={"synthesis_provider": provider.name},
            )
            wisdom_list.append(w)
        except (KeyError, ValueError) as e:
            logger.warning("Skipping malformed wisdom entry: %s", e)
            continue

    logger.info(
        "Synthesized %d wisdom entries, found %d contradictions",
        len(wisdom_list), len(contradictions),
    )
    return wisdom_list, contradictions
