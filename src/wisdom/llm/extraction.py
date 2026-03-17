"""LLM-powered knowledge extraction pipeline: Experience[] -> Knowledge[]."""

from __future__ import annotations

import json

from wisdom.exceptions import ExtractionError
from wisdom.llm.prompts import (
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM,
    format_experiences_for_prompt,
)
from wisdom.llm.provider import LLMProvider
from wisdom.logging_config import get_logger
from wisdom.models.common import ConfidenceScore, KnowledgeType
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge

logger = get_logger("llm.extraction")


def extract_knowledge(
    provider: LLMProvider,
    experiences: list[Experience],
    domain: str = "",
) -> list[Knowledge]:
    """Use an LLM to extract knowledge from experiences.

    Args:
        provider: The LLM provider to use
        experiences: List of experiences to analyze
        domain: Domain hint for extraction

    Returns:
        List of extracted Knowledge entries
    """
    if not experiences:
        return []

    experiences_block = format_experiences_for_prompt(experiences)
    prompt = EXTRACTION_PROMPT.format(experiences_block=experiences_block)

    logger.info("Extracting knowledge from %d experiences via %s", len(experiences), provider.name)

    try:
        response = provider.generate(prompt=prompt, system=EXTRACTION_SYSTEM)
    except Exception as e:
        raise ExtractionError(f"LLM generation failed: {e}") from e

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response[:500]}") from e

    entries = data.get("knowledge_entries", [])
    knowledge_list: list[Knowledge] = []

    for entry in entries:
        try:
            conf = entry.get("confidence", {})
            k = Knowledge(
                type=KnowledgeType(entry.get("type", "pattern")),
                statement=entry["statement"],
                explanation=entry.get("explanation", ""),
                preconditions=entry.get("preconditions", []),
                domain=entry.get("domain", domain),
                specificity=entry.get("specificity", 0.5),
                confidence=ConfidenceScore(
                    theoretical=conf.get("theoretical", 0.5),
                    empirical=conf.get("empirical", 0.5),
                    observational=conf.get("observational", 0.5),
                ),
                source_experience_ids=entry.get("source_experience_ids", [e.id for e in experiences]),
                supporting_count=len(entry.get("source_experience_ids", experiences)),
                tags=["llm_extracted"],
                metadata={"extraction_provider": provider.name},
            )
            knowledge_list.append(k)
        except (KeyError, ValueError) as e:
            logger.warning("Skipping malformed knowledge entry: %s", e)
            continue

    logger.info("Extracted %d knowledge entries from %d experiences", len(knowledge_list), len(experiences))
    return knowledge_list
