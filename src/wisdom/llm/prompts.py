"""Prompt templates and JSON schemas for LLM-powered pipelines."""

from __future__ import annotations

# ── System Prompts ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction engine. You analyze experiences and identify patterns, rules, facts, and heuristics.

You MUST respond with valid JSON matching the provided schema. No other text."""

SYNTHESIS_SYSTEM = """You are a wisdom synthesis engine. You analyze knowledge entries and existing wisdom to produce higher-order principles with trade-off awareness, confidence assessments, and applicability conditions.

You MUST respond with valid JSON matching the provided schema. No other text."""

VALIDATION_SYSTEM = """You are a wisdom validation engine. Given a wisdom entry and recent experiences, you assess whether the wisdom is still valid, needs revision, or should be deprecated.

You MUST respond with valid JSON matching the provided schema. No other text."""

# ── Extraction Prompt ───────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Analyze the following experiences and extract knowledge entries (patterns, rules, facts, heuristics).

## Experiences

{experiences_block}

## Instructions

For each knowledge entry you identify:
1. Classify it (fact, pattern, rule, principle, heuristic)
2. Write a clear statement
3. Provide an explanation
4. List conditions when it applies (preconditions)
5. Assess confidence (0.0-1.0) across three dimensions: theoretical, empirical, observational
6. List which experience IDs support this knowledge
7. Note any counterexamples or contradictions

## Response Schema

```json
{{
  "knowledge_entries": [
    {{
      "type": "pattern|fact|rule|principle|heuristic",
      "statement": "Clear, actionable statement",
      "explanation": "Why this is true / how it works",
      "preconditions": ["condition1", "condition2"],
      "confidence": {{
        "theoretical": 0.0-1.0,
        "empirical": 0.0-1.0,
        "observational": 0.0-1.0
      }},
      "source_experience_ids": ["id1", "id2"],
      "counterexamples": ["optional counterexample"],
      "domain": "domain if apparent",
      "specificity": 0.0-1.0
    }}
  ]
}}
```

Respond with ONLY the JSON object."""

# ── Synthesis Prompt ────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """Synthesize higher-order wisdom from the following knowledge entries and existing wisdom.

## Knowledge Entries

{knowledge_block}

## Existing Wisdom (for reference — avoid duplicates)

{existing_wisdom_block}

## Instructions

For each wisdom entry you synthesize:
1. Classify it (principle, heuristic, judgment_rule, meta_pattern, trade_off)
2. Write a clear, generalizable statement
3. Provide reasoning
4. List implications
5. Specify applicability conditions (when to apply, when NOT to apply)
6. Identify trade-offs (dimension, benefit, cost, magnitudes)
7. Assess confidence across dimensions
8. Note any contradictions with existing wisdom

## Response Schema

```json
{{
  "wisdom_entries": [
    {{
      "type": "principle|heuristic|judgment_rule|meta_pattern|trade_off",
      "statement": "Clear, generalizable principle",
      "reasoning": "Why this principle holds",
      "implications": ["implication1"],
      "applicability_conditions": ["when to apply"],
      "inapplicability_conditions": ["when NOT to apply"],
      "trade_offs": [
        {{
          "dimension": "e.g., speed vs correctness",
          "benefit": "what you gain",
          "benefit_magnitude": 0.0-1.0,
          "cost": "what you lose",
          "cost_magnitude": 0.0-1.0
        }}
      ],
      "confidence": {{
        "theoretical": 0.0-1.0,
        "empirical": 0.0-1.0,
        "observational": 0.0-1.0
      }},
      "source_knowledge_ids": ["id1", "id2"],
      "domains": ["domain1"],
      "counterexamples": ["optional"]
    }}
  ],
  "contradictions": [
    {{
      "new_wisdom_index": 0,
      "existing_wisdom_id": "id of conflicting existing wisdom",
      "description": "nature of the conflict"
    }}
  ]
}}
```

Respond with ONLY the JSON object."""

# ── Validation Prompt ───────────────────────────────────────────────────────

VALIDATION_PROMPT = """Evaluate whether the following wisdom entry is still valid given recent experiences.

## Wisdom Entry

Type: {wisdom_type}
Statement: {wisdom_statement}
Reasoning: {wisdom_reasoning}
Current confidence: {wisdom_confidence}
Application count: {application_count}
Success rate: {success_rate}

## Recent Relevant Experiences

{recent_experiences_block}

## Instructions

Assess:
1. Is this wisdom still valid?
2. If valid, should confidence be adjusted?
3. If invalid, should it be revised or deprecated?

## Response Schema

```json
{{
  "still_valid": true/false,
  "revised_confidence": 0.0-1.0,
  "reasoning": "explanation of your assessment",
  "suggested_revision": "revised statement if needed, or null",
  "should_deprecate": true/false
}}
```

Respond with ONLY the JSON object."""

# ── Helper Functions ────────────────────────────────────────────────────────


def format_experiences_for_prompt(experiences: list) -> str:
    """Format experience objects for inclusion in prompts."""
    lines = []
    for i, exp in enumerate(experiences, 1):
        lines.append(f"### Experience {i} (ID: {exp.id})")
        lines.append(f"- Domain: {exp.domain}")
        lines.append(f"- Type: {exp.type.value}")
        lines.append(f"- Result: {exp.result.value}")
        lines.append(f"- Description: {exp.description}")
        if exp.input_text:
            lines.append(f"- Input: {exp.input_text[:500]}")
        if exp.output_text:
            lines.append(f"- Output: {exp.output_text[:500]}")
        lines.append("")
    return "\n".join(lines)


def format_knowledge_for_prompt(knowledge_entries: list) -> str:
    """Format knowledge objects for inclusion in prompts."""
    lines = []
    for i, k in enumerate(knowledge_entries, 1):
        lines.append(f"### Knowledge {i} (ID: {k.id})")
        lines.append(f"- Type: {k.type.value}")
        lines.append(f"- Domain: {k.domain}")
        lines.append(f"- Statement: {k.statement}")
        lines.append(f"- Explanation: {k.explanation}")
        lines.append(f"- Confidence: {k.confidence.overall:.2f}")
        if k.preconditions:
            lines.append(f"- Preconditions: {'; '.join(k.preconditions)}")
        lines.append("")
    return "\n".join(lines)


def format_wisdom_for_prompt(wisdom_entries: list) -> str:
    """Format wisdom objects for inclusion in prompts."""
    lines = []
    for i, w in enumerate(wisdom_entries, 1):
        lines.append(f"### Wisdom {i} (ID: {w.id})")
        lines.append(f"- Type: {w.type.value}")
        lines.append(f"- Statement: {w.statement}")
        lines.append(f"- Confidence: {w.confidence.overall:.2f}")
        lines.append(f"- Lifecycle: {w.lifecycle.value}")
        lines.append("")
    return "\n".join(lines)
