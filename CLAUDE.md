# CLI Wisdom System

This is my personal project. Care for it like it is your baby. Every change should make it stronger, not just different. If something feels fragile, fix it. If something feels unclear, clarify it. If you're not sure a change is right, don't make it — think longer.

## What this is

A system for LLMs to accumulate, retain, and apply wisdom through the DIKW hierarchy (Data -> Information -> Knowledge -> Wisdom). Not RAG. Not memory. Wisdom — distilled principles with confidence tracking, applicability conditions, trade-off awareness, lifecycle management, and an immune system that actively challenges what it believes.

## How to run

```bash
pip install -e ".[dev]"       # Core + tests
pip install -e ".[mcp]"       # MCP server support
pip install -e ".[all]"       # Everything including LLM providers

wisdom --help                 # CLI
python -m pytest tests/ -v    # Tests (241 tests, all must pass)
python -m wisdom.mcp_server   # MCP server (stdio)

WISDOM_DATA_DIR=/path/to/data wisdom init --seed all   # Bootstrap
```

## Architecture — the dependency graph matters

```
WisdomSystem (composition root: src/wisdom/__init__.py)
  |
  +-- Storage: SQLiteStore + VectorStore (ChromaDB)
  |
  +-- LifecycleManager (SINGLE source of truth for state transitions)
  |     |
  +-- WisdomEngine -----> delegates lifecycle to LifecycleManager
  +-- EvolutionEngine --> delegates lifecycle to LifecycleManager
  |
  +-- Trust layer:
  |     ValidationEngine   -- external verification gates promotions
  |     AdversarialEngine  -- devil's advocate challenges wisdom (accepts risk profiles)
  |     PropagationEngine  -- cascades consequences when wisdom fails
  |     CoverageEngine     -- detects semantic blind spots
  |
  +-- Core engines:
  |     ExperienceEngine, KnowledgeEngine, RetrievalEngine
  |     TriggerEngine, GapAnalysisEngine
  |
  +-- Meta-learning:
  |     MetaLearningEngine -- failure patterns, risk scores, learning velocity, confidence volatility
  |
  +-- LLM: ProviderRegistry (Anthropic, OpenAI, Ollama — all optional)
```

## Rules — read these before changing anything

### The lifecycle state machine lives in ONE place
`engine/lifecycle.py` is the single authority. `WisdomEngine` and `EvolutionEngine` both delegate to it. Never add lifecycle transition logic anywhere else. This was a hard-won fix for a duplication bug where two copies of the state machine drifted apart with different thresholds, different logging, and different side effects.

### The system is structurally skeptical
Unvalidated wisdom gets a 40% confidence discount at retrieval time (`engine/retrieval.py`). Pipeline-created wisdom cannot be promoted to ESTABLISHED without external validation (`engine/lifecycle.py:_has_validation`). This is not a feature flag. This is the system's epistemology. Do not weaken it.

### Failures have real consequences
When wisdom is deprecated or found wrong, `engine/propagation.py:cascade_failure()` traces the provenance graph and applies penalties to sibling wisdom (proportional to knowledge overlap), source knowledge entries, and application experiences (marked contaminated). It also cascades through explicit relationships (SUPPORTS, DERIVED_FROM, COMPLEMENTS, etc.) with directional penalty logic — the semantics of the relationship determine who gets penalized and how much. Relationship penalties are intentionally lighter than provenance penalties (relationships are softer signals). Conflicting wisdom is NOT penalized — the failure of a rival is a positive signal. This is the immune response. A confidence score dropping by 0.08 is bookkeeping; contaminating 50 downstream entries is a consequence.

### The adversarial engine fights back
`engine/adversarial.py` runs five challenge batteries: counterexample search, vagueness detection, contradiction scan, blind spot detection, untested condition check. Wisdom that fails (any critical finding) does not receive adversarial validation. Do not soften the checks. The adversarial engine accepts an optional `risk_profile` dict from MetaLearningEngine to adjust its thresholds for high-risk profiles — but has zero import dependency on it.

### Temporal decay is computed, not stored
Confidence decay over time is applied at retrieval time, not by mutating stored values. The formula lives in `engine/evolution.py:_compute_temporal_decay()` and is used by both `retrieval.py` and `evolution.py`. One formula, one function. Keep it that way.

### Confidence is multi-dimensional
`ConfidenceScore.overall` is a `@computed_field` derived from three sub-dimensions: `empirical` (0.4 weight — field evidence from applications), `theoretical` (0.3 — logical/structural validation), `observational` (0.3 — weak signals). Never set `overall` directly — it is read-only. Use `confidence.apply_delta(dimension, delta)` to change confidence, which scales the sub-dimension change so the effective overall change approximates the requested delta. This ensures confidence provenance is always traceable: you can see WHERE confidence comes from.

### The system learns from its own mistakes
`engine/meta_learning.py:MetaLearningEngine` analyzes contamination logs, confidence history, and deprecation patterns to compute failure profiles and risk scores. It feeds risk-adjusted threshold hints to the adversarial engine, closing the meta-learning loop. It also computes learning velocity (how fast categories of wisdom mature) and confidence volatility (which entries have erratic histories). Volatility feeds directly into risk scoring — entries the system can't make up its mind about get scrutinized harder. All analysis is computed on-demand from existing tables — no new tables, no stored computed values.

### Constructor signatures encode the dependency graph
- `WisdomEngine(sqlite, vector, lifecycle)` — needs lifecycle
- `EvolutionEngine(sqlite, vector, config, lifecycle)` — needs config + lifecycle
- `ExperienceEngine(sqlite, vector)` — no config needed
- `KnowledgeEngine(sqlite, vector)` — no config needed
- `RetrievalEngine(sqlite, vector, config)` — needs config for weights/decay
- `ValidationEngine(sqlite)` — storage only
- `AdversarialEngine(sqlite, vector)` — needs vector for semantic search
- `PropagationEngine(sqlite, vector, config)` — needs everything
- `CoverageEngine(sqlite, vector)` — needs vector for semantic gap detection
- `MetaLearningEngine(sqlite, config)` — no vector needed (structured data analysis only)

If you add a dependency, add it to the constructor. If you need config, take config. Do not reach through `self.sqlite` to get something that should be injected.

## Key design decisions and why

| Decision | Why |
|----------|-----|
| SQLite + ChromaDB, not Postgres | Zero-dependency local deployment. Portable. WAL mode for concurrent reads. |
| ONNX MiniLM-L6-v2 via ChromaDB | sentence-transformers doesn't support Python 3.14. ChromaDB bundles ONNX. |
| Pydantic v2 for all models | Validation, serialization, and schemas from one source. |
| Asymmetric confidence (failures weigh more) | One counterexample is more informative than one confirmation. Epistemically sound. |
| Validation discount (0.6x for unvalidated) | The system should not trust itself. External evidence is required for full confidence. |
| Seeds start EMERGING with confidence 0.5 | Even bundled wisdom must earn its way. Seeds get a validation pass for first promotion only. |
| DEPRECATED is terminal | No zombie wisdom. Once deprecated, it stays deprecated. This prevents oscillation. |
| Confidence is multi-dimensional | `overall` is computed from empirical/theoretical/observational via `weighted_score()`. Mutations go through `apply_delta()` targeting the appropriate dimension. This makes confidence provenance traceable. |
| Meta-learning closes the loop | Failure patterns feed back into adversarial challenge thresholds. The system learns which types/methods/domains fail most. |
| Relationship cascade is directional | If A SUPPORTS B and A fails, B loses backing. If A CONFLICTS with B and A fails, B is vindicated (no penalty). Penalty magnitudes (0.01–0.05) are lighter than provenance penalties (up to 0.1) because relationships are softer signals than shared knowledge. |
| Content-hash dedup for import | SHA-256 of (statement + reasoning) enables cross-system dedup when IDs differ. Hashes are computed at import time, not stored — consistent with compute-at-read-time principle. |
| Relationships table is sole source of truth | The `relationships` column in the `wisdom` table is kept for schema compat but always `[]`. All relationship data flows through the `relationships` table. This eliminates a denormalization bug where the model field was never read but always serialized. |
| Coverage uses bigrams + suffix normalization | `_extract_concepts()` normalizes morphological variants (indexing/indexed/index) and extracts bigrams (connection_pooling). No external NLP dependencies — conservative suffix stripping only. Shared between coverage AND adversarial engines. |
| Coverage has two tiers | Token-level (`find_domain_blind_spots`) catches word-frequency gaps. Embedding-level (`find_semantic_gaps`) catches semantic gaps that words miss (e.g., "memoization" vs "caching"). Both run through the MCP `analyze_coverage` tool. |
| Volatility feeds into risk scoring | `compute_risk_score()` includes a volatility factor: entries with erratic confidence histories (many direction reversals, large swings) get higher risk scores, which feeds into tighter adversarial challenge thresholds. |
| Maintenance includes meta-learning analysis | When wisdom is deprecated during maintenance, `compute_risk_score()` runs on each deprecated entry. This closes the loop: the system learns WHY entries fail, not just THAT they fail. |

## File layout

```
src/wisdom/
  __init__.py          # WisdomSystem composition root — start here
  config.py            # All configuration (weights, thresholds, paths)
  exceptions.py        # Exception hierarchy
  logging_config.py    # Structured logging setup
  models/              # Pydantic v2 data models
    common.py          # Enums, ConfidenceScore, TradeOff, Relationship
    experience.py      # Experience (raw DIKW data layer)
    knowledge.py       # Knowledge (extracted patterns)
    wisdom.py          # Wisdom (principles with lifecycle)
  storage/
    base.py            # Protocol interfaces
    sqlite_store.py    # All CRUD, schema, migrations (v2: validation + contamination tables)
    vector_store.py    # ChromaDB wrapper (3 collections: experiences, knowledge, wisdom)
  engine/
    lifecycle.py       # THE lifecycle state machine (single source of truth)
    experience_engine.py
    knowledge_engine.py
    wisdom_engine.py   # Delegates lifecycle to LifecycleManager
    retrieval.py       # Multi-factor scoring with validation discount + relationship-aware composition
    evolution.py       # Reinforcement loop, delegates lifecycle to LifecycleManager
    validation.py      # External verification framework
    adversarial.py     # Devil's advocate challenge battery
    propagation.py     # Failure cascade through provenance graph + relationship graph
    coverage.py        # Semantic absence detection (bigrams, suffix normalization)
    triggers.py        # Auto-trigger rules for pipeline automation
    gap_analysis.py    # Quantity-based gap detection
    meta_learning.py   # Failure pattern analysis, risk scores, meta-learning loop
  llm/
    provider.py        # LLMProvider ABC + ProviderRegistry
    providers/         # Anthropic, OpenAI, Ollama implementations
    prompts.py         # Prompt templates + JSON schemas
    extraction.py      # Experience -> Knowledge (LLM-powered)
    synthesis.py       # Knowledge -> Wisdom (LLM-powered)
    injection.py       # Wisdom prompt injection + CLAUDE.md generator
  cli/
    app.py             # Main typer app + subcommand registration
    formatters.py      # Rich tables, panels, color-coded output
    experience_cmds.py # exp add|list|show|search|delete|stats
    knowledge_cmds.py  # know extract|list|show|search|validate|delete
    wisdom_cmds.py     # wis add|synthesize|list|show|search|reinforce|challenge|deprecate|relate|transfer|validate|validation-summary|devil-advocate|provenance|cascade-failure
    query_cmds.py      # query search|for-task|conflicts
    analytics_cmds.py  # analytics summary|domains|confidence|health|gaps|audit|coverage
    io_cmds.py         # io export|import|claude-md
  mcp_server/
    server.py          # FastMCP server (17 tools, 3 resources)
    __main__.py        # python -m wisdom.mcp_server
  seeds/               # Bundled starter wisdom (YAML)
tests/
  conftest.py          # Shared fixtures (system, sqlite, vector, lifecycle)
  test_models.py       # Model serialization and validation
  test_storage.py      # SQLite CRUD, migrations, relationships, confidence log
  test_engines.py      # Experience, knowledge, wisdom, evolution, triggers, gaps
  test_retrieval.py    # Multi-factor scoring, temporal decay, validation discount
  test_evolution.py    # Asymmetric confidence, feedback loop, lifecycle transitions
  test_trust.py        # Validation, adversarial, propagation, coverage, concept extraction (61 tests)
  test_meta_learning.py # Meta-learning engine (42 tests)
  test_cli.py          # CLI commands via CliRunner
  test_mcp.py          # MCP tool logic incl. meta-learning integration (15 tests)
```

## SQLite schema (v2)

7 tables: `experiences`, `knowledge`, `wisdom`, `relationships`, `confidence_log`, `validation_events`, `contamination_log`. Schema lives in `storage/sqlite_store.py`. Migration from v1 to v2 is automatic.

## Testing

```bash
python -m pytest tests/ -v                    # All 241 tests
python -m pytest tests/test_trust.py -v       # Trust layer (61 tests)
python -m pytest tests/test_meta_learning.py -v  # Meta-learning (42 tests)
python -m pytest tests/test_retrieval.py -v   # Retrieval + composition (17 tests)
python -m pytest tests/ -k "lifecycle"        # Specific tests
```

Every new engine, every new feature, every bug fix gets a test. The trust layer tests (`test_trust.py`) are the most important — they verify that the system's immune system works: skepticism, challenge, consequence, and absence detection. The meta-learning tests (`test_meta_learning.py`) verify that the system correctly analyzes its own failure patterns.

## What NOT to do

- Do not add lifecycle transition logic outside `engine/lifecycle.py`
- Do not allow confidence to increase without diminishing returns (`success_factor * (1 - current)`)
- Do not let self-reported reinforcement count as external validation
- Do not remove the unvalidated confidence discount from retrieval
- Do not make DEPRECATED a non-terminal state
- Do not duplicate the temporal decay formula — use `_compute_temporal_decay()` from `evolution.py`
- Do not bypass the adversarial engine for promotion convenience
- Do not store computed values (temporal decay, effective confidence) — compute them at read time
- Do not set `confidence.overall` directly — it is a computed property. Use `confidence.apply_delta(dimension, delta)` instead
- Do not add features that weaken the system's skepticism about itself
