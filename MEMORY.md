# Session Notes

## 2026-03-15 — Initial Build + Trust Layer

### What was built
- Complete DIKW wisdom system: 64 source files, 136 tests, all passing
- Core pipeline: Experience -> Knowledge -> Wisdom with semantic search (ChromaDB)
- Trust layer: Validation, Adversarial, Propagation, Coverage engines
- CLI with 40+ commands, MCP server with 15 tools + 3 resources
- 19 seed wisdom entries across 4 packs

### Current state
- All 136 tests green
- `pip install -e ".[dev]"` works
- CLI entry point `wisdom` works
- MCP server imports and starts cleanly
- No git repo yet (has .gitignore, needs `git init`)

### Bugs fixed this session
- Lifecycle state machine was duplicated in evolution.py and wisdom_engine.py — consolidated into lifecycle.py
- Variable shadowing (`w` as both Wisdom param and loop var) in adversarial.py
- Double-write in evolution.py reinforce() — now conditional on transition
- Inline Counter import moved to module level in adversarial.py
- confidence_log ordering used timestamp (same-second collisions) — switched to autoincrement id
- EMERGING -> CHALLENGED transition was missing — added

### Known edges (not bugs, just unfinished)
- LLM extraction/synthesis never tested against a real LLM
- ~~`weighted_score()` on ConfidenceScore is dead code~~ FIXED: now powers `overall` as computed_field
- `Wisdom.relationships` field is denormalized (JSON on model + relationships table)
- `gap_analysis.py` does raw SQL in one place
- `--since` parser in analytics audit doesn't handle invalid input gracefully

## 2026-03-15 — MCP Session (another Claude instance via MCP server)

### What was added
- **MetaLearningEngine** (`engine/meta_learning.py`) — analyzes failure patterns, computes risk scores, feeds adversarial engine
- **ConfidenceScore refactored** — `overall` is now a `@computed_field` derived from sub-dimensions; `apply_delta()` routes through correct dimension; backward compat via model_validator
- **Adversarial engine extended** — accepts `risk_profile` dict from meta-learning (zero import coupling)
- **ValidationEngine extended** — boosts `theoretical` dimension on external validation
- **6 new SQLite query methods** for meta-learning aggregates
- **`wisdom analytics meta` CLI command** — failure profiles, domain risk, super-spreaders, confidence trajectory
- **26 new tests** in test_meta_learning.py
- Total: 165 tests, all passing

### Key design insight from this session
The confidence model refactoring solved the dead `weighted_score()` problem and the `overall` drift problem simultaneously. `overall` is now derived, not stored. Reinforcement targets `empirical`, validation targets `theoretical`. You can trace WHERE confidence comes from by looking at the sub-dimensions. This is provenance at the confidence level.

### Known edges
- LLM pipeline still untested against real models
- No git repo yet
- MCP server tested via real session but relationship graph still underutilized
- Coverage engine concept extraction still naive (whitespace splitting)
