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
- ~~No git repo yet~~ FIXED 2026-03-16
- ~~MCP server tested via real session but relationship graph still underutilized~~ FIXED 2026-03-16
- Coverage engine concept extraction still naive (whitespace splitting)

## 2026-03-16 — Relationship Graph + Content Hash Dedup + Fixes

### What was built
- **Git initialized** — first commit with 70 files, 10,970 lines
- **Relationship graph fully wired** — all 6 relationship types (SUPPORTS, COMPLEMENTS, GENERALIZES, SPECIALIZES, DERIVED_FROM, CONFLICTS) now operational in both retrieval and propagation
- **compose_wisdom()** discovers and annotates relationships between result entries with typed notes (was only detecting CONFLICTS)
- **cascade_failure()** traces explicit relationships with directional penalty logic — relationship semantics respected (supporting a failed entry penalizes you; conflicting with a failed entry does not)
- **trace_provenance()** now includes relationship information in output
- **Content-hash dedup for import** — merge mode detects duplicates by SHA-256 content hash, enabling cross-system dedup when IDs differ
- **gap_analysis.py** no longer uses raw SQL — added get_task_type_counts() and count_wisdom_mentioning() to store API
- **analytics audit --since** handles invalid input gracefully (falls back to 7d with warning)
- 23 new tests: 9 relationship cascade, 7 relationship composition, 7 content-hash dedup
- Total: 188 tests, all passing

### Key design insight
Relationship cascade penalties must respect directionality and be lighter than provenance-based penalties. If A SUPPORTS B and A fails, B loses backing. But if B fails, A is fine — it was providing valid evidence for something that turned out wrong. Conflicts are positive signals: if A conflicts with B and A fails, B is vindicated. Penalty magnitudes (0.01–0.05 per relationship) are intentionally weaker than provenance overlap penalties (up to 0.1), because relationships are softer connections than shared knowledge derivation.

### Known edges remaining
- LLM pipeline still untested against real models
- MCP server needs real-world client testing
- Wisdom.relationships field still denormalized (JSON on model + relationships table)
- Coverage engine concept extraction still naive (whitespace splitting)
- Meta-learning could be exposed as MCP tools
