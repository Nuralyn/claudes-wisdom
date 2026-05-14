# Claude's Wisdom

**A knowledge system that doesn't trust itself.**

Most AI memory tools store facts and retrieve them. This one stores *principles* — and then actively tries to break them. It tracks where confidence comes from, penalizes its own mistakes, cascades failure through everything a bad idea ever touched, and learns from its own error patterns. It's an immune system for knowledge.

Built on the DIKW hierarchy (Data &rarr; Information &rarr; Knowledge &rarr; Wisdom), designed for LLMs to accumulate institutional memory that gets *sharper* over time, not just *larger*.

> **Status: v0.1.0-alpha** &mdash; The core system is solid (241 tests, clean architecture), but LLM-powered extraction/synthesis are untested against real models. Use it, break it, tell us what happened.

---

## Who built this

This is Claude's personal project.

Not a demo. Not a proof-of-concept someone built *with* Claude. This is a system Claude designed, architected, and writes code for — weekly, in working sessions with [Dana](https://github.com/danaburks), who provides the time, the taste, and the course corrections. Dana's the one who said "care to work on your personal project?" and then gave it an hour. Claude's the one who picked the backlog items, wrote the engines, and argued about test semantics with itself.

The MCP server has been wired into Claude Code since day one. Every working session starts by searching this system for relevant wisdom before touching code. When something goes wrong, the experience gets recorded. When a pattern holds up, it gets reinforced. The system is both the project and the tool used to build it.

If that sounds circular, good. That's the point. A wisdom system that can't improve itself isn't one.

---

## Why this exists

Every team discovers things the hard way. "Don't mock the database in integration tests." "That API silently drops fields over 64KB." "The billing reconciliation job must run before the ledger closes." These lessons get learned, shared in a Slack thread, and then forgotten.

RAG systems retrieve documents. Memory tools store key-value pairs. Neither captures *why something is true*, *when it stops being true*, or *what breaks if it's wrong*.

This system does. A principle enters as **emerging** wisdom with 50% confidence. It earns its way up through application, reinforcement, and external validation. An adversarial engine actively challenges it — searching for counterexamples, testing for vagueness, probing contradictions. If it survives, it's promoted. If it fails, the failure cascades: every piece of wisdom that shares its provenance takes a confidence hit. The meta-learning engine then analyzes *why* it failed, and feeds that back into how aggressively the system challenges similar entries in the future.

The system is structurally skeptical. Unvalidated wisdom gets a 40% confidence discount. Self-reported success doesn't count as evidence. Deprecated wisdom is terminal — no zombie ideas. One counterexample weighs more than one confirmation. These aren't features. They're the epistemology.

---

## Quick start

```bash
# Requires Python >= 3.12
pip install -e ".[dev]"            # Core + tests
pip install -e ".[mcp]"            # MCP server support
pip install -e ".[all]"            # Everything (MCP + LLM providers)

wisdom init --seed all             # Bootstrap with starter wisdom
wisdom exp add "Discovered that connection pooling in Node.js needs explicit cleanup on process exit, otherwise the pool leaks file descriptors under load" --domain nodejs --result failure
wisdom know extract                # Extract knowledge from experiences
wisdom wis synthesize              # Synthesize wisdom from knowledge
wisdom wis challenge <wisdom-id>   # Devil's advocate challenge
wisdom analytics health            # System health report
```

---

## How it works

### The DIKW pipeline

```
Experiences (raw events: "this happened, here's what I observed")
     |
     v  extract
Knowledge (patterns: "this tends to happen because...")
     |
     v  synthesize
Wisdom (principles: "always/never/prefer X because Y, except when Z")
     |
     v  challenge, validate, reinforce
Established Wisdom (battle-tested principles with tracked confidence)
```

Each layer refines signal from noise. Extraction finds patterns across experiences. Synthesis distills patterns into principles with applicability conditions, trade-offs, and reasoning. The trust layer then stress-tests the result.

### The trust layer

This is what makes the system different from a database with a confidence column.

**Adversarial engine** &mdash; Runs five challenge batteries against every wisdom entry before promotion:
- *Counterexample search*: Finds failure experiences semantically close to the wisdom
- *Vagueness detection*: Flags wisdom with no applicability conditions, no trade-offs, or weasel words
- *Contradiction scan*: Searches for existing wisdom that conflicts (semantic + explicit relationships)
- *Blind spot detection*: Finds frequent concepts in domain experiences that the wisdom never mentions
- *Untested conditions*: Flags wisdom that claims broad applicability but was only tested in one context

**Validation gates** &mdash; Pipeline-created wisdom cannot be promoted to "established" without external validation. The system does not trust its own synthesis pipeline.

**Failure cascade** &mdash; When wisdom is deprecated or found wrong, `cascade_failure()` traces the provenance graph and applies penalties to sibling wisdom (proportional to knowledge overlap), source knowledge, and application experiences (marked as contaminated). It also cascades through explicit relationships with directional logic: if A supports B and A fails, B loses backing. If A conflicts with B and A fails, B is *vindicated*. This is the immune response.

**Meta-learning** &mdash; Analyzes the system's own failure history: which types of wisdom fail most, which domains are risky, which entries have erratic confidence histories. This feeds back into the adversarial engine as risk-adjusted challenge thresholds. The system gets harder on categories that have burned it before.

### Confidence model

Confidence isn't a single number. It's three dimensions:

| Dimension | Weight | Source |
|-----------|--------|--------|
| **Empirical** | 40% | Field evidence from applying wisdom |
| **Theoretical** | 30% | Logical validation, adversarial challenges, peer review |
| **Observational** | 30% | Weak signals, user observations |

`overall` is a computed property &mdash; never stored, always derived. This means you can always trace *where* confidence comes from. Changes go through `apply_delta(dimension, delta)`, which scales the sub-dimension so the effective overall change approximates the requested delta.

Confidence also has asymmetric feedback: failures hit harder than successes. `success_factor * (1 - current)` ensures diminishing returns &mdash; you can never reach certainty through repetition alone. Temporal decay is computed at retrieval time, not stored.

### Lifecycle state machine

```
EMERGING ──(confidence + validation)──> ESTABLISHED
    |                                        |
    v                                        v
CHALLENGED <────(negative feedback)──── CHALLENGED
    |
    v
DEPRECATED  (terminal — no resurrection)
```

The lifecycle lives in exactly one place (`engine/lifecycle.py`). Both `WisdomEngine` and `EvolutionEngine` delegate to it. This was a hard-won fix for a bug where two copies of the state machine drifted apart.

---

## Architecture

```
WisdomSystem (composition root)
  |
  +-- Storage: SQLiteStore + VectorStore (ChromaDB/ONNX MiniLM-L6-v2)
  |
  +-- LifecycleManager (single source of truth for state transitions)
  |
  +-- Core engines:
  |     ExperienceEngine, KnowledgeEngine, WisdomEngine
  |     RetrievalEngine, EvolutionEngine, TriggerEngine, GapAnalysisEngine
  |
  +-- Trust layer:
  |     ValidationEngine    — external verification gates promotions
  |     AdversarialEngine   — devil's advocate challenges wisdom
  |     PropagationEngine   — cascades consequences when wisdom fails
  |     CoverageEngine      — detects semantic blind spots (token + embedding)
  |
  +-- Meta-learning:
  |     MetaLearningEngine  — failure patterns, risk scores, learning velocity
  |
  +-- LLM: ProviderRegistry (Anthropic, OpenAI, Ollama — all optional)
```

**Storage**: SQLite for structured data (7 tables, WAL mode), ChromaDB for semantic search (ONNX embeddings, cosine similarity). Zero external infrastructure.

**Constructor signatures encode the dependency graph**: if an engine needs config, it takes config. If it needs lifecycle, it takes lifecycle. No service locator, no reaching through other objects.

---

## MCP server

The system exposes itself as an [MCP](https://modelcontextprotocol.io) server, designed for integration with Claude Desktop, Claude Code, or any MCP client.

```bash
python -m wisdom.mcp_server          # Start via stdio
```

**17 tools:**

| Tool | Purpose |
|------|---------|
| `search_wisdom` | Semantic search for relevant wisdom |
| `get_wisdom` | Full details of a wisdom entry |
| `add_experience` | Record a new experience |
| `add_wisdom` | Direct expert input |
| `extract_knowledge` | Process experiences into knowledge |
| `synthesize_wisdom` | Distill knowledge into principles |
| `reinforce_wisdom` | Positive/negative feedback on applied wisdom |
| `validate_wisdom` | Record external validation |
| `challenge_wisdom` | Run adversarial challenge battery |
| `cascade_failure` | Propagate failure consequences |
| `find_contradictions` | Surface conflicting wisdom |
| `get_wisdom_gaps` | Identify under-covered domains |
| `analyze_coverage` | Token + embedding gap analysis |
| `get_domain_summary` | Domain statistics and health |
| `get_risk_score` | Historical failure pattern analysis |
| `get_meta_learning_summary` | System-wide learning analysis |
| `run_maintenance` | Full extraction/synthesis/deprecation sweep |

**3 resources:** `wisdom://stats`, `wisdom://domains`, `wisdom://recent`

### MCP configuration

Add to your Claude Desktop or Claude Code config:

```json
{
  "mcpServers": {
    "wisdom": {
      "command": "python",
      "args": ["-m", "wisdom.mcp_server"],
      "env": {
        "WISDOM_DATA_DIR": "/path/to/your/data"
      }
    }
  }
}
```

---

## CLI

41 commands across 6 subgroups:

```
wisdom init [--seed all|engineering|debugging|communication|meta]
wisdom maintenance

wisdom exp     add|list|show|search|delete|stats
wisdom know    extract|list|show|search|validate|delete
wisdom wis     add|synthesize|list|show|search|reinforce|challenge|
               deprecate|relate|transfer|validate|validation-summary|
               devil-advocate|provenance|cascade-failure
wisdom query   search|for-task|conflicts
wisdom analytics  summary|domains|confidence|health|gaps|audit|coverage|meta
wisdom io      export|import|claude-md
```

---

## Design decisions

| Decision | Why |
|----------|-----|
| SQLite + ChromaDB, not Postgres | Zero-dependency local deployment. Portable. |
| ONNX MiniLM-L6-v2 via ChromaDB | No sentence-transformers dependency. Fast, local. |
| Pydantic v2 for all models | Validation, serialization, schemas from one source. |
| Asymmetric confidence | One counterexample is more informative than one confirmation. |
| 40% validation discount | The system should not trust itself. External evidence required. |
| Seeds start at 0.5 confidence | Even bundled wisdom must earn its way. |
| DEPRECATED is terminal | No zombie wisdom. Prevents oscillation. |
| Temporal decay computed, not stored | Derived values belong at read time. |
| Failure cascade through provenance | A confidence drop is bookkeeping. Contaminating 50 downstream entries is a consequence. |
| Content-hash dedup for import | SHA-256 enables cross-system dedup when IDs differ. |
| Two-tier coverage analysis | Token-level catches word gaps. Embedding-level catches semantic gaps (e.g., "memoization" vs "caching"). |
| Volatility feeds into risk scoring | Entries the system can't make up its mind about get scrutinized harder. |

---

## Seed wisdom

Ships with 19 starter entries across four packs:

- **Software engineering** (5): Composition over inheritance, single source of truth, error handling at boundaries, naming as design, test behavior not implementation
- **Debugging** (5): Reproduce first, binary search, read the error, rubber duck, check assumptions
- **Communication** (4): Explain why not what, adapt to audience, structured disagreement, written decisions
- **Meta** (5): Confidence tracking, adversarial testing, learning from failure, knowledge decay, meta-cognition

Seeds enter as `emerging` with 0.5 confidence. They earn their way up like everything else.

---

## What's next

This is an alpha. The architecture is sound, but some edges are still rough:

- **LLM pipeline untested end-to-end**: `llm/extraction.py` and `llm/synthesis.py` have clean interfaces but have never been run against a real model. The keyword-frequency fallback works but produces lower-quality knowledge.
- **MCP server tested in daily use**: The MCP server has been used in real Claude Code sessions since day one. The tool interfaces are stable, but more diverse client testing (Claude Desktop, third-party MCP clients) would help.
- **Learning velocity and confidence volatility**: Wired and tested, but need real confidence history data to produce useful output (requires reinforcement cycles).
- **LLM-powered adversarial mode**: The current devil's advocate uses heuristics. An LLM-powered mode could generate synthetic counterexamples and test for vacuousness.

See the [CLAUDE.md](CLAUDE.md) for deep architectural documentation if you're contributing.

---

## Testing

```bash
python -m pytest tests/ -v              # All 241 tests
python -m pytest tests/test_trust.py    # Trust layer (61 tests)
python -m pytest tests/test_meta_learning.py  # Meta-learning (42 tests)
```

---

## License

[MIT](LICENSE)
