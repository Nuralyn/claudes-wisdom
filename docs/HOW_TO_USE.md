The MCP server does not do anything automatically in the background — it's a set of tools that Claude Code can call when it decides they're relevant. Here's how it
works in practice:

What the MCP server exposes

17 tools that Claude Code can invoke:

┌───────────────────────────┬──────────────────────────────────────────────┐
│           Tool            │                 What it does                 │
├───────────────────────────┼──────────────────────────────────────────────┤
│ search\_wisdom             │ Semantic search for relevant wisdom          │
├───────────────────────────┼──────────────────────────────────────────────┤
│ get\_wisdom                │ Full details on a specific entry             │
├───────────────────────────┼──────────────────────────────────────────────┤
│ add\_experience            │ Record something that happened               │
├───────────────────────────┼──────────────────────────────────────────────┤
│ add\_wisdom                │ Add a principle directly (expert input)      │
├───────────────────────────┼──────────────────────────────────────────────┤
│ extract\_knowledge         │ Process raw experiences into knowledge       │
├───────────────────────────┼──────────────────────────────────────────────┤
│ synthesize\_wisdom         │ Distill knowledge into wisdom                │
├───────────────────────────┼──────────────────────────────────────────────┤
│ reinforce\_wisdom          │ Positive/negative feedback on applied wisdom │
├───────────────────────────┼──────────────────────────────────────────────┤
│ validate\_wisdom           │ Record an external validation event          │
├───────────────────────────┼──────────────────────────────────────────────┤
│ challenge\_wisdom          │ Run the adversarial devil's advocate battery │
├───────────────────────────┼──────────────────────────────────────────────┤
│ cascade\_failure           │ Propagate consequences when wisdom is wrong  │
├───────────────────────────┼──────────────────────────────────────────────┤
│ find\_contradictions       │ Find conflicting wisdom entries              │
├───────────────────────────┼──────────────────────────────────────────────┤
│ get\_wisdom\_gaps           │ Identify blind spots                         │
├───────────────────────────┼──────────────────────────────────────────────┤
│ get\_domain\_summary        │ Stats for a domain                           │
├───────────────────────────┼──────────────────────────────────────────────┤
│ analyze\_coverage          │ Semantic coverage analysis                   │
├───────────────────────────┼──────────────────────────────────────────────┤
│ run\_maintenance           │ Extraction, synthesis, deprecation sweeps    │
├───────────────────────────┼──────────────────────────────────────────────┤
│ get\_risk\_score            │ Meta-learning risk assessment                │
├───────────────────────────┼──────────────────────────────────────────────┤
│ get\_meta\_learning\_summary │ System-wide failure pattern analysis         │
└───────────────────────────┴──────────────────────────────────────────────┘

3 resources: wisdom://domains, wisdom://stats, wisdom://recent

How to test it

It's not passive — Claude Code won't automatically call the tools while you work. You need to explicitly ask it to use the wisdom system. Some things to try:

1. Add an experience: "Record an experience: I found that using connection pooling in Python reduced API latency by 40% in the payments service"
2. Add wisdom directly: "Add wisdom: In microservices, prefer circuit breakers over retry loops when downstream services have variable latency"
3. Search: "Search the wisdom system for advice about error handling"
4. Run the pipeline: After adding a few experiences, ask "Extract knowledge from unprocessed experiences, then synthesize wisdom"
5. Challenge it: After adding wisdom, "Challenge wisdom \[ID] with the adversarial engine"
6. Validate it: "Validate wisdom \[ID] as confirmed with evidence: tested in production for 3 months"
7. Check health: "Show me the meta-learning summary" or "What wisdom gaps exist?"
8. Run maintenance: "Run wisdom maintenance" — this does extraction + synthesis + deprecation sweeps in one shot

The interesting loop to test is: add experiences -> extract knowledge -> synthesize wisdom -> challenge it -> reinforce it with feedback -> watch confidence evolve. That's the DIKW
hierarchy in action.

If the database is empty (freshly initialized without seeds), start by adding a few experiences or wisdom entries so there's something for the search and analysis tools to work
with. You can seed it with wisdom init --seed all from the CLI if you want starter content.

