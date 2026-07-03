# Epistemic Explorer

Minimal infrastructure for continuous research cycles.

This is not a chatbot. The first MVP is a deterministic research loop that
tests the project mechanics:

- append-only knowledge history;
- explicit knowledge graph;
- competing hypotheses;
- critiques and toy simulations;
- modular objectives;
- periodic meta-observation.

The original project brief is in `PROJECT.md`.

## Run

```bash
python -m epistemic_explorer.runner --cycles 10 --seed 1
```

Outputs are written to:

- `knowledge/events.jsonl`
- `knowledge/nodes.json`
- `knowledge/edges.json`
- `knowledge/cycles/*.json`
- `knowledge/summary.json`
- `knowledge/graph.dot`

## Current MVP Boundary

Agents are deterministic policy classes, not LLM agents. This keeps the first
version reproducible and makes it possible to test whether the research loop,
storage model and objective interface are structurally sound before adding
language-model calls.

## Evolutionary Ecosystem Experiment

The next research question is no longer "how does one explorer generate better
hypotheses?" It is "how do research strategies evolve under competition,
collaboration, energy limits and mutation?"

Run the first ecosystem benchmark:

```bash
python -m epistemic_explorer.evolution_runner --epochs 30 --seed 7 --populations 1 5 20 50
```

Outputs are written to `outputs/evolution/`. The comparison measures strategy
population size against selection pressure, experiment yield, bridge creation,
graph compression, research diversity and dead-end resource burn.

## Next Implementation Steps

1. Replace toy ecosystem payoffs with graph-backed research artifacts.
2. Track hypothesis survival curves explicitly across epochs.
3. Add niche pressure so species do not collapse into one dominant role.
4. Let strategies exchange artifacts and form isolated research communities.
5. Add optional LLM-backed strategy adapters behind the same genome interface.
