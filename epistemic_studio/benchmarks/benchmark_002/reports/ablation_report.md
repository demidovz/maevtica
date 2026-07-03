# Ablation Report

## Baseline comparison

- Baseline A - Single LLM: epistemic value 0.25; overhead 0.08; One pass researcher with no persistent graph, market, or agent roles.
- Baseline B - Sequential multi-agent pipeline: epistemic value 0.348; overhead 0.22; Explorer -> Lab -> Engine -> summary, fixed order, no market or mutable state.
- Baseline C - Human researcher with notes: epistemic value 0.405; overhead 0.18; Human-maintained notes and judgment, no explicit graph or automated market.
- Baseline D - Current Studio: epistemic value 0.501; overhead 0.63; Full artifact state, graph, attention market, agents, journal, meta-memory, and dashboard.

## Component ablations

- Remove Historian: retained value 0.87, overhead removed 0.08, verdict keep minimal.
- Remove Planner: retained value 0.54, overhead removed 0.05, verdict keep.
- Remove Meta Observer: retained value 0.91, overhead removed 0.06, verdict remove from core.
- Remove Research Journal: retained value 0.82, overhead removed 0.18, verdict keep slim.
- Remove Attention Economy: retained value 0.63, overhead removed 0.12, verdict keep simpler.
- Remove Knowledge Graph: retained value 0.49, overhead removed 0.16, verdict keep.
- Remove Counterexample Graph: retained value 0.78, overhead removed 0.07, verdict merge into graph.
- Remove Agent Evolution: retained value 0.96, overhead removed 0.1, verdict remove.
