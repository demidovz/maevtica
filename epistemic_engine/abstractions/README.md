# Abstraction Growth Experiment

Minimal experiment for testing whether reusable abstraction hierarchies appear
when abstractions are selected for lifetime and reuse rather than classification
accuracy.

Run:

```bash
python epistemic_engine/runner/run_abstraction_growth_experiment.py
```

Outputs are written to:

```text
epistemic_engine/outputs/abstraction_growth/
```

The experiment writes:

- `summary.json`
- `lifetime_reuse_metrics.csv`
- `error_baseline_metrics.csv`
- `lifetime_reuse_graph.dot`
- `error_baseline_graph.dot`

The first MVP deliberately avoids reinforcement learning, rewards, external
labels and predefined hierarchies. Effects are observed consequences of objects;
abstractions live or die by whether they survive reuse on future observations.

## Phase Diagram

Run the second-stage phase diagram:

```bash
python epistemic_engine/runner/run_abstraction_phase_diagram.py
```

Run the reviewer ablation suite:

```bash
python epistemic_engine/runner/run_abstraction_phase_diagram.py --reviewer-suite --seeds 3
```

Fast reviewer smoke:

```bash
python epistemic_engine/runner/run_abstraction_phase_diagram.py --quick --reviewer-suite --seeds 1
```

Outputs:

```text
epistemic_engine/outputs/abstraction_phase_diagram/phase_runs.csv
epistemic_engine/outputs/abstraction_phase_diagram/phase_map_standard.csv
epistemic_engine/outputs/abstraction_phase_diagram/phase_summary.json
```

The phase diagram treats `lifetime_reuse` as one objective among many. It maps:

```text
objective x world x ablation -> regime
```

Current automatic regime labels include:

- `flat_memory`
- `flat_reusable_features`
- `reusable_feature_graph`
- `abstraction_dag`
- `deep_hierarchy`
- `fragmentation`
- `collapse`
- `unstable_churn`
- `mixed`
