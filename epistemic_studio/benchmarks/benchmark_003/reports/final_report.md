# Benchmark Research #003 Final Report

This report describes organizational mechanisms, not software components.

## What organizational properties create better research?

The advantage comes from preserving claims as stable objects, separating generation from falsification, keeping counterexamples visible, and using refined questions plus compression to decide what deserves attention. The strongest effects were not UI, agent count, or a full market; they were institutional constraints on memory, criticism, and reuse.

Empirical mechanism ranking:

- M1 Persistent Research State: contribution 0.0503, confidence 0.672
- M10 Counterexample preservation: contribution 0.0404, confidence 0.635
- M5 Append-only memory: contribution 0.04, confidence 0.642
- M4 Knowledge Graph: contribution 0.0376, confidence 0.629
- M8 Compression: contribution 0.0357, confidence 0.626

## What properties are merely implementation details?

The Journal, Applicability tracking, and Planner-guided prioritization matter less as specific modules than as implementations of auditability, boundary tracking, and attention discipline. The exact dashboard, market shape, and per-agent ceremony are accidental.

Lowest-contribution mechanisms:

- M3 Planner-guided prioritization: contribution 0.0242
- M2 Explorer separated from Engine: contribution 0.0238
- M9 Applicability tracking: contribution 0.0179
- M6 Journal: contribution 0.0046

## Which mechanisms appear universal?

- M1 Persistent Research State

These appear universal because their contribution survived cross-validation across questions, seeds, and execution orders.

## Which mechanisms appear accidental?

Journal as a file format, applicability as a separate report, and planner as a named agent appear accidental. They can be replaced by any mechanism that preserves auditability, boundary conditions, and priority discipline.

## Interaction analysis

- M1 + M4: synergy 0.0069
- M10 + M7: synergy 0.0061
- M1 + M2: synergy 0.0057
- M1 + M5: synergy 0.0051
- M3 + M8: synergy 0.005

## Minimal architecture search

Reference score: 0.501; 90% threshold: 0.451; full mechanism set in this simulator: 0.501.

Best minimal candidate: M1, M2, M3, M4, M5, M7, M8, M10 with score 0.4709.

## If humanity could preserve only THREE principles

1. Persistent append-only research state: this combines M1 and M5 as the highest-memory principle. It protects against forgetting, rewriting, and untraceable drift.
2. Institutionalized adversarial separation: M2 plus M10. Generation and destruction must be separated, and counterexamples must remain first-class objects.
3. Compression through refined questions: M7 plus M8. Better research comes from dissolving bad questions and reducing many claims into fewer structures without hiding counterexamples.

These three preserve the measured advantage better than preserving named agents, dashboards, journals, or markets.
