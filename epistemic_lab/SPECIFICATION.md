# Epistemic Laboratory v0.1

## Research Goal

This project is **not** an AGI implementation.

It is a computational laboratory for experimentally studying the emergence of epistemic structures.

The objective is to discover, through exhaustive search and controlled experiments, the minimal conditions under which representations, concepts and ontologies emerge.

The implementation should prioritize mathematical clarity over performance.

---

# Fundamental Principle

The laboratory should never assume concepts.

Concepts must be allowed to emerge.

Initially there are only:

* a world
* an interaction history
* a projection
* elementary projection transformations
* an objective function

Everything else must emerge experimentally.

---

# 1. World

Implement a minimal deterministic world.

Initially use worlds consisting of only 3–6 states.

Example:

A → B

B → C

C → A

D → D

Later support random world generation.

The laboratory must be able to enumerate many different worlds.

---

# 2. History

History is simply

H = (s₀,s₁,s₂,...)

Initially actions may be omitted.

The first version should focus entirely on observation sequences.

Later action loops can be introduced.

---

# 3. Projection

Projection Π is the central mathematical object.

A projection maps histories into internal equivalence classes.

Initially implement Π simply as a partition over world states (or short histories).

Examples:

Π₀

{A,B,C,D}

↓

X

Π₁

{A,B}

↓

X

{C,D}

↓

Y

Π₂

A

B

C

D

all separate

Internally represent projections as partitions.

---

# 4. Elementary Projection Operators

Initially implement only three operators.

Split

Split one partition into two.

Merge

Merge two partitions.

Compose

Construct higher-level partitions from existing ones.

Do NOT implement ontology growth, language, reflection or reasoning yet.

---

# 5. Cost Function

Each projection receives a score.

Initial proposal:

Score = PredictionAccuracy − λ · Complexity

where

PredictionAccuracy

is prediction quality of future world states

and

Complexity

is description length of the partition.

Keep λ configurable.

---

# 6. Search

No learning algorithm initially.

Instead perform exhaustive search.

For every projection Π

apply every operator

obtain Π'

measure score

store transition.

The goal is to build the complete projection graph.

---

# 7. Projection Graph

Nodes:

all reachable projections.

Edges:

projection transformations.

Store:

parent

child

operator

score difference

complexity

prediction accuracy

---

# 8. Experiments

Experiment 1

Enumerate all projections.

Question:

How many local optima exist?

---

Experiment 2

Which operators improve score most frequently?

---

Experiment 3

Does Split dominate early evolution?

---

Experiment 4

Does Merge dominate later evolution?

---

Experiment 5

Do stable projections appear naturally?

---

Experiment 6

How does λ affect the landscape?

---

Experiment 7

Does the projection graph contain basins of attraction?

---

# 9. Logging

Every experiment should automatically produce

statistics

projection graph

operator frequencies

complexity distributions

prediction distributions

CSV files

JSON experiment dumps

Graphviz export

No visualization assumptions.

Everything should be exportable.

---

# 10. Architecture

world.py

projection.py

operators.py

cost.py

search.py

graph.py

experiments.py

analysis.py

No machine learning libraries.

Pure Python.

Deterministic.

Small.

Readable.

---

# 11. Success Criteria

The project succeeds if it can answer questions such as:

Why is this projection stable?

Why did Split occur here?

Why does Merge improve prediction?

How many local optima exist?

Can concepts emerge without being predefined?

The project fails if it merely implements another learning algorithm.

The objective is to experimentally investigate the mathematics of epistemic projection.

---

# 12. Future Extensions (Not Yet)

Actions

Rewards

Attention

Reflection

Ontology graphs

Language

Meta-projections

Self-modifying projections

These should not be implemented until the minimal laboratory is working.

---

# Final Requirement

The implementation should be treated as a scientific instrument.

Every design decision must maximize the ability to test hypotheses rather than maximize intelligence.

The laboratory should make it easy to falsify theoretical assumptions.

It is acceptable if the implementation reveals that the current theory is wrong.

That is considered a successful scientific outcome.
