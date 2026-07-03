# Research Critique for Epistemic Laboratory v0.1

This note reviews `SPECIFICATION.md` as a research instrument, not as an
implementation plan. The main question is whether the proposed lab can actually
test claims about emergent epistemic structure, rather than only optimize a
partition of observed states.

## 1. Research Critique

The specification has a good discipline: small deterministic worlds, exhaustive
search, explicit projection graph, and exportable experiment logs. That is the
right shape for a falsifiable lab.

The weak point is that the stated research goal is stronger than the initial
formal setup. The goal asks about representations, concepts and ontologies
emerging. The initial machinery only defines partitions that trade future-state
prediction against complexity. A high-scoring partition is not yet a concept.
It may be only a compression artifact.

The current setup risks proving a weaker and less interesting statement:

`some state partitions predict future states better than other state partitions
under a chosen complexity penalty.`

That is useful, but it is not yet evidence for emergent concepts or ontologies.
To make the lab scientifically strong, every experiment should separate these
claims:

1. predictive compression exists;
2. the compression is stable across histories and worlds;
3. the compression captures a latent regularity rather than labels or sampling
   bias;
4. the same projection supports counterfactual or out-of-distribution prediction;
5. calling the projection a concept adds explanatory power.

The current experiments mostly address the first two claims. They do not yet
test the third, fourth or fifth.

## 2. Hidden Assumptions

The specification currently assumes several things that should be made explicit.

First, it assumes that concepts can be represented as equivalence classes. This
may be true for the first lab, but it excludes graded concepts, relational
concepts, context-dependent concepts and procedural concepts.

Second, it assumes that prediction of the next world state is the right pressure
under which concepts emerge. Some useful concepts do not predict the next state;
they predict intervention effects, long-horizon outcomes, risk, controllability,
or hidden causes.

Third, it assumes that a deterministic world with direct state observation is a
good minimal substrate. This may be too easy. If the observed state is already
the world's true state, then the finest partition often wins unless complexity
dominates. That makes the lab mostly about compression, not epistemic discovery.

Fourth, it assumes that description length of a partition is a meaningful proxy
for conceptual complexity. But the description length depends on the encoding.
For example, `{A,C}` can be cheap if `A` and `C` share a hidden generator, but
expensive if states are just arbitrary labels.

Fifth, it assumes that exhaustive search over projections is not a learning
algorithm. Operationally that is fine, but scientifically it still induces an
optimization process. The important distinction should be: the lab searches over
all hypotheses for analysis, not because the agent is learning online.

Sixth, it assumes that local optima and basins of attraction are meaningful
without a specified dynamics over the projection graph. If the graph is complete
and exhaustive, "early" and "later" evolution are undefined unless a traversal
rule or improvement dynamics is defined.

Seventh, it assumes that `Split`, `Merge` and `Compose` are enough elementary
operators. `Compose` is especially underspecified and may smuggle in ontology
growth under another name.

Eighth, it assumes that stability of a projection means epistemic value. A
projection can be stable because it is over-regularized, because the world is
degenerate, or because the score hides errors that matter.

## 3. Better Minimal World for First Experiments

The initial world should be slightly less trivial than a fully observed cycle,
but still small enough for exhaustive enumeration.

Use a hidden-mode observation world:

```text
latent mode:   M in {0, 1}
visible state: O in {a, b, c}
world state:   (M, O)
observation:   only O
transition:
  if M = 0: a -> b -> c -> a
  if M = 1: a -> c -> b -> a
mode switch: optional fixed switch after k steps, or no switch in v0
```

Why this is better:

1. The lab cannot just partition directly observed true states.
2. A useful projection must infer hidden structure from history.
3. Short histories matter, so the difference between state partitions and
   history partitions becomes testable.
4. There is a clean false-abstraction trap: grouping observations by label can
   look good locally while failing when the hidden mode changes or when the same
   visible observation has different futures.

For the absolute first experiment, avoid stochasticity and actions. Use fixed
deterministic sequences generated from each hidden mode. Compare projections
over:

1. current observation only: `Pi(O_t)`;
2. short history: `Pi(O_{t-1}, O_t)`;
3. true hidden state oracle: `Pi(M_t, O_t)` as an upper bound, not as an allowed
   agent projection.

This gives a minimal ladder:

`label compression -> temporal concept -> hidden-state approximation -> oracle`

The first success criterion should not be "concepts emerge". It should be:

`Does exhaustive projection search recover a compact history partition that
predicts the next observation better than any current-observation partition?`

That is narrower, but it is a real first win.

## 4. Ambiguous Definitions

`World` is ambiguous. It can mean true latent state space, visible observation
space, transition system, generator of histories, or a family of environments.
The spec should distinguish `true_state`, `observation`, and `transition`.

`History` is ambiguous because it says `H = (s0, s1, s2, ...)`, but later the
projection can be over states or short histories. If observations differ from
true states, history should be `H = (o0, o1, o2, ...)` for the agent and
`S = (s0, s1, s2, ...)` for the evaluator.

`Projection` is ambiguous because it maps histories into equivalence classes,
but the first implementation represents it as a partition over world states or
short histories. These are different hypothesis spaces.

`Compose` is ambiguous. It could mean composition of two projections, forming a
product partition, building a hierarchy, or creating a derived feature. These
have different expressive power and different complexity costs.

`PredictionAccuracy` is ambiguous. It should specify target, horizon, scoring
rule and evaluation distribution. Examples: next true state, next observation,
next equivalence class, log loss, 0/1 accuracy, cross-entropy, train history, or
held-out generated histories.

`Complexity` is ambiguous. It could be number of blocks, number of bits needed
to encode block membership, number of split/merge operations from the trivial
projection, or minimum description length under a grammar.

`Stable projection` is ambiguous. It could mean local optimum under operators,
global optimum for a lambda range, recurring optimum across worlds, robust under
held-out histories, or invariant under relabeling of states.

`Concept` is undefined. The lab can postpone the philosophical definition, but
it still needs an operational test. A candidate definition for v0:

`A concept is a projection block or derived partition that improves held-out
prediction under compression and remains useful under at least one controlled
distribution shift.`

`Ontology` is undefined and should probably be removed from v0 success criteria.
Partitions alone do not yet form an ontology unless relations between concepts
are represented and tested.

`Early` and `later` evolution are ambiguous in experiments 3 and 4. Exhaustive
search has no intrinsic time. Define an improvement path, beam path, shortest
operator distance from the trivial partition, or lambda continuation path.

## 5. Counterexample: False Conclusions from the Lab

Here is a concrete failure mode where the lab may conclude that stable concepts
emerged, while the projection is only exploiting a superficial artifact.

World:

```text
true hidden state: (mode, phase)
mode A phases: A0 -> A1 -> A2 -> A0
mode B phases: B0 -> B1 -> B2 -> B0

observed labels:
  A0 -> x
  A1 -> y
  A2 -> z
  B0 -> x
  B1 -> z
  B2 -> y
```

Training histories mostly come from mode A. Evaluation is accidentally drawn
from the same distribution.

A projection over visible labels finds:

```text
{x}, {y}, {z}
```

It has high prediction accuracy and low complexity. The projection graph shows a
stable local optimum. Split appears early. The lab may report that natural
concepts emerged.

But this conclusion is false. The projection does not represent the hidden
regularity. It has learned the mode-A label cycle. In mode B, `y` and `z` have
different futures. The stable projection is a sampling artifact.

The false conclusion occurs because:

1. the train and evaluation histories share the same hidden mode bias;
2. prediction target is next visible label only;
3. complexity rewards the label partition;
4. there is no held-out mode or counterfactual transition test;
5. the lab treats local stability as conceptual validity.

A stricter lab would catch this by adding:

1. balanced evaluation across hidden modes;
2. held-out mode-switch histories;
3. comparison against short-history projections;
4. relabeling tests to ensure results do not depend on arbitrary state names;
5. reporting whether the best projection is robust or only in-distribution.

## 6. Recommended v0 Scope Change

Do not start with "concepts and ontologies emerge". Start with this narrower
claim:

`In small deterministic partially observed worlds, exhaustive search over
history projections can identify compact equivalence classes that improve
held-out prediction over raw observation partitions.`

This keeps the lab minimal, falsifiable and implementable. If this fails, the
larger theory should not be trusted yet. If it succeeds, the next step is to
test robustness under hidden-mode switches, relabeling and changes in lambda.

## 7. Immediate Spec Edits Suggested

1. Split `World` into `true_state`, `observation`, `transition`, and
   `history_generator`.
2. Define the first allowed projection domain exactly:
   `current observation`, `short observation history`, or `true state`.
3. Make `true state` projections oracle-only for evaluation in partially
   observed worlds.
4. Replace vague `PredictionAccuracy` with a specific scoring rule.
5. Replace vague `Complexity` with `number_of_blocks` for v0, then add MDL later.
6. Remove or postpone `ontology` from v0 success language.
7. Define local optimum relative to a specific neighborhood and lambda.
8. Define "emergence" operationally as held-out, compressed, robust predictive
   usefulness, not just high score.
