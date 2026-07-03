# Theoretical Attack on the Projection Hypothesis

This note follows the prompt: do not protect the projection theory, do not
improve it incrementally, and try to replace it with a deeper mathematical
foundation if possible.

## Executive Verdict

Projection should probably not be treated as the fundamental primitive.

The deeper primitive is distinction under interaction.

A projection `Pi` is better understood as a derived object:

```text
interaction process
  -> observable histories
  -> indistinguishability relation
  -> quotient / projection
  -> stable quotient
  -> invariant quotient
  -> candidate concept
```

So `Pi` is not wrong, but it is late. It is the quotient induced by a more basic
notion of distinguishability. The projection lab should therefore study
distinctions and quotient formation, not projection evolution as if projection
were ontologically first.

The strongest replacement:

`The fundamental epistemic object is an equivalence relation on histories induced
by future interaction consequences.`

Projection is then just the quotient map of that equivalence relation.

## Part 1. Is Projection Fundamental?

Projection can be eliminated.

Given a set of histories `H`, define a relation:

```text
h1 ~ h2 iff h1 and h2 are indistinguishable for all relevant future tests.
```

The "relevant future tests" may be predictions, actions, interventions,
questions, rewards, or continuation distributions. Once `~` exists, projection
is automatic:

```text
Pi(h) = [h]_~
```

This means projection is not primitive. It is the canonical quotient map induced
by indistinguishability.

The important mathematical object is not:

```text
Pi: H -> R
```

but:

```text
~ subset H x H
```

or, even deeper:

```text
test family T that induces ~
```

because different test families create different epistemologies.

Examples:

1. Predictive tests induce predictive equivalence.
2. Control tests induce behavioral equivalence.
3. Causal tests induce interventional equivalence.
4. Communicative tests induce semantic/pragmatic equivalence.
5. Compression tests induce descriptional equivalence.

So the real primitive may be:

```text
(H, T)
```

where `H` is the space of histories and `T` is the family of admissible
distinctions/tests.

Projection survives only as a quotient representation:

```text
Pi = quotient(H, ~_T)
```

This is a serious demotion. It means the lab should not ask first "which
projection emerges?" It should ask "which distinctions are forced by the
interaction structure?"

## Part 2. Missing Objects

The current chain:

```text
World -> Policy -> History -> Projection -> Decision
```

is missing several objects. The smallest complete picture is:

```text
Environment dynamics E
  hidden state space S
  action space A
  observation space O
  transition kernel delta: S x A -> S
  observation operator omega: S -> O

Agent interface
  policy pi: H -> A
  history space H = (O x A)*
  test family T over histories
  indistinguishability relation ~_T on H
  quotient q: H -> H/~_T
  decision rule d: H/~_T -> A or answer

Evaluation
  loss / utility L
  complexity C
  invariance criterion I
```

The most important missing object is the observation operator.

Without it, the lab cannot distinguish true world state from available
information. If the agent observes the true state directly, "concept formation"
collapses into choosing a compression of labels. Partial observability is not an
extension; it is required for epistemology.

The second missing object is the test family. A concept cannot be defined by a
projection alone. It must be defined by what distinctions matter. Distinctions
matter only relative to tests.

The third missing object is a generator of histories. Exhaustive search over
partitions is not meaningful unless there is a distribution, language, grammar,
or process that says which histories are possible and how they continue.

The fourth missing object is a notion of invariance. Stability under local
operators is too weak. A candidate concept should survive at least one of:

1. relabeling of raw states;
2. held-out histories;
3. intervention;
4. change of initial condition;
5. change of policy;
6. change of observation encoding.

## Part 3. Challenge Every Assumption

### 1. Projection as Partition

Partitions are mathematically clean, but they may be fundamentally too narrow.

A partition enforces crisp equivalence. Many epistemic states are not crisp:

1. probabilistic beliefs;
2. graded similarity;
3. overlapping concepts;
4. relational structures;
5. hierarchical abstractions;
6. contextual meanings;
7. topological neighborhoods;
8. causal mechanisms.

A better hierarchy:

```text
partition < preorder < metric / pseudometric < topology < sigma-algebra
< category / functor < causal model
```

Partitions are useful as v0, but the theory should not identify projection with
partition. A partition is only one representation of distinction.

### 2. Concepts From Stable Projections

Counterexample:

World has two hidden regimes with identical short-run observations but opposite
long-run consequences.

```text
Regime A: x -> y -> z -> reward
Regime B: x -> y -> z -> failure
```

If the observation window is short, the projection `{x,y,z}` may be stable and
predictively good locally. It is not a concept; it hides the only distinction
that matters.

A stable projection can be:

1. an artifact of limited tests;
2. a compression of labels;
3. a degenerate optimum caused by high complexity penalty;
4. a local optimum under the wrong operator neighborhood;
5. an invariant of the data collection policy, not of the world.

Therefore:

```text
stable projection != concept
```

At minimum:

```text
concept = stable + invariant + test-relevant + transferable quotient
```

### 3. Prediction as Driver of Concept Formation

Concepts can emerge without next-state prediction.

Examples:

1. Control concept: two states are equivalent if the same actions are available
   or the same goals are reachable.
2. Causal concept: two events are equivalent if interventions on them have the
   same effect.
3. Communicative concept: two signals are equivalent if they license the same
   responses in a language game.
4. Compression concept: two structures are equivalent if they share the same
   minimal program.
5. Normative concept: two cases are equivalent if they require the same rule.

Prediction is one pressure, not the foundation. The more general primitive is
future consequence under a test family.

### 4. Projection Evolves

Projection may not be what evolves.

The evolving object could be:

1. the policy that samples histories;
2. the test family that defines relevant distinctions;
3. the observation operator or attention function;
4. the loss function;
5. the hypothesis language;
6. the agent's computational budget;
7. the environment-agent coupled dynamical system.

If policy evolves, projections appear to evolve only because the sampled history
space changes. If attention evolves, projection changes because the observation
channel changes. If tests evolve, the quotient changes even when the world and
history are fixed.

So "self-evolving projection" may be a surface description of a deeper coupled
dynamics:

```text
(policy, observation, tests, quotient) co-evolve.
```

### 5. History as Domain

History is a good v0 domain, but it may not be the correct primitive domain.

Candidate domains:

1. trajectories: if continuous time or long-horizon structure matters;
2. interaction traces: if actions and observations are inseparable;
3. causal graphs: if interventions define meaning;
4. programs: if concepts are computational generators;
5. coalgebraic states: if behavior is defined by possible futures;
6. sheaves: if local observations must glue into global structure;
7. categories of experiments: if morphisms between tests matter.

The strongest generalization:

```text
domain = interaction trace category
```

Histories are then objects or paths inside a richer structure, not the whole
foundation.

## Part 4. Stronger Mathematics

### Lattice Theory

Partitions form a lattice under refinement.

Gained:

1. clean ordering from coarse to fine projections;
2. meet and join operations;
3. formal analysis of split and merge;
4. local/global optima over a finite lattice.

Simpler:

`Split` and `Merge` become movement in the partition lattice.

Provable:

1. monotonicity of prediction under refinement under some scoring rules;
2. tradeoff curves between accuracy and complexity;
3. existence of optimal partitions for finite worlds.

### Order Theory

More general than partition lattices. Distinctions can be ordered by
informativeness.

Gained:

1. abstraction levels as partial orders;
2. monotone operators;
3. fixed points by Tarski-style arguments.

Simpler:

Projection evolution becomes iteration of monotone refinement/coarsening
operators.

Provable:

1. existence of fixed abstractions;
2. convergence under monotone dynamics;
3. conditions where no nontrivial abstraction exists.

### Universal Algebra

If a world has operations, concepts may be congruences: equivalence relations
preserved by operations.

Gained:

1. a stronger notion than arbitrary partition;
2. "valid concepts" as operation-preserving quotients;
3. algebraic invariance.

Simpler:

False partitions are rejected if they do not respect world operations.

Provable:

1. quotient structures exist exactly for congruences;
2. concepts compose through homomorphisms;
3. stable abstractions preserve dynamics.

### Category Theory

Projection becomes a quotient, functor, or adjunction between concrete and
abstract systems.

Gained:

1. abstraction as structure-preserving map;
2. invariance under isomorphism;
3. compositionality;
4. separation between representation and represented process.

Simpler:

Instead of asking "what is a concept?", ask which functors preserve behavior
relevant to a test family.

Provable:

1. when abstraction preserves transition structure;
2. when two abstraction paths commute;
3. universal properties of minimal sufficient abstractions.

### Dynamical Systems

The environment-agent loop is a dynamical system. Projections are factors of
that system.

Gained:

1. attractors;
2. basins;
3. stability;
4. invariant sets;
5. factor maps.

Simpler:

"Stable projection" becomes "factor system preserving relevant dynamics."

Provable:

1. when a quotient dynamics is well-defined;
2. when an abstraction is Markovian;
3. when two histories have identical futures.

### Fixed Point Theory

A stable epistemic structure can be modeled as a fixed point of an update
operator.

Gained:

1. precise stability;
2. convergence conditions;
3. separation between transient and stable structures.

Simpler:

Self-evolving projection becomes:

```text
Pi* = F(Pi*)
```

Provable:

1. existence and uniqueness under contraction-like conditions;
2. multiplicity of fixed points;
3. dependence on lambda or test family.

### Information Geometry

Belief states live on probability manifolds. Projections may be information
preserving maps or coarse-grainings.

Gained:

1. graded uncertainty;
2. information loss;
3. sufficient statistics;
4. divergence measures.

Simpler:

Concepts become sufficient compressed coordinates for prediction/control.

Provable:

1. minimal sufficient statistics;
2. information bottleneck tradeoffs;
3. optimal compression under KL-like objectives.

### Computational Mechanics

This may be the closest existing mathematical home.

Computational mechanics defines causal states as equivalence classes of pasts
that induce the same distribution over futures:

```text
past1 ~ past2 iff P(future | past1) = P(future | past2)
```

Gained:

1. projection is derived from predictive equivalence;
2. concepts become causal states;
3. minimality is formal;
4. stochastic and deterministic systems both fit.

Simpler:

The first theorem almost writes itself: causal states are the minimal sufficient
predictive representation.

Provable:

1. causal states are unique up to isomorphism;
2. they are minimal sufficient statistics of history for future prediction;
3. any other predictive representation refines or maps through them.

This is the strongest candidate foundation if prediction remains central.

### Coalgebras

Coalgebras model systems by their observable behavior over time.

Gained:

1. behavior-first definition of state;
2. bisimulation as indistinguishability;
3. quotient by behavioral equivalence.

Simpler:

Projection becomes quotient by bisimulation.

Provable:

1. final coalgebra semantics;
2. behavioral equivalence;
3. minimal automata-like reductions.

This is a strong candidate if interaction and observation are more fundamental
than prediction.

### Sheaf Theory

Useful if concepts are assembled from local observations that may or may not
glue into a global structure.

Gained:

1. local-to-global consistency;
2. context dependence;
3. obstruction detection.

Simpler:

Contradictory partial concepts can be formalized as failed gluing.

Provable:

1. when local projections compose into a global projection;
2. when no coherent global ontology exists;
3. where inconsistencies live.

Probably too heavy for v0, but important later.

## Part 5. Replace the Projection Problem

The Projection Problem:

```text
How does a projection emerge?
```

is probably not fundamental.

A deeper problem:

```text
How does an interaction system induce distinctions that are stable,
invariant and useful under future tests?
```

Even shorter:

```text
How does distinguishability emerge?
```

The proposed replacement:

## The Distinction Problem

Given:

```text
history space H
test family T
cost functional C
evaluation loss L
```

find the coarsest equivalence relation `~` on `H` such that histories equivalent
under `~` cannot be distinguished by tests in `T` beyond tolerance epsilon, while
the quotient remains cheaper than raw history.

Projection becomes a consequence:

```text
Pi(h) = [h]_~
```

Concept becomes a stronger consequence:

```text
concept = equivalence class that remains stable under changes in T,
distribution, encoding or intervention.
```

This is deeper because it explains why projection exists at all.

## Part 6. First Theorem Worth Proving

If starting from zero, the first theorem should be:

## Minimal Sufficient Quotient Theorem

Let `H` be a finite set of histories and `F` a finite set of future tests. Each
history `h in H` induces a response vector:

```text
r(h) = (response_f(h)) for f in F
```

Define:

```text
h1 ~ h2 iff r(h1) = r(h2)
```

Then:

1. `~` is an equivalence relation.
2. The quotient map `q: H -> H/~` is sufficient for all tests in `F`.
3. `q` is the coarsest sufficient projection: any projection that is sufficient
   for all tests in `F` must refine `q` or factor through it.
4. Therefore `q` is unique up to relabeling of quotient classes.

Why this theorem matters:

1. It derives projection from tests instead of assuming projection.
2. It defines the minimal representation without using the word "concept".
3. It gives a falsifiable baseline for the lab.
4. It separates arbitrary compression from epistemic compression.
5. It says exactly when a projection is not a discovery: when it is not the
   coarsest sufficient quotient for the chosen tests.

This theorem is the foundation because it gives the first non-mystical bridge:

```text
interaction consequences -> distinctions -> quotient -> projection
```

Only after this theorem should the lab ask whether some quotient classes deserve
to be called concepts.

## Final Position

Projection fails as the deepest primitive.

It survives as an important derived object: the quotient map induced by a
distinction relation.

The corrected theoretical stack should be:

```text
World / environment dynamics
  -> observation and interaction channel
  -> history / trace space
  -> test family over possible futures
  -> indistinguishability relation
  -> minimal sufficient quotient
  -> projection map
  -> stable/invariant quotient
  -> candidate concept
```

The Epistemic Laboratory should therefore be renamed conceptually from
"projection dynamics lab" to:

```text
laboratory for distinction, quotient and invariance dynamics
```

Implementation can still use projections, partitions and exhaustive search. But
the theory should not say projection is fundamental. The theory should say:

```text
Distinction is fundamental.
Projection is the quotient shadow of distinction.
Concept is a robust invariant quotient class under meaningful tests.
```
