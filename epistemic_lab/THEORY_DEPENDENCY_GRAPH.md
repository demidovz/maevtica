# Chapter 1: Dependency Graph of the Theory

This document fixes the current foundation temporarily. It does not search for
deeper primitives and does not introduce new theory unless needed to remove
ambiguity.

The current working program:

```text
interaction world
  -> interaction traces
  -> constructible tests
  -> test families
  -> distinctions
  -> equivalence relations
  -> quotient structures
  -> robust quotient candidates
```

The word `concept` is not yet a theorem. At this stage it is at most a target
interpretation for robust quotient candidates.

## Part 1. Definitions

### D1. Possibility Space

Status: primitive in the background foundation.

Definition: a finite set `S` of possible true states.

Dependencies: none inside the finite theory.

### D2. Constraint / Dynamics

Status: primitive relative to the experiment.

Definition: a deterministic transition map:

```text
delta: S -> S
```

or with actions:

```text
delta: S x A -> S
```

Dependencies: `S`.

### D3. Observation Interface

Status: primitive relative to the experiment.

Definition:

```text
omega: S -> O
```

where `O` is a finite observation set.

Dependencies: `S`.

### D4. Interaction World

Status: primitive package for the current theory.

Definition:

```text
W = (S, delta, omega)
```

or with actions:

```text
W = (S, A, delta, omega)
```

Dependencies: `S`, `delta`, `omega`, optionally `A`.

### D5. Interaction Policy

Status: derived/admissible input.

Definition:

```text
pi: O* -> A
```

Dependencies: action set `A`, observation set `O`.

### D6. Interaction Trace

Status: derived.

Definition: for world `W`, policy `pi`, horizon `n`, and initial state `s`,

```text
trace_n^pi(s) = (o0, a0, o1, a1, ..., on)
```

where:

```text
o_i = omega(s_i)
a_i = pi(o0, ..., oi)
s_{i+1} = delta(s_i, a_i)
```

In the no-action case:

```text
trace_n(s) = (omega(s), omega(delta(s)), ..., omega(delta^n(s)))
```

Dependencies: `W`, `pi`, `n`.

### D7. Test

Status: derived if constructible; otherwise external.

Definition:

```text
t: S -> R_t
```

where `R_t` is a finite result set.

Dependencies: `S`; if admissible/constructible, also `W`, traces, and a
label-independent finite recoding.

### D8. Constructible Test

Status: derived.

Definition:

```text
t(s) = f(trace_n^pi(s))
```

where `f` is independent of state labels and `pi` is independent of the
candidate quotient.

No-action version:

```text
t(s) = f(omega(s), omega(delta(s)), ..., omega(delta^n(s)))
```

Dependencies: `W`, `pi`, `n`, `f`.

### D9. Test Family

Status: derived or external.

Definition: a finite set or tuple of tests:

```text
T = {t1, ..., tk}
```

Dependencies: tests.

### D10. Admissible Generator

Status: derived only after `W` and constructibility rules are fixed.

Definition: a quotient-independent rule `g` that produces a finite test family
from the interaction world:

```text
g(W) = T_g
```

and is equivariant under isomorphisms of `W`.

Dependencies: `W`, constructible tests, allowed policies, allowed recodings.

### D11. Outcome Signature

Status: derived.

Definition:

```text
r_T(s) = (t(s))_{t in T}
```

Dependencies: `S`, `T`.

### D12. Distinction

Status: derived.

Definition:

```text
D_T(s, s') = 1 iff exists t in T such that t(s) != t(s')
```

Dependencies: `T`, outcome signatures.

### D13. Test-Induced Equivalence Relation

Status: derived.

Definition:

```text
s ~_T s' iff for every t in T, t(s) = t(s')
```

Dependencies: `T`, distinctions.

### D14. Quotient Set / Quotient Partition

Status: derived.

Definition:

```text
Q_T = S / ~_T
```

Dependencies: `S`, `~_T`.

### D15. Quotient Dynamics

Status: derived when well-defined.

Definition:

```text
delta_Q([s]) = [delta(s)]
```

Well-defined iff:

```text
s ~ s' implies delta(s) ~ delta(s')
```

Dependencies: quotient `Q`, dynamics `delta`.

### D16. Dynamical Validity

Status: derived predicate.

Definition: `Q` is dynamically valid iff quotient dynamics is well-defined.

Dependencies: `Q`, `delta`.

### D17. Sufficiency

Status: derived predicate.

Definition: `Q` is sufficient for `T` iff each test in `T` is constant on every
block of `Q`.

Dependencies: `Q`, `T`.

### D18. Minimal Sufficiency

Status: derived predicate.

Definition: `Q` is minimally sufficient for `T` iff `Q` is sufficient for `T`
and no strictly coarser quotient is sufficient for `T`.

Dependencies: `Q`, `T`, refinement order on partitions.

### D19. Refinement Order

Status: derived.

Definition:

```text
Q1 <= Q2
```

means `Q1` refines `Q2`: every block of `Q1` is contained in a block of `Q2`.

Dependencies: quotient partitions.

### D20. Stability Under Test Change

Status: derived predicate; not enough for concept status.

Definition: under `T -> T+`, `Q_T` is stable iff it remains sufficient for `T+`
and dynamically valid.

Dependencies: `Q_T`, `T+`, `delta`.

### D21. Split

Status: derived event.

Definition: a block `B in Q_T` must split under `T+` iff:

```text
exists s, s' in B and exists t in T+ such that t(s) != t(s')
```

Dependencies: `Q_T`, `T+`.

### D22. Merge

Status: derived event.

Definition: blocks merge under weakening `T -> T-` when they have the same
outcomes for all tests in `T-` and the merged quotient is dynamically valid.

Dependencies: `Q_T`, `T-`, `delta`.

### D23. Automorphism of a World

Status: derived.

Definition: a bijection `g: S -> S` preserving the world structure:

```text
omega(g(s)) = omega(s)
g(delta(s)) = delta(g(s))
```

with the obvious action-aware generalization.

Dependencies: `W`.

### D24. Equivariant / Natural Generator

Status: derived predicate.

Definition:

```text
g(h.W) = h.g(W)
```

for every isomorphism `h` of interaction worlds.

Dependencies: world isomorphisms, generator `g`.

### D25. Robust Quotient Candidate

Status: derived predicate, not yet a concept theorem.

Definition: a quotient `Q` is a robust quotient candidate iff it is recoverable
as a canonical common factor of quotients induced by quotient-independent
admissible generators.

Dependencies: admissible generators, quotient factors, canonical factor maps.

### D26. Concept

Status: not defined as a theorem yet.

Working restraint: do not identify `concept` with stable quotient. At most:

```text
concept may later be defined as a robust quotient candidate satisfying further
semantic, causal, or transfer criteria.
```

Dependencies: intentionally unresolved.

## Part 2. Dependency DAG

Minimal DAG:

```text
D1 Possibility Space
  -> D2 Dynamics
  -> D4 Interaction World

D1 Possibility Space
  -> D3 Observation Interface
  -> D4 Interaction World

D4 Interaction World
  -> D23 Automorphisms
  -> D24 Equivariant Generator

D4 Interaction World
  -> D5 Policy
  -> D6 Interaction Trace
  -> D8 Constructible Test
  -> D10 Admissible Generator
  -> D9 Test Family

D7 Test
  -> D9 Test Family
  -> D11 Outcome Signature
  -> D12 Distinction
  -> D13 Equivalence Relation
  -> D14 Quotient

D14 Quotient
  -> D15 Quotient Dynamics
  -> D16 Dynamical Validity

D14 Quotient
  -> D17 Sufficiency
  -> D18 Minimal Sufficiency

D14 Quotient
  -> D19 Refinement Order
  -> D20 Stability
  -> D21 Split
  -> D22 Merge

D10 Admissible Generator
  -> D25 Robust Quotient Candidate

D14 Quotient
  -> D25 Robust Quotient Candidate

D25 Robust Quotient Candidate
  -> D26 Concept (open)
```

Results DAG:

```text
D7-D14
  -> L1 Test-Induced Quotient Lemma

D9 nested families + D19
  -> L2 Refinement Monotonicity

D14 + arbitrary block-labeling tests
  -> I1 Quotient-Only Impossibility

I1 + first experiment
  -> R1 Stable Quotient != Concept

D4 + D23
  -> I2 Bare-State Impossibility

D4 + D6 + D8 + D24
  -> P1 Interaction-Derived Admissibility

I1 + P1
  -> C1 Robust Quotient Candidate Program
```

## Part 3. Known Results

### L1. Test-Induced Equivalence Lemma

Statement: every deterministic finite test family `T` induces an equivalence
relation `~_T` by equality of outcome signatures.

Assumptions: finite `S`, deterministic tests.

Proof sketch: equality of tuples is reflexive, symmetric and transitive.

Confidence: 100%.

### L2. Test-Induced Quotient Lemma

Statement: every test family induces a canonical quotient `S / ~_T`.

Assumptions: L1.

Proof sketch: quotient by an equivalence relation exists.

Confidence: 100%.

### L3. Nested Refinement Lemma

Statement: if `T subset T+`, then `Q_T+` refines `Q_T`.

Assumptions: deterministic tests, set inclusion of test families.

Proof sketch: equality on more coordinates implies equality on fewer
coordinates.

Confidence: 100%.

### L4. Minimal Sufficient Quotient Lemma

Statement: for finite `S` and finite `T`, `Q_T` is the coarsest quotient
sufficient for preserving all outcomes of `T`.

Assumptions: deterministic tests.

Proof sketch: any sufficient quotient cannot place states with different
outcome signatures in the same block; therefore it refines `Q_T`.

Confidence: 100%.

### L5. Quotient Dynamics Validity Criterion

Statement: quotient dynamics is well-defined iff:

```text
s ~ s' implies delta(s) ~ delta(s')
```

Assumptions: deterministic dynamics.

Proof sketch: the image of a block must not depend on representative.

Confidence: 100%.

### I1. Quotient-Only Impossibility Lemma

Statement: no criterion depending only on a quotient and its minimal sufficiency
for some test family can exclude arbitrary injected quotients.

Assumptions: arbitrary tests are allowed.

Proof sketch: for any partition `P`, define `t_P(s)=block_id(P,s)`. Then
`Q_{t_P}=P`.

Confidence: 100%.

### R1. Stable Minimally Sufficient Quotient Is Not Concept

Statement: stability and minimal sufficiency are not enough for concept status.

Assumptions: concept status should exclude arbitrary direct test injection.

Proof sketch: the parity test in the first experiment injects a stable
minimally sufficient quotient.

Confidence: 95%. The mathematical counterexample is exact; the only dependency
is the normative assumption that direct arbitrary injection should not count as
concept emergence.

### I2. Bare-State Impossibility

Statement: no nontrivial admissible generator class can be derived from a bare
finite set `S` without extra structure or arbitrary label injection.

Assumptions: invariance under bijections of bare finite sets.

Proof sketch: the automorphism group is `Sym(S)`; no nontrivial proper
partition is canonically selected by a fully symmetric set.

Confidence: 95%. Needs formalization of "reasonable" as functorial/invariant.

### P1. Interaction-Derived Admissibility

Statement: finite observable trace tests are admissible in the minimal derived
sense from an interaction world.

Assumptions: world `W=(S,A,delta,omega)`, quotient-independent policies,
label-independent recodings, equivariance under world isomorphisms.

Proof sketch: trace tests use only observable consequences of interaction and
commute with isomorphisms of `W`.

Confidence: 85%. The result is solid once the allowed policy and recoding
classes are specified.

### R2. Parity Is Not Derivable From Passive First World

Statement: in the first world `(S,delta,omega)`, parity is not constructible
from finite observable traces.

Assumptions: no extra mode-sensitive instrument; tests must be functions of
finite observable futures.

Proof sketch: states with identical finite observable futures cannot be
distinguished by any trace-constructible test. Parity distinguishes such states.

Confidence: 95%.

## Part 4. Open Conjectures

### C1. Robust Quotient Candidate Adequacy

Statement: admissible independent recoverability is the weakest condition that
eliminates direct injected quotients without excluding natural trace-derived
quotients.

Why it matters: this is the current best replacement for `stable quotient =>
concept`.

How falsified: find a direct injected quotient that passes admissible
independent recoverability, or find a natural trace-derived quotient that fails
it for non-technical reasons.

Experiment: define two or more admissible trace generators and compare hidden
mode, observation phase and parity quotients.

### C2. Constructible Trace Quotients Are Exactly Behavioral Equivalences

Statement: for finite deterministic interaction worlds, equality of all finite
observable traces coincides with behavioral equivalence/bisimulation.

Why it matters: it connects this theory to automata theory and coalgebra.

How falsified: construct states with equal finite trace tests but different
behavioral status under the chosen notion of behavior.

Experiment: exhaustive enumeration of tiny deterministic worlds with actions.

### C3. Non-Arbitrariness Requires Provenance

Statement: every successful non-arbitrariness criterion must depend on
test provenance, world structure or allowed transformations; quotient-internal
criteria are impossible.

Why it matters: prevents wasted effort on better quotient-only scores.

How falsified: give a quotient-internal property that excludes all direct
injections while preserving all natural quotients.

Experiment: not computational first; prove or refute abstractly.

### C4. Concept Requires More Than Robust Quotient Candidate

Statement: robust quotient candidate is still not enough for concept status;
additional criteria such as transfer, intervention, utility, or semantic role
may be required.

Why it matters: prevents premature relabeling of mathematical quotients as
concepts.

How falsified: prove that robust quotient candidates satisfy all desired
conceptual roles in a clearly specified finite setting.

Experiment: after C1-C2, add controlled task transfer or intervention tests.

### C5. Minimal Instrument Extension

Statement: hidden structures become admissibly discoverable exactly when the
interaction interface is extended by an instrument whose trace algebra separates
the hidden equivalence classes.

Why it matters: it formalizes when adding a test is discovery versus injection.

How falsified: find a hidden quotient separated by an instrument but not by the
generated trace algebra, or generated without any separating interface.

Experiment: add `mu` mode-sensitive instrument to the first world and compare
derived admissible quotients.

## Part 5. Minimal Research Program

### Theorem 1. Finite Test Quotient Theorem

Prove: finite deterministic test families induce unique coarsest sufficient
quotients.

Depends on: D7-D18.

Purpose: foundation of the quotient calculus.

### Theorem 2. Quotient-Only Impossibility Theorem

Prove: no quotient-only criterion can eliminate arbitrary injection when
arbitrary tests are allowed.

Depends on: Theorem 1.

Purpose: forces the theory to account for test provenance.

### Theorem 3. Bare-State No-Free-Admissibility Theorem

Prove: no nontrivial admissible generator can be derived from a bare finite set
under bijection invariance.

Depends on: Theorem 2 plus invariance under `Sym(S)`.

Purpose: shows that admissibility requires world/interface structure.

### Theorem 4. Interaction Trace Admissibility Theorem

Prove: finite observable trace tests are equivariant constructible tests of an
interaction world, and cannot distinguish trace-equivalent states.

Depends on: D4-D10 and Theorem 3.

Purpose: provides the first non-arbitrary admissible generator class.

### Theorem 5. Robust Candidate Separation Theorem

Prove in the first four-state world: observation-phase quotient is
trace-constructible, parity is not trace-constructible, and hidden mode becomes
constructible iff the interface is extended by a mode-sensitive instrument.

Depends on: Theorem 4.

Purpose: gives the first clean mathematical separation between natural,
injected and instrument-dependent quotients.

## Part 6. Critical Review

### Circularity 1. "Admissible" Can Smuggle the Answer

Risk: define admissible generators to exclude parity because parity is unwanted.

Repair: define admissibility before candidate quotients, using only `W`,
constructibility and equivariance.

### Circularity 2. "Concept" Is Used Before It Is Defined

Risk: call robust quotient candidates concepts and then use that as evidence
that concepts emerged.

Repair: reserve `concept` as an open target term. Prove results only about
quotients and robust quotient candidates.

### Circularity 3. "Natural" Means "What We Wanted"

Risk: natural concepts are those that pass our test, and the test is justified
because it keeps natural concepts.

Repair: replace "natural" with formal properties: trace-constructible,
equivariant, dynamically valid, transferable, intervention-robust.

### Circularity 4. Hidden Mode Is Treated as Real but Unobservable

Risk: declaring hidden mode a genuine concept while the interface cannot
distinguish it.

Repair: hidden mode is not admissibly discoverable until the interface contains
mode-sensitive consequences.

### Circularity 5. Stability Is Interpreted as Significance

Risk: stable fixed points are treated as meaningful.

Repair: stability is only a dynamical predicate. It must be combined with
provenance/constructibility.

### Hidden Assumption 1. Finite Determinism

The current theory assumes finite deterministic systems. Stochastic systems may
require probability kernels, sigma-algebras and statistical sufficiency.

### Hidden Assumption 2. Tests Are Total Functions

Partial, noisy, costly or destructive tests are excluded. This is acceptable for
chapter one but not for the full theory.

### Hidden Assumption 3. Observation Interface Is Given

The theory currently does not derive `omega`. It treats interface as primitive.

### Hidden Assumption 4. Policy Class Is Given

In action worlds, admissible traces depend on allowed policies. A bad policy
class can hide or inject distinctions.

### Hidden Assumption 5. Canonical Common Factor Is Not Fully Formalized

Robust quotient candidate depends on canonical factor maps. This needs a formal
definition, probably via lattice meet/join or category-theoretic factorization.

### Hidden Assumption 6. Recoding Independence

The phrase "label-independent recoding" must be formalized as invariance under
renaming of states and outcomes.

### Theorem With Suspicious Assumption

`Independent Recoverability Theorem` assumes independence of generators. This is
not yet mathematically defined. Without a definition, it can smuggle in the
desired conclusion.

Repair: define independence as separate construction from `W` through distinct
allowed trace policies or instruments, not as non-collusion by assertion.

## Monograph Table of Contents

### Chapter 1. Finite Interaction Worlds

Definitions of `S`, `delta`, `omega`, traces and constructible tests.

### Chapter 2. Test-Induced Quotients

Equivalence relations, quotient partitions, sufficiency, minimal sufficiency
and refinement monotonicity.

### Chapter 3. Failure of Quotient-Only Concept Criteria

Quotient-only impossibility, parity counterexample and stable quotient
falsification.

### Chapter 4. Admissibility and Provenance

Bare-state impossibility, interaction-derived admissibility and equivariance.

### Chapter 5. Robust Quotient Candidates

Independent recoverability, canonical common factors and separation examples.

### Chapter 6. Toward Concepts

Only after robust quotient candidates are formalized: transfer, intervention,
semantic role and possible definition of concept.
