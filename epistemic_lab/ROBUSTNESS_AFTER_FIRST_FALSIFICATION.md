# Robustness After the First Falsification

The first experiment falsified:

```text
stable minimally sufficient quotient => concept
```

The counterexample was an arbitrary `parity` test. It induced a stable
minimally sufficient quotient, but the quotient was injected by the test family
rather than discovered as a genuine structure.

This document asks for the smallest additional mathematical property needed
after that falsification.

## 0. Starting Point

Let:

```text
S = finite state space
delta: S -> S = deterministic dynamics
T = finite test family
Q_T = S / ~_T = test-induced quotient
```

where:

```text
s ~_T s' iff for every t in T, t(s) = t(s')
```

The failed criterion was:

```text
Q_T is stable + minimally sufficient
```

The failure teaches an important negative lesson:

```text
No property of Q_T alone can distinguish a genuine concept from an injected
quotient.
```

Reason: an arbitrary test can be constructed whose kernel is exactly any chosen
partition. Therefore every partition can be made minimally sufficient and stable
relative to a custom test.

### Quotient-Only Impossibility Lemma

Let `P` be any partition of finite `S`. Define a test:

```text
t_P(s) = block_id of the unique block B in P containing s
```

Then:

```text
Q_{ {t_P} } = P
```

and `P` is minimally sufficient for `{t_P}`.

Therefore any criterion that examines only:

```text
P itself
```

or only:

```text
P plus the fact that it is minimally sufficient for some test family
```

cannot rule out arbitrary injection. The missing condition must constrain the
test family, its origin, or its behavior under transformations that were fixed
before the candidate quotient was known.

So the missing property cannot be only about the quotient. It must mention at
least one of:

1. the admissible source of tests;
2. transformations under which tests are considered equivalent;
3. perturbations of the test family;
4. dynamics/interactions independent of the injected test;
5. independent recoverability from multiple test sources.

## Part 1. Plausible Robustness Notions

### 1. Equivalent-Test-Family Invariance

Definition:

Two test families `T` and `T'` are outcome-equivalent if they induce the same
equivalence relation:

```text
~_T = ~_T'
```

A quotient is invariant under equivalent test families if:

```text
Q_T = Q_T'
```

Assessment:

This is too weak. It is almost tautological because equal kernels imply equal
quotients.

Counterexample:

`parity` and any relabeled version of `parity` are outcome-equivalent and induce
the same injected quotient.

First experiment:

It would not distinguish genuine from injected quotients.

### 2. Stability Under Small Test Perturbation

Definition:

Let `d(T, T')` be a distance between test families, for example the number of
state-test outputs changed. A quotient `Q_T` is epsilon-stable if:

```text
d(T, T') <= epsilon implies Q_T and Q_T' are isomorphic or have a common factor.
```

Assessment:

Stronger than equivalent-test-family invariance. Still dangerous because
arbitrary tests can be locally stable if the perturbation does not cross a
kernel-changing boundary.

Counterexample:

If `parity` outputs are encoded as real values `0.1` and `0.9`, small numeric
perturbations preserve the red/blue split.

First experiment:

The current first experiment does not implement graded perturbations, so it
would not distinguish this notion.

### 3. Persistence Under Refinement

Definition:

For `T subset T+`, quotient `Q_T` persists under refinement if every block of
`Q_T` is a union of blocks of `Q_T+`.

Equivalently:

```text
Q_T+ refines Q_T
```

Assessment:

This is guaranteed for nested deterministic test families. It is not a concept
criterion; it is a theorem about adding tests.

Counterexample:

The observation quotient persists when `parity` is added: its blocks split into
singletons. That persistence does not make the injected refinement conceptual.

First experiment:

Yes, it distinguishes refinement behavior, but it does not eliminate parity.

### 4. Persistence Under Coarsening

Definition:

For `T- subset T`, quotient `Q_T` persists under coarsening if `Q_T` remains
sufficient for `T-`.

Assessment:

Too strong if equality is required; too weak if factor survival is enough.

Counterexample:

The identity quotient induced by `obs + parity` remains sufficient after
removing `parity`, because identity preserves all tests. It survives but is
still not a concept.

First experiment:

It partially distinguishes disappearance of specific splits, but not injected
identity-like quotients.

### 5. Recoverability After Test Removal

Definition:

A quotient `Q` is recoverable from a reduced family `T-` if `Q` can be computed
as a canonical function of `Q_T-`, `delta`, and admissible operations.

Assessment:

Potentially useful, but only after "canonical" and "admissible operations" are
defined. Otherwise it can smuggle in the desired quotient.

Counterexample:

Given any target partition, define an "admissible operation" that reconstructs
it. Recoverability becomes vacuous.

First experiment:

Not yet. The first experiment reports valid coarser quotients, but does not
select a canonical one.

### 6. Universality

Definition:

A quotient `Q` is universal for a property `P` if every representation that
preserves `P` factors through `Q`.

Example:

Minimal sufficient quotient for a test family is universal for preserving that
test family's outcomes.

Assessment:

Mathematically strong, but target-relative. It can certify the wrong thing if
`P` is "preserve arbitrary parity labels".

Counterexample:

The parity quotient is universal for preserving parity outcomes.

First experiment:

Yes. It already shows that universality relative to the wrong property does not
imply conceptual significance.

### 7. Minimal Fixed Point

Definition:

Let `F` be an operator on quotients, for example:

```text
F(Q) = minimally sufficient valid refinement of Q under T and delta
```

`Q` is a minimal fixed point if:

```text
F(Q) = Q
```

and no strictly coarser quotient is also fixed.

Assessment:

Stronger than stability if `F` is meaningful. But arbitrary tests can define
operators whose fixed points are arbitrary injected partitions.

Counterexample:

Let `F` refine every partition until it preserves `parity`. The parity quotient
is a minimal fixed point.

First experiment:

Not directly. A second experiment would need explicit operators.

### 8. Dynamical Validity

Definition:

`Q` is dynamically valid if the quotient transition is well-defined:

```text
s ~_Q s' implies delta(s) ~_Q delta(s')
```

Assessment:

Necessary for quotient dynamics, but not sufficient for concepthood.

Counterexample:

In the four-state world, the parity partition:

```text
{{A0, B1}, {A1, B0}}
```

is dynamically valid. It alternates under `delta`.

First experiment:

Yes. It already shows many dynamically valid quotients, including suspicious
ones.

### 9. Temporal Invariance / Conserved Quotient

Definition:

`Q` is temporally invariant if every block is preserved by dynamics:

```text
delta(B) = B for every block B in Q
```

Assessment:

Stronger than dynamical validity. It eliminates phase-like and parity-like
alternating quotients.

Counterexample:

Observation phase can be a legitimate concept in many worlds, but it fails this
criterion because it changes over time.

First experiment:

Yes. It would select the hidden-mode quotient and reject parity. But it is too
strong as a general concept criterion.

### 10. Automorphism Invariance

Definition:

Let `Aut(B)` be the automorphism group of a base experimental structure `B`.
A quotient `Q` is automorphism-invariant if:

```text
s ~_Q s' iff g(s) ~_Q g(s') for every g in Aut(B)
```

Assessment:

Promising, but depends critically on what belongs to `B`. If `B` includes the
arbitrary parity test, parity becomes invariant by construction.

Counterexample:

Put `parity` into the base structure. Then any automorphism preserving `B` must
preserve parity, so the injected quotient passes.

First experiment:

It can distinguish parity only if the base structure excludes arbitrary tests
from `B` and treats them as candidate instruments.

### 11. Categorical Invariance / Naturality

Definition:

Let experimental presentations form a category `C`. A quotient assignment `Q`
is natural if for every morphism `f: X -> Y`, the following diagram commutes:

```text
X        ->        Y
|                  |
q_X                q_Y
v                  v
Q(X)     ->        Q(Y)
```

Assessment:

Very strong and mathematically clean. It prevents representation-specific
artifacts if the category is correctly chosen.

Counterexample:

Choose a category whose morphisms preserve parity. Then parity is natural.

First experiment:

No. The current experiment has no category of presentations.

### 12. Information-Theoretic Invariance

Definition:

`Q` is information-theoretically robust if it preserves a target information
quantity across admissible distributions, for example:

```text
I(Q(S); F) is maximal or stable over distributions P in P_adm
```

Assessment:

Useful when stochastic distributions matter. Too much machinery for the first
deterministic experiment.

Counterexample:

If the target `F` is parity, the parity quotient is information-optimal.

First experiment:

No. There are no distributions except implicit exhaustive enumeration.

### 13. Intervention Robustness

Definition:

Given a set of interventions `I`, `Q` is intervention-robust if it remains valid
and sufficient under each intervened dynamics:

```text
delta_i for i in I
```

Assessment:

Strong. Good for causal concepts. Too strong for concepts that are not causal
or where interventions are unavailable.

Counterexample:

If interventions are chosen to preserve parity, parity survives. If they break
mode but preserve parity, mode fails.

First experiment:

No. It has no interventions.

### 14. Independent Recoverability

Definition:

Let `G1, ..., Gk` be independent admissible sources of tests. A quotient `Q` is
independently recoverable if it appears as a common factor of quotients induced
by at least two independent sources:

```text
Q <= Q_{G_i} for at least two independent i
```

where `<=` means "is coarser than or equal to", i.e. recoverable as a factor.

Assessment:

This directly targets injected tests. A one-off arbitrary test should not count
unless another independent route recovers the same quotient.

Counterexample:

Two arbitrary tests can collude to encode the same injected quotient. So
independence must be external, not merely syntactic.

First experiment:

No. It has only hand-specified tests and no independent test generators.

### 15. Admissible-Test Invariance

Definition:

Let `A` be a specified class of admissible test-generating procedures that do
not depend on the candidate quotient. A quotient `Q` is admissibly robust if:

1. `Q` is induced by some `T in A`;
2. `Q` is stable under allowed transformations of `A`;
3. `Q` is not induced only by tests that directly encode `Q`.

Assessment:

This is the smallest notion that attacks the parity failure at its source:
test provenance.

Counterexample:

If `A` is chosen too broadly, parity is admitted. If `A` is chosen too narrowly,
real concepts are excluded.

First experiment:

Only partially. The first experiment labels `parity` arbitrary, but does not yet
formalize `A`.

## Part 2. Relative Strengths

The notions form rough layers:

```text
equivalent-test invariance
  < refinement persistence
  < dynamical validity
  < perturbation stability
  < temporal invariance
  < automorphism invariance
  < naturality
```

But this is not a total order.

Independent recoverability and admissible-test invariance are orthogonal to
dynamical validity. A quotient can be dynamically valid but inadmissibly
injected. A quotient can be admissibly generated but dynamically invalid.

Information-theoretic invariance is also orthogonal: it depends on a target
variable and distribution family.

Intervention robustness implies a strong form of dynamical validity across
multiple dynamics, but it does not imply automorphism invariance.

Temporal invariance implies dynamical validity, but not conversely:

```text
delta(B) = B for each block B
  => quotient transition is well-defined
```

The converse fails:

```text
{{A0, B1}, {A1, B0}}
```

has a well-defined alternating quotient transition but its blocks are not fixed.

Minimal sufficiency is independent of dynamical validity:

1. a minimally sufficient quotient can fail quotient dynamics;
2. a dynamically valid quotient can preserve no useful tests.

## Part 3. Equivalences, Implications and Independence

### Equivalence 1

For deterministic tests:

```text
same outcome signatures <=> same induced equivalence relation
```

Therefore equivalent-test-family invariance is identical to kernel equality.
It adds no robustness beyond the quotient definition.

### Equivalence 2

For nested deterministic test families `T subset T+`:

```text
Q_T+ refines Q_T
```

This follows directly from equality over more tests. So refinement persistence
is automatic, not a concept criterion.

### Implication 1

Temporal invariance implies dynamical validity:

```text
forall B, delta(B)=B
  => if s ~ s', then delta(s) ~ delta(s')
```

Converse is false.

### Implication 2

Intervention robustness over a set containing the original dynamics implies
dynamical validity for the original dynamics.

Converse is false because validity for one dynamics says nothing about altered
dynamics.

### Independence 1

Minimal sufficiency and automorphism invariance are independent.

Counterexample:

An arbitrary parity quotient can be minimally sufficient but fail automorphism
invariance relative to a base structure that excludes parity. Conversely, the
trivial quotient may be automorphism-invariant but not sufficient for a
nontrivial test.

### Independence 2

Dynamical validity and admissible-test invariance are independent.

Counterexample:

Parity can be dynamically valid but inadmissibly injected. A legitimate sensor
can induce a quotient that is admissible but not dynamically valid if it groups
states with different futures.

### Independence 3

Recoverability and temporal invariance are independent.

A phase concept may be recoverable from several sensors while not being
temporally invariant. A conserved hidden variable may be temporally invariant
but unrecoverable from available tests.

## Part 4. Recommended Weakest Property

The weakest property that eliminates the parity counterexample without
excluding natural concepts is not a property of quotients alone.

Recommended property:

## Admissible Independent Recoverability

Let `G` be a quotient-independent class of admissible test generators. Each
generator `g in G` produces a finite test family:

```text
T_g
```

Let:

```text
Q_g = Q_{T_g}
```

A quotient `Q` is robust enough to survive the parity counterexample iff:

1. `Q` is a common factor of `Q_g` for at least two independent generators
   `g1, g2 in G`;
2. `G` is defined without reference to `Q` or to the labels of Q's blocks;
3. the factor map from each `Q_g` to `Q` is canonical under the allowed
   symmetries of the base world;
4. `Q` is unchanged by recoding test outcomes;
5. a single direct block-labeling test `t_Q` is not an admissible generator.

Short form:

```text
robust quotient candidate = quotient-independent common factor
```

Why this is weakest:

1. It adds only one missing ingredient: test provenance.
2. It does not require temporal invariance, so phase-like concepts are not
   excluded.
3. It does not require intervention robustness, so non-causal concepts are not
   excluded.
4. It does not require categorical machinery, only an admissible generator class
   `G`.
5. It directly kills the parity counterexample because `parity` appears only
   when a direct arbitrary block-labeling generator is allowed.

Important limitation:

This property forces the lab to define admissible generators. That is unavoidable.
Without such a definition, any quotient can be injected.

## Part 5. Revised Theorem

The revised theorem must not say that stability alone creates concepts.

It should say:

## Independent Recoverability Theorem

Let `S` be a finite state space. Let `G` be a quotient-independent class of
admissible test generators. Each `g in G` produces a finite test family `T_g`.
Let:

```text
Q_g = S / ~_{T_g}
```

Let `Q` be a quotient of `S`.

Assume:

```text
1. Q is a common factor of Q_g for at least two independent generators g.
2. G was specified without reference to Q.
3. direct block-labeling tests t_Q are excluded from G.
4. the factor maps Q_g -> Q are invariant under allowed recodings of test
   outcomes and relabelings of states that preserve the base world.
```

Then `Q` is not explained by a single injected test family.

Equivalently:

```text
independent admissible recoverability rules out the first experiment's parity
counterexample.
```

This theorem deliberately stops short of saying:

```text
Q is a concept.
```

The mathematically safe conclusion is:

```text
Q is a robust quotient candidate.
```

Only after additional experiments should the lab decide whether robust quotient
candidates deserve the name "concept".

## Final Position

The first falsification does not kill the whole program. It kills a quotient-only
criterion.

The minimal missing assumption is:

```text
admissibility of test provenance
```

or equivalently:

```text
non-arbitrary recoverability across admissible test transformations
```

Without this, the theory is mathematically underdetermined. With it, the next
experiment becomes clear: formalize a small admissible class of test generators
and check whether hidden-mode, phase and parity quotients separate under
recoverability.
