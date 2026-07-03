# Representation Completeness Program

Status: representation-strength classification.

The Reduction Program is closed.

The Equivalence Program is complete.

We no longer ask:

```text
Which object is the true primitive?
```

We now ask:

```text
What can each representation prove that the others cannot?
```

Representations:

```text
R = recurrence / presentation relation
K = admissible configuration set
P = compatibility profile family
D = distinction relation
S = substitutability preorder
B = boundary representation
G = transformation / invariance representation
```

No code. No algorithms. No new objectives.

## 1. Status Labels

Each cell uses one of four labels:

```text
complete
```

The theorem class can be stated and proven directly inside the representation.

```text
equivalent
```

The theorem class can be stated and proven after a previously established
lossless translation to an equivalent representation.

```text
incomplete
```

The representation can express a shadow of the theorem, but loses information
needed for the full theorem.

```text
impossible
```

There are two structures with the same representation but different truth
values for the theorem class. Therefore no theorem in that representation alone
can decide it.

## 2. Theorem Classes

Columns:

1. `admissibility`: inside/outside status of configurations.
2. `reconstruction`: recovery of a representation from another.
3. `abstraction`: stable profile classes or reusable admissibility roles.
4. `substitutability`: safe replacement / profile inclusion.
5. `hierarchy`: nontrivial ordered/refined structure of roles.
6. `composition`: combination of structures into further structures.
7. `invariance`: preservation under transformations.
8. `open_growth`: emergence/change of admissibility over stages.
9. `fixed_points`: structures stable under change.
10. `transfer`: preservation across representation/world/context changes.
11. `generalization`: status of unobserved configurations from observed ones.

## 3. Expressive Matrix

| Representation | admissibility | reconstruction | abstraction | substitutability | hierarchy | composition | invariance | open_growth | fixed_points | transfer | generalization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R` | incomplete | incomplete | impossible | impossible | impossible | incomplete | incomplete | complete | incomplete | incomplete | complete |
| `K` | complete | complete | incomplete | complete | incomplete | incomplete | incomplete | impossible | impossible | impossible | incomplete |
| `P` | equivalent | complete | incomplete | complete | incomplete | incomplete | incomplete | impossible | impossible | impossible | incomplete |
| `D` | impossible | impossible | incomplete | impossible | incomplete | impossible | incomplete | impossible | impossible | impossible | impossible |
| `S` | impossible | incomplete | incomplete | complete | incomplete | incomplete | incomplete | impossible | impossible | incomplete | impossible |
| `B` sharp | equivalent | equivalent | incomplete | equivalent | incomplete | incomplete | incomplete | impossible | impossible | impossible | incomplete |
| `B` partial | incomplete | complete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete | incomplete |
| `G` | impossible | incomplete | incomplete | impossible | incomplete | impossible | complete | incomplete | incomplete | complete | incomplete |

This matrix is not a ranking. It says which theorem classes each
representation can carry without smuggling in another representation.

## 4. Proofs and Counterexamples by Representation

### 4.1 R: Recurrence / Presentation

`R` can prove open-growth and generalization claims only when they are phrased
as presentation growth or closure over recurrence:

```text
R_alpha -> R_beta
Cl(R) covers unpresented configurations
```

It cannot prove admissibility without a closure principle:

```text
same R, different Cl -> different K
```

Counterexample:

```text
K_min = observed presentations only
K_sym = symmetry closure of observed presentations
```

Both share the same `R`, but disagree on unobserved admissibility.

Therefore `R` is complete for genesis questions, not for admissibility.

### 4.2 K: Admissible Configurations

`K` is complete for admissibility:

```text
X is admissible iff X in K
```

`K` is complete for substitutability:

```text
x >= y iff for all C, C union {y} in K implies C union {x} in K
```

`K` is incomplete for abstraction because abstraction requires stability across
change, not just one boundary.

Counterexample:

Two systems with the same `K` at one stage can have different future
boundaries:

```text
K -> K
K -> K'
```

Same current `K`, different stability.

Therefore fixed points, transfer and open growth are impossible from static
`K` alone.

### 4.3 P: Profile Family

Coherent `P` is equivalent to `K` up to empty configuration status:

```text
P <-> K
```

Therefore `P` inherits admissibility and substitutability completeness when
coherence holds.

`P` is directly complete for substitutability:

```text
x >= y iff P(y) subset P(x)
```

`P` is incomplete for invariance, transfer and fixed points because those
require transformations or stage comparison.

### 4.4 D: Distinctions

`D` can prove only separation facts:

```text
D(a,b)
not D(a,b)
```

It cannot recover profiles, admissible configurations or substitutability.

Counterexample:

Two different profile families can induce the same distinction relation:

```text
P1(a) != P1(b)
P2(a) != P2(b)
```

but with different context sets and different admissibility.

Therefore `D` is mostly a lossy observational representation.

It can express an abstraction shadow as clustering by non-distinction, but
cannot prove abstraction in the strong sense.

### 4.5 S: Substitutability

`S` is complete for preorder-theoretic substitution facts:

```text
S(x,y)
S(y,z)
therefore S(x,z)
```

if transitivity is part of the representation.

It is incomplete for hierarchy because hierarchy may require graded levels,
composition, or profile content beyond order.

It cannot recover admissibility:

Different `K` can induce isomorphic `S`.

Therefore `S` is proof-efficient for replacement arguments, but weak for
reconstruction.

### 4.6 B: Boundary

Sharp `B` is equivalent to `K`.

Partial `B` is complete for underdetermination:

```text
inside / outside / undetermined
```

but incomplete for sharp admissibility unless a completion principle is added.

Partial `B` is stronger than `K` for representing epistemic indeterminacy, but
weaker than sharp `K` for deciding membership.

### 4.7 G: Transformations / Invariants

`G` is complete for invariance claims:

```text
structure is invariant under g in G
```

It is complete for transfer only when transfer is defined as preservation under
a transformation in `G`.

It cannot prove admissibility because many different boundaries share the same
transformation group.

To recover `K`, `G` needs orbit labels:

```text
Fin(U) / G -> {inside, outside}
```

## 5. Weakest Sufficient Representation by Theorem Class

| Theorem class | Weakest sufficient representation | Extra conditions |
| --- | --- | --- |
| admissibility | sharp `B` or `K` | none |
| profile reconstruction | coherent `P` | empty-status convention |
| substitutability | `P` or `S` | `S` enough for preorder facts; `P` needed for context proofs |
| quotient by profile equality | coherent `P` | or `K` via `P_K` |
| distinction | `D` | only separation facts |
| abstraction | `P` plus stage stability | `K` alone insufficient |
| hierarchy | `S` plus nontrivial order and composition/refinement | `S` alone gives preorder shadow |
| composition | explicit composition structure over `P` or `K` | not derivable from either alone |
| invariance | `G` | plus object acted on |
| transfer | `G` plus profile/boundary preservation | `G` alone insufficient for content |
| open growth | staged `R` or staged `K` | stage comparability |
| fixed points | staged representation plus preservation criterion | static representations insufficient |
| generalization | `R` plus closure principle | `K` can state result, not justify from observations |

## 6. Dependency Lattice of Theorem Strength

Approximate dependency lattice:

```text
D
  separation only

S
  substitution preorder
  requires less than P for order proofs

P
  profiles
  -> S
  -> D

K / sharp B
  -> P
  -> S
  -> D

partial B
  -> family of K completions
  -> underdetermination theorems

G
  invariance / transfer form
  needs K/P/B as acted-on content

R
  genesis / presentation growth
  needs closure to reach K

staged R or staged K + G
  open growth
  stabilization
  fixed points
  transfer
```

This lattice has no single top that is always best. The top depends on theorem
class.

## 7. Translation Operators

### Exact Translations

```text
T_KB: K -> sharp B
T_BK: sharp B -> K
T_KP: K -> coherent P
T_PK: coherent P -> K
T_PS: P -> S
T_PD: P -> D
T_KS: K -> S
T_KD: K -> D
```

Exact and reversible:

```text
K <-> sharp B
K <-> coherent P
```

Exact but lossy:

```text
P -> S
P -> D
K -> S
K -> D
K -> Aut(K)
```

### Conditional Translations

```text
R -> K
```

requires closure:

```text
Cl(R) = K
```

```text
G -> K
```

requires orbit boundary:

```text
Fin(U)/G -> {inside, outside}
```

```text
partial B -> K
```

requires completion:

```text
K+ subset K subset K+ union K?
```

### Impossible Translations Without Extra Data

```text
D -> K
S -> K
G -> K
R -> K
D -> P
S -> P
Aut(K) -> K
```

Each impossibility is witnessed by multiple non-isomorphic `K` or `P`
structures inducing the same source representation.

## 8. Translation Graph

```text
sharp B <--> K <--> coherent P
                 |       |
                 v       v
                 S       D

K ---------> Aut(K)

R --Cl?--> K

G + orbit boundary <--> G-invariant K

partial B --completion?--> sharp B/K
partial B <--> family of sharp completions
```

Legend:

```text
<--> exact reversible
---> exact lossy
--?-> conditional
```

## 9. Representation Complexity

Do not assume one complexity measure.

Candidate costs:

1. primitive count;
2. information content;
3. reconstruction overhead;
4. proof length;
5. description length;
6. empirical measurement cost;
7. closure cost;
8. transformation bookkeeping cost.

Different representations optimize different costs.

### 9.1 R

Low primitive cost for raw presentation data.

High closure cost for admissibility.

Best for genesis and generalization-from-observation.

### 9.2 K

High information content because it stores membership over configurations.

Low proof cost for admissibility and non-exit.

Weak for genesis because it hides source.

### 9.3 P

Potentially more redundant than `K`.

Low proof cost for substitutability and equivalence.

Requires coherence checks if treated as independent.

### 9.4 D

Low information cost.

Very low expressive power.

Good for observational regime separation, bad for reconstruction.

### 9.5 S

Lower information than `P`.

Efficient for replacement/order proofs.

Cannot recover contexts.

### 9.6 B

Sharp `B` has same content as `K`.

Partial `B` has higher expressive value for uncertainty and underdetermination.

### 9.7 G

Low content if only transformations are stored.

High explanatory power for invariance/transfer.

Needs acted-on structure to make content claims.

## 10. Universality

Question:

```text
Is there a meta-object M whose faithful images, quotients or projections yield
R, K, P, D, S, B and G?
```

Do not assume categories or algebra.

Weakest candidate:

```text
M = (U, Fin(U), R, B_partial, G, Ctx)
```

where:

1. `U` is carrier;
2. `Fin(U)` is configuration domain;
3. `R` is presentation/recurrence;
4. `B_partial` is partial boundary;
5. `G` is transformation structure;
6. `Ctx` is context-domain selection.

From `M`:

```text
sharp completions of B_partial -> K
K -> P
P -> S
P -> D
G -> invariance / transfer
R -> genesis / generalization
```

This `M` is not a new primitive claim. It is a bookkeeping object for
representation completeness.

### Universality Verdict

A meta-object exists if we allow it to contain the independent data that
equivalence failed to recover:

```text
R, partial boundary, transformation class, context domain
```

No smaller faithful meta-object is currently justified, because:

1. `R` is not recoverable from `K`.
2. `G` is not recoverable uniquely from `K`.
3. partial boundary is not recoverable from sharp `K`.
4. context restriction is not recoverable from `S` or `D`.

Therefore universality is possible as integration, not as reduction.

## 11. Predictive Power

Predictive power here means:

```text
the representation licenses an experimentally distinguishable expectation
that another representation cannot license without extra data.
```

### 11.1 R

Unique predictions:

1. Which new configurations should become admissible under a closure principle.
2. Whether admissibility should stabilize as presentations accumulate.
3. Whether multiple `K` remain underdetermined by the same recurrence data.

Minimal example:

Observed recurrence supports two closures. `R` predicts ambiguity; sharp `K`
alone cannot explain why ambiguity existed.

### 11.2 K

Unique predictions:

1. Immediate admissibility of untested configurations already in `K`.
2. Non-exit under replacement.

If two representations translate exactly to the same `K`, they are empirically
equivalent for admissibility tests.

### 11.3 P

Unique predictions:

1. Which substitutions should work by profile inclusion.
2. Which items should be indistinguishable by profile equality.

`K` can recover these if complete, so `P` has proof efficiency rather than
strictly unique empirical content relative to `K`.

### 11.4 D

Unique predictions:

Only separation/non-separation.

If experiments observe only pairwise distinction, `D` may be empirically
equivalent to richer representations after projection. It cannot predict
context-specific failures.

### 11.5 S

Unique predictions:

Replacement success/failure as order facts.

But if context-specific outcomes are measured, `S` is weaker than `P` because
it cannot say which contexts cause the replacement relation.

### 11.6 B

Partial `B` uniquely predicts underdetermination:

```text
some configurations should remain undecidable by current evidence
```

Sharp `B` is empirically equivalent to `K`.

### 11.7 G

Unique predictions:

1. Invariance under relabeling/transformation.
2. Transfer across transformed presentations.
3. Failure modes when a transformation is not in `G`.

Minimal example:

Two systems share the same `K`, but one genesis theory uses a smaller
presentation-preserving group. They predict different transfer invariances.

## 12. Empirical Equivalence

Representations are empirically equivalent relative to an experiment class if
their projections to the measured observables coincide.

Examples:

```text
K and sharp B
```

are empirically equivalent for admissibility membership.

```text
K and coherent P
```

are empirically equivalent for admissibility and substitution if all contexts
are measured.

```text
D and projection of K to distinctions
```

are empirically equivalent only for experiments that measure distinctions and
nothing about contexts or admissibility.

```text
S and projection of P to profile inclusion
```

are empirically equivalent only for experiments that measure replacement
success and not context structure.

## 13. Fundamental Status Rule

A representation deserves to be called fundamental only if it satisfies both:

1. minimality for a theorem class;
2. unique explanatory or predictive power not recoverable from a weaker
   equivalent representation.

Current classification:

```text
R is fundamental for genesis and generalization-from-presentation.
K/sharp B is fundamental for admissibility and non-exit.
P is fundamental for efficient substitutability and profile equivalence proofs.
D is not fundamental when P or K is available.
S is not fundamental for admissibility, but may be minimal for pure replacement order.
G is fundamental for invariance and transfer.
partial B is fundamental for underdetermination.
```

No single representation is globally fundamental.

## 14. Next Tasks

1. Formalize theorem classes into named theorem schemas.
2. Produce minimal counterexamples for every `impossible` matrix cell.
3. Define context-domain cost and closure cost more precisely.
4. Identify experiment classes and their observable projections.
5. Decide whether `M = (U, Fin(U), R, B_partial, G, Ctx)` is only bookkeeping
   or a genuine representation with its own theorems.

