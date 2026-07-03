# Equivalence Program

Status: theoretical equivalence program.

The Reduction Program is closed. We are no longer looking for a weaker
primitive.

The new hypothesis:

```text
distinction, invariant, recurrence relation R, admissible configuration set K,
admissibility boundary and related objects may be different mathematical
representations of the same underlying structure.
```

The task is not to protect any one representation. The task is to define each
object as independently as possible, build maps among them, and determine when
the maps are reversible.

No code. No algorithms. No new objectives.

## 1. Common Carrier Without Privileging an Object

Let `U` be a carrier of interaction-items, roles or marks.

Let:

```text
Fin(U)
```

be finite configurations over `U`.

This common carrier is not an admissibility boundary. It is only the place
where the candidate objects can be compared.

Objects to compare:

1. Distinction structure `D`.
2. Invariant structure `Inv`.
3. Recurrence / presentation relation `R`.
4. Admissible configuration set `K`.
5. Boundary structure `B`.
6. Profile structure `P`.
7. Substitutability structure `S`.

## 2. Independent Definitions

### 2.1 Distinction Structure D

A distinction structure is a relation saying when two items or configurations
are separated:

```text
D subset A x A
```

where `A` may be `U`, `Fin(U)`, or another explicitly chosen domain.

Read:

```text
D(a, b) means a and b are distinguished.
```

Minimal requirements:

1. Irreflexive: not `D(a, a)`.
2. Symmetric if distinction is undirected.

No admissibility, consequence, profile or test is assumed.

### 2.2 Invariant Structure Inv

An invariant structure is a family of transformations and the quantities,
relations or partitions preserved by them.

Minimal form:

```text
G acts on A
Inv = structures fixed by G
```

No admissibility is assumed. `G` may be relabeling, presentation-preserving
transformation or another transformation class.

### 2.3 Recurrence / Presentation Relation R

A recurrence or presentation relation is a raw relation over finite
configurations:

```text
R subset Fin(U)
```

Read:

```text
R(X) means X is presented, encountered, repeated or co-available as raw data.
```

This is not admissibility. It says only that the configuration occurs in the
presentation structure.

### 2.4 Admissible Configuration Set K

An admissible configuration set is:

```text
K subset Fin(U)
```

with nontriviality:

```text
K != empty
K != Fin(U)
```

Read:

```text
X in K means X is admissibly realizable.
```

This is stronger than raw presentation because it classifies configurations as
inside or outside a boundary.

### 2.5 Boundary Structure B

A boundary structure is a classification:

```text
B: Fin(U) -> {inside, outside}
```

or, in partial form:

```text
B: Fin(U) -> {inside, outside, undetermined}
```

Sharp boundaries are equivalent to `K` by:

```text
K_B = {X : B(X) = inside}
B_K(X) = inside iff X in K
```

So sharp `B` and `K` are definitionally equivalent.

Partial `B` is strictly richer than sharp `K` because it can represent
undetermined configurations.

### 2.6 Profile Structure P

A profile structure assigns to each item its contexts of admissible insertion:

```text
P(x) subset Fin(U)
```

Read:

```text
C in P(x) means C union {x} is allowed by the represented structure.
```

Defined independently, `P` need not come from any `K`. A profile family is
`K`-representable only if there exists `K` such that:

```text
P(x) = {C : C union {x} in K}
```

for all `x`.

### 2.7 Substitutability Structure S

A substitutability structure is a preorder candidate:

```text
S subset U x U
```

Read:

```text
S(x, y) means x can stand for y.
```

Defined independently, `S` need not be profile inclusion. It is
`K`-representable only if:

```text
S(x, y) iff P_K(y) subset P_K(x)
```

possibly after quotienting profile-equal items.

## 3. Maps From K

Given sharp `K`, several objects are forced.

### 3.1 K to Boundary

```text
B_K(X) = inside iff X in K
```

This is reversible for sharp boundaries.

### 3.2 K to Profiles

```text
P_K(x) = {C : C union {x} in K}
```

This map is canonical.

### 3.3 K to Distinctions

```text
D_K(x, y) iff P_K(x) != P_K(y)
```

Distinction emerges as profile difference.

### 3.4 K to Equivalence

```text
x ~=_K y iff P_K(x) = P_K(y)
```

### 3.5 K to Substitutability

```text
S_K(x, y) iff P_K(y) subset P_K(x)
```

### 3.6 K to Invariants

Let:

```text
Aut(K) = {g : Fin(U) -> Fin(U) | X in K iff g(X) in K}
```

Then every profile, equivalence class and substitutability relation induced by
`K` is invariant under `Aut(K)`.

So:

```text
K -> Inv(K)
```

is canonical once a transformation class is fixed.

## 4. Can K Be Recovered?

### 4.1 Boundary to K

Sharp boundary and `K` are equivalent:

```text
B <-> K
```

Partial boundary is not equivalent to one sharp `K`. It corresponds to a family
of sharp completions:

```text
K+ subset K subset K+ union K?
```

### 4.2 Profiles to K

Given a family `P(x)`, define a candidate:

```text
K_P^1 = {C union {x} : C in P(x)}
```

This recovers only configurations with a distinguished inserted item.

Problem:

If configurations can be decomposed in multiple ways:

```text
X = C union {x} = C' union {y}
```

the profiles must agree on membership.

Representability condition:

```text
for every X and every x,y in X:
  X - {x} in P(x) iff X - {y} in P(y)
```

If this coherence holds, profiles reconstruct a unique `K` for all nonempty
configurations:

```text
X in K iff X - {x} in P(x) for any x in X
```

The empty configuration requires separate status.

Verdict:

```text
P <-> K
```

only under profile coherence plus empty-configuration convention.

### 4.3 Distinctions to K

Distinctions alone do not recover `K`.

Counterexample:

Let `U = {a, b}`.

Two different `K` can induce no distinction between `a` and `b`:

```text
K1 = {empty, {a}, {b}}
K2 = {empty, {a}, {b}, {a,b}}
```

Both may give equal single-item profiles for `a` and `b` depending on the
chosen context domain, but they disagree on whether `{a,b}` is admissible.

More generally, a distinction relation records inequality of profiles, not the
profiles themselves.

Verdict:

```text
D is strictly weaker than K.
```

To recover `K`, distinctions need enrichment with profile content, not just
separation facts.

### 4.4 Substitutability to K

Substitutability preorder alone does not recover `K`.

It records inclusion among profiles:

```text
P(y) subset P(x)
```

but not the actual context sets.

Different profile families can have the same inclusion order.

Counterexample:

Let:

```text
P1(a) = {C1}
P1(b) = {C1, C2}

P2(a) = {D1}
P2(b) = {D1, D2}
```

The inclusion order is the same, but the admissible configurations may differ.

Verdict:

```text
S is weaker than P and weaker than K.
```

Recovering `K` from `S` requires choosing concrete profile realizers.

### 4.5 Invariants to K

Invariants alone do not recover `K`.

Knowing the automorphism group of a structure usually does not determine the
structure uniquely.

Counterexample:

Many different boundaries may have the same automorphism group, especially the
trivial group.

Verdict:

```text
Inv is not equivalent to K without a reconstruction principle.
```

### 4.6 Recurrence R to K

Raw recurrence does not equal admissibility.

The same `R` can support multiple closures:

```text
minimal K containing R
maximal K consistent with R
symmetry-closed K
future-stable K
```

Verdict:

```text
R is not equivalent to K without closure and invariance principles.
```

## 5. Equivalence Theorems

### Theorem 1. Sharp Boundary and K Are Equivalent

For sharp two-valued boundaries:

```text
B <-> K
```

Proof:

`B` maps each configuration to inside/outside. `K` is exactly the inverse image
of inside. Conversely, `K` defines the inside/outside boundary.

### Theorem 2. Coherent Profiles and K Are Equivalent

If profile family `P` satisfies decomposition coherence, then:

```text
P <-> K
```

up to the status of the empty configuration.

This is a real equivalence: profiles are not merely consequences of `K`; under
coherence they carry the same information.

### Theorem 3. Distinction Is Not Equivalent to K

There is no reconstruction of `K` from `D` alone.

Reason:

`D` loses profile content and keeps only whether profiles differ.

Therefore distinction is not a fundamental independent process if `K` or `P`
is available; it is a lossy shadow. But it is also not equivalent to `K`.

### Theorem 4. Substitutability Is Not Equivalent to K

There is no reconstruction of `K` from preorder `S` alone.

Reason:

`S` loses concrete contexts and keeps only inclusion order.

### Theorem 5. R Generates K Only With Closure Principles

`R` and `K` are equivalent only if the theory supplies a closure operator:

```text
Cl: R -> K
```

and a reconstruction of `R` from `K`, for example:

```text
R = observed or generating basis of K
```

This reconstruction is not unique unless `R` is required to be a canonical
basis such as the minimal invariant generating family.

Thus:

```text
R <-> K
```

requires:

1. invariant closure;
2. canonical basis/minimal generator;
3. treatment of unobserved configurations.

## 6. Role of Invariants

Invariance is not equivalent to `K`, but it is necessary for non-injected
genesis of `K`.

There are two directions:

```text
K -> Aut(K)
```

always works.

```text
Aut(K) -> K
```

does not work in general.

Therefore invariants are not another representation of `K` unless enriched
with orbit labels or boundary status per orbit.

Equivalence condition:

Let a transformation group `G` act on `Fin(U)`. A `G`-invariant boundary is
determined by assigning inside/outside status to orbits:

```text
Fin(U) / G
```

Then:

```text
G + orbit boundary <-> K_G-invariant
```

So invariance alone is too weak. Invariance plus orbit classification is
equivalent to a `G`-invariant `K`.

## 7. Is Distinction Fundamental?

Distinction can be defined independently:

```text
D(a, b)
```

But if admissibility/profile structure is present, distinction is forced:

```text
D_K(a, b) iff P_K(a) != P_K(b)
```

This shows:

```text
distinction is not primitive relative to K/P.
```

However, distinction is not equivalent to `K` because it forgets which
contexts distinguish the items.

The precise result:

```text
distinction is a quotient of admissibility structure, not an equivalent
representation of it.
```

It is fundamental only in theories that refuse to represent compatibility
profiles or admissibility boundaries. In the current theory, distinction is an
emergent projection of admissibility.

## 8. Independence Results

### Independent Object 1. Partial Boundary

Partial boundary:

```text
(K+, K-, K?)
```

is not equivalent to a single sharp `K`.

It represents underdetermination. A sharp `K` is a completion.

### Independent Object 2. Recurrence R

`R` is independent from `K` unless a closure/generator relation is fixed.

Same `R` can yield multiple `K`; same `K` can have multiple generating `R`.

### Independent Object 3. Transformation Class G

`G` is not recoverable from `K` uniquely.

`Aut(K)` is the maximal symmetry of `K`, but a genesis theory may use a smaller
presentation-preserving group.

Therefore the transformation class used in genesis is additional unless it is
defined canonically.

### Independent Object 4. Context Domain Omega

Profiles and substitutability depend on the chosen context domain:

```text
P_Omega(x)
```

`K` over all finite contexts fixes the maximal profile, but restricted
epistemic domains require explicit `Omega`.

## 9. Equivalence Diagram

Current diagram:

```text
sharp B  <->  K  <->  coherent P
              |
              v
             D_K        lossy
              |
              v
             S_K        lossy

R + invariant closure + canonical basis  <->  K_generated

G + orbit boundary  <->  G-invariant K

partial B  <->  family of sharp completions
```

Arrows downward are generally lossy unless extra reconstruction data is added.

## 10. Minimal Conditions for Full Equivalence

To treat all objects as equivalent representations of one structure, the theory
must assume:

1. Sharpness: no undetermined boundary zone.
2. Profile coherence: profiles reconstruct configuration membership.
3. Canonical context domain: the domain of contexts is fixed.
4. Invariant closure: recurrence determines admissibility.
5. Canonical generator: `R` is recoverable from `K`.
6. Orbit boundary: invariants include status per orbit, not only symmetries.
7. Nondegeneracy: profiles are not all identical unless the theory accepts
   trivial distinction.

Without these, equivalence fails.

## 11. Verdict

The objects are not all equivalent in full generality.

Strong equivalences:

```text
sharp boundary <-> K
K <-> coherent profile family
G + orbit boundary <-> G-invariant K
partial boundary <-> family of sharp K completions
```

Lossy projections:

```text
K -> distinction
K -> substitutability preorder
K -> automorphism invariants
```

Non-equivalences without extra structure:

```text
R not equivalent to K
D not equivalent to K
S not equivalent to K
Inv not equivalent to K
```

Most important answer:

```text
distinction is not the fundamental process in the current theory.
```

It inevitably arises as profile difference once admissibility/profile structure
exists, but it is not equivalent to that structure because it forgets the
contexts that create the difference.

So distinction is best understood as:

```text
a visible projection of admissibility structure.
```

The next task is not reduction. It is classification of which representation is
most useful for each theorem:

1. `K` for admissibility and non-exit.
2. `P` for equivalence and substitutability.
3. `R` for genesis.
4. `B_partial` for underdetermination.
5. `G + orbit boundary` for invariance.

