# Theorems From an Admissibility Boundary

Status: theoretical theorem inventory.

This document asks what becomes inevitable if the project accepts the current
irreducible axiom:

```text
K = a nontrivial boundary between admissibly realizable and non-realizable
finite configurations
```

The goal is to separate:

1. theorems forced by `K` alone;
2. stronger theorems requiring additional nondegeneracy assumptions;
3. claims that are not theorems and would smuggle in extra structure.

No code. No algorithms. No new objectives.

## 1. Minimal Setup

Let `U` be a collection of possible items, roles, marks, structures or
configuration-parts.

Let:

```text
Fin(U) = finite configurations over U
K subset Fin(U)
```

Read:

```text
X in K     means X is admissibly realizable
X notin K means X crosses the admissibility boundary
```

The minimal nontriviality axiom:

```text
K is not empty
K is not all of Fin(U)
```

No time, causality, tests, future, objective, model space, probability,
topology, metric, order or dynamics is assumed.

## 2. What K Alone Forces

### Theorem 1. A Failure Boundary Exists

If `K` is nontrivial, then there exist configurations whose status differs:

```text
exists A, B in Fin(U):
  A in K and B notin K
```

This is trivial formally, but nontrivial conceptually: the theory has negative
space. Some configurations fail.

Without this, no distinction between preservation and breakage can be defined.

### Theorem 2. Constraint Is Derivable

Every `K` induces a constraint predicate:

```text
Adm(X) iff X in K
```

So "constraint" is not primitive once `K` is accepted. It is just the boundary
between admitted and non-admitted configurations.

### Theorem 3. Incompatibility Is Derivable

Define finite joint incompatibility:

```text
Incomp(X) iff X notin K
```

Then every non-admitted finite configuration is an incompatibility witness.

This is stronger than binary incompatibility. Binary incompatibility is only
the special case:

```text
Incomp({a, b})
```

### Theorem 4. Profiles Are Canonical

For each item `x`, define its compatibility profile:

```text
P(x) = {C in Fin(U) : C union {x} in K}
```

For a domain of contexts `Omega subset Fin(U)`, define:

```text
P_Omega(x) = {C in Omega : C union {x} in K}
```

This construction is forced. Once `K` exists, every item has a profile relative
to every context domain.

### Theorem 5. Distinguishability Is Derivable

Define:

```text
x !=_K y iff P(x) != P(y)
```

Then `K` induces a structural distinction relation.

Important limitation:

`K` alone does not guarantee that any two items are distinguishable. It only
guarantees that distinguishability, if present, is determined by profiles.

### Theorem 6. Equivalence Is Derivable

Define:

```text
x ~=_K y iff P(x) = P(y)
```

Then `~=_K` is an equivalence relation.

Proof:

1. Reflexive because `P(x) = P(x)`.
2. Symmetric because equality is symmetric.
3. Transitive because equality is transitive.

So quotient structure is inevitable:

```text
U / ~=_K
```

This quotient is not assumed. It is forced by profile equality.

### Theorem 7. Substitutability Preorder Is Derivable

Define:

```text
x >=_K y iff P(y) subset P(x)
```

Read:

```text
x can stand wherever y can stand, relative to K
```

Then `>=_K` is a preorder:

1. Reflexive because `P(x) subset P(x)`.
2. Transitive because subset inclusion is transitive.

Antisymmetry is not guaranteed. If `P(x) = P(y)` and `x != y`, then:

```text
x >=_K y and y >=_K x
```

but `x` and `y` remain distinct as raw items.

Therefore a partial order is forced only on equivalence classes:

```text
[x] >= [y] iff P(y) subset P(x)
```

### Theorem 8. Preservation Is Derivable as Non-Exit

For a context domain `Omega`, `x >=_Omega y` means:

```text
for every C in Omega:
  if C union {y} in K
  then C union {x} in K
```

This is preservation without importing a separate preservation primitive:

```text
replacement preserves membership in K
```

So preservation is not fundamental once `K` is accepted. It is non-exit from
the admissibility boundary.

### Theorem 9. Minimal Quotient Is Forced

The quotient:

```text
Q_K = U / ~=_K
```

is the coarsest representation of items that preserves all single-item
compatibility profiles.

Proof sketch:

Any representation that identifies `x` and `y` while `P(x) != P(y)` loses a
profile distinction. Therefore it does not preserve all profile facts.

Any representation that separates profile-equal items is finer than necessary.

Thus `Q_K` is canonical relative to `K`.

### Theorem 10. Compression Has a Formal Meaning

If an item or class `[x]` satisfies:

```text
[x] >= [y]
```

then `[x]` can replace `[y]` without exiting `K` across the chosen context
domain.

If `[x]` is simpler under some later size or description measure, this becomes
compression.

Important limitation:

`K` alone does not define simplicity. It defines safe replacement. Compression
requires an additional size, cost or description relation.

### Theorem 11. Irrelevance Is Derivable as Profile Invisibility

An item `z` is invisible relative to a context domain `Omega` if adding it
does not change profile membership over `Omega`.

One form:

```text
z is Omega-irrelevant to x iff
P_Omega(x) = P_Omega(x with z included)
```

More generally, two modifications are equally irrelevant if they induce the
same profile.

Thus a primitive relevance score is not necessary for a first distinction
between relevant and irrelevant variation. Relevance begins as profile change.

### Theorem 12. Context-Relativity Is Inevitable

For different context domains `Omega1` and `Omega2`, the induced relations may
differ:

```text
P_Omega1(x) may differ from P_Omega2(x)
x >=_Omega1 y may hold while x >=_Omega2 y fails
```

Therefore no absolute substitutability or consequence relation is forced by
`K` alone. Context-relativity is not a defect; it is inevitable.

## 3. Theorems Requiring Mild Extra Assumptions

The following are not forced by nontrivial `K` alone. They require explicit
nondegeneracy assumptions.

### Extra Assumption A. Item Separability

There exist `x, y` such that:

```text
P(x) != P(y)
```

Then:

```text
nontrivial distinctions exist
```

Without this, all items may have identical profiles even though `K` itself is
nontrivial.

### Extra Assumption B. Profile Inclusion Is Nontrivial

There exist `x, y` such that:

```text
P(y) proper_subset P(x)
```

Then:

```text
nontrivial substitutability exists
```

Without this, substitutability may collapse into equivalence or incomparability.

### Extra Assumption C. Higher-Order Constraint

There exist configurations `A, B` such that every small subconfiguration is in
`K`, but `A` itself is not in `K`.

Then:

```text
higher-order incompatibility exists
```

This matters because it proves binary compatibility is insufficient.

### Extra Assumption D. Multiple Context Domains

There exist `Omega1, Omega2` such that profile relations differ.

Then:

```text
context-sensitive epistemic structure exists
```

Without this, context-relativity is definable but not active.

### Extra Assumption E. Transformations

Let `U` have allowed transformations:

```text
tau: U -> U'
```

If transformations preserve membership:

```text
X in K iff tau(X) in K'
```

then profile structure transports:

```text
P_K(x) corresponds to P_K'(tau(x))
```

This yields transport/invariance theorems. But transformations are not
contained in `K` alone unless encoded as configurations.

## 4. Non-Theorems

These do not follow from `K` alone.

### Non-Theorem 1. Concepts Exist

`K` induces profile equivalence classes. But a concept requires additional
stability, invariance, admissible origin or compositional role.

So:

```text
profile class != concept
```

without extra axioms.

### Non-Theorem 2. Learning Occurs

`K` is static. It does not imply change, update or growth.

To get learning, one needs:

```text
K_t -> K_{t+1}
```

or transformations over `K`.

### Non-Theorem 3. Causality Exists

`K` says what can hold together. It does not say what produces what.

Causal direction is additional structure.

### Non-Theorem 4. Probability Exists

`K` is qualitative. It does not assign weights, frequencies or measures.

Probability requires additional structure.

### Non-Theorem 5. Optimization Exists

No objective, utility or preference follows from `K`.

Optimization is not forced.

### Non-Theorem 6. Hierarchy Exists

Preorders and quotients exist, but hierarchy in the strong epistemic sense
requires compositional or refinement structure beyond raw profile inclusion.

## 5. Strongest Inevitable Package

From `K` alone, the unavoidable package is:

```text
admissibility boundary
  -> incompatibility witnesses
  -> compatibility profiles
  -> profile equivalence
  -> canonical quotient
  -> profile-inclusion preorder
  -> substitutability as non-exit from K
  -> context-relativity
```

This is already nontrivial.

It means that accepting one admissibility boundary inevitably generates:

1. distinctions;
2. equivalence classes;
3. quotient structure;
4. preorder/substitution structure;
5. a minimal preservation relation;
6. a formal notion of irrelevance;
7. a sharp distinction between forced structure and injected structure.

## 6. Why These Theorems Matter

The result is important because the project no longer needs to assume
substitutability, equivalence, quotient or preservation as independent
primitives.

They are theorems of admissible realizability:

```text
K
  entails profile structure
  entails equivalence
  entails quotient
  entails preorder
  entails substitutability
  entails preservation-as-non-exit
```

The next reduction target is therefore not consequence or substitution.

It is:

```text
admissibility itself
```

## 7. Current Boundary of the Theory

The first irreducible axiom currently remains:

```text
There exists a nontrivial admissibility boundary K.
```

Everything above follows once that boundary exists.

But the source of the boundary is still not explained.

If `K` is arbitrary, the theory collapses back into injected structure.

Therefore the next theorem program must ask:

```text
What conditions make K non-arbitrary?
```

Possible candidates:

1. invariance under relabeling;
2. generation by interaction constraints;
3. closure under admissible transformations;
4. independent recoverability from multiple presentations;
5. non-injectability from post-hoc labels.

These are not assumed here. They are the next target.

