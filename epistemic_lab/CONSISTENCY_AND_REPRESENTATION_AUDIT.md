# Consistency and Representation Theorem Program

Status: internal consistency audit.

Research State Space Structure is the current mathematical foundation.

Rule 0:

```text
Assume the current theory is wrong.
Attempt to derive contradictions.
Do not extend the framework.
Audit it.
```

No new primitives. No new theories. No new scientific concepts.

## 1. Minimal Axiom System

The current framework can be reduced to the following axiom layers.

### A0. Research-State Records

There is a class `RS` of admissible research states:

```text
S = (T, A, E, F, U, R)
```

where the components are records, not necessarily sets with extra structure.

### A1. No-Worse Relation

There is a binary relation:

```text
>=_E subset RS x RS
```

read:

```text
S' is epistemically no worse than S.
```

### A2. Reflexivity

```text
S >=_E S
```

for every admissible research state.

### A3. Transitivity

```text
S2 >=_E S1 and S1 >=_E S0 imply S2 >=_E S0.
```

This requires monotonic evidence accounting or faithful preservation of prior
records.

### A4. Representation Equivalence

Define:

```text
S1 ~=_E S2 iff S1 >=_E S2 and S2 >=_E S1
```

Lossless representational variants are identified by this equivalence.

### A5. Optional Join Closure

Compatible research states have conflict-preserving least common refinements.

This is not required for the preorder or quotient poset.

### A6. Optional Directed-Union Closure

Directed coherent research programs have suprema.

This is required only for dcpo claims.

### A7. Optional Localization

Applicability records can be restricted to regions of `Omega` and glued when
overlaps agree.

This is required only for sheaf-like claims.

## 2. Theorem-to-Axiom Map

| Theorem | Required axioms |
| --- | --- |
| raw states form reflexive relation | A0, A1, A2 |
| raw states form preorder | A0, A1, A2, A3 |
| `~=_E` is equivalence | A2, A3 |
| quotient is poset | A2, A3, A4 |
| joins exist | A0-A4 plus A5 |
| directed suprema exist | A0-A4 plus A6 |
| dcpo | A0-A4 plus A6 |
| information topology | A0 plus finite-record containment; independent of A5/A6 |
| sheaf-like localization | A7 |
| fixed points | transition operator/policy, not intrinsic to A0-A4 |
| morphisms preserve improvement | A1-A3 plus chosen mapping class |

Immediate redundancy:

```text
join closure, dcpo closure, topology and sheaf-like structure are not needed
for the base research-state order.
```

They are optional enrichments.

## 3. Consistency Report

### 3.1 Order Axioms

No contradiction in taking `>=_E` as a preorder.

Potential inconsistency:

If the framework allows transitions that erase failed tests while still
counting as improvements, transitivity can fail.

Counterexample:

```text
S1 = S0 + counterexample c
S2 = S1 translated into a representation that drops c
```

If:

```text
S1 >=_E S0
S2 >=_E S1
```

but `S2` no longer contains the improvement over `S0`, then:

```text
S2 >=_E S0
```

is unjustified.

Resolution:

Transitivity requires faithful translation or monotonic evidence accounting.
This must be an explicit axiom, not a background hope.

### 3.2 Quotient Construction

If `>=_E` is a preorder, then `~=_E` is an equivalence relation and the quotient
is a poset.

No contradiction.

But if transitivity is not assumed, `~=_E` may fail transitivity.

Therefore:

```text
quotient-poset theorem depends on A3.
```

### 3.3 Joins

The previous framework correctly stated joins as conditional.

Counterexample to unconditional joins:

```text
S1: T holds on A.
S2: T fails on A.
```

If contradiction records or theorem splitting are forbidden, no least common
refinement exists.

If contradiction records are allowed, a join can be constructed by recording
the conflict.

Verdict:

No inconsistency, but join-semilattice is not base structure.

### 3.4 Meets

Unconditional meets fail when vocabularies cannot be aligned.

Counterexample:

```text
S1 uses theorem vocabulary V1.
S2 uses theorem vocabulary V2.
No translation V1 <-> V2 exists.
```

There is no canonical greatest common abstraction except the empty state, and
even that requires a shared initial schema.

Verdict:

Meet-semilattice requires canonical projection/translation assumptions.

### 3.5 DCPO Assumption

Directed suprema are not guaranteed.

Counterexample:

A directed chain adds increasingly refined theorem splits:

```text
T -> T_1 -> T_2 -> ...
```

If the state language permits only finite theorem records and no limit object,
the supremum is not an admissible state.

Verdict:

DCPO is optional and requires closure under coherent limits.

### 3.6 Topology

The information topology from finite records is consistent.

But it requires a notion of finite record containment:

```text
r subset S
```

This is already present in the record structure if states are record-bearing.

Verdict:

Topology is a conservative representation of record containment, not an
additional scientific axiom.

### 3.7 Applicability Atlas

No internal contradiction if:

```text
Suf(T), Fail(T), Und(T)
```

are allowed to be partial and revisable.

Potential contradiction:

```text
Suf(T) intersect Fail(T) != empty
```

for the same theorem and same interpretation.

Resolution:

Such overlap must be represented as conflict/artifact/theorem-splitting, not
as a valid sharp atlas.

### 3.8 Morphisms

Order-preserving maps are consistent.

But "lossless" and "lossy" morphisms must not be conflated.

If a lossy morphism is treated as equivalence, theorem preservation fails.

Verdict:

Morphism theory is consistent if morphism type is explicit.

## 4. Independence Report

### Remove A2 Reflexivity

Fails:

1. no non-strict improvement relation;
2. `~=_E` may not be reflexive;
3. quotient construction fails.

A2 is necessary for preorder/poset structure.

### Remove A3 Transitivity

Fails:

1. preorder theorem;
2. equivalence relation `~=_E`;
3. quotient poset;
4. path-composition reasoning.

A3 is necessary unless the theory downgrades to a directed graph of states.

### Remove A4 Representation Equivalence

Raw states can still form a preorder.

Fails:

1. quotient poset;
2. representation-invariant value;
3. loop-as-equivalence interpretation.

A4 is necessary for representation-independent structure.

### Remove A5 Join Closure

Base preorder and quotient poset remain.

Fails:

1. guaranteed merging of independent paths;
2. join-semilattice claims.

A5 is not base-necessary.

### Remove A6 Directed-Union Closure

Base preorder and quotient poset remain.

Fails:

1. dcpo claims;
2. domain-theoretic limit interpretation.

A6 is not base-necessary.

### Remove A7 Localization

Base order remains.

Fails:

1. sheaf-like localization;
2. local theorem gluing.

A7 is not base-necessary.

## 5. Representation Theorem Candidates

### Domain Theory

Representable if:

1. quotient poset exists;
2. directed suprema exist;
3. finite research records act as compact approximants.

Without these, domain theory is only an analogy.

### Lattice Theory

Representable if:

1. joins and meets exist;
2. they are least/greatest with respect to `>=_E`.

The current base structure is not a lattice.

### Category Theory

Representable if:

1. research states are objects;
2. transitions or morphisms compose associatively;
3. identity morphisms exist.

This is plausible for histories and translations, but not necessary for the
base preorder.

A preorder itself can be viewed as a thin category, but that adds no theorem
content.

### Sheaf Theory

Representable if:

1. applicability domains form a site/topological base;
2. theorem records restrict to subdomains;
3. compatible local records glue uniquely or with controlled ambiguity.

Not guaranteed by the base framework.

### Formal Concept Analysis

Representable locally for relations between:

```text
research states / theorem records / applicability properties
```

but not a representation of the full dynamic research-state space unless one
chooses a formal context.

### Abstract Interpretation

Representable if:

1. concrete research states and abstract summaries are related;
2. soundness relation is defined;
3. abstraction/concretization maps exist.

Useful for comparing compressed research summaries, not base structure.

## 6. Redundancy Analysis

Redundant or optional in the base theory:

1. lattice language;
2. complete lattice language;
3. dcpo/domain language;
4. sheaf language;
5. metric geometry;
6. simplicial geometry;
7. fixed points without a specified transition operator.

Keep only as conditional representation theorems.

Base theory reduces to:

```text
record-bearing states + preorder + equivalence quotient
```

with optional closure/localization enrichments.

## 7. Equivalence Analysis

### Applicability Atlas vs Sheaf-Like Localization

Not equivalent.

Applicability atlas only stores domains and theorem records.

Sheaf-like localization additionally requires restriction and gluing laws.

So:

```text
sheaf-like structure = applicability atlas + locality/gluing axioms
```

### Research Paths vs Directed Systems

Not equivalent.

A path is sequential:

```text
S0 -> S1 -> ...
```

A directed system allows branching with upper bounds.

Every monotone path is directed. Not every directed system is a path.

### Research Value vs Order Enrichment

Mostly equivalent at the base level.

Epistemic Value Theory's non-scalar value is precisely the order/enrichment:

```text
>=_E
```

Numerical value is extra and not equivalent.

### Counterexample Records vs Compact Elements

Not equivalent without compactness axiom.

Counterexamples are finite records, but finite record does not automatically
mean compact in the order-theoretic sense unless directed suprema behave
properly.

## 8. Conservative Extension Meta-Theorem

### Claim

Research State Space Structure is a conservative extension of Applicability
Theory plus Epistemic Value Theory.

### Meaning

It adds mathematical organization:

```text
preorder
quotient
conditional joins
conditional dcpo
topology
morphisms
```

but does not add new epistemic content about worlds, agents or experiments.

### Verdict

Mostly true for the base structure:

```text
RS + >=_E + quotient
```

is conservative. It reorganizes existing Applicability/Value records.

False for optional enrichments if treated as facts:

1. guaranteed joins;
2. dcpo structure;
3. sheaf gluing;
4. compact counterexample elements;
5. intrinsic fixed points.

Those add genuine mathematical assumptions.

Therefore:

```text
base framework = conservative extension
enriched framework = non-conservative unless assumptions are explicit
```

## 9. Minimal Clean Foundation

The cleaned foundation should be:

```text
A0 research-state records
A1 no-worse relation
A2 reflexivity
A3 transitivity
A4 representation equivalence quotient
```

Then prove:

```text
raw RS is a preorder
RS / ~=_E is a poset
finite-record containment induces an information topology if records are
set-like
```

Everything else must be labeled conditional:

```text
join-semilattice under A5
dcpo under A6
sheaf-like localization under A7
fixed points under specified transition operators
domain theory under compact-approximation axioms
```

## 10. Unresolved Mathematical Questions

1. What is the weakest faithful-translation condition that guarantees
   transitivity?
2. Can conflict-preserving joins be made canonical?
3. Do finite counterexample records satisfy compactness in any natural dcpo
   completion?
4. Is there a canonical common abstraction operation for unrelated theorem
   vocabularies?
5. Can applicability atlases be given restriction/gluing laws without adding
   non-conservative assumptions?
6. Is the information topology enough for convergence of real research
   histories?
7. Which optional enrichments are empirically useful rather than merely elegant?

## 11. Final Audit Verdict

No contradiction was found in the base framework.

But the framework was over-described.

The rigorous core is smaller:

```text
preorder of research states
quotient poset of epistemic content
conditional information topology
```

The following must be demoted from conclusions to conditional enrichments:

```text
lattice
complete lattice
dcpo/domain
sheaf-like structure
fixed points
canonical joins/meets
compact counterexamples
```

This is progress: the foundation becomes smaller, cleaner and less vulnerable.

