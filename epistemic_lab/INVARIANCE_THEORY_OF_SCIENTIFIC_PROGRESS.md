# Invariance Theory of Scientific Progress

Status: invariant-core audit.

Research State Space Structure and Consistency Audit are the current
mathematical foundation.

No new primitives. No new representations. No new enrichments.

Base framework:

```text
RS = research states
>=_E = reflexive/transitive no-worse relation
~=_E = mutual epistemic equivalence
Q = RS / ~=_E
```

The objective is to identify what every faithful scientific process must
preserve.

## 1. Faithful Transitions

An admissible transition:

```text
S -> S'
```

is faithful only if:

```text
S' >=_E S
```

and the transition does not erase prior evidence/failure records except through
lossless translation or explicit reinterpretation.

Thus the first invariant is not a number. It is:

```text
order non-decrease in Q
```

where:

```text
[S'] >= [S]
```

in the quotient poset.

## 2. Candidate Invariant Catalogue

| Candidate | Invariant under faithful transition? | Reason |
| --- | --- | --- |
| theorem text | No | theorem rewriting may change syntax |
| theorem content up to translation | Yes, unless explicitly refuted/refined | preserved in quotient or recorded as failure/split |
| eliminated counterexamples | Yes | monotonic evidence accounting |
| failed tests | Yes as records | interpretations may change, records remain |
| applicability refinement | Yes as order improvement | `Und` may shrink, `Fail/Suf` may expand/refine |
| exact applicability regions | No | regions may refine or split |
| representation quotient class `[S]` | Yes under lossless representation change | definition of `~=_E` |
| raw representation | No | translation/compression can change it |
| partial-order relation | Yes under admissible morphisms | morphisms must preserve `>=_E` |
| scalar value | No | scalar value is noncanonical unless derived |
| evidence monotonicity | Yes if faithful | needed for transitivity |
| information preservation | Yes only for epistemic content, not syntax | compression may remove redundancy |
| causal dependencies | Not intrinsic | causality is not base structure |
| benchmark score | No | implementation/benchmark artifact |
| confirmation count | No | not invariant under theorem splitting |
| artifact eliminations | Yes as records | cannot be erased faithfully |

## 3. Accidental Regularities vs Genuine Invariants

### Accidental Regularities

Patterns that may hold in a history but are not required by faithful progress:

1. increasing theorem count;
2. increasing confirmation count;
3. monotonic confidence scalar;
4. fixed vocabulary;
5. fixed representation;
6. fixed benchmark performance.

### Implementation Artifacts

Patterns dependent on tools or encodings:

1. feature names;
2. cluster labels;
3. objective scores;
4. metric coordinates;
5. storage format.

### Genuine Invariants

Properties required by faithful scientific progress:

1. order non-decrease in `Q`;
2. preservation of failed-test records;
3. preservation of counterexample records;
4. preservation of theorem content up to declared refinement/refutation;
5. preservation of lossless translation equivalence;
6. preservation of morphism order;
7. explicit accounting for any loss.

## 4. Morphism Invariance

A morphism:

```text
f: RS1 -> RS2
```

is admissible if:

```text
S' >=_E S implies f(S') >=_E f(S)
```

and record loss is either absent or explicitly marked.

### Invariants Under Every Faithful Morphism

1. preorder structure;
2. epistemic equivalence:

   ```text
   S ~=_E S' implies f(S) ~=_E f(S')
   ```

   for lossless morphisms;

3. order of histories;
4. existence of improvement paths;
5. irreversibility of recorded failures.

### Not Invariant Under All Morphisms

1. theorem syntax;
2. number of theorem records;
3. chosen representation architecture;
4. scalar confidence;
5. metric distances;
6. join/meet structure unless morphism preserves joins/meets.

### Maximal Invariant Class

The maximal invariant under lossless morphisms is:

```text
the isomorphism class of the ordered research-content state [S] in Q,
together with its preserved evidence/failure records.
```

If morphisms are lossy, no complete invariant survives except what the morphism
explicitly preserves.

## 5. Composition Invariance

Suppose two programs compose:

```text
H1 ; H2
```

or merge through a common refinement when a join exists.

### Invariants That Compose

1. order-preserving transitions;
2. preserved failure/evidence records;
3. lossless representation equivalences;
4. theorem refinements if vocabularies align or translate faithfully.

### Invariants Preserved But Not Additive

1. applicability precision;
2. artifact elimination;
3. representation completeness.

They may improve or remain unchanged, but they do not simply add.

### Invariants Destroyed By Composition

None under faithful composition.

But apparent invariants can be destroyed:

1. fixed vocabulary;
2. theorem count;
3. local confidence ranking;
4. local coordinate system.

### Invariants Emerging Only After Composition

1. cross-program translation equivalence;
2. contradiction records between programs;
3. theorem splits caused by combined evidence;
4. merged applicability domains.

These are not invariants of either component alone.

## 6. Minimal Complete Invariant

Question:

```text
What is the smallest invariant that reconstructs epistemic content up to
representation equivalence?
```

Answer:

```text
[S] in Q = RS / ~=_E
```

plus the order relation inherited from `>=_E`.

This is complete by construction:

```text
[S1] = [S2] iff S1 ~=_E S2
```

No scalar invariant can be complete in general.

### Why No Scalar Is Complete

If two incomparable states exist:

```text
S1 not>= S2
S2 not>= S1
```

any scalar that totally orders them either:

1. assigns equal values and loses distinction; or
2. orders them artificially.

Thus a scalar cannot faithfully represent all order/content structure unless
the quotient poset is already embeddable into a scalar order without loss.

## 7. Universal Property Proposal

Research-state content is characterized by the following universal property:

```text
Q = RS / ~=_E is the universal representation-invariant quotient of research
states through which every lossless, order-preserving interpretation factors.
```

Form:

If:

```text
f: RS -> X
```

is lossless with respect to epistemic equivalence:

```text
S ~=_E S' implies f(S) = f(S')
```

then there exists a unique:

```text
f_bar: Q -> X
```

such that:

```text
f = f_bar o q
```

where:

```text
q: RS -> Q
```

is the quotient map.

This is the strongest universal property justified by the cleaned foundation.

It does not require category theory, though it can be expressed categorically.

## 8. No-Go Theorems

### No-Go 1. No Scalar Complete Invariant

Unless `Q` is scalar-order embeddable without losing incomparability and
content, no scalar invariant is complete.

### No-Go 2. No Finite Fixed Invariant List Is Complete In General

If research states can contain arbitrarily many theorem/evidence/failure
records, any fixed finite list of summaries loses information.

Counterexample:

Two states agree on the finite summaries but differ on an unlisted
counterexample.

### No-Go 3. No Invariant Survives Arbitrary Lossy Morphisms

A lossy morphism can forget any chosen record unless constrained.

Therefore invariance claims must specify faithful/lossless morphism class.

### No-Go 4. Theorem Count Is Not Invariant

Theorem splitting and merging change theorem count while preserving or
improving epistemic content.

### No-Go 5. Confidence Scalars Are Not Invariant

Confidence can be recalibrated under representation translation without
changing applicability content.

## 9. Representation Independence

### Theorem Rewriting

Invariant:

```text
content class [S]
```

Not invariant:

```text
syntax, proof order, theorem names
```

### Vocabulary Translation

Invariant if translation is lossless:

```text
~=_E class, order position, evidence/failure records
```

### Abstraction

Invariant only if abstraction is sound and records declared loss.

Lossless abstraction preserves `[S]`; lossy abstraction does not.

### Compression

Invariant if compression is reversible or proof/prediction preserving.

Otherwise only compressed summaries survive.

### Refinement

Refinement is order-increasing:

```text
S' >=_E S
```

The old content survives as a lower element, not necessarily as identical
syntax.

### Decomposition

Splitting a theorem preserves content if the subtheorems cover the original
domain and record all failure/success regions.

### Composition

Composition preserves component invariants only under faithful merge. It may
introduce new conflict records.

## 10. Reformulating Prior Theories Invariantly

### Applicability Theory

Invariant form:

```text
applicability records are order-preserved content components of [S]
```

not specific table formats.

### Epistemic Value Theory

Invariant form:

```text
value = order movement in Q
```

not scalar gain.

### Research State Space

Invariant form:

```text
scientific progress = monotone path in Q
```

with raw syntactic paths projected to quotient paths.

### Counterexample Theory

Invariant form:

```text
counterexamples are irreversible failure records under faithful transitions.
```

## 11. Preservation Theorems

### Theorem 1. Faithful Transition Preserves Order Content

If:

```text
S -> S'
```

is faithful, then:

```text
[S'] >= [S]
```

in `Q`.

### Theorem 2. Lossless Morphisms Preserve Equivalence

If `f` is lossless and order-preserving, then:

```text
S ~=_E S' implies f(S) ~=_E f(S')
```

### Theorem 3. Counterexample Records Are Faithful Invariants

If a counterexample record belongs to `F(S)`, then every faithful successor
must contain it or a lossless reinterpretation of it.

### Theorem 4. Refinement Does Not Destroy Prior Content

If `S'` refines `S` faithfully, then the content of `S` appears as a lower
element below `[S']`.

## 12. Open Conjectures

### Conjecture A. Complete Invariant Is Exactly the Quotient State

No smaller representation than `[S] in Q` reconstructs all epistemic content
for arbitrary research states.

### Conjecture B. Counterexamples Generate Irreversible Suborders

The set of states containing a fixed counterexample record is upward closed in
`Q`.

### Conjecture C. Scientific Revolutions Preserve More Than They Rewrite

Large representation changes are still monotone in `Q` if faithful; their
apparent discontinuity is syntactic or stratificational.

### Conjecture D. Lossy Compression Has No Universal Invariant Guarantee

Every lossy compression can destroy some theorem class unless constrained by
the target theorem class.

## 13. Final Verdict

The invariant core of scientific progress is:

```text
monotone movement in the quotient poset Q = RS / ~=_E
```

plus:

```text
monotonic preservation of evidence/failure records under faithful transitions.
```

Complete invariant:

```text
[S] in Q
```

Incomplete but useful invariants:

1. theorem validity up to translation;
2. counterexample records;
3. applicability refinement;
4. order position;
5. translation equivalence class.

Not genuine invariants:

1. theorem syntax;
2. theorem count;
3. scalar confidence;
4. benchmark score;
5. metric distance;
6. fixed representation.

The current framework can therefore be simplified:

```text
Scientific progress = faithful monotone path in Q.
```

Everything else is representation, bookkeeping or optional enrichment.

