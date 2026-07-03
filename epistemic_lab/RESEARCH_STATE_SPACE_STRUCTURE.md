# Structural Theory of Research State Space

Status: mathematical structure program.

Applicability Theory and Epistemic Value Theory are considered complete.

Research states:

```text
S = (T, A, E, F, U, R)
```

where:

```text
T = theorem records
A = applicability atlas
E = empirical support records
F = failed tests / counterexamples
U = undecidable regions
R = representation and translation records
```

Epistemic improvement:

```text
S' >=_E S
```

The objective is not to construct another epistemic theory.

The objective is to identify the mathematical object:

```text
(ResearchStates, >=_E)
```

without assuming it is merely a partially ordered set.

No code. No algorithms. No new objectives.

## 1. Base Object

Let:

```text
RS = all admissible research states
```

and:

```text
S' >=_E S
```

mean:

```text
S' is epistemically no worse than S
```

according to Applicability Theory and Epistemic Value Theory.

Because value is not assumed scalar, the structure begins as a relation, not a
metric or numeric landscape.

## 2. Order-Theoretic Analysis

### 2.1 Reflexivity

Expected:

```text
S >=_E S
```

because a state is no worse than itself.

If reflexivity fails, `>=_E` is not an improvement relation but a strict
preference. The non-strict relation should be reflexive.

### 2.2 Transitivity

Expected under stable accounting:

```text
S2 >=_E S1 and S1 >=_E S0 implies S2 >=_E S0
```

This requires evidence accounting not to erase failures or hidden assumptions.

Counterexample if accounting is unstable:

```text
S1 improves S0 by adding evidence.
S2 improves S1 by changing representation.
The representation change drops the evidence record.
```

Then transitivity fails.

Therefore transitivity requires monotonic evidence accounting or faithful
translation of prior records.

### 2.3 Antisymmetry

Antisymmetry need not hold on raw states.

Two research states may be mutually no worse:

```text
S1 >=_E S2 and S2 >=_E S1
```

while differing syntactically:

```text
different theorem names
different but losslessly translated representations
different proof ordering
```

Thus raw `RS` is at best a preorder.

Define epistemic equivalence:

```text
S1 ~=_E S2 iff S1 >=_E S2 and S2 >=_E S1
```

Then:

```text
RS / ~=_E
```

is a partial order if transitivity holds.

### Verdict

The strongest default structure is:

```text
preorder on raw states
poset on epistemic-equivalence classes
```

## 3. Lattice Questions

### 3.1 Join: Least Common Refinement

Given independent states:

```text
S1, S2
```

a join is:

```text
S1 join S2 = least state S such that S >=_E S1 and S >=_E S2
```

Interpretation:

```text
least common refinement / merge of research results
```

### Join Existence Conditions

A join exists if:

1. theorem records can be unioned or reconciled;
2. evidence records are compatible or conflict is explicitly represented;
3. applicability atlases can merge by refining domains;
4. representation records can include both translation systems;
5. contradictions do not force collapse, or are recorded as failures/undecided
   regions.

If conflicts are representable, joins often exist by adding conflict records.

If conflicts are forbidden, joins may fail.

### Join Counterexample

`S1` records:

```text
T holds on domain A
```

`S2` records:

```text
T necessarily fails on overlapping domain A
```

If the state formalism cannot represent contradiction or split `T`, there is no
least common refinement.

If it can split:

```text
T_A_support
T_A_fail
artifact_or_assumption_conflict
```

then a join may exist.

### 3.2 Meet: Greatest Common Abstraction

A meet is:

```text
S1 meet S2 = greatest state S such that S1 >=_E S and S2 >=_E S
```

Interpretation:

```text
common content retained by both states
```

### Meet Existence Conditions

Meet exists if research states have a well-defined common projection:

1. shared theorem records can be identified;
2. common evidence can be intersected;
3. common applicability domains can be computed;
4. representation differences can be quotient-translated.

Meet fails when there is no canonical alignment between theorem vocabularies or
representations.

### Lattice Verdict

`RS / ~=_E` is not necessarily a lattice.

It becomes a join-semilattice if conflict-preserving merges are always
admissible.

It becomes a meet-semilattice if common-content projections are always
admissible and canonical.

It becomes a lattice only if both conditions hold.

## 4. Completeness and Domains

### Complete Lattice

A complete lattice requires arbitrary joins and meets.

This is too strong by default because arbitrary collections of research states
may contain incompatible vocabularies, unbounded evidence records or no
canonical common abstraction.

### Directed Complete Partial Order

A directed set of states is a family where every finite subset has an upper
bound.

If states are ordered by cumulative evidence/refinement and all directed
unions are admissible, then:

```text
RS / ~=_E
```

is a dcpo.

This requires:

1. directed evidence unions exist;
2. theorem records have limits;
3. applicability domains have limiting regions;
4. representation translations are coherent along the directed system.

### Domain

A domain structure requires approximation:

```text
finite or compact research states approximate larger states
```

This is plausible if finite theorem/evidence records approximate infinite
research programs.

But it is not automatic. It requires a way to say that a finite state is a
compact approximation.

### Verdict

The strongest justified structure under reasonable scientific closure axioms:

```text
dcpo-like preorder quotient with partial joins for compatible programs
```

Not a complete lattice by default.

## 5. Geometry

No Euclidean geometry is intrinsic.

Possible geometries arise from additional structure.

### 5.1 Topology

Natural topology:

```text
basic open set = states containing a finite research record r
```

This is an information/order topology.

Convergence:

```text
S_n -> S
```

means finite parts of `S` eventually appear in `S_n`.

### 5.2 Metric Geometry

A metric requires a distance:

```text
d(S1,S2)
```

No canonical metric exists unless the theory chooses weights over theorem
records, evidence, failures and translations.

Thus metric geometry is noncanonical.

### 5.3 Simplicial / Hypergraph Geometry

If theorem records are treated as vertices and jointly consistent subsets as
simplices, each state induces a complex:

```text
K_S = compatible theorem/evidence subcomplex
```

This is useful for conflict and compatibility, not intrinsic to `>=_E`.

### 5.4 Stratified Space

Research states naturally stratify by:

1. representation architecture;
2. theorem vocabulary;
3. failure taxonomy;
4. confidence/evidence type;
5. degree of boundary sharpness.

Transitions that change vocabulary or split theorems move between strata.

### 5.5 Sheaf-Like Structure

Applicability Theory suggests a sheaf-like view:

```text
local theorem records over domains in Omega
gluing = consistency of theorem records on overlaps
```

This is not assumed as category theory. It is a structural analogy:
local applicability claims must agree on overlaps to form a global research
state.

### Geometry Verdict

The most natural intrinsic geometry is:

```text
order/topological geometry of information refinement
```

with optional stratified and sheaf-like enrichments.

Metric geometry is not intrinsic.

## 6. Paths

A scientific history is a path:

```text
S0 -> S1 -> S2 -> ...
```

where each transition is an experiment, conjecture, proof, counterexample,
translation or revision.

### Can Incomparable Paths Merge?

Yes, if a join exists.

Two programs:

```text
S0 -> S1
S0 -> S2
```

merge at:

```text
S3 >=_E S1, S2
```

when their records can be reconciled.

### Can Loops Occur?

In raw states, yes:

```text
renaming
translation
temporary reformulation
```

In the quotient poset `RS / ~=_E`, nontrivial improvement loops cannot occur if
`>=_E` is antisymmetric after quotienting.

Cycles then represent equivalence, not progress.

### Can Irreversible Transitions Exist?

Yes.

Adding a valid counterexample or failed test is irreversible under monotonic
evidence accounting:

```text
F' contains F plus new failure
```

The interpretation may change, but the record remains.

### Can Dead Ends Exist?

Yes.

A state is a dead end relative to a resource or representation class if no
available transition improves it without unacceptable loss.

Dead ends may disappear after changing representation or adding resources.

## 7. Boundary Objects

### Minimal States

A minimal state contains only the empty research record or a fixed initial
schema.

Not unique if different initial vocabularies are incomparable.

### Maximal States

A maximal state would settle all theorem domains, failures, translations and
evidence.

Usually not expected to exist.

If it exists, it is relative to a bounded possibility space and theorem
language.

### Fixed Points

A state `S` is fixed under a research operator or policy `F` if:

```text
F(S) ~=_E S
```

Intrinsic fixed points require a specified transition class. Without a policy
or transition relation, fixed point is not intrinsic.

### Frontier States

A frontier state has many valuable outgoing transitions and large structured
`Und(T)`.

This is intrinsic only relative to the available transition set.

### Singularities

A singularity is a state where a small research transition forces a large
reorganization:

1. theorem split;
2. representation change;
3. failure taxonomy revision;
4. collapse of a universal claim.

This is visible as a nonlocal jump in dependency graph or applicability atlas.

### Bifurcation Points

A bifurcation point is a state with two or more incomparable high-value
continuations that cannot yet be merged.

This formalizes research branching.

## 8. Morphisms

A morphism between research programs should preserve:

1. theorem records or their translations;
2. applicability domains;
3. evidence/failure accounting;
4. improvement order;
5. representation translation structure.

Define:

```text
f: RS1 -> RS2
```

as structure-preserving if:

```text
S' >=_E S implies f(S') >=_E f(S)
```

and theorem/evidence records translate without loss or with declared loss.

### Equivalence of Histories

Two histories are equivalent if there is a pair of morphisms preserving their
state sequence up to `~=_E`.

### Simulation

History `H1` simulates `H2` if every improvement step in `H2` has a
corresponding no-worse step in `H1`.

### Composition

Histories compose when the terminal state of one translates into the initial
state of another, or when their joins exist.

## 9. Meta-Theorem

### Candidate

Every admissible scientific process can be represented as a path inside a
research-state structure:

```text
(RS, >=_E, transitions)
```

### True Under Assumptions

This holds if the process can record:

1. theorem-like claims;
2. evidence and failures;
3. applicability domains or uncertainty;
4. representation translations;
5. state transitions.

### Counterexample Classes

Processes not representable without loss:

1. tacit skill practices with no recordable theorem/evidence structure;
2. purely aesthetic exploration with no applicability claims;
3. destructive processes that erase failed tests with no trace;
4. systems whose state identity cannot be compared across transitions.

Thus universality requires a broad definition of "scientific process" as
record-bearing and applicability-oriented.

## 10. Consequences for Prior Theories

### Applicability Theory

Applicability Theory is local coordinate structure on `RS`:

```text
each S contains an atlas A
```

### Epistemic Value Theory

Epistemic Value Theory provides the order:

```text
>=_E
```

### Research Policy

Research Policy studies admissible outgoing paths from a state:

```text
S -> possible S'
```

and dominance among them.

### Counterexample Theory

Counterexample Theory studies irreversible transitions that expand `Fail(T)`
and refine applicability boundaries.

### Representation Completeness

Representation Completeness studies strata of `RS` indexed by representation
architecture and translation power.

## 11. Formal Structure Theorem

### Theorem Attempt

Under:

1. reflexive no-worse comparison;
2. transitive monotonic evidence accounting;
3. quotienting by lossless representation equivalence;
4. admissibility of conflict-preserving joins for compatible programs;
5. directed unions of coherent cumulative programs;

the research-state space is:

```text
a dcpo-enriched join-semilattice of epistemic-equivalence classes,
with an information topology and optional stratified/sheaf-like structure.
```

### Without These Assumptions

Only the weaker statement holds:

```text
raw research states form a preorder under >=_E.
```

## 12. Open Mathematical Conjectures

### Conjecture A. Directed Scientific Programs Have Suprema

Every directed, coherence-preserving research program has a supremum state.

### Conjecture B. Counterexamples Are Irreversible Compact Events

Valid counterexample records are compact elements in the research-state domain.

### Conjecture C. Scientific Revolutions Are Stratum Changes

Large theory changes correspond to transitions between representation strata,
not merely long paths inside one stratum.

### Conjecture D. Local Theorem Gluing

Compatible local applicability records glue into global research states exactly
when their overlap restrictions agree.

### Conjecture E. Dead Ends Are Relative, Not Absolute

Every dead end is relative to a transition/resource/representation class.

## 13. Final Verdict

The mathematical nature of research-state space is more precise than:

```text
a partially ordered set.
```

Default:

```text
preorder of raw states
```

After quotienting by epistemic equivalence:

```text
poset of research-state content
```

With scientific closure assumptions:

```text
dcpo-like join-semilattice with information topology
```

With applicability-domain localization:

```text
stratified / sheaf-like refinement structure
```

The strongest justified object is therefore:

```text
an order-topological, partially join-complete space of research-state
equivalence classes, enriched by strata of representation architecture and
local applicability records.
```

