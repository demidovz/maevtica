# Applicability Theory Program

Status: scientific-method program.

Engine has completed multiple independent falsification programs.

Observed pattern:

1. strong universality repeatedly failed;
2. elegant hypotheses collapsed under broader validation;
3. local structural principles survived;
4. every discovered law appears to have a limited validity domain.

The goal is not to invent another epistemic law.

The goal is to build a formal theory of applicability.

Every theorem should answer:

```text
When is it true?
When does it necessarily fail?
Where is current evidence undecidable?
```

No code. No algorithms. No new objectives.

## 1. Possibility Space

Let an epistemic system be described observationally by:

```text
sigma = (W, I, A)
```

where:

```text
W = observable world/interface properties
I = observable interaction properties
A = observable agent properties
```

This is not implementation detail. It excludes source code, internal hidden
state names, arbitrary feature labels and benchmark-specific identifiers unless
they are observable at the agent-world-interaction level.

Let:

```text
Omega = set of possible observable agent-world-interaction systems sigma
```

A theorem is no longer treated as a universal proposition. It is treated as a
claim with a domain:

```text
T = (statement, domain, evidence, failure_boundary)
```

## 2. Applicability Domain

### Definition

An applicability domain for theorem `T` is a subset:

```text
App(T) subset Omega
```

such that:

```text
sigma in App(T) implies T is expected to hold for sigma
```

The domain must be specified using observable properties of:

1. world/interface;
2. interaction stream;
3. agent behavior/capacity;

not implementation details.

### Four-Zone Version

A mature theorem should specify four regions:

```text
Suf(T) = systems where sufficient conditions are known
Nec(T) = properties known necessary for T
Fail(T) = systems where T necessarily fails
Und(T) = systems not yet classified
```

with:

```text
Suf(T) subset App(T)
Fail(T) disjoint from App(T)
Und(T) = Omega - (Suf(T) union Fail(T)) unless finer evidence is available
```

`Nec(T)` is not a region by itself. It is a constraint that every valid region
must satisfy.

## 3. Boundary Conditions

Every theorem record must include:

```text
Theorem:
Assumptions:
Sufficient conditions:
Necessary conditions:
Failure conditions:
Undecidable region:
Observable variables:
Evidence:
Counterevidence:
Confidence:
```

This format forbids free-floating universality claims.

## 4. Applicability Atlas

| Theorem class | Sufficient conditions | Necessary conditions | Failure conditions | Undecidable region |
| --- | --- | --- | --- | --- |
| distinction | stable profile differences under chosen context domain | observable separation criterion or profile/admissibility representation | all profiles equal; distinction source is arbitrary label | partial profiles or sparse contexts |
| invariants | transformation class `G` and acted-on structure | nontrivial transformation/comparison relation | no comparable transformations; invariance depends on token names | multiple plausible `G` |
| hierarchy | nontrivial profile/order inclusion plus compositional/refinement structure | at least preorder-like role comparison | flat profiles; no composition/refinement | shallow order without compositional evidence |
| reuse | recurrent contexts and identifiable role persistence | recurrence or repeated comparable situations | one-shot unrelated interactions | recurrence present but identity uncertain |
| transfer | `G` plus profile/boundary preservation across transformed contexts | transformation relation and measurable preserved content | representation mismatch; no transport map; anti-invariant change | partial transport evidence |
| representation completeness | explicit representation set plus theorem classes | defined translation/reconstruction targets | lossy representation claimed complete without reconstruction | missing counterexamples for impossible cells |
| semantic roles | stable profile class with context-sensitive substitution | profile/content representation | roles collapse to labels; no profile stability | sparse role contexts |
| natural coordinates | low-loss representation stable under transformations and ablations | observable invariance under coordinate changes | feature-ablation changes theorem truth | partial stability over tested worlds |
| admissibility genesis | recurrence/presentation `R`, invariance, conservative closure | relation among raw interactions | bare unstructured set; token-label injection | multiple closures compatible with evidence |
| substitutability | profile inclusion or non-exit from `K` | acceptability boundary or replacement criterion | no success/failure contrast | partial `K` or unknown contexts |
| abstraction | stable profile/equivalence class across stage changes | stability criterion and context domain | one-stage profile only; no persistence | stability observed locally only |
| fixed points | staged representation plus preservation under stage transition | stage comparability and preservation criterion | static-only theory; no transition relation | limited horizon |

## 5. Failure Taxonomy

Failure is not a single category.

### F1. World / Interface Failure

The world/interface lacks the structure required by the theorem.

Example:

Transfer theorem assumes a transformation relation; the world supplies no
comparable transformed contexts.

### F2. Interaction Failure

The interaction stream does not present enough recurrence, variation or
coverage.

Example:

Hierarchy theorem fails because only one-shot unrelated cases are available.

### F3. Agent Capacity Failure

The world and interaction support the theorem, but the agent cannot represent,
preserve or exploit the required structure.

Example:

Substitution exists in `K`, but the agent cannot track profiles.

### F4. Representation Failure

The chosen representation loses the information required by the theorem.

Example:

`D` cannot prove context-specific substitution because it forgets profiles.

### F5. Optimization Artifact

The theorem appears true only because the optimization objective or search
pressure favors the observed pattern.

Example:

Depth appears as a law but disappears when the objective changes.

### F6. Benchmark Artifact

The theorem holds because of benchmark construction, not because of the
epistemic structure.

Example:

An injected parity-like test creates a clean quotient.

### F7. Theorem Incompleteness

The theorem omits a necessary condition.

Example:

Claiming `R -> K` without closure/invariance principles.

### F8. Boundary Underdetermination

Evidence supports multiple incompatible applicability domains.

Example:

Two closures of the same recurrence data remain possible.

## 6. Epistemic Phase Space

Represent theorem domains inside common possibility space `Omega`.

Each theorem `T` has:

```text
App(T), Fail(T), Und(T)
```

Relations among theorem domains:

```text
T1 includes T2 iff App(T2) subset App(T1)
T1 overlaps T2 iff App(T1) intersect App(T2) != empty
T1 incompatible T2 iff App(T1) intersect App(T2) = empty
T1 boundary-neighbor T2 iff their closures touch in observed property space
```

Do not assume sharp transitions.

Continuous applicability can be represented by:

```text
app_T: Omega -> [0,1]
```

where:

```text
0 = known failure
1 = known applicability
between = graded/uncertain applicability
```

This is not probability by default. It may encode confidence, coverage or
degree of condition satisfaction depending on the theorem record.

## 7. Theorem Dependency Graph

Current dependencies:

```text
raw interactions
  -> recurrence/presentation R
  -> partial boundary B_partial
  -> sharp K completions
  -> profiles P
  -> substitutability S
  -> distinction D

G + acted-on content
  -> invariance
  -> transfer

staged R/K/P + preservation criterion
  -> abstraction
  -> fixed points
  -> hierarchy if composition/refinement exists

representation set + translation operators
  -> representation completeness
  -> necessity analysis
```

Applicability inheritance:

```text
If T2 depends on T1, then App(T2) subset App(T1)
```

unless T2 supplies an alternative proof path.

## 8. Meta-Theorem: No Nontrivial Universal Epistemic Law

### Candidate Statement

No nontrivial epistemic theorem is universal over all possible
agent-world-interaction systems.

### Needed Meaning of Nontrivial

A theorem is nontrivial if it excludes at least one structurally possible
system:

```text
exists sigma in Omega such that theorem content would be false in sigma
```

Pure tautologies are excluded.

### Proof Sketch

Let `T` be a theorem requiring any structural condition:

1. recurrence;
2. admissibility boundary;
3. profile difference;
4. transformation;
5. stage comparison;
6. agent capacity;
7. optimization pressure;
8. representation completeness.

Construct a system `sigma0` lacking that condition:

1. bare unstructured interactions for recurrence-dependent claims;
2. all configurations admitted or none admitted for boundary-dependent claims;
3. all profiles equal for distinction claims;
4. no comparable transformations for transfer claims;
5. static one-stage system for fixed-point/growth claims;
6. agent with insufficient memory for profile tracking.

Then `T` fails or is undefined on `sigma0`.

Therefore no theorem requiring nontrivial structure is universal over all
`Omega`.

### Weakest Assumption

The meta-theorem requires:

```text
Omega contains degenerate systems lacking each candidate structure.
```

If `Omega` is restricted to systems already satisfying a structural axiom, then
universal theorems inside that restricted class may exist.

### Counterexample Class

Tautological representation theorems can be universal:

```text
If K exists, then K defines a boundary.
```

But this is conditional and structural, not universal over all agent-world
systems without `K`.

## 9. Formal Scientific Framework

Every future theorem should be recorded as:

```text
id:
statement:
representation:
assumptions:
observable variables:
applicability domain:
sufficient conditions:
necessary conditions:
failure conditions:
undecidable region:
empirical support:
failed tests:
surviving tests:
known artifacts:
confidence:
revision history:
```

### Confidence

Confidence is not a scalar truth value by default.

Use a structured record:

```text
coverage:
replication:
counterexample_pressure:
artifact_risk:
domain_precision:
```

### Updating

A new experiment can:

1. expand `Suf(T)`;
2. expand `Fail(T)`;
3. shrink `Und(T)`;
4. split one theorem into multiple domain-specific theorems;
5. reclassify failure type;
6. lower confidence due to artifact risk.

## 10. Open Conjectures

### Conjecture A. Locality of Epistemic Laws

Every nontrivial epistemic law has a proper applicability domain:

```text
App(T) proper_subset Omega
```

### Conjecture B. Failure Boundary Informativess

A theorem with a precisely specified failure boundary is scientifically stronger
than a theorem with broader but vague apparent success.

### Conjecture C. Representation-Dependent Domains

Applicability domains differ by representation. The same phenomenon may have a
sharp domain in `K` and only a lossy domain in `D`.

### Conjecture D. Domain Transfer

If a theorem is proven in representation `X` and translated losslessly to `Y`,
then:

```text
App_X(T) = App_Y(T_translated)
```

If translation is lossy:

```text
App_Y(T_projected)
```

may be larger but less predictive.

### Conjecture E. Falsification Improves Domain Precision

Repeated falsification does not merely lower confidence. It refines the
boundary:

```text
Und(T) decreases
Fail(T) and Suf(T) become more structured
```

## 11. Future Falsification Targets

1. Test whether profile-based distinction fails under sparse context coverage.
2. Test whether transfer predictions fail when `G` is misspecified.
3. Test whether hierarchy claims fail in flat but recurrent worlds.
4. Test whether recurrence-derived `K` remains underdetermined under multiple
   closure rules.
5. Test whether representation completeness claims fail when partial boundary
   information is omitted.
6. Test whether natural coordinates survive feature ablation across worlds.
7. Test whether substitutability predicted by `S` fails in contexts hidden by
   the preorder projection.
8. Test whether abstraction stability requires stage comparability or can be
   reformulated without it.

## 12. Success Criterion

The project succeeds when claims are no longer written as:

```text
T is true.
```

but as:

```text
T holds on App(T),
fails on Fail(T),
requires Nec(T),
and remains undecided on Und(T).
```

The goal is not stronger universal theorems.

The goal is stronger science.

