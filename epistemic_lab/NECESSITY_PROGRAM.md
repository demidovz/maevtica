# Necessity Program

Status: architecture necessity analysis.

Previous stages are closed:

1. Reduction.
2. Genesis.
3. Representation Equivalence.
4. Representation Completeness.
5. Literature Comparison.

The new question is not:

```text
Which representation is primitive?
```

The new question is:

```text
Which representation classes are mathematically necessary for an epistemic
theory with the target theorem classes?
```

Current architecture:

```text
{R, K, P, D, S, B, G}
```

where:

```text
R = recurrence / presentation
K = admissible configurations
P = compatibility profiles
D = distinctions
S = substitutability preorder
B = boundary representation
G = transformations / invariance
```

No code. No algorithms. No new objectives.

## 1. Target Capability Class

Call a theory `E*`-complete if it can express and prove the following theorem
classes:

1. genesis of admissibility;
2. sharp admissibility;
3. partial admissibility / uncertainty;
4. abstraction as stable role/profile;
5. substitutability;
6. transfer/invariance;
7. representation translation;
8. generalization from presentation;
9. fixed/stable structures;
10. experimental prediction classes.

The necessity question is relative to this capability class. A weaker theory
may omit some representations legitimately.

## 2. Ablation Table

| Removed | Theorem classes lost or weakened | Proofs longer | Translations lost | Predictions lost |
| --- | --- | --- | --- | --- |
| `R` | genesis, generalization from presentation, recurrence-based stabilization | deriving `K` source requires external narrative | `R -> K`, `R -> partial B`, `R -> genesis M` | ambiguity from recurrence, closure predictions, stabilization from accumulated presentations |
| `K` | direct sharp admissibility, non-exit preservation, direct `K -> P/S/D` | admissibility must route through sharp `B` or coherent `P` | `K <-> B`, `K <-> P`, `K -> S/D` unless replacements exist | immediate membership predictions, safe replacement from boundary |
| `P` | direct profile equivalence, context-sensitive substitutability, profile-based abstraction | substitution proofs must expand to `K` membership checks | `P -> S`, `P -> D`, `P <-> K` | context-specific substitution failures and profile equality predictions |
| `D` | no major theorem class if `P` or `K` remains | pairwise distinction claims require projecting from `P` or `K` | direct observational projection `D` removed | cheap pairwise separation predictions |
| `S` | no major theorem class if `P` or `K` remains; pure order reasoning weakened | replacement proofs require profile or `K` expansion | direct preorder translation target removed | cheap replacement-order predictions |
| `B` | underdetermination if partial `B`; sharp boundary notation if sharp `B` | uncertainty proofs must be encoded as families of `K` completions | `partial B <-> completions`, `B <-> K` | predictions of undecidable configurations |
| `G` | invariance, transfer, feature-ablation stability, equivariant reconstruction | every transfer proof must be restated externally | `G + orbit boundary -> K`, `K -> Aut(K)` comparison | transformation invariance and transfer failure predictions |

Immediate conclusion:

```text
D and S are not necessary as stored representations if P or K is retained.
```

They remain proof-efficient views.

By contrast:

```text
R, partial B, G, and at least one admissibility/profile representation are
necessary for E*-completeness.
```

## 3. Independence

Question:

```text
Can each representation be reconstructed from all remaining ones?
```

### 3.1 R From Others

No.

Same final `K`, `P`, `D`, `S`, `B`, `G` can arise from different recurrence
histories:

```text
R1 -> K
R2 -> K
```

where `R1 != R2`.

Thus genesis source is not recoverable from final admissibility structure.

### 3.2 K From Others

Yes, if sharp `B` remains.

Yes, if coherent `P` remains and empty-configuration status is fixed.

No, from `{R,D,S,G,partial B}` without a completion/closure principle.

Therefore `K` as a named representation is replaceable, but the admissibility
content class is not.

### 3.3 P From Others

Yes, from `K`.

Yes, from sharp `B`.

No, from `D` and `S` alone. They lose context content.

### 3.4 D From Others

Yes, from `P` or `K`:

```text
D(a,b) iff P(a) != P(b)
```

Therefore `D` is reconstructible if profile/admissibility content remains.

### 3.5 S From Others

Yes, from `P` or `K`:

```text
S(x,y) iff P(y) subset P(x)
```

Therefore `S` is reconstructible if profile/admissibility content remains.

### 3.6 B From Others

Sharp `B` is reconstructible from `K`.

Partial `B` is not reconstructible from a chosen sharp `K`, because it records
underdetermination among completions.

### 3.7 G From Others

No.

`Aut(K)` can be recovered from `K`, but the transformation class used in
genesis or transfer may be smaller, larger, typed or presentation-preserving in
ways not determined by `K`.

### 3.8 Dependency Graph

```text
R  independent genesis source
|
| with closure
v
partial B  independent underdetermination layer
|
| with completion
v
sharp B <-> K <-> coherent P
                 |       |
                 v       v
                 S       D

G independent transfer/invariance layer
```

## 4. Alternative Bases

### Basis A: `{R, K, G}`

Coverage:

1. genesis via `R`;
2. admissibility via `K`;
3. transfer/invariance via `G`;
4. profiles, distinction and substitutability reconstructed from `K`.

Failure:

Partial underdetermination is not represented unless encoded as a family of
candidate `K`.

Verdict:

```text
E*-complete only if uncertainty is represented as sets/families of K.
```

### Basis B: `{R, B_partial, G}`

Coverage:

1. genesis via `R`;
2. uncertainty via `B_partial`;
3. sharp `K` as completions;
4. transfer via `G`.

Derived:

```text
K, P, S, D
```

from selected completions.

Verdict:

This is a strong alternative basis. It may be more faithful than the original
architecture because sharp `K` becomes a completion, not a primitive layer.

### Basis C: `{P, S}`

Coverage:

Substitution and profile equivalence.

Failure:

No genesis, no underdetermination, no transformation/transfer, no guarantee
that `P` reconstructs `K` unless coherence holds, and no source of profiles.

Verdict:

Not E*-complete.

### Basis D: `{R, G}`

Coverage:

Genesis data and transformations.

Failure:

No admissibility without closure and orbit boundary. No sharp or partial
boundary unless extra principles are added.

Verdict:

Not E*-complete.

### Basis E: `{M}`

Let:

```text
M = (U, Fin(U), R, B_partial, G, Ctx)
```

Coverage:

All current representation classes are recoverable as views or completions.

Verdict:

E*-complete as an integrated architecture, but not a smaller primitive in
content. It packages the independent classes.

### Basis F: `{R, Cl, G, Uc}`

where:

```text
Cl = closure/completion rule
Uc = uncertainty/completion policy
```

Coverage:

Can generate `B_partial`, sharp `K`, profiles, substitution and distinction.

Verdict:

Potential alternative if `Cl` and `Uc` are explicit. It replaces boundary
storage with boundary generation. It is not strictly fewer in mathematical
content.

## 5. Necessary Classes

The current seven representations are not individually necessary.

The necessary classes for `E*`-complete theories appear to be:

### Class C1. Presentation / Genesis

Needed for:

```text
genesis
generalization from observations
closure ambiguity
```

Represented by:

```text
R
```

or an equivalent presentation history/source.

### Class C2. Admissibility Content

Needed for:

```text
inside/outside claims
non-exit
profile derivation
```

Represented by:

```text
K, sharp B, coherent P, or a completion of partial B
```

### Class C3. Underdetermination / Partiality

Needed for:

```text
uncertainty
multiple K completions
non-sharp genesis
```

Represented by:

```text
partial B or family of candidate K
```

### Class C4. Transformation / Invariance

Needed for:

```text
transfer
feature-ablation stability
representation change
```

Represented by:

```text
G
```

or an equivalent transformation/equivariance layer.

### Class C5. Role/Profile or Replacement View

Needed for efficient proofs of:

```text
substitutability
abstraction
hierarchy
```

Represented by:

```text
P, S, or derivation from K
```

This class is necessary as a theorem-view, not necessarily as stored data.

## 6. Necessity Theorem Attempt

### Theorem

Every `E*`-complete epistemic theory must contain, either primitively or
reconstructibly, representations from each of:

```text
C1 presentation/genesis
C2 admissibility content
C3 underdetermination/partiality
C4 transformation/invariance
C5 role/replacement view
```

### Proof Sketch

1. Without C1, two theories with the same final admissibility content but
   different recurrence histories are indistinguishable. Genesis and
   generalization-from-presentation cannot be stated.
2. Without C2, inside/outside status and non-exit preservation cannot be
   stated.
3. Without C3, multiple compatible completions collapse into one sharp boundary,
   so uncertainty and underdetermination cannot be represented.
4. Without C4, transfer across representation or feature change cannot be
   distinguished from arbitrary relabeling.
5. Without C5, substitution, abstraction and hierarchy proofs must be either
   impossible or expanded into lower-level admissibility checks. If C2 is rich
   enough, C5 can be derived, so C5 is necessary as a view, not necessarily as
   independent storage.

Therefore an `E*`-complete theory needs at least these five classes.

### Limit

This theorem does not prove that the exact architecture:

```text
{R,K,P,D,S,B,G}
```

is necessary.

It proves only necessity of representation classes.

## 7. Universality Classification

Current architecture:

```text
{R,K,P,D,S,B,G}
```

is:

```text
sufficient: yes
necessary as exact seven-part decomposition: no
necessary up to representation classes: plausibly yes
minimal: no
```

Why not minimal?

Because:

```text
D and S are reconstructible from P/K.
K and sharp B are equivalent.
P and K are equivalent under coherence.
```

Why sufficient?

Because it contains all five necessary classes:

```text
C1 through C5.
```

## 8. Falsification Search

### Counterarchitecture 1: Partial-Boundary Architecture

```text
A1 = {R, B_partial, G}
```

with:

```text
sharp K = completion(B_partial)
P,S,D = projections from K
```

This violates the original seven-part decomposition but appears `E*`-complete
if completion policies are included.

Result:

```text
The exact architecture is falsified as necessary.
```

### Counterarchitecture 2: Generated-Boundary Architecture

```text
A2 = {R, Cl, G}
```

where `Cl` maps recurrence structures to partial or sharp boundaries.

If `Cl` also represents underdetermination, then `B` is generated rather than
stored.

Result:

This is potentially `E*`-complete, but only because `Cl` absorbs boundary and
uncertainty content.

### Counterarchitecture 3: Profile-Centric Architecture

```text
A3 = {R, P_partial, G}
```

where partial profiles represent uncertain context insertion.

Can recover:

```text
K completions, S, D
```

if coherence conditions hold.

Result:

Another viable alternative. It shows `K` is not necessary as the central
representation.

### Failed Counterarchitecture: Distinction-Centric

```text
A4 = {R, D, G}
```

Fails because `D` cannot reconstruct contexts, admissibility or substitution.

### Failed Counterarchitecture: Substitution-Centric

```text
A5 = {R, S, G}
```

Fails because `S` cannot reconstruct context profiles or sharp admissibility.

## 9. Final Verdict

The current architecture is not mathematically inevitable as an exact list:

```text
{R,K,P,D,S,B,G}
```

It is overcomplete.

However, the Necessity Program supports a weaker theorem:

```text
Any E*-complete theory requires at least one representation from each of:
presentation/genesis,
admissibility content,
underdetermination,
transformation/invariance,
role/replacement view.
```

The strongest current counterexample to exact necessity is:

```text
{R, B_partial, G}
```

plus completion/projection machinery.

This architecture can recover the other representations as views:

```text
B_partial -> K completions -> P -> S,D
```

Therefore the current seven-representation architecture is best understood as:

```text
sufficient and proof-convenient,
but not minimal and not exact-necessary.
```

The next task is to decide whether the class-level necessity theorem can be
made fully rigorous, or whether even the five class decomposition has a
counterexample.

