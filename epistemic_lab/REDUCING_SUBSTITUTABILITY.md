# Reducing Substitutability

Status: theoretical reduction attempt.

This document attacks the current primitive candidate:

```text
Sub(a, b | c, u)
```

read as:

```text
a can stand for b in context c under change u
```

The goal is to determine whether substitutability can be derived from something
weaker without assuming preservation, equivalence, consequence or models.

If this fails, we must prove why any definition of substitutability inevitably
contains a preservation relation. Then we must ask whether preservation itself
can be reduced further.

No code. No algorithms. No new objectives.

## 1. Forbidden Starting Points

Do not assume:

1. model spaces;
2. consequences;
3. tests;
4. future;
5. equivalence;
6. preservation;
7. objectives;
8. utility;
9. semantic identity;
10. a pre-given order.

The word "substitute" itself is already suspicious because it suggests
preservation. The reduction must expose whether this is avoidable.

## 2. What Substitutability Must Mean

Any nontrivial substitutability claim has the form:

```text
a may replace b
```

But "replace" is incomplete. Replace for what?

Without a respect in which replacement is licensed, the relation is either:

1. arbitrary permission;
2. physical exchange;
3. syntactic renaming;
4. equality disguised as substitution.

None is epistemic substitutability.

Therefore every meaningful substitution statement has the hidden form:

```text
a may replace b without failure of R
```

where `R` is whatever must remain acceptable.

This is already preservation:

```text
R is preserved under replacing b by a.
```

So the first theorem attempt is negative.

## 3. Impossibility Lemma

### Lemma

There is no nontrivial definition of epistemic substitutability that does not
contain, explicitly or implicitly, a preservation relation.

### Proof Sketch

Assume a relation:

```text
Sub(a, b)
```

with no preserved property, relation, constraint, role, compatibility profile,
behavior, inference, admissibility status or failure condition.

Then for any pair `a, b`, the assertion `Sub(a, b)` cannot distinguish:

```text
a replaces b successfully
```

from:

```text
a replaces b and everything relevant breaks
```

because "breaks" has no defined meaning.

Therefore `Sub(a, b)` can only mean one of:

1. all things substitute for all things;
2. no things substitute for any things;
3. some arbitrary relation names substitute pairs;
4. identity: only `a = b` substitutes for `b`.

Cases 1 and 2 are trivial. Case 3 is injected structure. Case 4 assumes
identity, which is stronger than substitutability and does not explain
abstraction.

Thus nontrivial substitutability requires a criterion of successful replacement.
That criterion is preservation.

QED attempt.

## 4. Can Preservation Be Weakened?

The lemma does not say preservation must mean consequence preservation,
semantic preservation or objective preservation. It only says:

```text
some acceptability condition remains unbroken under replacement.
```

So we reduce:

```text
Substitution -> preservation under replacement
```

Now attack preservation.

Possible weaker candidates:

1. non-breakage;
2. compatibility;
3. invariance;
4. indistinguishability;
5. co-realizability;
6. continuation of admissibility;
7. absence of contradiction.

Invariance and indistinguishability are not weaker; they already contain
preservation. Absence of contradiction is a form of compatibility. The serious
candidate is:

```text
non-breakage of admissible joint realizability
```

## 5. Preservation From Non-Breakage

Preservation can be defined negatively.

Instead of:

```text
a preserves P
```

say:

```text
replacing b by a does not destroy admissible realizability
```

Given a collection `X` containing `b`, replacement produces:

```text
X[b := a]
```

Then:

```text
a substitutes for b over domain Omega
iff
for every X in Omega:
  if X is admissibly realizable,
  then X[b := a] is admissibly realizable.
```

This uses no models, no future, no consequences and no equivalence.

But it uses:

```text
admissible realizability
```

and a replacement operation on configurations.

So preservation has been reduced to:

```text
non-destruction of admissible realizability under replacement
```

## 6. Can Non-Breakage Be Reduced Further?

Non-breakage requires two things:

1. a distinction between realizable and non-realizable configurations;
2. a way to compare original and replacement configurations.

If realizability is removed, "breakage" is undefined.

If replacement comparison is removed, "under replacement" is undefined.

Therefore:

```text
non-breakage = realizability boundary + replacement schema
```

Now attack both.

## 7. Can Realizability Be Reduced?

Realizability says:

```text
configuration X can hold together
```

This is equivalent to joint compatibility.

If we remove realizability entirely, there is no contrast between possible
configuration and impossible configuration. Then there is no failure, no
constraint, no exclusion and no reason one replacement is invalid.

Pairwise compatibility is too weak because higher-order conflicts exist:

```text
{a, b} realizable
{a, c} realizable
{b, c} realizable
{a, b, c} not realizable
```

So the minimal structure is not binary compatibility. It is:

```text
K = admissible jointly realizable finite configurations
```

This matches the previous reduction from consequence.

Verdict:

```text
realizability / joint compatibility is not reducible without making breakage
undefined.
```

## 8. Can Replacement Schema Be Reduced?

Replacement schema says there is a way to form:

```text
X[b := a]
```

from a configuration `X`.

Is this primitive?

Maybe not. Replacement can be derived from positions, occurrences or roles:

```text
configuration = structured occurrence pattern
replacement = same pattern with one occupant changed
```

But this requires at least:

1. distinguishable occurrences or roles;
2. a relation saying `a` can occupy the same slot-type as `b`;
3. preservation of the rest of the configuration.

If no occurrence/role structure exists, replacement collapses into simply
adding `a` and removing `b`, which is still a primitive transformation on
configurations:

```text
(X - {b}) union {a}
```

For unstructured finite sets, this is enough. For structured configurations,
one needs slots or incidence.

Verdict:

```text
some contrast operation between original and modified configuration is
necessary.
```

It need not be called replacement, but without it substitutability cannot be
stated.

## 9. First Irreducible Axiom Candidate

At this point the reduction reaches:

```text
K = admissible joint realizability of configurations
Delta = admissible local variation of configurations
```

where `Delta(X, X')` means:

```text
X' is an allowed variant of X
```

This is weaker than substitutability because it does not say the variant
preserves anything. It only says which local variations are even comparable.

Then substitutability can be derived:

```text
Sub(a, b | Omega)
iff
for all X in Omega:
  if b occurs in X and X in K,
  then there exists X' such that
    Delta_b_to_a(X, X') and X' in K
```

Informally:

```text
a substitutes for b when every admissibly realizable configuration in the
comparison domain remains admissibly realizable after the allowed local
variation replacing b by a.
```

This uses preservation only as derived non-exit from `K`.

## 10. Is This Still Preservation?

Yes.

The definition says:

```text
membership in K is preserved under Delta.
```

So preservation has not disappeared. It has been reduced to the most minimal
form:

```text
non-exit from an admissible realizability boundary under allowed variation
```

Any further attempt to remove preservation removes the contrast between staying
inside and leaving `K`.

Then substitutability becomes arbitrary.

## 11. Can K Be Removed?

Suppose there is no `K`, no admissible realizability boundary.

Then every variant is equally acceptable or acceptability is undefined.

If every variant is acceptable:

```text
everything substitutes for everything
```

If acceptability is undefined:

```text
substitution has no success condition
```

If an arbitrary subset is later chosen as acceptable:

```text
admissibility has been reintroduced under another name
```

Verdict:

```text
K cannot be removed.
```

## 12. Can Delta Be Removed?

Suppose there is no local variation relation.

Then there is no formal connection between `X` and the alleged replacement
configuration `X'`.

One may still compare profiles:

```text
profile(a) superset profile(b)
```

But a profile is already defined through all contexts in which adding `a` or
`b` remains in `K`. This hides variation by comparing two possible insertions
into the same background.

So Delta can sometimes be derived from shared context:

```text
C union {b}  ->  C union {a}
```

For unstructured configurations, explicit Delta is not primitive.

For structured configurations with roles or typed slots, Delta is necessary
unless role structure is encoded into `K`.

Verdict:

```text
Delta is not always primitive. It can be derived from shared-context
replacement in unstructured finite configuration spaces.
```

The irreducible core is therefore not `Delta` itself, but:

```text
the ability to compare two configurations as alternatives over a common
background.
```

Call this:

```text
Alt_C(a, b)
```

where `a` and `b` are alternative insertions into context `C`.

## 13. Reduced Definition

The deepest non-circular definition found so far:

```text
Sub(a, b | C-domain)
iff
for every context C in C-domain:
  if C union {b} in K,
  then C union {a} in K
```

This is profile inclusion:

```text
profile(b) subset profile(a)
```

where:

```text
profile(x) = {C : C union {x} in K}
```

No consequence, model, equivalence, future, test or objective is assumed.

But preservation remains as:

```text
membership in K is preserved when replacing b by a across contexts.
```

So:

```text
substitutability = universal non-breakage of admissible joint realizability
under common-context variation
```

## 14. Circularity Boundary

Try to reduce further:

```text
K membership preservation
```

to something weaker.

Options:

1. compatibility;
2. consistency;
3. possibility;
4. non-contradiction;
5. co-instantiability;
6. admissibility.

But these are all names for the same boundary:

```text
which configurations can hold together
```

If "can hold together" is removed, the theory has no negative space. Nothing
can fail. If nothing can fail, replacement cannot be successful or unsuccessful.
If replacement has no success/failure contrast, substitutability is undefined.

Therefore the reduction becomes circular exactly here:

```text
substitutability
  -> preservation
  -> non-breakage
  -> staying inside admissible realizability
  -> admissible realizability
```

Attempting to define admissible realizability via substitutability would close
the circle:

```text
K = configurations stable under allowed substitutions
```

That is not a reduction. It presupposes what it tries to define.

## 15. Final Minimal Axiom

The first genuinely irreducible axiom encountered is:

```text
There exists a nontrivial boundary between jointly realizable and
non-realizable configurations.
```

More carefully:

```text
K is a nontrivial family of admissibly realizable finite configurations.
```

Nontrivial means:

```text
some configurations are in K;
some configurations are not in K;
K is not selected post-hoc to encode a desired substitution relation.
```

This axiom is weaker than consequence, weaker than substitution, weaker than
equivalence and weaker than modelhood.

But it is not eliminable. Without it, no replacement can break or preserve
anything.

## 16. Updated Hierarchy

The current reduction hierarchy is:

```text
admissible joint realizability K
  -> compatibility profiles
  -> preservation as non-exit from K
  -> substitutability as profile inclusion
  -> equivalence as mutual substitutability
  -> distinction as profile difference
  -> consequence as profile change / exclusion
  -> quotient as profile-equivalence classes
  -> compression as smaller substitute
  -> transport as substitutability across allowed representation changes
  -> dynamics as changing K or moving through K-indexed transformations
```

This suggests the project should stop treating substitutability as primitive.

The deeper primitive is:

```text
admissible joint realizability
```

with the unresolved problem:

```text
Can admissibility be derived without reintroducing arbitrary injection?
```

## 17. Verdict

Substitutability is not primitive.

It can be derived as:

```text
Sub(a, b | Omega)
iff profile_Omega(b) subset profile_Omega(a)
```

where profiles are generated from admissible joint realizability:

```text
profile_Omega(x) = {C in Omega : C union {x} in K}
```

However, every meaningful definition of substitutability contains preservation
in at least this minimal form:

```text
non-exit from K under replacement / common-context variation
```

Preservation itself reduces to:

```text
staying inside an admissible realizability boundary.
```

Further reduction becomes circular if it tries to define that boundary using
substitutability, preservation, consequence or equivalence.

Therefore the current irreducible axiom is:

```text
Nontrivial admissible joint realizability exists.
```

