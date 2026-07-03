# Deriving Consequence From Compatibility

Status: theoretical reduction attempt.

This document attacks the word `consequence`.

The previous stress test concluded that the deepest current candidate was:

```text
Sub(a, b | c, u)
```

read as contextual substitutability under admissible change.

But this still hides a consequential notion: `a` can stand for `b` because
something is preserved. The question here is whether "consequence" can be
derived from a weaker relation that does not assume time, causality, tests or
future interaction.

The target is not to invent a new known formalism. The target is to ask what the
weakest pre-consequential relation could be.

No code. No algorithms. No new objectives.

## 1. Forbidden Starting Points

Do not assume:

1. time;
2. future;
3. causality;
4. tests;
5. prediction;
6. agents;
7. utility;
8. update rules;
9. model spaces;
10. logical entailment.

Logical entailment already contains consequence. Starting there would be
circular.

## 2. Candidate Weaker Relations

Possible pre-consequential primitives:

```text
compatibility
consistency
co-satisfiability
constraint satisfaction
exclusion
co-instantiability
realizability
```

These are not identical, but they share a minimal form:

```text
some configurations can coexist;
some configurations cannot coexist.
```

This suggests a very weak primitive:

```text
Comp(x, y)
```

read:

```text
x is compatible with y
```

or equivalently:

```text
Incomp(x, y)
```

read:

```text
x excludes y
```

At this level, no one says that `x` causes `y`, predicts `y`, entails `y`, is
tested by `y`, or occurs before `y`. There is only coexistence or exclusion
inside some possible configuration.

## 3. Why Bare Compatibility Is Too Weak

Bare pairwise compatibility is not enough.

Counterexample:

```text
Comp(a, b)
Comp(b, c)
not Comp(a, c)
```

Pairwise compatibility does not determine whether larger collections can
coexist.

Therefore the primitive cannot merely be binary compatibility unless all
higher-arity compatibility is derivable. The weaker object should be:

```text
K = set of jointly realizable finite configurations
```

where:

```text
X in K means X is jointly compatible / realizable
```

No order, topology, time, causality or test family is assumed.

## 4. Deriving Constraint From Compatibility

Given `K`, a constraint is not primitive.

A constraint is any condition that separates realizable from non-realizable
configurations.

For finite configurations:

```text
constraint r holds over X iff X belongs to a selected region of K
```

or negatively:

```text
r excludes X iff X notin K_r
```

Thus:

```text
constraints = boundaries in compatibility space
```

This is weaker than consequence. It only says what can coexist.

## 5. Deriving Distinction

A distinction appears when two elements have different compatibility profiles.

Define:

```text
profile(x) = {Y : Y union {x} in K}
```

Then:

```text
x and y are distinguishable iff profile(x) != profile(y)
```

This derives distinction without tests or future.

The distinction is purely structural:

```text
x differs from y because substituting one for the other changes what can
coexist with it.
```

This is the first point where epistemic content appears.

## 6. Deriving Consequence

Now consequence can be reconstructed as preservation or exclusion over
compatibility profiles.

### 6.1 Positive Consequence

`a` has `b` as a consequence if every realizable configuration containing `a`
also supports `b` in the relevant compatibility sense.

One non-temporal form:

```text
a => b iff for every X, if X union {a} in K, then X union {a, b} in K
```

This is not time, causality or prediction. It says:

```text
adding b never breaks a-compatible configurations.
```

But this is still weak. It makes harmless additions consequences. A second
condition is needed:

```text
b is not arbitrary relative to a
```

That requires relevance.

### 6.2 Negative Consequence

`a` excludes `b` if no realizable configuration contains both:

```text
a # b iff {a, b} notin K
```

This is the cleanest pre-temporal consequence:

```text
to accept a is to rule out b
```

No future is required. Consequence begins as exclusion.

### 6.3 Contextual Consequence

Absolute consequence is too strong. Most epistemic consequences are contextual.

Define:

```text
a =>_C b iff for every X compatible with C,
if X union C union {a} in K,
then X union C union {a, b} in K
```

and:

```text
a #_C b iff C union {a, b} notin K
```

Now consequence is derived from compatibility relative to a context `C`.

This is important: `C` is not a future test. It is just a configuration
background.

## 7. Deriving Substitutability

Once compatibility profiles exist, substitutability can be derived.

`a` can substitute for `b` in context `C` if every compatibility relation needed
by `b` in `C` is preserved by `a`.

Profile form:

```text
a >=_C b iff profile_C(b) subset profile_C(a)
```

where:

```text
profile_C(x) = {Y : C union Y union {x} in K}
```

Equivalence is mutual substitutability:

```text
a ~=_C b iff profile_C(a) = profile_C(b)
```

Thus the earlier candidate:

```text
Sub(a, b | c, u)
```

can be reduced to:

```text
profile comparison in a compatibility structure after a context change
```

At this point `consequence` has not been assumed. It has been reconstructed as
profile preservation and exclusion.

## 8. Where Admissibility Enters

Compatibility alone admits arbitrary elements.

If any symbol can be added to the universe, then arbitrary labels again become
structure.

Therefore admissibility cannot be derived from compatibility alone unless the
compatibility structure itself includes a boundary on allowable configurations.

There are two options:

### Option A: Admissibility Is Primitive

```text
K_adm subset K
```

Only configurations in `K_adm` count epistemically.

This keeps the theory clean but makes admissibility independent.

### Option B: Admissibility Is Closure Under Compatibility

A configuration is admissible if it is generated by repeated compatible
extension from a seed set.

This is tempting, but dangerous. It can smuggle arbitrary seeds and arbitrary
extension rules back into the theory.

### Verdict

Compatibility can derive consequence, but not admissibility.

Some admissibility boundary still appears necessary to prevent arbitrary
injection.

## 9. Where Dynamics Enters

The compatibility structure `K` is static.

Can dynamics be derived from it?

Only if change is represented internally as compatibility with change markers:

```text
K contains configurations like {state_1, state_2, transition_marker}
```

But this encodes dynamics as structure. It does not derive dynamics from
compatibility alone unless temporal or transformation roles are already present
as elements.

Therefore:

```text
static compatibility derives static consequence;
dynamic consequence requires compatibility across transformations.
```

A weaker dynamic primitive would be:

```text
K_U = compatibility structure indexed by transformations U
```

or:

```text
Realizable(a, b, u, C)
```

where `u` is not necessarily time. It is any allowed change of configuration.

## 10. Minimal Reduction

The deepest current reduction is:

```text
consequence = compatibility-profile invariance / exclusion
```

More explicitly:

```text
Given a realizability structure K,
the consequences of x in context C are the changes in realizability
forced by adding x to C.
```

So:

```text
Cons(x | C) = profile_C(x)
```

or, more exactly:

```text
Cons(x | C) = difference between profile_C(empty) and profile_C(x)
```

This avoids future, tests and causality.

## 11. Destruction Tests

### 11.1 Remove Compatibility

If no compatibility or exclusion relation exists, there is no way to say that
one structure constrains another.

Then all collections are equally possible, or possibility is undefined.

Consequence cannot be recovered.

Verdict: compatibility or exclusion is necessary.

### 11.2 Use Only Pairwise Compatibility

Pairwise compatibility fails for higher-order constraints.

Example:

```text
a compatible with b
a compatible with c
b compatible with c
but {a, b, c} is not jointly realizable
```

Many epistemic constraints are higher-order. Therefore pairwise compatibility is
not enough.

Verdict: the primitive must handle finite joint realizability, not just binary
compatibility.

### 11.3 Remove Context

Without context, consequence becomes absolute.

But most distinctions matter only under background constraints. Removing
context forces either false universality or trivial fragmentation.

Verdict: context is necessary, but it need not be temporal or agent-relative.

### 11.4 Remove Relevance

Compatibility profiles can be huge. Many compatible additions are irrelevant.

If relevance is not derived or bounded, consequence becomes bloated:

```text
everything harmless becomes a consequence
```

Verdict: relevance is not solved by compatibility. It must be derived from
admissibility, compression, invariance or another boundary.

### 11.5 Remove Admissibility

Arbitrary symbols can be introduced with arbitrary compatibility profiles.

Then any desired consequence can be manufactured.

Verdict: admissibility remains independent unless generated by a non-arbitrary
source.

## 12. Revised Primitive Candidate

The new candidate is not "consequence".

It is:

```text
joint realizability under admissible context
```

or:

```text
K_adm = admissible finite compatibility structure
```

From it:

```text
distinction = difference of compatibility profiles
consequence = profile change / exclusion under adding a structure
substitution = profile inclusion
equivalence = profile equality
quotient = classes of profile equality
compression = replacement by smaller profile-preserving structure
```

To recover the full previous theory, one still needs:

```text
allowed transformations U
```

Then:

```text
transport = profile preservation across U
dynamics = path through compatibility structures or compatibility under U
open growth = expansion/revision of K_adm across U
```

## 13. Final Verdict

Yes: `consequence` can be derived from a weaker structure.

The weaker structure is not bare compatibility and not mere consistency. It is:

```text
admissible joint realizability of finite configurations
```

Consequence is then:

```text
the compatibility-profile change forced by adding a structure to a context
```

This derivation avoids assuming time, causality, tests or future.

However, two things are not eliminated:

1. admissibility;
2. context.

And for open-ended epistemic growth, a third thing must be added:

```text
allowed transformation / change
```

The new hierarchy is therefore:

```text
joint realizability
  -> compatibility profiles
  -> distinctions
  -> contextual consequence
  -> substitutability
  -> quotients / compression
  -> transport and dynamics under allowed transformations
```

The next question is whether admissibility itself can be derived from
realizability, or whether it is the true irreducible boundary of epistemic
structure.

