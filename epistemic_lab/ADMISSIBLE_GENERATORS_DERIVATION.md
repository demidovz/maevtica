# Can Admissible Generators Be Derived?

This note continues after the first falsification and the robustness analysis.

Question:

```text
Can a reasonable class of admissible test generators be derived from more
general principles, or must it be stipulated?
```

Short answer:

```text
From a bare state set, no.
From a structured interaction world, partially yes.
```

The strongest defensible conclusion is:

```text
Admissible generators cannot be derived from "state space" alone.
They can be constrained by invariance of interaction structure.
The weakest derived constraint is naturality/equivariance under automorphisms of
the base world that are fixed before candidate quotients are considered.
```

This does not give all admissible generators. It gives the minimal filter that
rules out arbitrary injected tests such as `parity`.

## 1. Impossibility From Bare State Space

Let `S` be a finite set with no additional structure.

A test is:

```text
t: S -> R
```

A generator is any rule that produces tests on `S`.

If `S` has no structure, every permutation of `S` is an automorphism. Therefore
any intrinsic construction from `S` alone must be invariant under the full
symmetric group:

```text
Sym(S)
```

But under `Sym(S)`, no nontrivial state can be distinguished from any other.
The only invariant partitions are:

```text
{S}
```

and, if equality itself is allowed as primitive:

```text
{{s}: s in S}
```

There is no canonical way to derive:

```text
{{A0, A1}, {B0, B1}}
```

or:

```text
{{A0, B0}, {A1, B1}}
```

from the set alone.

### Bare-State Impossibility Proposition

Let `F` be any rule that assigns to each finite set `S` a family of tests
`F(S)`, using no structure beyond equality of elements. If `F` is invariant
under bijections of finite sets, then `F(S)` cannot canonically select any
nontrivial proper partition of `S`.

Reason:

For any two states `s, s' in S`, there is a bijection swapping them. An
intrinsic rule must commute with that bijection. Therefore it cannot prefer a
test that separates `s` from `s'` unless it also admits all permutations of that
test. The orbit of such a test contains arbitrary labelings. This gives no
non-arbitrary admissibility filter.

Conclusion:

```text
No reasonable admissible generator class can be derived from S alone.
```

Some structure must be supplied.

## 2. What Structure Is Minimally Needed?

A world is not just a set. The smallest world that can constrain tests is:

```text
W = (S, delta, omega)
```

where:

```text
S      = finite state space
delta  = transition structure
omega  = observation/interface map
```

If actions exist, use:

```text
W = (S, A, delta, omega)
```

For the first experiment, the base world was:

```text
S = {A0, A1, B0, B1}
delta(A0)=A1
delta(A1)=A0
delta(B0)=B1
delta(B1)=B0
omega(A0)=0
omega(B0)=0
omega(A1)=1
omega(B1)=1
```

This structure has automorphisms: permutations of `S` that preserve `delta` and
`omega`.

The admissibility filter should be derived from these automorphisms.

## 3. Interaction Invariance Principle

Minimal principle:

```text
A test generator is admissible only if it is invariant under automorphisms of
the base interaction structure.
```

More precisely, let `Aut(W)` be the automorphism group of world `W`. A generator
`G` is admissible only if:

```text
G(g.W) = g.G(W)
```

for every structure-preserving isomorphism `g`.

For a fixed world:

```text
T in G(W) implies g.T in G(W) for every g in Aut(W)
```

This is equivariance.

Interpretation:

The generator may use the world structure, but it may not use arbitrary names of
states. If two states are indistinguishable by the base interaction structure,
the generator cannot separate them by label fiat.

## 4. Does This Rule Out Parity?

In the first world, consider the automorphism:

```text
sigma(A0)=B0
sigma(B0)=A0
sigma(A1)=B1
sigma(B1)=A1
```

This swaps hidden modes while preserving:

```text
delta
omega
```

The arbitrary parity test was:

```text
parity(A0)=red
parity(A1)=blue
parity(B0)=blue
parity(B1)=red
```

Under `sigma`, it becomes:

```text
parity(sigma(A0)) = parity(B0) = blue
```

but:

```text
parity(A0) = red
```

So `parity` is not invariant under the mode-swap automorphism of the observed
interaction structure.

Therefore:

```text
parity is not derivable from (S, delta, omega)
```

It can be admitted only by adding extra structure that breaks the symmetry.

That is exactly the desired result: if an instrument directly observes an extra
hidden property, that property must be added to the world/interface structure.
It cannot appear for free.

## 5. Does This Exclude Natural Concepts?

No, if the concept is genuinely supported by the interaction structure.

Observation-phase quotient:

```text
{{A0, B0}, {A1, B1}}
```

is derived from `omega`, so it is admissible relative to `(S, delta, omega)`.

Hidden-mode quotient:

```text
{{A0, A1}, {B0, B1}}
```

is not derivable from `(S, delta, omega)` because the two modes are perfectly
symmetric and observationally indistinguishable.

This is not a bug. It is the correct conclusion:

```text
hidden mode is not an admissible concept unless the interaction structure
contains a mode-sensitive test, intervention, or asymmetry.
```

If we add a mode-sensitive instrument:

```text
mu(A0)=A
mu(A1)=A
mu(B0)=B
mu(B1)=B
```

then the base world becomes:

```text
W' = (S, delta, omega, mu)
```

Now hidden mode is part of the interaction interface. The admissible generator
may derive the mode quotient from `mu`.

So the criterion does not exclude natural concepts. It says:

```text
concepts are relative to available interaction structure.
```

## 6. Why Invariance Alone Is Not Enough

Equivariance is necessary but not sufficient.

If a generator includes every orbit of every arbitrary test under `Aut(W)`, it
is equivariant but still admits arbitrary junk.

Example:

Let `G(W)` be:

```text
all tests S -> {0,1}
```

This class is invariant under automorphisms. But it includes parity.

Therefore the derived principle must be:

```text
tests must be constructible from the interaction structure by allowed
structure-preserving operations.
```

Not merely closed under automorphisms.

## 7. Constructibility From Interaction Structure

Define the base observable algebra generated by the world:

```text
Obs(W)
```

For the first deterministic world without actions, the primitive observable is:

```text
omega
```

and time-shifted observations:

```text
omega(delta^k(s))
```

for `k >= 0`.

An admissible test is any finite function constructible from a finite tuple:

```text
(omega(delta^0(s)), omega(delta^1(s)), ..., omega(delta^n(s)))
```

by a relabeling-invariant finite operation.

So:

```text
t(s) = f(omega(s), omega(delta(s)), ..., omega(delta^n(s)))
```

where `f` is fixed independently of state labels.

This is a derivation from interaction:

```text
world dynamics + observation interface -> admissible tests
```

In the base world, this generates only phase/observation information. It cannot
generate hidden mode, because both modes have identical observable trajectories.
It also cannot generate arbitrary parity.

This gives a concrete derived admissible class:

```text
G_obs(W) = tests computable from finite observable futures.
```

## 8. General Derived Class

For an interactive world:

```text
W = (S, A, delta, omega)
```

where:

```text
delta: S x A -> S
omega: S -> O
```

define an experiment policy:

```text
pi: O* -> A
```

A finite interaction trace from state `s` under policy `pi` is:

```text
trace_n^pi(s) = (o0, a0, o1, a1, ..., on)
```

where:

```text
o_i = omega(s_i)
a_i = pi(o0, ..., oi)
s_{i+1} = delta(s_i, a_i)
```

Derived admissible tests are:

```text
t(s) = f(trace_n^pi(s))
```

where:

1. `n` is finite;
2. `pi` is chosen from an admissible policy class independent of candidate
   quotient `Q`;
3. `f` is a finite recoding independent of state labels;
4. all constructions are equivariant under isomorphisms of `W`.

This is the smallest meaningful derivation:

```text
admissible tests = finite observable consequences of admissible interactions.
```

## 9. Impossibility and Possibility Together

We can now state the precise result.

### No-Free-Admissibility Theorem

No nontrivial admissible generator class can be derived from a bare finite state
set `S` without adding structure or allowing arbitrary label injection.

Proof sketch:

The automorphism group of a bare finite set is `Sym(S)`. Any intrinsic rule must
be invariant under `Sym(S)`. Such a rule cannot pick nontrivial partitions
without admitting their full arbitrary permutation orbits. Therefore it cannot
separate natural from injected tests.

### Interaction-Derived Admissibility Theorem

Given a structured world:

```text
W = (S, A, delta, omega)
```

the class of finite trace tests:

```text
t(s) = f(trace_n^pi(s))
```

with quotient-independent `pi` and label-independent `f`, is admissible in the
minimal derived sense:

1. it is generated from interaction structure only;
2. it is invariant under isomorphisms of `W`;
3. it cannot distinguish states with identical finite interaction traces;
4. it excludes arbitrary tests not expressible as observable interaction
   consequences.

This class is not the only possible admissible class. But it is the weakest
general class that follows from interaction invariance.

## 10. Consequence for the First Experiment

In the first experiment without actions:

```text
W = (S, delta, omega)
```

the derived admissible tests are:

```text
t(s) = f(omega(s), omega(delta(s)), ..., omega(delta^n(s)))
```

Since both hidden modes generate the same infinite observation sequence up to
phase, hidden mode is not derivable.

Parity is also not derivable.

Observation phase is derivable.

Therefore the next experiment should not ask:

```text
Can the lab discover hidden mode from passive observation?
```

It should ask:

```text
Given a chosen interaction interface, which quotients are derivable from finite
observable traces, and which require adding a new instrument?
```

This makes failure informative rather than embarrassing.

## 11. Final Answer

Can admissible generators be derived from more general principles?

Answer:

```text
Not from state space alone.
Yes, partially, from interaction invariance.
```

The minimal derived class is:

```text
finite observable trace tests
```

or:

```text
tests computable from finite interaction consequences.
```

The minimal property is:

```text
equivariance under isomorphisms of the base interaction structure
plus constructibility from that structure.
```

This is the correct stopping point. If a test cannot be generated from the
interaction structure, then admitting it is an extra experimental assumption,
not a mathematical consequence.
