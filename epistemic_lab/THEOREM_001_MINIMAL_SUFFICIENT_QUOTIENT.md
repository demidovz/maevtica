# Theorem 1: Minimal Sufficient Quotient for a Finite Test Family

## Abstract

This theorem establishes the first formal result of the theory. A finite family
of deterministic tests on a finite state space induces a canonical equivalence
relation. The corresponding quotient is the unique coarsest representation that
preserves all test outcomes.

The theorem does not mention concepts, emergence, intelligence, learning, or
admissibility. It is only the foundational quotient theorem.

## 1. Definitions

### Definition 1. State Space

A state space is a finite nonempty set `S`.

Elements of `S` are called states.

### Definition 2. Test

A test on `S` is a total function:

```text
t: S -> R_t
```

where `R_t` is a set of possible outcomes for `t`.

In this theorem, tests are deterministic and total.

### Definition 3. Finite Test Family

A finite test family on `S` is a finite indexed family:

```text
T = (t_i: S -> R_i)_{i in I}
```

where `I` is a finite index set.

The empty test family is allowed unless explicitly excluded.

### Definition 4. Outcome Signature

For a state `s in S`, the outcome signature of `s` under `T` is:

```text
r_T(s) = (t_i(s))_{i in I}
```

This is an element of the finite product:

```text
prod_{i in I} R_i
```

If `I` is empty, `r_T(s)` is the unique empty tuple for every `s`.

### Definition 5. Test-Induced Equivalence

Define a relation `~_T` on `S` by:

```text
s ~_T s'  iff  r_T(s) = r_T(s')
```

Equivalently:

```text
s ~_T s'  iff  for every i in I, t_i(s) = t_i(s')
```

### Definition 6. Quotient

The test-induced quotient of `S` by `T` is:

```text
Q_T = S / ~_T
```

Its elements are equivalence classes:

```text
[s]_T = {s' in S : s' ~_T s}
```

The quotient map is:

```text
q_T: S -> Q_T
q_T(s) = [s]_T
```

### Definition 7. Representation

A representation of `S` is a function:

```text
q: S -> Q
```

where `Q` is any set and `q` is surjective.

This convention loses no information: any non-surjective representation can be
replaced by its image.

It induces an equivalence relation `~_q` on `S`:

```text
s ~_q s' iff q(s) = q(s')
```

### Definition 8. Sufficiency for a Test Family

A representation `q: S -> Q` is sufficient for `T` if every test in `T` factors
through `q`.

Formally, for every `i in I`, there exists a function:

```text
u_i: Q -> R_i
```

such that:

```text
t_i = u_i o q
```

Equivalently:

```text
q(s) = q(s') implies t_i(s) = t_i(s') for every i in I.
```

### Definition 9. Coarser and Finer Representations

Let:

```text
q: S -> Q
p: S -> P
```

be representations.

We say `q` is coarser than `p` if `p` factors through `q`; that is, if there
exists a function:

```text
h: Q -> P
```

such that:

```text
p = h o q
```

Equivalently:

```text
q(s) = q(s') implies p(s) = p(s')
```

Thus a coarser representation identifies at least as many states.

## 2. Theorem

### Minimal Sufficient Quotient Theorem

Let `S` be a finite nonempty state space and let:

```text
T = (t_i: S -> R_i)_{i in I}
```

be a finite family of deterministic total tests.

Then:

1. `~_T` is an equivalence relation on `S`.
2. The quotient map:

   ```text
   q_T: S -> Q_T
   ```

   is sufficient for `T`.
3. `q_T` is the coarsest sufficient representation of `S` for `T`.
   Equivalently, if `q: S -> Q` is sufficient for `T`, then `q` refines `q_T`;
   that is, there exists a unique function:

   ```text
   h: Q -> Q_T
   ```

   such that:

   ```text
   q_T = h o q
   ```

4. Therefore `Q_T` is unique up to canonical isomorphism among coarsest
   sufficient quotients.

## 3. Proof

### Part 1. `~_T` Is an Equivalence Relation

We must prove reflexivity, symmetry and transitivity.

Reflexivity:

For any `s in S`, we have:

```text
r_T(s) = r_T(s)
```

Therefore:

```text
s ~_T s
```

Symmetry:

Suppose:

```text
s ~_T s'
```

Then:

```text
r_T(s) = r_T(s')
```

Equality is symmetric, so:

```text
r_T(s') = r_T(s)
```

Therefore:

```text
s' ~_T s
```

Transitivity:

Suppose:

```text
s ~_T s'
```

and:

```text
s' ~_T s''
```

Then:

```text
r_T(s) = r_T(s')
```

and:

```text
r_T(s') = r_T(s'')
```

By transitivity of equality:

```text
r_T(s) = r_T(s'')
```

Therefore:

```text
s ~_T s''
```

Thus `~_T` is an equivalence relation.

### Part 2. `q_T` Is Sufficient for `T`

Fix `i in I`.

We must construct a function:

```text
u_i: Q_T -> R_i
```

such that:

```text
t_i = u_i o q_T
```

Define:

```text
u_i([s]_T) = t_i(s)
```

We must show this is well-defined.

Suppose:

```text
[s]_T = [s']_T
```

Then:

```text
s ~_T s'
```

By definition of `~_T`:

```text
t_j(s) = t_j(s') for every j in I
```

In particular:

```text
t_i(s) = t_i(s')
```

So `u_i([s]_T)` does not depend on the representative `s`.

Now for any `s in S`:

```text
(u_i o q_T)(s) = u_i([s]_T) = t_i(s)
```

Therefore:

```text
t_i = u_i o q_T
```

Since this holds for every `i in I`, `q_T` is sufficient for `T`.

### Part 3. `q_T` Is Coarsest Among Sufficient Representations

Let:

```text
q: S -> Q
```

be any representation sufficient for `T`.

We need to construct a unique function:

```text
h: Q -> Q_T
```

such that:

```text
q_T = h o q
```

Define:

```text
h(q(s)) = [s]_T
```

We must prove this is well-defined.

Suppose:

```text
q(s) = q(s')
```

Since `q` is sufficient for `T`, for every `i in I` there exists:

```text
u_i: Q -> R_i
```

with:

```text
t_i = u_i o q
```

Therefore:

```text
t_i(s) = u_i(q(s)) = u_i(q(s')) = t_i(s')
```

for every `i in I`.

Thus:

```text
r_T(s) = r_T(s')
```

and hence:

```text
s ~_T s'
```

So:

```text
[s]_T = [s']_T
```

Therefore `h` is well-defined on the image of `q`.

Since representations are surjective by convention, every element of `Q` has
the form `q(s)` for some `s in S`. Thus `h` is defined on all of `Q`.

For every `s in S`:

```text
(h o q)(s) = h(q(s)) = [s]_T = q_T(s)
```

Therefore:

```text
q_T = h o q
```

The function `h` is unique. If `h': Q -> Q_T` also satisfies:

```text
q_T = h' o q
```

then for any `y in Q`, choose `s in S` with `q(s)=y`. Surjectivity gives such an
`s`. Then:

```text
h(y) = h(q(s)) = q_T(s) = h'(q(s)) = h'(y)
```

So `h=h'`.

So every sufficient representation factors onto `q_T`. Equivalently, every
sufficient representation refines `q_T`.

Thus `q_T` is the coarsest sufficient representation.

### Part 4. Uniqueness Up to Canonical Isomorphism

Let:

```text
p: S -> P
```

be another coarsest sufficient representation.

Since `p` is sufficient, by Part 3 there exists:

```text
a: P -> Q_T
```

such that:

```text
q_T = a o p
```

Since `q_T` is sufficient and `p` is coarsest sufficient, there exists:

```text
b: Q_T -> P
```

such that:

```text
p = b o q_T
```

Then:

```text
q_T = a o b o q_T
```

and:

```text
p = b o a o p
```

If `P` and `Q_T` are taken as images of their quotient maps, then `p` and `q_T`
are surjective. Therefore:

```text
a o b = id_{Q_T}
```

and:

```text
b o a = id_P
```

Thus `a` and `b` are inverse bijections. Hence `P` and `Q_T` are canonically
isomorphic as quotients of `S`.

This completes the proof.

## 4. Corollaries

### Corollary 1. Distinctions Are Test-Induced

For any `s, s' in S`:

```text
q_T(s) != q_T(s')
```

if and only if there exists a test `t_i in T` such that:

```text
t_i(s) != t_i(s')
```

Proof:

`q_T(s) != q_T(s')` iff `[s]_T != [s']_T` iff not `s ~_T s'` iff the outcome
signatures differ, which happens iff at least one coordinate differs.

### Corollary 2. Adding Tests Can Only Refine the Quotient

If:

```text
T subset T'
```

then:

```text
Q_{T'} refines Q_T
```

Proof:

If two states have equal outcomes under all tests in `T'`, then they have equal
outcomes under all tests in `T`.

### Corollary 3. Empty Test Family Gives the Trivial Quotient

If `T` is empty, then:

```text
Q_T = {S}
```

Proof:

All states have the same empty outcome signature.

### Corollary 4. Identity Test Gives the Discrete Quotient

If `T` contains an injective test:

```text
t: S -> R
```

then:

```text
Q_T = {{s}: s in S}
```

Proof:

If `s != s'`, injectivity gives `t(s) != t(s')`, so `s` and `s'` are not
equivalent.

### Corollary 5. Arbitrary Quotients Can Be Injected by Tests

For every partition `P` of `S`, there exists a test family `T_P` such that:

```text
Q_{T_P} = P
```

Proof:

Define:

```text
t_P(s) = the block of P containing s
```

Then two states have the same test outcome iff they lie in the same block of
`P`.

This corollary is the formal source of the first experimental falsification:
minimal sufficiency alone cannot distinguish discovered structure from injected
classification.

## 5. Limitations

### Limitation 1. No Concept Claim

The theorem proves the existence and uniqueness of a coarsest sufficient
quotient for a finite deterministic test family. It does not prove that the
quotient is a concept.

### Limitation 2. No Admissibility Claim

The theorem allows arbitrary tests. Therefore arbitrary partitions can be
injected as quotients. Admissibility of tests must be handled by a separate
theory.

### Limitation 3. Deterministic and Total Tests

The theorem assumes each test is a total deterministic function. Noisy, partial,
costly or destructive tests require a different formulation.

### Limitation 4. Finite Presentation

The theorem is stated for finite test families and finite state spaces. Infinite
versions require additional structure, such as sigma-algebras, topologies or
measurability conditions.

### Limitation 5. No Dynamics

The theorem does not use world dynamics. It says nothing about whether the
quotient transition is well-defined.

## 6. Indispensable Assumption

The indispensable assumption is:

```text
tests are well-defined functions on the same state space S.
```

This assumption is what makes outcome signatures comparable across states.
Without it, equality of signatures is not defined and the quotient construction
does not follow.

Finiteness is convenient but not conceptually indispensable. Determinism can be
generalized to probability kernels. Totality can be generalized to partial
functions with domain conditions. But the existence of a shared domain and
well-defined test outcomes is essential.

## 7. Can This Be Strengthened or Generalized?

Natural strengthening questions:

1. Can the theorem be generalized from deterministic tests to stochastic tests
   using equality of conditional distributions?
2. Can the finite quotient be replaced by a measurable quotient for infinite
   state spaces?
3. Can admissibility be added without destroying the universal property?
4. Can quotient dynamics be incorporated so that the coarsest sufficient
   quotient is also dynamically valid?
5. Can the quotient be characterized categorically as a universal coequalizer
   of test outcome maps?

The immediate next theorem should probably be:

```text
No quotient-only criterion can distinguish a genuine quotient from an injected
quotient when arbitrary tests are allowed.
```

That result follows directly from Corollary 5.
