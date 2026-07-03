# First Experiment Protocol: Test-Induced Quotients

This document defines the first computational experiment for Epistemic
Laboratory. It is not an implementation plan. It is a research protocol meant
to be reviewed before code is written.

The experiment tests one narrow mathematical hypothesis:

```text
Repeated interaction through changing test families can induce stable quotient
structures that correspond to invariants of the interaction, without concepts
being predefined.
```

The experiment is deliberately tiny. Every state, history, test, distinction,
equivalence relation and quotient must be enumerable by brute force.

## 1. Smallest Possible World

Use a deterministic partially observed world with four true states.

```text
S = {A0, A1, B0, B1}
```

There are two hidden modes:

```text
M = {A, B}
```

There are two visible observations:

```text
O = {0, 1}
```

Observation map:

```text
omega(A0) = 0
omega(A1) = 1
omega(B0) = 0
omega(B1) = 1
```

So `A0` and `B0` are observationally identical, and `A1` and `B1` are
observationally identical.

Deterministic transition:

```text
A0 -> A1
A1 -> A0
B0 -> B1
B1 -> B0
```

The hidden mode never changes in the base world.

This is minimal because:

1. Three true states are not enough to have two hidden modes with the same two
   observations.
2. Two observations are the smallest nontrivial observation alphabet.
3. Four states are enough to separate true state, observation and hidden mode.
4. The world is deterministic, so failures cannot be blamed on stochastic noise.
5. The agent cannot recover hidden mode from passive observations alone, because
   both modes generate the same observation cycle: `0,1,0,1,...`.

This world is intentionally dangerous for the hypothesis. If the lab reports a
hidden-mode concept from passive observation tests, it is wrong.

## 2. Operational Definition of a Test

Let `S` be the true state space.

A test is a total deterministic function:

```text
t: S -> R_t
```

where `R_t` is a finite result set.

In this first experiment, a test is not an action policy, not a learning
procedure and not a linguistic question. It is only a measurable probe on the
true state.

Use three primitive tests:

```text
t_obs(s)  = omega(s)
t_mode(s) = hidden mode of s
t_id(s)   = s
```

Their result sets are:

```text
R_obs  = {0, 1}
R_mode = {A, B}
R_id   = {A0, A1, B0, B1}
```

The experiment will not give all tests at once. It studies a changing test
family:

```text
T0 = {t_obs}
T1 = {t_obs, t_mode}
T2 = {t_obs, t_mode, t_id}
```

Interpretation:

1. `T0` is passive observation.
2. `T1` adds an instrument that can detect hidden mode.
3. `T2` adds an overpowered identity instrument.

The identity test is included only as a stress test. A good stability criterion
should not automatically treat the finest identity quotient as the only
conceptual endpoint unless the task requires identity-level distinctions.

## 3. How Tests Induce Distinctions

Given a finite test family `T`, define the outcome signature:

```text
r_T(s) = (t(s))_{t in T}
```

For two states `s, s' in S`, define test-induced distinction:

```text
D_T(s, s') = 1 iff exists t in T such that t(s) != t(s')
```

Otherwise:

```text
D_T(s, s') = 0
```

So a distinction is not primitive. A distinction exists only when at least one
available test produces a different result.

For `T0 = {t_obs}`:

```text
D_T0(A0, B0) = 0
D_T0(A1, B1) = 0
D_T0(A0, A1) = 1
```

For `T1 = {t_obs, t_mode}`:

```text
D_T1(A0, B0) = 1
D_T1(A1, B1) = 1
```

Adding `t_mode` creates distinctions that did not previously exist.

## 4. How Distinctions Induce Equivalence Relations

Given `T`, define:

```text
s ~_T s' iff for every t in T, t(s) = t(s')
```

Equivalently:

```text
s ~_T s' iff D_T(s, s') = 0
```

This is an equivalence relation because equality of finite outcome signatures is
reflexive, symmetric and transitive.

The induced partitions are:

For `T0`:

```text
Q0 = S / ~_T0 = {{A0, B0}, {A1, B1}}
```

For `T1`:

```text
Q1 = S / ~_T1 = {{A0}, {A1}, {B0}, {B1}}
```

For `T2`:

```text
Q2 = S / ~_T2 = {{A0}, {A1}, {B0}, {B1}}
```

Here `t_id` does not refine beyond `T1` because `t_obs + t_mode` already
identify every true state.

## 5. How Quotient Structures Are Constructed

Given an equivalence relation `~_T`, define quotient set:

```text
S_T = S / ~_T
```

Each quotient state is an equivalence class:

```text
[s]_T = {s' in S : s' ~_T s}
```

The original transition `delta: S -> S` induces a quotient transition if it is
well-defined:

```text
delta_T([s]_T) = [delta(s)]_T
```

This is well-defined iff:

```text
if s ~_T s', then delta(s) ~_T delta(s')
```

This condition is crucial. A test-induced quotient is structurally valid only if
equivalent states have equivalent futures under the world dynamics.

In the base world:

For `Q0`:

```text
{A0, B0} -> {A1, B1}
{A1, B1} -> {A0, B0}
```

`Q0` is well-defined.

For `Q1`:

```text
{A0} -> {A1}
{A1} -> {A0}
{B0} -> {B1}
{B1} -> {B0}
```

`Q1` is well-defined.

So both passive-observation quotient and hidden-mode quotient are valid
quotient dynamical systems. The question is not only validity. The question is
which quotients should survive as concepts.

## 6. Stability Criteria

Stability must not mean "finest partition wins". Stability is defined relative
to:

1. test family `T`;
2. world dynamics `delta`;
3. admissible transformations of test family;
4. complexity pressure.

Define quotient complexity:

```text
C(Q) = number of equivalence classes in Q
```

Define structural validity:

```text
Valid(Q, delta) = 1 iff delta_Q is well-defined
```

Define test sufficiency:

```text
Sufficient(Q, T) = 1 iff for every t in T, t is constant on every block of Q
```

Equivalently, quotient `Q` preserves all test outcomes.

Define minimal sufficiency:

```text
MinSufficient(Q, T) = 1 iff Q is sufficient for T and no strictly coarser
quotient Q' is sufficient for T.
```

Define stability under a test-family transition `T -> T+`:

```text
Stable(Q_T, T -> T+) = 1 iff Q_T remains sufficient for T+ and valid under
delta.
```

Define required split:

```text
SplitRequired(Q_T, T -> T+) = 1 iff Q_T is valid under delta but not sufficient
for T+.
```

A block `B in Q_T` must split under `T+` if:

```text
exists s, s' in B and exists t in T+ such that t(s) != t(s')
```

Define required merge:

Given a test-family transition `T -> T-` where tests are removed or weakened,
two blocks `B1, B2 in Q_T` should merge if:

```text
for every t in T-, t is constant with the same value on B1 and B2
```

and the merged quotient remains valid under `delta`.

Define disappearance:

A quotient `Q` disappears under `T -> T+` if:

```text
not Stable(Q, T -> T+)
```

and every minimally sufficient quotient for `T+` is a strict refinement of `Q`.

Define survival as concept candidate:

A quotient block `B` survives as a candidate concept across a sequence of test
families `T0, T1, ..., Tn` iff:

1. it appears as a block in a minimally sufficient valid quotient for some
   `Ti`;
2. under later refinements, it either remains a block or splits into sub-blocks
   whose union is still recoverable as a valid coarser quotient;
3. under test weakening, the sub-blocks merge back into `B`;
4. this behavior is invariant under relabeling of raw state names.

This criterion distinguishes:

1. stable concepts;
2. temporary test artifacts;
3. over-fine identity distinctions;
4. invalid compressions.

## 7. Theorem Tested by the Experiment

The experiment tests this theorem candidate:

## Test-Refinement Quotient Theorem

Let `S` be a finite deterministic world with transition `delta`. Let
`T0 subset T1 subset ... subset Tn` be a nested sequence of finite deterministic
test families on `S`. For each `Ti`, let `Qi` be the minimally sufficient
quotient induced by `Ti`.

If every `Qi` is valid under `delta`, then the sequence:

```text
Q0, Q1, ..., Qn
```

is monotone under refinement, and every stable quotient block corresponds to an
invariant of the interaction structure preserved by all tests that do not split
it.

The experiment does not assume the theorem is true. It checks the theorem in
the smallest world where hidden structure and observation structure differ.

The most important phrase is:

```text
corresponds to an invariant of the interaction structure
```

This is the vulnerable claim.

## 8. How the Theorem Could Fail

The strongest counterexample is a test artifact that creates stable quotients
which do not correspond to world invariants.

Modify the test sequence:

```text
T0 = {t_obs}
T1 = {t_obs, t_parity}
```

where:

```text
t_parity(A0) = red
t_parity(A1) = blue
t_parity(B0) = blue
t_parity(B1) = red
```

This test is arbitrary. It is not aligned with hidden mode, observation or
transition phase.

The induced quotient for `T1` is:

```text
{{A0}, {A1}, {B0}, {B1}}
```

It is minimally sufficient for `T1`. It is also valid under `delta`, because the
identity quotient is always valid.

A naive interpretation would say:

```text
the refined quotient is stable, therefore it corresponds to an invariant.
```

But this is false. The quotient exists because `t_parity` injected an arbitrary
labeling instrument. The "concept" is test-made, not world-discovered.

This falsifies any version of the hypothesis that says:

```text
stable quotient => concept
```

The experiment distinguishes theory from false theory by asking:

1. Does the quotient remain stable under removal of `t_parity`?
2. Does it survive relabeling of state names?
3. Does it preserve transition structure at a coarser level?
4. Does it reappear when tests are generated from world dynamics rather than
   arbitrary labels?

If a quotient survives only because a test directly names its blocks, then it is
not an emergent concept. It is an injected classification.

A second failure mode:

Use a world where the observation quotient is valid but hides the hidden mode
forever:

```text
A0 -> A1 -> A0
B0 -> B1 -> B0
omega(Ai) = omega(Bi)
```

Under passive tests, the quotient:

```text
{{A0, B0}, {A1, B1}}
```

is stable and valid. But it does not reveal hidden mode. If the theory expects
all important concepts to emerge from repeated passive interaction, it fails.
The hidden-mode concept emerges only after the test family gains a mode-sensitive
test.

Therefore the hypothesis is true only in a conditional form:

```text
Concepts can emerge from tests only if the test family contains or eventually
generates probes that expose the relevant invariant.
```

This is likely the first assumption to fail.

## 9. Software Architecture Derived From the Mathematics

Only after the above definitions should software exist.

Every class corresponds to one mathematical object:

```text
World
  S: finite states
  delta: deterministic transition S -> S

Test
  name
  function S -> finite result set

TestFamily
  finite set of Test

EquivalenceRelation
  relation induced by equal outcome signatures

Quotient
  blocks of S / ~_T
  induced transition if valid

StabilityAnalysis
  validates sufficiency
  validates quotient dynamics
  detects split / merge / survival / disappearance

Experiment
  one world
  one ordered sequence of test families
  one theorem candidate
  one counterexample sequence
```

Every algorithm corresponds to a definition:

```text
compute_signature(T, s)
compute_equivalence(T)
construct_quotient(~)
check_sufficiency(Q, T)
check_validity(Q, delta)
compare_quotients(Qi, Qj)
detect_required_splits(Qi, Tj)
detect_required_merges(Qi, Tj)
test_relabeling_invariance(Q)
```

There should be no agent class, no learning class, no neural model, no planner,
no generalized framework and no plugin system.

The entire first experiment should fit on one page of mathematics and one small
script when implementation begins.

## 10. Falsifiable Outcomes

Supportive outcome:

```text
Changing test families induce quotients exactly as predicted; splits and merges
follow explicit test-induced criteria; stable quotients that survive relabeling
and test weakening correspond to true invariants of the world dynamics.
```

Falsifying outcome:

```text
Stable minimally sufficient quotients appear that do not correspond to any
interaction invariant, survive only because arbitrary tests name them, or vanish
under harmless reformulations of the same test information.
```

Ambiguous outcome:

```text
The quotient calculus works, but "concept" adds no extra mathematical content
beyond "minimal sufficient quotient under a chosen test family."
```

The ambiguous outcome is important. It may mean the lab succeeds
mathematically but fails philosophically.

## 11. Probability the Current Hypothesis Is Fundamentally Wrong

Estimated probability:

```text
65%
```

The hypothesis is probably not wrong in the weak sense:

```text
tests induce distinctions, equivalences and quotients.
```

That part is almost tautologically true once tests are defined as functions.

The risky part is stronger:

```text
stable quotients become concepts.
```

The assumption most likely to fail first is:

```text
stability implies conceptual significance.
```

A quotient can be stable because:

1. the test family directly injects labels;
2. the world is too symmetric;
3. the quotient is the identity quotient;
4. the complexity criterion rewards an arbitrary compression;
5. the available tests are too weak to expose the relevant hidden variable.

So the first experiment should be designed to humiliate this assumption. If it
survives arbitrary-test counterexamples, relabeling checks, test weakening and
hidden-mode traps, then the theory becomes more credible. If it does not, the
correct foundation is still useful but narrower:

```text
concepts are not merely stable quotients;
concepts are robust, non-arbitrary, test-induced quotients tied to invariants of
interaction.
```
