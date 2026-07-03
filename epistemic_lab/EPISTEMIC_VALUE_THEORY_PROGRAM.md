# Epistemic Value Theory Program

Status: value theory program built on Applicability Theory.

Applicability Theory is the current scientific framework.

The objective is not to discover another epistemic theorem.

The objective is to construct a formal theory of epistemic value:

```text
What makes one experiment, conjecture or question more valuable than another?
```

The answer must not depend on external utility functions, human preferences,
benchmark scores or engineered rewards.

Assumption:

```text
S0 -> S1 -> S2 -> ...
```

There is a sequence of research states.

Not assumed:

```text
V: research_state -> number
```

If a numerical value functional exists, it must be derived as a representation
of epistemic ordering, not postulated.

No code. No algorithms. No new objectives.

## 1. Research State

A research state is:

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

Each theorem record follows Applicability Theory:

```text
statement
assumptions
observable variables
Suf(T)
Nec(T)
Fail(T)
Und(T)
evidence
counterevidence
confidence record
```

This makes value internal to scientific structure rather than external reward.

## 2. Primitive Candidate: Epistemic Improvement, Not Value

The primitive candidate is not a number.

Define a relation:

```text
S' >=_E S
```

read:

```text
S' is epistemically no worse than S.
```

This relation is acceptable only if it is representation-independent:
equivalent translations of the same research state must preserve the ordering.

### Basic Improvement Moves

`S'` improves `S` when it performs at least one of the following without
uncompensated loss:

1. shrinks an undecidable region `Und(T)`;
2. expands a failure region `Fail(T)` with a valid counterexample;
3. expands a sufficient region `Suf(T)` with non-artifact evidence;
4. refines a theorem boundary;
5. splits an overbroad theorem into sharper domain-specific theorems;
6. discovers a hidden necessary assumption;
7. eliminates an artifact explanation;
8. improves translation/reconstruction power between representations;
9. replaces a vague theorem with an applicability record;
10. reduces redundancy while preserving theorem content.

The phrase "without uncompensated loss" matters. An experiment can clarify one
theorem while destroying evidence quality elsewhere.

## 3. Is Epistemic Value Primitive?

Epistemic value should not be primitive.

It is derivable from structural change in the research state:

```text
value(experiment e at S) = epistemic improvement induced by S --e--> S'
```

The primitive is:

```text
research-state transition plus comparison of applicability structures
```

Numerical value is optional and secondary.

## 4. Experiment Value

An experiment is a transition:

```text
e: S -> S'
```

Its value is first an ordered difference:

```text
Delta_E(e, S) = compare(S', S)
```

not a scalar.

### Structural Components of Delta_E

1. `delta_und`: reduction of undecidable regions.
2. `delta_fail`: new necessary failure regions.
3. `delta_suf`: new sufficient applicability regions.
4. `delta_nec`: discovered necessary conditions.
5. `delta_split`: theorem splitting / boundary refinement.
6. `delta_artifact`: artifact explanations eliminated.
7. `delta_translation`: improved representation translation.
8. `delta_compression`: reduced theorem redundancy without lost predictions.
9. `delta_prediction`: new experimental predictions made possible.
10. `delta_risk`: new ambiguity, artifact risk or contradiction introduced.

An experiment is valuable when the positive structural deltas dominate
`delta_risk` under the partial order of research-state improvement.

## 5. Counterexample Value

A counterexample is valuable because it can move a theorem from:

```text
universal / overbroad
```

to:

```text
domain-delimited
```

### Counterexample Greater Than Confirmation

A counterexample is more valuable than a confirmation when it:

1. discovers a missing necessary condition;
2. expands `Fail(T)` more than the confirmation expands `Suf(T)`;
3. splits a theorem into sharper subtheorems;
4. eliminates a high-confidence false universal claim;
5. reveals a benchmark, optimization or representation artifact;
6. changes the applicability atlas for multiple dependent theorems.

Form:

```text
counterexample > confirmation
```

when:

```text
Delta_boundary(counterexample) dominates Delta_support(confirmation)
```

in downstream theorem refinement.

### Confirmation Greater Than Counterexample

A confirmation can be more valuable when:

1. the theorem already has a precise failure boundary;
2. the experiment tests a sparse or high-uncertainty region of `Suf(T)`;
3. confirmation links two previously disconnected theorem domains;
4. confirmation eliminates a serious artifact explanation;
5. the counterexample affects only an already-known failure region.

Therefore counterexamples are not automatically more valuable. They are more
valuable when they refine applicability more deeply.

## 6. Value Axioms

These are axioms for epistemic improvement, not for scalar utility.

### A1. Representation Invariance

If two research states are losslessly translated:

```text
S ~= S_tilde
```

then:

```text
S' >=_E S iff S'_tilde >=_E S_tilde
```

### A2. Applicability Refinement

All else equal, shrinking `Und(T)` by moving systems into justified `Suf(T)` or
`Fail(T)` is an improvement.

### A3. Artifact Elimination

Removing an artifact explanation without losing valid theorem content is an
improvement.

### A4. Boundary Precision

A theorem with sharper sufficient/failure/undecidable regions is better than an
equally supported theorem with vaguer regions.

### A5. Non-Monotonic Belief, Monotonic Evidence Accounting

Beliefs may be revised non-monotonically, but evidence accounting is monotonic:
failed tests and counterexamples are not erased. They may be reinterpreted, but
not deleted.

### A6. Translation Preservation

Improving exact translation between representations is valuable when it
preserves predictive or proof power.

### A7. No Confirmation Bias

Confirmation alone has value only through structural effects on applicability,
artifact risk, prediction or domain precision.

## 7. When Does a Numerical Functional Exist?

A scalar:

```text
V(S)
```

or:

```text
V(S -> S')
```

exists only if the partial order `>=_E` admits a numerical representation.

This requires additional assumptions such as:

1. comparability of research improvements;
2. tradeoff rates between boundary refinement, artifact elimination and
   prediction expansion;
3. aggregation across theorem records;
4. cost normalization;
5. stability of preferences over scientific state features.

These are not epistemically free. They are extra structure.

Therefore the default value object is:

```text
partial order / preorder of research-state transitions
```

not a number.

## 8. Research Policy

Suppose possible experiments:

```text
e1, e2, ..., en
```

from state `S`.

A policy is derivable if each experiment induces a forecast set of possible
state transitions:

```text
e_i: S -> {S_i1, S_i2, ...}
```

The policy should prefer experiments whose possible outcomes dominate in
epistemic improvement.

### Dominance Rule

Prefer `e1` over `e2` if for every plausible outcome of `e2`, there is a
plausible outcome of `e1` at least as epistemically valuable, and for some
outcome strictly more valuable.

This avoids scalar expected utility.

### When Dominance Is Incomplete

If experiments are incomparable under `>=_E`, a scalar policy cannot be derived
without extra assumptions.

Then the rational output is:

```text
incomparable experiments with different epistemic profiles
```

not an arbitrary ranking.

## 9. Scientific Economics

Experiments consume finite resources.

Let:

```text
Cost(e, S)
```

be a resource profile, not necessarily a scalar:

1. time;
2. compute;
3. data;
4. agent budget;
5. opportunity cost;
6. risk of artifact generation.

Scientific return is:

```text
Delta_E(e,S) / Cost(e,S)
```

only if both value and cost have compatible scalar representations.

Default form:

```text
e1 dominates e2 economically
```

if:

```text
Delta_E(e1,S) >=_E Delta_E(e2,S)
and
Cost(e1,S) <= Cost(e2,S)
```

with at least one strict inequality.

This is a partial-order analogue of return on investment.

## 10. Meta-Theorem

### Claim

A rational scientific process asymptotically prefers experiments with maximal
expected epistemic value rather than maximal expected confirmation.

### Status

False without assumptions.

### Counterexamples

1. If the process is rewarded for publication or benchmark score, confirmation
   can dominate epistemic value.
2. If the process has no memory of failed tests, counterexamples lose future
   value.
3. If research states cannot record applicability boundaries, confirmations
   are easier to count than refinements.
4. If cost is extreme, a low-value confirmation may rationally be chosen over a
   high-value but impossible experiment.

### Conditional Theorem

The claim becomes true under these assumptions:

1. research states record failed tests and applicability boundaries;
2. the process uses `>=_E` or a faithful scalar representation of it;
3. long-term progress is evaluated by cumulative applicability refinement, not
   confirmation count;
4. costs are bounded or included in the dominance relation;
5. artifact risk is penalized as a structural loss.

Then confirmation-maximizing policies are dominated whenever they preserve
larger `Und(T)` or fail to expose reachable `Fail(T)` boundaries.

## 11. Open Conjectures

### Conjecture A. Deep Questions Are Boundary Questions

A question is scientifically deep when every plausible answer substantially
changes the applicability atlas.

### Conjecture B. Counterexample Leverage

The value of a counterexample grows with the number of dependent theorem
records whose applicability domains it refines.

### Conjecture C. Artifact Risk Dominance

Experiments that cannot distinguish theorem support from artifact support have
lower epistemic value than experiments with equal confirmation but artifact
discrimination.

### Conjecture D. Scalar Value Is Usually Noncanonical

Most scientific states admit only a partial order of value, not a unique scalar
functional.

### Conjecture E. Long-Term Value Favors Domain Precision

Over long horizons, policies that reduce `Und(T)` and artifact risk dominate
policies that only expand confirmation counts.

## 12. Deliverables Summary

1. Formal definition:

   ```text
   epistemic value = research-state improvement relation >=_E
   ```

2. Value axioms:

   representation invariance, applicability refinement, artifact elimination,
   boundary precision, monotonic evidence accounting, translation preservation,
   no confirmation bias.

3. Experiment value functional:

   ```text
   Delta_E(e,S) = compare(S',S)
   ```

   scalar only under additional representability assumptions.

4. Counterexample value theory:

   counterexamples are valuable when they refine boundaries, reveal hidden
   assumptions or eliminate false universality.

5. Research policy theorem:

   dominance over possible state transitions, not arbitrary scalar
   optimization.

6. Scientific economics:

   partial-order return on resource profiles.

7. Meta-theorem:

   confirmation preference is not rational under applicability-refinement
   assumptions; otherwise the claim is false.

8. Open conjectures:

   boundary questions, counterexample leverage, artifact risk, noncanonical
   scalar value and long-term domain precision.

## 13. Success Criterion

The program succeeds if next-experiment choice can be justified by internal
epistemic dynamics:

```text
Which transition from S to S' most improves the applicability atlas?
```

not by:

```text
external reward,
benchmark score,
human preference,
confirmation count.
```

The objective is to formalize why some questions are scientifically deeper than
others.

