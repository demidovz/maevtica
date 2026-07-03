# Fundamental Variables Program

Status: theoretical program only.

This document suspends the search for new algorithms, objectives and
implementations. The current experimental evidence is treated as a warning:
regimes found by Engine and Explorer are sensitive to the feature basis. If
feature ablation changes clustering, the measured quantities are probably
coordinates on a deeper process, not the fundamental variables of that process.

The goal is to find a minimal set of epistemic macrovariables that can describe
epistemic dynamics across objective choice, representation choice, world choice,
feature choice and agent implementation.

## 1. Working Verdict

The current metrics are not fundamental.

They are observables induced by a representation of an underlying process:

```text
interaction stream
  -> admissible tests
  -> distinctions
  -> quotient candidates
  -> creation, reuse, revision and destruction of quotient structure
  -> measured coordinates
```

The best current hypothesis is that most observed metrics are functions of five
more basic macrovariables:

```text
G: generative pressure
R: retention pressure
K: coupling strength
C: compression load
M: mobility / transformability
```

These are not proposed as final names. They are placeholders for quantities that
should be definable without referring to a specific objective, feature list,
agent implementation or world encoding.

## 2. Candidate Macrovariables

### G. Generative Pressure

Meaning: the rate at which the interaction process forces new distinctions or
new quotient candidates.

It is not merely `births`. Birth count is representation-dependent. Generative
pressure is the necessity of adding distinguishable structure under the
admissible test family.

Expected observables:

```text
births
abstraction_count
alive_count when death is weak
branching_factor under compositional representation
early hierarchy_emergence_time when pressure is structured
```

### R. Retention Pressure

Meaning: the degree to which distinctions remain useful under continued
interaction.

It is not identical to lifetime. Lifetime also depends on deletion policy,
memory budget and representation granularity. Retention is the invariance of a
distinction's consequences under future tests.

Expected observables:

```text
mean_lifetime
mean_exposure_lifetime
survival
valid_reuse
low death rate
```

### K. Coupling Strength

Meaning: the degree to which a retained distinction participates in many future
contexts rather than remaining locally isolated.

Reuse and transfer are different projections of coupling. Reuse measures
within-stream recurrence. Transfer measures recurrence across a changed stream
or world. Both should be derivable from a single coupling variable plus the
distance between contexts.

Expected observables:

```text
mean_reuse
valid_reuse
raw_transfer_reuse
valid_transfer_reuse
transfer_correctness
```

### C. Compression Load

Meaning: the pressure to explain many admissible distinctions with fewer
quotient degrees of freedom.

This is not the same as graph compression ratio. The graph ratio depends on the
chosen graph encoding. Compression load is the demand for a smaller sufficient
description under the current test family and interaction constraints.

Expected observables:

```text
graph_compression_ratio
semantic_subsumption_ratio
lower alive_count at fixed explanatory coverage
depth when compression is achieved through hierarchy
```

### M. Mobility / Transformability

Meaning: the ease with which existing epistemic structure can be transformed
when the world, objective, representation, observation interface or feature
basis changes.

This variable is currently the least directly observed. It is required because
feature-ablation sensitivity cannot be explained by generation, retention,
coupling and compression alone. A structure can be retained and reusable inside
one coordinate system but immobile under representational change.

Expected observables:

```text
cluster instability under feature ablation
objective sensitivity
world sensitivity
transfer drop under anti-compositional worlds
regime changes under representation change
```

## 3. Metric Classification

| Existing metric | Fundamental? | Likely derivation |
| --- | --- | --- |
| `births` | No | Coordinate expression of `G` under an entity-count representation. |
| `deaths` | No | Function of low `R`, memory/budget constraints and destructive policy. |
| `abstraction_count` | No | Integral of `G - death_rate`, modulated by `C`. |
| `alive_count` | No | Current stock: accumulated generation minus removals. |
| `graph_depth` / `depth` | Probably no | Emerges when `C` is high and coupling is compositional; also constrained by allowed representation. |
| `dag_width` | No | Cross-section of accumulated candidates; mainly `G` versus `C`. |
| `branching_factor` | No | Local expression of structured `G` plus compositional coupling. |
| `mean_lifetime` | No | Observable projection of `R`, censored by horizon and deletion rules. |
| `mean_exposure_lifetime` | No | Retention under exposure; closer to `R` than raw lifetime but still coordinate-dependent. |
| `survival` | No | Thresholded or binary expression of `R`. |
| `mean_reuse` | No | Within-context projection of `K`. |
| `valid_reuse` | No | `K` filtered by `R` and test validity. |
| `raw_transfer_reuse` | No | Coupling across contexts before validity filtering. |
| `valid_transfer_reuse` | No | `K * R` across context shift. |
| `transfer_correctness` | No | Retention of consequences after mobility demand; roughly `R * M` conditional on transfer attempts. |
| `graph_compression_ratio` | No | Encoding-specific proxy for `C`. |
| `semantic_subsumption_ratio` | No | Compression through inclusion structure; a representation-specific face of `C`. |
| `hierarchy_emergence_time` | No | First-passage time under `G`, `C` and compositional `K`. |
| `entropy` | Ambiguous | Not fundamental by default; may measure uncertainty over distinctions, diversity of candidates or volatility depending on state space. |
| `regime` | No | Label assigned to regions in the observable coordinate system. |

None of these should be treated as a primitive until it survives changes of
objective, representation, world, features and agent implementation.

## 4. Dependency Sketch

A first dependency hypothesis:

```text
birth_rate        = coordinate(G, representation, admissible_tests)
death_rate        = coordinate(1 - R, budget, deletion_policy)
alive_stock       = integral(birth_rate - death_rate)
lifetime          = horizon_censored(R)
reuse             = coordinate(K | same_context)
transfer          = coordinate(K * M | shifted_context)
transfer_accuracy = coordinate(R * M | shifted_context)
compression       = coordinate(C, representation_language)
depth             = coordinate(C * structured_K, compositional_language)
branching         = coordinate(G * structured_K, graph_language)
entropy           = coordinate(distribution over live distinctions or outcomes)
```

This implies that depth, reuse, lifetime, transfer, compression and branching
should not be independent axes. They are projections of fewer pressures.

The important theoretical move is to stop asking whether "high depth" or "high
reuse" defines a regime. Instead ask which combinations of `G`, `R`, `K`, `C`
and `M` force those observables in any representation.

## 5. Minimal Basis Attempt

### Basis v0: Five Variables

```text
B0 = {G, R, K, C, M}
```

This is the current best candidate because it explains the observed metrics
without treating any measured feature as ontologically first.

Interpretation:

1. `G` says how much new epistemic structure must be generated.
2. `R` says how much of that structure continues to matter.
3. `K` says how broadly retained structure couples to future contexts.
4. `C` says how strongly the system is forced to compress structure.
5. `M` says how well structure survives coordinate or world transformations.

### Could the basis be smaller?

Possibly.

`R` and `K` may collapse if retention always means future coupling. But current
evidence suggests they should remain separate: a distinction can survive for a
long time while being rarely reused, and a frequently reused pattern can fail
under transfer.

`C` may be derivable from a ratio between `G` and resource bounds. But that
would make compression a constraint rather than a variable. The current theory
needs `C` because two systems can face the same generative pressure and memory
budget while differing in their admissible quotient language.

`M` might be derivable from higher-order `K`: coupling not only across contexts
but across transformations of representation. However, feature-ablation
sensitivity makes this too important to hide inside transfer. Keep it explicit
until a stronger derivation exists.

### Could the basis be larger?

Maybe one variable is missing:

```text
V: volatility / drift of the test-induced equivalence relation
```

If world changes are not just harder transfer cases but true changes in the
future consequence relation, then volatility is not reducible to low retention.
For now, volatility is treated as an external condition that modulates `G` and
`R`, not as a sixth primitive.

## 6. Axiomatic Sketch

These are not final axioms. They are constraints a future theory must satisfy.

### A1. Coordinate Invariance

A fundamental macrovariable must be invariant under relabeling of states,
features, observations and internal representation.

If a quantity changes only because a graph encoding or feature list changes, it
is not fundamental.

### A2. Test-Origin Dependence

Fundamental macrovariables may depend on the admissible source of tests, but not
on arbitrary injected tests chosen after the target quotient is known.

This continues the lesson from the parity counterexample: no quotient-only
property is enough.

### A3. Coarse-Graining Consistency

If two descriptions are related by a sufficient quotient that preserves all
admissible tests, the macrovariables must either agree or transform by a
well-defined law.

This is the epistemic analogue of requiring a thermodynamic variable to survive
coarse-graining.

### A4. Predictive Closure

The basis must make existing metrics predictable:

```text
observed_metrics = F(G, R, K, C, M, boundary_conditions)
```

where boundary conditions include horizon, budget, representation language,
world family and admissible test generator.

If a metric cannot be expressed this way, either the basis is incomplete or the
metric is measuring an artifact.

### A5. Non-Algorithmicity

The macrovariables are properties of epistemic dynamics, not procedures for
constructing agents.

A theory that requires a specific algorithm to define `G`, `R`, `K`, `C` or `M`
has failed the invariance requirement.

## 7. Analogies Without Copying

### Thermodynamics

The useful analogy is not "entropy equals epistemic entropy." The useful analogy
is separation between microstate coordinates and macroscopic state variables.
Depth, reuse and lifetime currently look like micro-coordinate summaries. The
program seeks variables analogous to pressure or temperature: quantities that
retain meaning across many microscopic implementations.

### Statistical Physics

The useful analogy is order parameters and phase transitions. But a regime label
found by clustering is not an order parameter. A genuine order parameter should
predict how observables change when objective, representation or world changes.

### Evolutionary Dynamics

Birth, death and survival are not fundamental by themselves. They are population
coordinates. The deeper object is selective persistence of distinctions under
interaction. `G` and `R` are closer to that level than raw births and deaths.

### Information Theory

Entropy and compression are dangerous because they are easy to import. The
project should ask: entropy of what quotient, under which admissible test
family, and invariant under which transformations? Without those answers,
entropy is a coordinate.

### Dynamical Systems

The important analogy is invariant structure under transformation, not
trajectories in a chosen coordinate chart. The feature-ablation problem is
exactly the warning that the current phase portrait may be chart-dependent.

## 8. Predictions to Demand From the Theory

The basis should eventually predict claims of the following form:

1. If `G` is high and `C` is low, abstraction count and width increase, but
   transfer remains unstable unless `K` is also high.
2. If `R` is high and `K` is low, lifetime increases without strong transfer.
3. If `K` is high and `M` is low, within-world reuse is high but feature
   ablation or representation change destabilizes regimes.
4. If `C` and structured `K` are both high, depth should emerge as a derived
   coordinate.
5. If `M` is high, cluster assignments should become less sensitive to feature
   ablation even if the measured coordinates change.
6. If apparent regimes vanish under feature ablation, at least one alleged
   regime coordinate is not an order parameter.

These are theory-level predictions. They do not require proposing new
objectives or algorithms.

## 9. Immediate Research Tasks

No code. No new objectives. No new algorithms.

Theoretical tasks:

1. Define `G`, `R`, `K`, `C` and `M` without referencing current feature names.
2. Prove which existing metrics are coordinate projections of those variables.
3. Search for counterexamples where two systems share the same `B0` values but
   differ in an important observed metric.
4. If such counterexamples exist, decide whether the observed metric is an
   artifact or whether the basis is missing a variable.
5. Formalize the transformation law for macrovariables under quotient,
   representation change and test-family change.
6. Determine whether volatility is external boundary condition or sixth
   macrovariable.

## 10. Success Criterion

The program succeeds if it produces 3-7 macrovariables such that most observed
metrics become natural consequences rather than independent features.

The target is not to explain only the current experiments. The target is a
predictive theory: a basis that says what should happen in experiments not yet
run.

