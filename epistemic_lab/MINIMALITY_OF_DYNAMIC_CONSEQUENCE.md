# Minimality Test for Dynamic Consequence Structure

Status: theoretical stress test.

This document attacks the current candidate from Research Program IV:

```text
D = dynamic structure of admissible consequences
```

The question is whether `D` is genuinely minimal, or whether it can be derived
from a weaker object.

The method:

1. State the strongest possible minimality argument for `D`.
2. Try to destroy it by removing each component:
   consequence, admissibility, dynamics, transport, composition and
   substitution.
3. If open-ended knowledge growth survives without a component, that component
   is not fundamental.
4. If it does not survive, explain precisely which capacity is lost.

No code. No algorithms. No new objectives.

## 1. What Must Be Preserved

We need a criterion for "open-ended knowledge growth" strong enough to avoid
trivial counterexamples.

Finite memorization is not enough. A lookup table can grow forever by appending
facts, but that is storage growth, not knowledge growth.

For this document, open-ended knowledge growth means a process can repeatedly:

1. encounter new interaction conditions;
2. form or revise distinctions that matter beyond the current instant;
3. preserve some structures through future interaction;
4. ignore or collapse irrelevant variation;
5. reuse prior structure in new contexts;
6. combine prior structures into more powerful structures;
7. remain identifiable across allowed changes of representation.

If a weaker system cannot satisfy these seven clauses, it may still learn,
adapt, memorize or optimize locally, but it does not satisfy the project's
target notion of open-ended epistemic accumulation.

## 2. The Candidate to Prove Minimal

The candidate object is not a bare operator. It is:

```text
D = (E, A)
```

where:

```text
E = possible epistemic consequence structures
A = admissible transitions among them
```

An epistemic consequence structure is whatever is minimally sufficient to
answer:

```text
Which differences matter for future admissible interaction?
Which structures can stand in for which others?
Which structures can be combined?
Which structures remain the same under allowed transformation?
```

This is intentionally type-neutral. `D` may later be representable as a
relation, graph, transition system, category, preorder, operator family or
something else. Those are representations, not assumptions.

## 3. Attempted Minimality Theorem

### Theorem Attempt

Any system satisfying open-ended knowledge growth, in the strong sense above,
induces a dynamic structure of admissible consequences.

Therefore `D` is minimal up to representational equivalence.

### Proof Sketch

Let `S` be any system that supports open-ended knowledge growth.

#### Step 1. Consequential Difference Is Induced

Because `S` can form or revise distinctions that matter beyond the current
instant, there must be some way to separate differences that affect future
interaction from differences that do not.

Call two internal structures equivalent when substituting one for the other
does not change any relevant future interaction available to the system.

This induces consequence classes, even if the system never explicitly stores
them.

Therefore a consequence structure is implicit in `S`.

#### Step 2. Admissibility Is Induced

Because not every arbitrary difference counts as knowledge, `S` must have some
boundary between variations that are allowed to affect epistemic status and
variations that are arbitrary injections.

If all tests, labels, encodings and post-hoc distinctions are equally valid,
then every partition can be made "knowledge" by defining a test that names its
blocks.

Therefore an admissibility boundary is implicit in `S`.

#### Step 3. Dynamics Is Induced

Because `S` grows through repeated encounters, its consequence structure before
an encounter and after an encounter can differ.

Even if no explicit update map exists, the process determines allowed
successor structures.

Therefore a dynamic transition structure is implicit in `S`.

#### Step 4. Substitution Is Induced

Because `S` can ignore or collapse irrelevant variation, there must be some
criterion by which one structure can replace another while preserving what
matters.

Without this, there is no distinction between compression and loss, or between
abstraction and deletion.

Therefore substitution is implicit in `S`.

#### Step 5. Composition Is Induced

Because `S` can reuse prior structures in new contexts and combine them into
more powerful structures, there must be some rule or relation determining when
structures jointly have consequences.

Without this, old knowledge can only be recalled, not used as material for new
knowledge.

Therefore composition is implicit in `S`.

#### Step 6. Transport Is Induced

Because the project requires invariance across representation, world interface,
objective, feature basis and agent implementation, `S` must say when a
structure before transformation and a structure after transformation count as
the same epistemic role.

Without this, all claims are coordinate-relative.

Therefore transport is implicit in `S`.

#### Conclusion

Every system satisfying the strong target notion of open-ended knowledge growth
induces:

```text
consequence + admissibility + dynamics + substitution + composition + transport
```

This is exactly the content of `D`.

So `D` is minimal if, and only if, none of these components can be removed
without weakening the target notion.

## 4. Stress Test 1: Remove Consequence

### Weakened Object

```text
D - consequence = changing structures with admissibility, transitions,
substitution, composition and transport, but no notion of future consequence.
```

### Can Knowledge Still Grow?

Only as syntactic accumulation.

The system may add symbols, models, constraints or files. It may preserve and
combine them. But it cannot say whether a difference matters.

### Counterexample Attempt

A formal grammar expands strings:

```text
a -> ab
b -> a
```

It grows indefinitely without explicitly representing consequences.

### Why This Does Not Destroy Minimality

The grammar is a generator, not yet a knowledge system. It becomes epistemic
only when generated differences have consequences for prediction, compression,
control, explanation, communication or another admissible future role.

If no consequence relation exists even implicitly, then any generated structure
is epistemically indistinguishable from arbitrary noise.

### Verdict

Consequence cannot be removed for open-ended knowledge growth.

It can be implicit, probabilistic, behavioral or operational. But some
matter/non-matter distinction is necessary.

## 5. Stress Test 2: Remove Admissibility

### Weakened Object

```text
D - admissibility = dynamic consequence structure where any test, distinction,
encoding or relevance criterion may be introduced.
```

### Can Knowledge Still Grow?

Growth can occur, but it is no longer distinguishable from arbitrary injection.

### Counterexample Attempt

Let every possible partition of a state space be admitted as a consequence
structure. The system can always "learn" a new distinction by naming any block:

```text
t_P(s) = block_id of P containing s
```

This creates endless growth.

### Why This Destroys Epistemic Status

The parity counterexample already shows the failure:

```text
arbitrary test -> minimally sufficient quotient
```

Without admissibility, every arbitrary label becomes knowledge. No theory can
separate discovered structure from injected structure.

### Verdict

Admissibility cannot be removed.

However, admissibility need not be a fixed test list. It may be generated by
interaction, embodiment, allowed interventions, communicative constraints or
future consequence. What is fundamental is the existence of a boundary, not the
specific mechanism.

## 6. Stress Test 3: Remove Dynamics

### Weakened Object

```text
D - dynamics = static admissible consequence structure.
```

### Can Knowledge Still Grow?

No, not as growth.

A static structure may encode a large amount of knowledge. It may even encode
all future consequences in advance. But then learning is only lookup or
unfolding of a pre-given object.

### Counterexample Attempt

Suppose there is a complete theory `T*` containing all consequences. The system
does not need dynamics; it merely queries `T*`.

### Why This Does Not Destroy Minimality

This is omniscience or precompilation, not open accumulation. It removes the
problem by assuming the final structure already exists.

Open-ended knowledge growth requires that future interaction can change what is
distinguished, preserved, substituted or composed.

### Verdict

Dynamics cannot be removed unless the target is weakened from knowledge growth
to static knowledge possession.

## 7. Stress Test 4: Remove Transport

### Weakened Object

```text
D - transport = dynamic admissible consequence structure within one fixed
coordinate system.
```

### Can Knowledge Still Grow?

Yes, in a closed representation.

A system can accumulate knowledge inside one world, one feature basis, one
agent implementation and one objective interface without needing transport.

### Counterexample

A theorem prover working in a fixed formal language can accumulate theorems,
definitions and lemmas. It has consequence, admissibility, dynamics,
substitution and composition. It may not need representation transport.

### Why This Partially Destroys Minimality

Transport is not necessary for all open-ended knowledge growth.

It is necessary only for the stronger project target:

```text
invariance across objective, representation, world, feature basis and agent
implementation
```

Without transport, `M` cannot be derived and regime claims remain
coordinate-relative. But knowledge growth itself can still occur in a fixed
formal frame.

### Verdict

Transport is not absolutely fundamental for open-ended growth.

It is fundamental for representation-invariant epistemic theory. Therefore the
minimal object splits:

```text
D_core = consequence + admissibility + dynamics + substitution + composition
D_invariant = D_core + transport
```

This is the first real weakening of `D`.

## 8. Stress Test 5: Remove Composition

### Weakened Object

```text
D - composition = dynamic admissible consequence structure with substitution
and transport, but no rule by which structures combine into further structures.
```

### Can Knowledge Still Grow?

Yes, but only as non-compositional accumulation.

The system may accumulate many independent facts, cases or local policies. It
may preserve and substitute them. But prior structures do not become material
for later structures.

### Counterexample Attempt

An episodic memory system stores every solved case and retrieves similar cases.
It grows indefinitely and can improve behavior through recall.

### Why This Weakens the Target

This is open-ended storage plus retrieval, not open-ended theory growth.

Without composition:

1. no hierarchy is forced;
2. no theory-like object is forced;
3. no systematic recombination is forced;
4. `K` collapses to recurrence or lookup frequency;
5. later knowledge does not use earlier knowledge as a constituent.

### Verdict

Composition is not necessary for unbounded adaptive accumulation.

It is necessary for the stronger notion of open-ended epistemic growth where
knowledge becomes reusable material for further knowledge.

So composition belongs to `D_core` only if the target includes theory formation,
hierarchies and generative reuse. For weaker learning, it is not fundamental.

## 9. Stress Test 6: Remove Substitution

### Weakened Object

```text
D - substitution = dynamic admissible consequence structure with composition
and transport, but no criterion by which one structure can replace another.
```

### Can Knowledge Still Grow?

It can accumulate, but it cannot abstract safely.

The system may add and combine structures, but cannot say when a smaller,
coarser, different or transported structure preserves what matters.

### Counterexample Attempt

A symbolic system never deletes or compresses anything. It only adds new
theorems and composes them by inference.

### Why This Does Not Fully Remove Substitution

Inference itself usually contains a substitution principle: a lemma can be used
in place of its proof, a definition can replace an expression, an isomorphic
form can replace another form, or a derived theorem can stand in for many
instances.

If no replacement relation exists at all, every use must carry the entire
history of its construction. Then there is no abstraction, no quotient, no
compression and no stable identity under transformation.

### Verdict

Substitution cannot be removed for cumulative knowledge that abstracts.

It can be absent only in pure append-only accumulation. But append-only
accumulation does not explain concepts, abstraction, compression or invariant
identity.

## 10. Component-by-Component Result

| Component | Removable for weak growth? | Removable for project target? | Reason |
| --- | --- | --- | --- |
| Consequence | No | No | Without it, growth is syntactic or arbitrary. |
| Admissibility | No | No | Without it, injected distinctions count as knowledge. |
| Dynamics | No | No | Without it, there is possession, not growth. |
| Transport | Yes | No | Closed-frame growth survives; invariance does not. |
| Composition | Yes | No, if theory growth is required | Storage/retrieval survives; hierarchy and theories do not. |
| Substitution | Only for append-only accumulation | No | Without it, abstraction and safe replacement disappear. |

## 11. Revised Minimality Claim

The original `D` is too strong if "open-ended knowledge growth" means only
unbounded adaptive accumulation inside one fixed frame.

The minimal object for that weaker target is:

```text
D_weak = dynamic admissible consequence structure
```

But `D_weak` can describe only:

```text
new relevant distinctions over time
```

It does not derive the full `B0`, because:

```text
C requires substitution
K requires composition if coupling means more than recurrence
M requires transport
```

The minimal object for the actual Epistemic Lab target is:

```text
D_project =
  dynamic admissible consequence structure
  + substitution
  + composition
  + transport
```

This is essentially the earlier `D`, but now with a clearer status:

1. Consequence, admissibility and dynamics are fundamental for any nontrivial
   open epistemic growth.
2. Substitution is fundamental for abstraction and compression.
3. Composition is fundamental for theories, hierarchies and generative reuse.
4. Transport is fundamental for representation-invariant epistemology.

Therefore `D` is not absolutely minimal for every possible learning system.

It is minimal for the stronger target:

```text
open-ended, abstraction-forming, composition-capable,
representation-invariant knowledge growth
```

## 12. Can D Be Derived From Something Weaker?

The only plausible weaker source is:

```text
context-indexed consequential substitutability under change
```

This would treat consequence, admissibility, dynamics, substitution and
transport as faces of one relation:

```text
a can stand for b in context c after change u
```

Possible reading:

```text
Sub(a, b | c, u)
```

where:

1. `a` and `b` are structures or roles;
2. `c` is a future interaction context;
3. `u` is a change of situation, representation or time;
4. the relation means that replacing `b` by `a` preserves what matters in `c`
   after `u`.

Then:

```text
consequence = equivalence or preorder induced by Sub over contexts
admissibility = allowed contexts c and changes u
dynamics = allowed changes u over time
substitution = the relation itself
transport = substitution across representation-changing u
composition = higher-order substitutability of compound roles
```

This is weaker than `D` as a package, but not weaker in content. It compresses
the components into one primitive relation.

### Verdict on the Weaker Source

The deepest current candidate is not `D` as a tuple.

It is:

```text
contextual substitutability under admissible change
```

`D` is the expanded form of that primitive.

So the minimality proof for `D` partly fails:

```text
D is not primitive-minimal.
```

But a revised claim survives:

```text
D is structure-minimal as the unfolded form required to derive B0 and the
project's target phenomena.
```

## 13. Final Verdict

The destruction test produces a sharper result:

```text
Absolute minimal primitive:
  contextual substitutability under admissible change

Minimal unfolded structure:
  D_project = dynamic admissible consequence structure
              + substitution
              + composition
              + transport
```

If the project wants the single deepest object, it should now study:

```text
Sub(a, b | c, u)
```

not `Phi`, not a model space and not even `D` as a tuple.

If the project wants the minimal explicit structure from which the existing
macroscopic variables can be derived, it should keep `D_project`.

The next theoretical problem is therefore:

```text
Can consequence, admissibility, dynamics, transport, composition and
substitution all be reconstructed from contextual substitutability under
admissible change without smuggling them back in as hidden assumptions?
```

