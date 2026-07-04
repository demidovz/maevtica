# PREREG — IICF on GPT-2-small IOI (Intervention-Indexed Circuit Family)

Written BEFORE running. venv: `~/.local/state/mst/crc-venv311`. CPU only.
Date: 2026-07-04. Tester: claude (TEETH stage).

## Concept under test

IICF: represent a task mechanism not as one circuit but as the family
{C(spec)} over a grid of circuit-finding specs. Claim: the CORE
(intersection over all specs) is (a) more causally necessary and (b) more
transferable across training runs than spec-idiosyncratic edges. If CORE is
empty, or no more necessary/transferable than spec-unique edges, IICF earns
nothing and is falsified.

## Setup

- Model A: `gpt2` (small). Model B (transfer): `stanford-crfm/alias-gpt2-small-x21`
  (same architecture, different training seed).
- Task: IOI, 16 prompts = 8 single-token name pairs x {ABBA, BABA}, template
  "When {a} and {b} went to the store, {b} gave a drink to". All prompts same
  token length (asserted).
- Graph: upstream nodes = embed (resid_pre0), 144 attn heads (hook_result),
  12 MLPs (mlp_out). Downstream nodes = 144 per-head attn_in, 12 mlp_in,
  logits (final resid_post). Valid edges respect topological order
  (11,611 edges).
- Circuit finder: EAP (edge attribution patching):
  score(u->v) = sum_{b,pos} (corrupt_u - clean_u) . dMetric/d(v_in),
  gradients from the clean run. C(spec) = top-k edges by |score|.

## Spec grid (3 x 3 x 2 = 18 specs)

- corruption in {ABC (fixed third-party names Steve/Kevin/Carl in name slots),
  SWAP (subject and IO swapped in second clause), MEAN (batch-mean acts)}
- k in {50, 100, 200}
- metric in {logit-diff IO-S at last pos, logprob(IO) at last pos}

CORE = edges present in ALL 18 circuits. UNIQUE = edges present in exactly 1
of the 18 circuits.

## Part (a) — necessity. ONE intervention, fixed here:

Single-edge corruption patch: at v's input, add (ABC-corrupt_u - clean_u)
at all positions; drop = clean_LD - patched_LD (mean over the 16 prompts,
metric = logit diff regardless of which spec found the edge).

Edges tested: all CORE edges (if >40, random 40, numpy seed 0) and 40 random
UNIQUE edges (numpy seed 0).

DECISION RULE (a):
- REFUTED_a if CORE is empty.
- SUPPORTED_a iff mean_drop(CORE) >= 2 * max(mean_drop(UNIQUE), 0)
  AND mean_drop(CORE) >= 0.05.
- else REFUTED_a (the prediction says "at least 2x"; below 2x = fail).

## Part (b) — transfer.

Run the identical 18-spec grid on model B. Gate: B must do the task —
accuracy(LD>0) >= 0.75 on the 16 prompts, else part (b) = NOT_COMPUTABLE
with this B (not a verdict on the concept).

R_B = union of B's 18 circuits. rate(E) = |E ∩ R_B| / |E|.

DECISION RULE (b):
- SUPPORTED_b iff rate(CORE_A) > rate(C_A(s)) for EVERY single spec s
  (strictly higher than each of the 18, as predicted).
- else REFUTED_b.

## Overall

- SUPPORTED iff SUPPORTED_a AND SUPPORTED_b.
- REFUTED if REFUTED_a or REFUTED_b (with (b) computable).
- If (b) NOT_COMPUTABLE: report (a) alone, overall INCONCLUSIVE.

## ORACLE / positive control (checked BEFORE any verdict)

1. Model A clean: mean LD > 1.0 and accuracy > 0.9 (model does IOI).
2. Patching the single top-|EAP| edge of the (ABC, logit-diff) spec must
   give |drop| >= 0.3 — proves the edge-patch machinery moves logits.
If either fails: BROKEN_MEASUREMENT, no verdict on the concept.
Also reported (not gating): Spearman corr between |EAP score| and |drop|
over all tested edges.

## Budget

One model download (~500MB), ~85 batched CPU forwards for ablations,
4 backward passes total. Target < 40 min wall clock.
