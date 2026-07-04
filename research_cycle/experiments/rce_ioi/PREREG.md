# PREREG — Repair-Closed Effect (RCE) on GPT-2-small IOI

Written 2026-07-04 BEFORE running. Thresholds fixed here; they do not move.

## Concept under test

RCE(h): instead of the raw total effect of ablating head h, ablate the
*repair-closure* of h — iteratively add the head that most compensates
(in direct-logit-attribution terms) for the current ablation set, until no
head compensates above threshold — and measure the effect of ablating the
whole closed set. Claim: this quantity is protocol-invariant where raw
total effect is not (the zero/mean/resample rank flip of arXiv:2402.15390),
and the closures of the IOI name movers are exactly the known backup name
movers.

## Operationalization (fixed)

- Model: gpt2 (small), CPU, transformer_lens.
- Data: 16 IOI prompts, one template ("When {A} and {B} went to the store,
  {S} gave a drink to"), 8 single-token name pairs × {ABBA, BABA}. All
  prompts identical token length → positions align.
- Metric: logit diff LD = logit(IO) − logit(S) at final position, mean over
  the 16 prompts. Effect of an ablation = LD_clean − LD_ablated (signed).
- Ablation of a head = overwrite its hook_z at ALL positions:
  - zero: z ← 0
  - mean: z ← per-position mean of clean z over the 16 prompts
  - resample: z ← clean z of the batch rolled by 1 (fixed derangement)
- Raw total effect: ablate {h} alone, per protocol. All 144 heads.
- DLA of head g in a run: centered per-head hook_result at final position,
  divided by that run's ln_final scale, times ln_final w, dotted with
  (W_U[:,IO] − W_U[:,S]), mean over prompts.
- Repair closure of seed h under protocol P: S = {h}; loop:
  run with S ablated; score(g) = sign(DLA_clean(h)) × (DLA_S(g) − DLA_clean(g))
  for g ∉ S; g* = argmax; if score(g*) < 0.1 (logit-diff units) stop;
  else S ← S ∪ {g*}; cap at 6 added heads. RCE(h) = effect of ablating
  final S. k(h) = |S| − 1 (number of added repairers).
- Evaluation subset for P1 (fixed rule, chosen from raw effects only):
  top-12 heads by mean |raw effect| across the 3 protocols, union the three
  name movers {L9H9, L9H6, L10H0}. Spearman is computed on SIGNED values,
  pairwise across the 3 protocols; report the MINIMUM pairwise rho.
  (Secondary, reported not decided on: raw Spearman over all 144 heads.)
- Known backup name movers B = {9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2,
  11.9}; negative name movers N = {10.7, 11.10} (Wang et al. 2022). The
  algorithm is NOT told these; they are only used to grade the output.

## Oracle / positive control (gate before any verdict)

- Clean mean LD > 1.0 and per-prompt argmax(IO vs S) accuracy > 90%.
- Zero-ablating L9H9 alone must change LD by ≥ 0.5.
If either fails → BROKEN_MEASUREMENT, no verdict.

## Decision rule (fixed)

P1 (protocol invariance):
- SUPPORTED iff min-pairwise Spearman(RCE, subset) ≥ 0.9 AND
  min-pairwise Spearman(raw, same subset) ≤ 0.6.
- REFUTED iff Spearman(RCE) ≤ Spearman(raw) (closure did not help) OR
  Spearman(RCE) < 0.7.
- Otherwise INCONCLUSIVE (e.g. raw already invariant → the prediction's
  premise fails on this operationalization).

P2 (name-mover closures):
- SUPPORTED iff for each seed in {9.9, 9.6, 10.0} under the mean protocol
  (primary; others reported): k ≤ 3, AND every recovered repairer ∈ B ∪ N,
  AND at least one ∈ B.
- REFUTED iff any seed needs k > 3 (hits cap without stabilizing at ≤3) or
  any recovered repairer outside B ∪ N.
- Otherwise INCONCLUSIVE.

Overall verdict: supported only if BOTH P1 and P2 supported; refuted if
EITHER is refuted; else inconclusive.

## Budget

One script, one run, CPU, ≤ ~15 min. No new dependencies.
