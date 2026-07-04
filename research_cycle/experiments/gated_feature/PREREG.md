# PREREGISTRATION — Gated Feature (direction x context contract)
Date: 2026-07-04 (written BEFORE the run; thresholds fixed here, cannot move).

## Hypothesis under test
"Causally dead but interpretable" SAE latents are largely context-marginalization
artifacts: for >= 25% of latents in the bottom quintile of unconditional
mediation effect (their layer), there exists a learnable context gate
(<= 5 literals, held-out precision >= 0.7) under which their mediation effect
exceeds the layer median by >= 3x. Falsified if < 5% of dead latents are rescued.

## Setup
- Model: gpt2 (CPU), layer 7, hook `blocks.7.hook_resid_pre`.
- SAE: jbloom/GPT2-Small-SAEs-Reformatted, same hook (cached locally).
- Corpus: 120 fixed short English prompts (diverse topics), listed in exp.py.
  A "context" = one prompt. BOS position (index 0) excluded everywhere.
- Latent sample: latents that fire (relu SAE act > 0, standard encoding
  x_cent = x - b_dec) at any non-BOS position in >= 40 prompts. Randomly
  sample M = 256 of them (seed 0). If < 256 qualify, lower the firing
  threshold to >= 25 prompts and take all; report the change.

## Mediation effect (ONE intervention type)
For latent i and prompt p where i fires: ablate i's contribution — subtract
a_i(t) * W_dec[i] from the residual stream at every non-BOS position t —
one forward pass; effect[i,p] = KL(clean || ablated) of the next-token
distribution at the last position (nats).
- Unconditional effect U_i = mean of effect[i,p] over i's firing prompts.
- LayerMedian = median of U_i over the M sampled latents.
- Dead set = bottom quintile of U_i (lowest 20% of the M latents).
- Top quintile kept for the oracle.

## Gate family (fixed before run)
- Context features: prompt-level binary firing indicators of the OTHER
  sampled latents j != i (fires at any non-BOS position). Latent i's own
  activation is NOT a permitted literal (that rescue would be trivial and
  is excluded by design).
- Gate = conjunction of <= 5 literals, each literal = (latent j fires) or
  (latent j does not fire).
- Learning: per dead latent i, its firing prompts are split 60/40
  train/test (seed 1). Label(p) = [effect[i,p] >= 3 * LayerMedian].
  Greedy forward selection of literals maximizing train precision, each
  step must keep train support >= 6; candidates = gate states after
  0..5 literals; accept the candidate with max train precision if that
  precision >= 0.7.

## Rescue criterion (per dead latent, all on HELD-OUT test contexts)
1. gate fires on >= 3 test contexts, AND
2. test precision (fraction of gated test contexts with label 1) >= 0.7, AND
3. mean effect[i,p] over gated test contexts >= 3 * LayerMedian.

## DECISION RULE (fixed now)
- rescue_rate = rescued / |dead set|.
- SUPPORTED if rescue_rate >= 0.25.
- REFUTED if rescue_rate < 0.05.
- INCONCLUSIVE otherwise.

## Sanity / controls (checked before any verdict)
- ORACLE (measurement works): max U_i >= 0.01 nats AND median U_i of the
  top quintile >= 0.005 nats. Below that => BROKEN_MEASUREMENT, no verdict.
- NOISE FLOOR: KL(clean || clean-with-zero-delta-hook) < 1e-6.
- SHUFFLE CONTROL (gate procedure does not overfit): rerun the entire
  rescue pipeline with effect[i,·] permuted across i's firing contexts
  (seed 2). If shuffled_rescue_rate >= max(0.05, 0.5 * rescue_rate) while
  rescue_rate >= 0.05, the positive result is untrustworthy =>
  INCONCLUSIVE_OVERFIT instead of SUPPORTED. (REFUTED is immune: shuffle
  only guards fake positives.)

## AMENDMENT 1 (2026-07-04, after run 1 tripped the oracle — measurement fix ONLY)
Run 1 (exp.py, result.json.run1): oracle_topq_median = 0.0016 < 0.005 =>
BROKEN_MEASUREMENT. Cause: KL was read only at the LAST position while
latents mostly fire mid-prompt; the effect attenuates before the final
token, so even strong latents barely register. Fix (exp2.py):
effect[i,p] = MAX over non-BOS positions t >= first firing position of i
of KL(clean || ablated) of the next-token distribution at position t.
Noise floor likewise max over positions. NOTHING ELSE CHANGES: decision
rule (25% / 5%), gate family, precision 0.7, 3x LayerMedian, splits,
seeds, oracle thresholds all stay as preregistered above.

## Regime / honesty limits
gpt2 only, layer 7 only, one corpus of 120 short prompts, one ablation type
(decoder-row subtraction), prompt-level contexts. "Survived" != proven.
