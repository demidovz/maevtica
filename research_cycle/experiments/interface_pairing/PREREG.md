# Preregistration — Interface Pairing (reader–writer dual representation)

Committed BEFORE running. 2026-07-04.

## Concept under test
Each concept c has TWO representations at a layer: a writer-side subspace W_c
(what upstream layers deposit; what probes/diff-in-means decode) and a
reader-side subspace R_c (what downstream computation consumes). The pairing
claim: causal effects of interventions are governed by the READER side.
Prediction (verbatim core, rescoped): the causal effect (change in behavior
logit-diff) of a unit-norm perturbation v at layer L is predicted by
||proj_{R_c}(v)|| with r^2 >= 0.8 across random v, while ||proj_{W_c}(v)||
predicts with r^2 lower by >= 0.2 whenever the first principal angle between
R_c and W_c exceeds ~20 degrees.
KILL (the prediction's own clause): FALSIFIED if the writer-side projection
predicts intervention effects as well as the reader-side projection across
concepts — i.e. the pairing adds nothing over one representation.

## Honest scope reduction
Prediction names Llama-3-8B refusal + Arditi steering vector. 8B is out of
CPU/budget for this box. Rescoped to gpt2-small (CPU) with 6 word-category
concepts and the same category-contrast behavior metric as the sibling
experiment read_write_split (same logical structure: writer=deposit,
reader=consumption, causal probe=perturbation). The Arditi corollary
(cos of best refusal steering vector to R_c vs diff-of-means) is
status: designed_not_run — needs Llama-3-8B + refusal prompts; design:
compute R_c from refusal logit-diff gradients at the Arditi layer, W_c from
harmful/harmless activation deposits, compare cosines.

## Operationalization (gpt2-small, CPU, ONE layer, ONE intervention type)
- Layer L=8 (strongest effects in read_write_split), hook resid_pre, LAST token.
- Concepts: 6 word categories (colors animals fruits countries drinks body).
- Behavior B(x) = mean final-logit of concept answer tokens minus mean
  final-logit over the union of all 6 categories' tokens (contrast), on 6
  neutral base prompts.
- Perturbation (THE one intervention): x' = x + alpha_c * v at (L, last),
  v unit-norm; alpha_c = mean_i ||x_src_mean(c) - x_base_i|| (the natural
  concept-writing scale, fixed per concept BEFORE sampling v).
  E(v) = mean over 6 base prompts of B(x') - B(x). Response variable |E(v)|.
- R_c (reader): top-4 left singular vectors of the 6x768 matrix of gradients
  dB/dx at (L,last), one gradient per base prompt.
- W_c (writer): top-4 left singular vectors of the 4x768 matrix of concept
  deposits: resid(L,last) of each of 4 concept source prompts minus the grand
  mean of all 6 concepts' source-mean activations.
- Predictors: pR(v) = ||R_c^T v||, pW(v) = ||W_c^T v|| (orthonormal bases).
- Random v, n=80 per concept, stratified so projections vary (pure isotropic v
  would have both projections ~sqrt(4/768) and measure only noise):
  mass shares (cR,cW,cA) ~ Dirichlet(1,1,1);
  v = unit( sqrt(cR)*u_R + sqrt(cW)*u_W + sqrt(cA)*u_amb ), where u_R, u_W are
  random unit vectors inside R_c, W_c and u_amb is a random unit vector in the
  full 768-d space. Seed fixed (0).
- Per concept: r2_R = R^2 of OLS |E| ~ pR (with intercept); r2_W likewise;
  theta1 = first (smallest) principal angle between R_c and W_c
  = arccos(sigma_max(R_c^T W_c)).
- Speed: batched forward from layer L via start_at_layer (oracle O0 verifies
  it matches the full forward).

## Decision rule (FROZEN before results)
Eligible groups: theta1 > 20 deg. Let D = r2_R - r2_W per group.
- SUPPORTED iff median over all 6 groups r2_R >= 0.8 AND eligible set nonempty
  AND median over eligible groups D >= 0.2.
- REFUTED iff median r2_R < 0.5 OR (eligible nonempty AND median_eligible
  D <= 0.05) — the writer side predicts (essentially) as well: pairing adds
  nothing.
- INCONCLUSIVE otherwise (including: no eligible group, or 0.5 <= median
  r2_R < 0.8 with D > 0.05).

## ORACLES / positive controls (any gating fail => BROKEN_MEASUREMENT, no verdict)
- O0 start_at_layer correctness: |B_full_forward - B_start_at_layer(clean)|
  < 1e-3 on every base prompt x concept.
- O1 effect exists: for each concept, |E(v*)| at v* = top reader direction
  (sign chosen to increase B); mean over concepts >= 0.5 logit. A near-zero
  O1 = alpha too small or hook broken, NOT concept failure.
- O2 projection code (guards the 2026-07-03 .norm-bug class): for 20 random
  unit vectors inside R_c: min pR >= 0.99; for 20 random ambient unit
  vectors: mean pR < 0.3 (expected ~0.072). Same for W_c.
- O3 (reported, non-gating): sign of E(+v*) vs E(-v*); r2 of |E| ~ |g_mean.v|
  (full-gradient linear ceiling); E(diff-in-means dir) as the Arditi-analog
  steering reference.

## Honest scope limits
One small model, one layer, resid stream, last-token additive perturbation
only, gradient-linearized reader side (favors the concept), k=4 subspaces,
6 categories. Survival != proven; refutation here = refuted in this regime.
