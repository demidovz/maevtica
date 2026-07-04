# Preregistration — Causal Quotient Feature (null-space-shifted copies)

Date: 2026-07-04. Written BEFORE running. Decision rule fixed here.

## Concept under test
The probe direction, the difference-of-means steering vector, and the matched
SAE decoder row for the same concept are the SAME causal object shifted inside
the causal null space. Prediction: where their pairwise cosine < 0.6,
(A) >= 90% of the squared norm of the pairwise DIFFERENCE vectors lies in the
causal null space (bottom singular subspace of the perturbation-effect
Jacobian), and (B) projecting all three onto the top-k causal subspace raises
pairwise cosines to >= 0.9.
FALSIFIED if the difference vectors carry substantial causal effect (the three
objects genuinely do different things).

## Setup
- Model: gpt2-small (TransformerLens, CPU), layer 6,
  hook `blocks.6.hook_resid_post`, d=768, V=50257.
- SAE: OpenAI v5 TopK 32768-latent SAE for gpt2-small L6 (cached from the
  2026-07-04 quotient_bq run), TopK k=32.
- Concepts (3), each 24 pos + 24 neg sentences whose LAST token carries the
  concept; split 18/6 per class train/test (seeded):
  1. sentiment (positive vs negative final adjective),
  2. tense (past vs present final verb),
  3. animal vs vehicle (final noun).
  Representations: last-token resid_post L6.
- Three objects per concept (train split only):
  m = mean(pos) − mean(neg);
  p = logistic-regression probe weight (raw features, L2 1e-2, Adam);
  s = decoder row of the SAE latent maximizing mean TopK activation
      difference (pos − neg) on train last tokens.
- Canonical orientation: unit-normalize each object; flip p and s to have
  non-negative cosine with m (m is reference). All cosines below are signed
  after this alignment.
- Qualifying pairs: within-concept object pairs with cos < 0.6 (the premise).
  If < 3 qualifying pairs total across the 3 concepts → INCONCLUSIVE
  (premise population empty here).

## Causal Jacobian (decoupled from the concepts)
- 4 NEUTRAL probe prompts (weather/city/said/science — harness PROBES[0:4]),
  none containing the concept words.
- J per probe = finite-difference linearized map, direction at L6 last-pos
  resid → last-token logit change: column_j = (logits(x + eps·e_j) −
  logits(x))/eps over the full 768 basis, batched; eps = 0.02 × mean clean
  last-token resid norm over the 4 probes.
- G = Σ_probes JᵀJ (768×768, = stacked-Jacobian Gram). Eigendecomp of G gives
  right singular subspace: top-k = eigvecs covering 90% of Σλ (k = k90);
  null space = the remaining 768−k eigvecs.

## Metrics (per qualifying pair, u,v unit + aligned; d = u − v)
- nf(d) = 1 − dᵀP_top d / ||d||²  (fraction of squared norm in null space)
- cos_topk = cos(V_kᵀu, V_kᵀv)
- rel_ce = sqrt(d̂ᵀGd̂) / max(sqrt(ûᵀGû), sqrt(v̂ᵀGv̂))  (does the difference
  direction carry substantial causal effect relative to the objects?)
- Calibration (reported, gates clause A): nf and sqrt(vᵀGv) for 200 random
  unit directions; nf of the objects themselves. If median random nf >= 0.9,
  clause A is vacuous at this k — it is then dropped from the SUPPORTED
  requirement and the verdict rests on cos_topk and rel_ce (stated in report).

## Oracle / positive control (MANDATORY — no verdict past a broken oracle)
- O1 linearity: 8 random directions on probe 0, effects at eps and 2·eps:
  median cos >= 0.98 and median norm ratio in [1.7, 2.3], else
  BROKEN_MEASUREMENT.
- O2 null space is behaviorally null at REAL scale: steer each of the 4
  probes with alpha = 0.5 × mean resid norm along top-1 vs bottom-1 eigvec;
  median ||Δlogits||_top / median ||Δlogits||_bottom >= 5, else
  BROKEN_MEASUREMENT (the "null space" isn't null; clause A meaningless).
- O3 concept gate: probe held-out accuracy >= 0.8 (10/12) per concept;
  concepts failing this are dropped (objects not meaningful there).

## Decision rule (fixed; medians over qualifying pairs)
BROKEN_MEASUREMENT if O1 or O2 fails.
SUPPORTED iff median nf >= 0.90 (waived only if random-nf calibration >= 0.9)
          AND median cos_topk >= 0.90 AND median rel_ce < 0.5.
REFUTED  iff median cos_topk < 0.60 OR median rel_ce >= 0.5
          (difference vectors carry substantial causal effect — the objects
          genuinely do different things).
INCONCLUSIVE otherwise, or if < 3 qualifying pairs, or all concepts fail O3.

## Honest limits (in advance)
One model (gpt2-small), one layer, one linearized effect map at 4 neutral
prompts, last-token logits only, add-perturbations only, 3 concepts, one SAE.
Survival = "survived a minimal decisive test in this regime", not proven.

## Budget
~100 batched CPU forwards for the Jacobian + ~150 small forwards for
activations; SAE already on disk. ~10 min.
