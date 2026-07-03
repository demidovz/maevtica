# Preregistration — Read-fraction (causal load of a direction)

Committed BEFORE seeing results. Concept: R(d) = "read-fraction" = normalized
causal load of a unit direction d = (d^T G d) / lambda_max(G), where G is the
downstream reader Gram (sum of ww^T over reader weight columns W_Q,W_K,W_V,W_in
of layers >= probe layer, LN-folded). R in [0,1]; R=1 is the maximally-read dir.

## Prediction under test (verbatim core)
Rank directions by R. On a downstream causal task, high-R atoms beat linear
probes while low-R atoms do not; the per-property GAP (SAE-atom-usefulness minus
probe-usefulness) correlates POSITIVELY with R (predict Pearson r > 0.4).
Sharpest form: probe directions with R below median have NO causally-matching
SAE atom (steering via the matched atom moves output < 10% as much as via the
probe direction). FALSIFIED if the SAE/probe usefulness gap is uncorrelated w/ R.

## Operationalization (single, honest) — gpt2-small, jbloom SAE resid_pre L8
- Properties = 16-20 semantic categories (colors, animals, ... single-token
  answer words). One property = one data point.
- Probe direction p_c: diff-in-means at blocks.8.hook_resid_pre, last token, of
  "w1, w2 and" continuation prompts for the category vs the grand mean over all
  categories. Unit u_p.
- Matched SAE atom a_c: the SAE decoder atom (of 24576) with MAX cosine to u_p.
- R_c: read-fraction of the PROBE direction u_p = (u_p^T G u_p)/lambda_max(G).
- Usefulness(dir): steer at blocks.8.hook_resid_pre by adding dir scaled to
  STEER_C * n (n = mean resid-norm; SAME scale for probe & atom), measure mean
  logit lift of the category's answer tokens over neutral prompts (intervened -
  clean). usefulness_probe_c, usefulness_atom_c.
- gap_c = usefulness_atom_c - usefulness_probe_c.

## Decision rule (FROZEN before results)
PRIMARY = Pearson(gap_c, R_c) across categories.
- SUPPORTED  iff PRIMARY >= 0.40 and p < 0.05.
- REFUTED    iff PRIMARY <  0.20  (uncorrelated or negative = prediction's own
             falsification clause).
- INCONCLUSIVE iff 0.20 <= PRIMARY < 0.40, or >=0.40 with p >= 0.05.
Also report (secondary, not gating): Pearson(usefulness_atom,R),
Pearson(usefulness_probe,R), and the sharp-form pass rate = fraction of
below-median-R categories with usefulness_atom/usefulness_probe < 0.10.

## ORACLE / positive controls (BROKEN_MEASUREMENT if any fail — refuse verdict)
- OA (steering measurement works): mean_c usefulness_probe_c must be strongly
  positive AND beat a random unit direction on the same tokens:
  oracle_probe = mean usefulness_probe; oracle_rand = mean usefulness of a random
  unit dir. BROKEN if oracle_probe <= 0 OR oracle_probe < 3 * |oracle_rand|.
- OB (reader metric works; guards the 2026-07-03 .norm bug): rho_top/rho_bot
  ratio of top vs bottom eigenvector of G must be > 5, and rho_rand ~ sqrt(tr/d).

## Honest scope limits
Single small model, single SAE release, single layer/position, ONE intervention
(additive steering), one usefulness metric, ~18 categories (small n for Pearson),
matched-atom = max-cosine (one matching rule). A survival is one data point.
