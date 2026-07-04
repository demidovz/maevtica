# PREREG — Compensation Matrix K vs self-repair on Pythia-160m / wikitext

Written 2026-07-04 BEFORE running. Thresholds fixed here; they do not move.

## Concept under test

"Compensation Matrix K": self-repair after ablating a head c is attributable to
identifiable downstream absorbers. K[c,d] = signed change in downstream
component d's direct (frozen-LN) contribution to the correct-token logit when c
is ablated. Prediction (as given): row-norm ‖K[c,·]‖₁ predicts each head's
self-repair magnitude (DE − TE) with R² > 0.7 across heads; heads with
row-norm ≈ 0 show DE ≈ TE within noise. Falsified if repair is uncorrelated
with row-norm (e.g. repair carried by LayerNorm-scale change or by cancelling
±compensations, which the paper arXiv:2402.15390 itself reports as a large
component — that is the real kill-shot this test aims at).

## Deviations from the stated prediction (budget, fixed up front)

- Model: pythia-160m (cached, CPU) instead of pythia-410m. Verdict is scoped
  to 160m; the concept claims a mechanism, not a size, so a clean refutation
  at 160m on a natural distribution is evidence against, and support at 160m
  is support-in-a-smaller-regime, not proof of the 410m number.
- Prompt distribution: 24 wikitext-103 snippets (cached) instead of The Pile.
  Natural text either way.
- ONE intervention: mean ablation (batch mean of clean hook_z at the final
  position) of head c at the final input position only.

## Operationalization (fixed)

- 24 prompts: first 24 wikitext-103 test-split paragraphs with ≥ 400 chars;
  tokenize, keep those with ≥ 33 tokens; input = first 32 tokens, target =
  token 33. Metric = logit of target token at final position (pos 31).
- TE(c) = L_clean − L_abl (real forward passes), per prompt.
- Frozen-LN contribution of a residual-stream write v (per prompt):
  contrib(v) = ((v − mean(v)) / s_clean) · W_U[:, target], s_clean = the clean
  run's ln_final scale at pos 31. Biases cancel in all differences used.
- DE(c) = contrib_clean(c) − contrib_abl(c) where c's write = z[:,-1,h] @ W_O.
- Downstream components of head (l,h): heads AND MLPs of layers l+1..11 only
  (parallel attn+MLP in Pythia ⇒ same-layer MLP is not downstream).
  K[c,d] = contrib_abl(d) − contrib_clean(d), per prompt.
- ΔLN(c) = L_abl_frozen − L_abl_real, where L_abl_frozen uses s_clean on the
  ablated resid_post of layer 11. Exact identity that must close:
  TE = DE − Σ_d K[c,d] + ΔLN  ⇒  repair := DE − TE = Σ_d K[c,d] − ΔLN.
- Heads graded: all (l,h) with l ∈ 0..10 (132 heads). Layer-11 heads have no
  downstream components (row-norm structurally 0) and are excluded from the
  regression; reported descriptively only.
- Per-head aggregates over the 24 prompts:
  repair(c) = mean_p (DE − TE);  rownorm(c) = mean_p Σ_d |K[c,d]|;
  sem(c) = std_p(DE − TE)/√24.
- PRIMARY: OLS with intercept of repair(c) on rownorm(c) over the 132 heads;
  report R². (Signed repair, as the prediction literally states "(DE − TE)".)
- SECONDARY (descriptive, not part of the verdict): same regression on
  |repair(c)|; Pearson r; share of repair carried by ΔLN
  (mean_c |mean_p ΔLN| / mean_c |repair|).
- ZERO-ROW-NORM CLAUSE: among the bottom-decile-rownorm heads of the 132
  (13 heads), the clause holds iff ≥ 75% of them have |repair(c)| ≤ 2·sem(c).

## Oracle / positive controls (gate; any failure ⇒ BROKEN_MEASUREMENT or
## NO_PHENOMENON, no concept verdict)

1. Mechanics-closure: per (head, prompt), err = |(DE − ΣK + ΔLN) − TE|.
   Gate: median err < 1e-3 and 99th-percentile err < 0.02 (float32 sums over
   ~144 components). A broken hook / wrong frozen-LN math fails this loudly.
2. Frozen-formula check: |L_clean_frozen − L_clean_model| < 1e-3 (median).
3. Effect exists: max_c |mean_p TE| ≥ 0.05 logits.
4. Phenomenon exists: ≥ 10 of the 132 heads have |repair(c)| > 0.02 AND
   > 2·sem(c). If not ⇒ NO_PHENOMENON ⇒ status ran, verdict inconclusive
   (nothing to predict; prediction vacuous here, not refuted).
5. Clean model sanity: mean clean logprob of target > −7 nats.

## Decision rule (fixed)

Given oracles pass:
- SUPPORTED iff primary R² ≥ 0.7 AND zero-row-norm clause holds.
- REFUTED iff primary R² < 0.3 (repair not attributable to identifiable
  absorbers via ℓ1 row mass) OR (R² < 0.7 AND zero-row-norm clause fails).
- Otherwise INCONCLUSIVE (0.3 ≤ R² < 0.7 with clause holding = partial
  attribution, weaker than claimed).

Known caveat, stated up front: repair ≡ ΣK − ΔLN is an identity, so a high R²
for ‖K‖₁ is NOT automatic — it fails exactly when ΔLN dominates or when K's
signed entries cancel. Those are the two escape routes the prediction denies;
the test checks whether it gets away with that denial.

## Budget

One script, 1 + 132 batched forwards (batch 24, seq 32) on CPU, ~2-5 min.
No new dependencies.
