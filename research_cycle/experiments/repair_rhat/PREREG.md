# PREREG — Weight-Native Repair Predictor (R̂) vs measured self-repair

Date: 2026-07-04 (written BEFORE running the experiment; thresholds frozen here).

## Prediction under test

R̂, computed from WEIGHTS ALONE, correlates with measured self-repair at r > 0.5
across heads; the LayerNorm term alone reproduces the ~30-50% repair share of
arXiv:2402.15390 without any ablation. Falsified if repair is dominated by heads
with near-zero computed subspace overlap (repair would be distributional, not
weight-readable).

## Minimal design (time-boxed)

- Model: **pythia-160m** only (cached, CPU). 12 layers × 12 heads, d=768.
- Prompts: 16 fixed natural sentences (in script). Target token = clean top-1
  next token at last position (decoupled: R̂ never sees prompts).
- Heads tested: all 72 heads in layers 6–11 (self-repair concentrates late).
- Intervention (ONE): zero-ablate `hook_z` of one head at the LAST position,
  rerun downstream.

## Measured quantities (per head h, layer l; means over 16 prompts)

- `DE`  = frozen-LN direct effect = ((o_h − mean(o_h))/σ_clean · γ) @ W_U[:,tok]
  where o_h = z_h @ W_O_h at last pos, σ_clean from clean final-LN.
- `TE`  = clean_logit − ablated_logit (full downstream rerun).
- `SR`  = DE − TE  (self-repair).  Repair fraction `RF = 1 − mean(TE)/mean(DE)`.
- `LN_repair` = DE − DE_recomp, where DE_recomp = clean_logit − LN(r−o_h)@W_U
  (recomputed LN, no downstream rerun).

## Weight-native predictor (weights only, no forward passes)

- O_h = W_V[l,h] @ W_O[l,h]  (rank ≤ 64).
- `R̂_LN(h)`   = ||O_h||_F  (proxy: bigger write ⇒ bigger norm drop ⇒ more LN repair).
- `R̂_down(h)` = mean over downstream readers W ∈ {W_V[l',h'] for l'>l} ∪
  {W_in[l'] for l'>l} of  ||O_h W||_F · √d / (||O_h||_F ||W||_F)
  (≈1 for random alignment; layer-11 heads get 0 — no downstream readers).
- `R̂(h)` = zscore(R̂_LN) + zscore(R̂_down) over the 72 heads.

## Decision rule (FROZEN)

Eligible heads: mean DE > 0.05 logits.

- **SUPPORTED**: Pearson r(R̂, RF) > 0.5 over eligible heads AND Spearman > 0.4.
- **REFUTED**: Pearson r < 0.2, OR falsification clause: among the top-quartile
  heads by SR, median R̂_down is below the 25th percentile of all tested heads
  AND median R̂_LN is below its 25th percentile (repair without weight-readable
  signal).
- **INCONCLUSIVE**: anything between.
- Secondary (LN claim, reported, not gating the main verdict): aggregate
  LN share = Σ LN_repair / Σ SR over eligible heads with SR>0; the claim's band
  is [0.30, 0.50]; also report r(R̂_LN, LN_repair).

## Oracle / positive controls (gate — refuse verdict if broken)

1. **Analytic-LN oracle**: LN(r_clean)@W_U + b_U must reproduce the model's own
   clean logits, max |diff| < 1e-3 (guards the .norm-bug class).
2. **Layer-11 identity oracle**: for layer-11 heads, downstream is empty, so
   TE must equal DE_recomp: max |TE − DE_recomp| < 0.01 logits. Validates DE,
   TE, and the LN decomposition simultaneously.
3. **Effect-size oracle**: max over heads of mean DE > 0.5 logits (ablation and
   the direct path actually move something).
