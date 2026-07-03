# Preregistration — Legibility coordinate (auto-interp fidelity as its own axis)

Committed BEFORE seeing results. 2026-07-04.

## Concept
Legibility λ of an SAE atom = auto-interp fidelity: how well a short label of
the atom predicts (simulates) its activations on held-out text. The concept
claims λ is its OWN axis of the code — near-independent of how strongly the
direction is read downstream (read-fraction R) and of how sparsely the atom
codes (κ).

## Prediction under test (verbatim core)
Across atoms, λ is near-independent of read-fraction R and of coding sparsity
κ: predict |Pearson r| < 0.2 for both. Corollaries: (i) among top-decile
most-causal atoms a substantial share are illegible ("dark computation");
(ii) among the most legible atoms a substantial share are causally inert.
FALSIFIED if λ correlates strongly (|r| > 0.5) with R or κ.

## Operationalization (single, honest) — gpt2-small, jbloom SAE resid_pre L8
Corpus: wikitext-103-raw-v1 TEST split (local parquet), texts >= 400 chars,
tokenized and packed into 72 sequences x 128 tokens. First 64 sequences (8192
positions) for λ/κ stats — even-index sequences = TRAIN half, odd = TEST half.
Last 8 sequences (1024 positions) held out for the causal measure C.

Atom sample: alive atoms = fire (act > 0) on >= 20 positions in EACH half and
firing rate < 0.5. Sample N = 200 uniformly at random (numpy seed 0).

Per-atom quantities:
- λ_j (legibility, unigram label simulation — cheap local stand-in for LLM
  auto-interp; stated as a scope limit): on TRAIN half compute mean activation
  per token id (over all occurrences incl. zeros; tokens with >= 2 train
  occurrences eligible). Label = top-K tokens by that mean, K = 10. Simulated
  activation at a TEST position = train mean act of its token if token in
  label, else 0. λ_j = Pearson(simulated, actual) over TEST positions.
  If simulated is constant (label never appears in test) → λ_j = 0 by
  definition (label explains nothing). Robustness (not gating): K = 50.
- κ_j (coding sparsity axis): firing rate f_j over all 8192 positions.
  PRIMARY uses log10(f_j) (rates span orders of magnitude; raw-f Pearson and
  Spearman reported as robustness). Sign is irrelevant: the rule is on |r|.
- R_j (read-fraction): (u^T G u)/λ_max(G) for the unit decoder direction
  u = W_dec[j]/||W_dec[j]||, G = downstream reader Gram over layers >= 8
  (W_Q, W_K, W_V, W_in columns), exactly as in ../read_fraction/exp.py.
- C_j (causal effect, SECONDARY — corollaries only): on the 8 held-out
  sequences, hook blocks.8.hook_resid_pre, recompute act_j inside the hook and
  subtract act_j * W_dec[j] at every position; C_j = mean over positions of
  (ablated next-token loss − clean loss).

## Decision rule (FROZEN before results)
PRIMARY: r_R = Pearson(λ, R) and r_κ = Pearson(λ, log10 f), across N atoms.
- SUPPORTED    iff |r_R| < 0.20 AND |r_κ| < 0.20.
- REFUTED      iff |r_R| > 0.50 OR  |r_κ| > 0.50 (prediction's own clause).
- INCONCLUSIVE otherwise.
Secondary (reported, not gating): Spearman versions; raw-f Pearson; K=50
variant; corollary shares — (i) dark-computation share = fraction of
top-decile-C atoms with λ below the overall median λ; (ii) inert-legible
share = fraction of top-decile-λ atoms with C below the overall median C.
"Substantial" is reported descriptively against 0.25; corollaries do not gate.

## ORACLE / positive controls (BROKEN_MEASUREMENT if any fail — refuse verdict)
- O1 (λ simulator works): synthetic atom whose activation = 1.0 exactly at
  occurrences of the corpus's most frequent token → measured λ must be > 0.9.
  Synthetic noise atom (|N(0,1)| activations, no token link) → λ must be < 0.15.
- O2 (λ axis not degenerate): std(λ) over sampled atoms > 0.03.
- O3 (reader Gram works; guards the 2026-07-03 .norm bug): rho_top/rho_bot
  eigvec ratio of G > 5 and rho_rand ≈ sqrt(tr(G)/d).
- O4 (causal measure works — gates ONLY the corollaries, not the primary):
  mean C over sampled atoms > 0 and >= 70% of C_j >= 0 (removing a really-used
  feature should mostly hurt loss). If O4 fails, corollaries reported as
  not-measurable; primary verdict still stands on O1–O3.

## Honest scope limits
Single small model, single SAE release, single layer/site, one corpus slice.
λ uses a UNIGRAM-label simulator — a lower bound on full LLM auto-interp
legibility; atoms legible only via multi-token/contextual labels score low.
A mechanical λ–sparsity link, if found, is a real property of this
operationalization, not a bug — but it caps the claim at "unigram legibility".
N = 200, one seed. A survival is one data point, not proof.
