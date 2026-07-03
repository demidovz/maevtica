# Preregistration — functional variable (causal-abstraction slot)

Committed BEFORE seeing results. Thresholds frozen in `das_vs_sae_iia.py:PREREG`.

## Prediction (concept: "functional variable / causal-abstraction slot")
For tasks where DAS finds a k-dim subspace S with IIA > 0.9, the best single
axis-aligned SAE latent scores IIA < 0.6, and the winning k > 1 — i.e. causally
real variables are typically non-monosemantic and lie OFF the SAE axes.
FALSIFIED if single monosemantic SAE latents routinely reach IIA parity with DAS
subspaces (k=1 suffices and axis-aligned wins).

## Operationalization (single, honest)
- Model: gpt2-small. Behavior: subject-verb NUMBER agreement (a real causal var).
- Hook: blocks.6.hook_resid_pre, last (subject) token. Var V = subject number.
- IIA: patched is/are prediction follows the SOURCE's number (counterfactual flip).
- Fair fight, same hook/pos, additive resid edits:
  - DAS: base + P_R(source-base), P_R projector onto learned k-dim S (grad-trained).
  - SAE (real, gpt2-small-res-jb, 24576 latents): base + (a_src_j-a_base_j)*d_j, best single j.
  - ORACLE (pos control): full last-token resid swap — must be high or measurement broken.
  - NEG control: random 1-dim direction — must be ~chance (0.5).

## Decision rule (frozen)
- oracle_min=0.75, das_high=0.85, sae_ceiling=0.60, parity_margin=0.10, k1_margin=0.10, chance=0.50.
- BROKEN if oracle < oracle_min (refuse to conclude).
- SUPPORTED iff DAS_best>=das_high AND bestSAE<sae_ceiling AND (DAS_best-DAS_k1)>=k1_margin.
- REFUTED iff bestSAE>=DAS_best-parity_margin (SAE parity) OR (DAS_k1>=das_high AND bestSAE>=das_high).
- else INCONCLUSIVE.

## Honest scope limits (stated up front)
Single small model, single behavior, single layer/position, one DAS operationalization,
one SAE release. "Majority of nontrivial behaviors" in the prediction is NOT tested —
this probes ONE nontrivial behavior. A survival here is a single data point, not proof.
