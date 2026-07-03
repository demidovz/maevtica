# Preregistration — Reader-direction (weight-grounded feature)

## Concept
A dictionary element is a "reader-direction" if downstream WEIGHTS read from it,
independent of how much it contributes to RECONSTRUCTION of activations.
Claim: reconstruction-salience and reader-gain are near-orthogonal properties.

## Operationalization (GPT-2 small, SAE = jbloom resid_pre L8, d_sae=24576, d_in=768)
- reconstruction salience  s_i = freq_i * ||W_dec_i||   (freq from SAE log-sparsity)
- reader-gain              rho_i = sqrt( d_i^T G d_i ),  d_i = W_dec_i / ||W_dec_i||
    G = sum over downstream readers (layers >= 8: W_Q,W_K,W_V,W_in, LN-folded)
        of w w^T  (d_in x d_in Gram). rho_i = norm of reader projections onto d_i.

## Decision rule (preregistered, fixed before seeing result)
- PRIMARY corr = Spearman(s, rho) over all 24576 features.
- SUPPORTED  if |Spearman| <= 0.2 AND |Pearson| <= 0.5
- REFUTED    if |Pearson| > 0.5  OR |Spearman| > 0.5  (salient == high-rho)
- INCONCLUSIVE otherwise (0.2 < corr <= 0.5)

## ORACLE / positive controls (must pass or verdict = BROKEN_MEASUREMENT)
- O1 reader-metric works: rho(top eigenvector of G) = sqrt(lambda_max) must be
     >> rho(bottom eigenvector) = sqrt(lambda_min); ratio must be > 5.
     A random unit dir gives rho ~ sqrt(tr(G)/d_in). If the projection/.norm is
     wrong (the 2026-07-03 .norm bug), this ordering collapses.
- O2 ranking pipeline works: Spearman(s, s+tiny_noise) ~ 1 (>0.99),
     Spearman(s, random_perm(s)) ~ 0 (|.|<0.05).
Report O1 ratio and O2 numbers. Broken if O1 ratio < 5.

## Scaling clause (16k->65k->256k reader-dim constant) : Gemma-Scope, NOT run
  status designed_not_run — needs multi-GB Gemma-Scope SAEs + gemma model,
  out of CPU/budget here. Design: repeat rho on Gemma-Scope widths, compute
  participation-ratio PR(G_features) = (sum lam)^2/sum lam^2 over the rho-weighted
  decoder set; check PR roughly constant across widths.
