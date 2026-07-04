# Preregistration — Width-Persistent Latent (renorm fixed point across dictionary width)

Date: 2026-07-04. Model: gpt2, resid_post layer 6, ~20k token activations (wikitext-103).

## Concept
As SAE dictionary width grows, some latents are stable fixed points (persist across
widths) while others split/absorb/reorganize. Claim: persistence estimated from a
width ladder W1..W4 predicts which latents survive vs split/absorb at held-out W5.

## Operationalization
- Train ReLU-L1 SAEs at widths W = [128, 192, 288, 432, 648] on the SAME activations.
- Decoder columns unit-normalized ⇒ each latent = a direction. Match = max cosine of a
  latent's decoder direction to any decoder direction in the other dictionary.
- Anchor = W1 (128 latents). For anchor latent i:
    - c_k = max-cosine to width W_k dict, k=2,3,4  → persistence_i = mean(c2,c3,c4)
    - c5  = max-cosine to held-out W5=648 dict (survival)
    - survival label: SURVIVES if c5 >= 0.5, else SPLIT/ABSORB (reorganized).
- Metric A: AUC of persistence (W1..W4) predicting SURVIVES label at W5.
- Metric B: split/absorb rate in bottom vs top persistence quartile.

## Positive control / ORACLE (must pass or verdict = BROKEN_MEASUREMENT)
Two independently-seeded SAEs at the SAME width (192): mean best cross-match cosine.
Random unit dirs in 768-d give max-cosine ~0.11. Require oracle >= 0.30 (real shared
structure detectable by the cosine matcher). Near-zero ⇒ pipeline broken, not concept dead.

## Negative control
Shuffle persistence vs label pairing ⇒ AUC must collapse to ~0.5 (confirms AUC code honest).

## Base-rate gate
If survival-event rate (fraction with c5<0.5) < 8% or > 92% ⇒ INCONCLUSIVE (degenerate,
too few events to estimate quartile ratio).

## DECISION RULE (frozen before running)
- Oracle < 0.30 ................................ BROKEN_MEASUREMENT
- else base-rate degenerate ................... INCONCLUSIVE
- else AUC >= 0.60 AND bottomQ_split >= 2*topQ_split ... SUPPORTED
- else AUC <= 0.55 ............................ REFUTED ("persistence dead")
- else ....................................... INCONCLUSIVE
