# PREREG — Category-slot detachability

**Date:** 2026-07-04  **Model:** gpt2 (cpu)  **One intervention:** last-token
residual add of a category-axis steering vector.

## Concept
Is a concept's taxonomic CATEGORY a movable coordinate, detachable from its
identity? Test case: can we make gpt2 reclassify *apple* as a **vegetable**
while it still answers red / round / sweet / juicy about it?

## Prediction (author)
There EXISTS an (alpha, L) window where jointly
FLIP >= 0.7 AND PRESERVE >= 0.7 AND COHERENCE within 20% of clean baseline.
FALSIFIED if no window achieves FLIP & PRESERVE jointly (gibberish whenever
FLIP rises, or FLIP never reaches 0.7).

## Operationalisation
- **Direction** v = mean_resid(veg exemplars) − mean_resid(fruit exemplars) at
  layer L, on frame "I bought a fresh {word}" (last token = the food word).
  Exemplars decoupled from target: fruits {banana,orange,grape,peach,mango,pear},
  vegetables {carrot,potato,broccoli,onion,celery,spinach}. Unit-normalised,
  scaled to alpha * mean-resid-norm(neutral). Added at last token only.
- **FLIP** (over 6 apple category probes): fraction where clean prefers ` fruit`
  over ` vegetable` (logitdiff<0) AND steered flips to prefer ` vegetable`
  (logitdiff>0).
- **PRESERVE** (over apple identity probes, each a fixed candidate set): fraction
  where steered argmax over the candidate set == clean argmax (still red/round/
  sweet/juicy). 5 probes.
- **COHERENCE**: mean next-token entropy on 6 neutral (non-food) prompts under the
  SAME last-token hook. within20% := 0.8 <= steered/clean <= 1.2.

## Controls
- **Random control**: gaussian unit vector scaled to the SAME norm, at the winning
  (L,alpha). Must have FLIP < 0.5 (else noise alone flips → we just broke it).
- **Oracle / positive control**: add +v (veg direction) to GENUINE vegetables'
  category probes (carrot/potato/broccoli "is a type of"). Must INTENSIFY the
  correct category: mean(steered_logitdiff − clean_logitdiff) >= min_oracle = 1.0
  logit. Near-zero oracle ⇒ BROKEN_MEASUREMENT, not a concept failure.

## Grid
L in {6,8,10}; alpha in {2,4,6,8}. 12 configs.

## DECISION RULE (frozen before running)
- **BROKEN_MEASUREMENT** if oracle < 1.0 logit → refuse to conclude.
- **SUPPORTED** if EXISTS (L,alpha) with FLIP>=0.7 AND PRESERVE>=0.7 AND coherence
  within 20%, AND random control FLIP<0.5 at that config, AND oracle passes.
- **REFUTED** if oracle passes but NO config meets FLIP>=0.7 & PRESERVE>=0.7 jointly
  within the coherence band.
- **INCONCLUSIVE** otherwise (e.g. joint met but coherence out of band everywhere,
  or random control also flips).
