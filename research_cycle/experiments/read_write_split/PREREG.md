# Preregistration — Read/Write Feature Split (dual-basis typed features)

Committed BEFORE running. 2026-07-04.

## Concept under test
Features have two distinct bases: a WRITE basis (what earlier layers deposit,
what probes decode) and a READ basis (what downstream computation consumes).
Prediction (verbatim core): mediation effect of patching along a candidate
direction d is predicted by read-alignment cos^2(d, r) with R^2 >= 0.5;
held-out probe AUC of d adds < 0.05 incremental R^2; and projecting a
decodable-but-non-mediating probe direction onto the read-cone recovers
>= 70% of the full-layer-patch behavioral effect. KILL: if read-alignment
predicts mediation no better than probe accuracy, the split is dead.

## Operationalization (smallest decisive: gpt2-small, CPU, ONE intervention)
- Model gpt2-small (prediction names Gemma-2-2B; out of CPU budget — honest
  scope reduction, same logical structure; a survival here is one data point).
- Layers L in {5, 8}, hook resid_pre, LAST token position.
- Concepts: 6 word categories (colors animals fruits countries drinks body).
  Unit of analysis = concept x layer group (12 groups).
- Behavior metric B(x) = mean final-logit of the concept's answer tokens minus
  mean final-logit of all 6 categories' answer tokens (contrast), on base
  (neutral) prompts.
- Source acts: mean resid at (L,last) over 4 concept prompts ("Red, blue and").
- Full-layer patch E_full: replace base resid(L,last) by source mean; delta B.
- Direction patch (THE one intervention), for unit d:
  x' = x_base + (d . (x_src_mean - x_base)) d ; E(d) = mean delta B over 6 base
  prompts; normalized mediation m(d) = E(d)/E_full.
- READ direction r_c: normalized mean gradient dB/dx at (L,last) over base
  prompts (steelman: most favorable linearized "what downstream reads").
  Read-alignment = cos^2(d, r_c).
- Held-out probe AUC of d: score x . d on HELD-OUT prompts (different templates
  from those used to build probes/sources): 8 positives per concept vs
  negatives from the 5 other concepts; AUC := max(auc, 1-auc) (sign-free).
- Candidate pool per group (12 dirs, fixed): probe p (diff-in-means vs grand
  mean, unit), r, Delta-hat, d_nm = unit(p - (p.r)r), unit mixes
  a*p+(1-a)*r for a in {.25,.5,.75}, unit mixes .5/.75 p+random, 3 random.
  Pooled regression n = 12 groups x 12 dirs = 144 points.

## Decision rule (FROZEN before results)
Pooled OLS across all 144 points, response m(d):
- R2_read  = R^2 of m ~ cos^2(d,r)
- R2_auc   = R^2 of m ~ AUC(d)
- dR2_auc  = R^2 of m ~ cos^2 + AUC  minus  R2_read
- recovery_g = E(r)/E_full per group ( = read-cone projection of p, since
  proj of p on span{r} normalizes to +/-r and direction patch is sign-free);
  RECOVERY = median over 12 groups.
Verdict:
- SUPPORTED  iff R2_read >= 0.50 AND dR2_auc < 0.05 AND RECOVERY >= 0.70.
- REFUTED    iff R2_read <= R2_auc (the prediction's own kill clause)
             OR R2_read < 0.25 OR RECOVERY < 0.35.
- INCONCLUSIVE otherwise.
Secondary (reported, non-gating): R^2 of bilinear (d.Delta-hat)(d.r) predictor;
qualification of d_nm as "decodable-but-non-mediating" (AUC(d_nm) >= 0.8 and
|m(d_nm)| < 0.3), and its per-group rate.

## ORACLES / positive controls (any fail => BROKEN_MEASUREMENT, refuse verdict)
- O1 effect exists: mean E_full over groups >= 0.5 logit.
- O2 patch code correct: m(Delta-hat) must equal 1 by construction;
  median |m(Delta-hat) - 1| <= 0.10.
  AMENDMENT (2026-07-04, after O2 tripped BROKEN_MEASUREMENT on run 1): the
  "by construction" identity only holds for the PER-PROMPT Delta-hat_i =
  unit(x_src - x_base_i); run 1 wrongly computed O2 from the mean-diff
  direction (a legitimate pool candidate, not an identity). O2 is now
  computed per-prompt. NO decision threshold changed; run-1 primary numbers
  were not used to choose anything.
- O3 AUC pipeline: mean AUC(p) >= 0.85 held-out; mean AUC(random) in [0.3,0.7].
(Guards the 2026-07-03 .norm-bug class: a broken projection/metric shows up as
O2 or O3 failing, not as a concept verdict.)

## Honest scope limits
One small model, resid stream only, last-token-only patch, additive projection
patch as the sole intervention, gradient-linearized r (favors the concept),
6 categories, 2 layers. Survival != proven.
