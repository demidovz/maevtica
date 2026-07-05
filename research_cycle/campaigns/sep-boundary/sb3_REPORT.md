# sep-boundary sb3 — FAITHFUL 410m run (2026-07-06)

**Verdict: REFUTED (clean, adequately powered — NOT underpowered).**

Our internal grad-overlap predictor does **not** beat the Park orthogonality
baseline at saying where a clean concept-swap holds vs breaks. It is below the
chance floor and has the wrong sign; the field's existing baseline actually works.

## Measurement was sound (the point of the rerun)
- Model: real pythia-410m (24 layers), layer 9, steering mults [1,1.5,2,3,4]
  (calibrated by sb_window_probe / sb_calib — the sb3-v1 "BROKEN" was a 10x
  under-crank of steering strength, not a dead outcome).
- Oracle positive control strong: mean +3.45 (fruit +3.13, country +3.76,
  tree +4.12, insect +1.81). bird→fish oracle failed (-0.33) → family correctly
  dropped. Random-direction controls flipped 0/2 in every family.
- Windows opened with real variance (country: France/Spain clean, Japan/India
  flip category but lose their continent → window 0).
- Judged WITHIN family (fix for sb2's Simpson's-paradox pooling).

## Numbers (within-family primary)
| family | n | rho_OURS | rho_PARK | oracle |
|---|---|---|---|---|
| fruit | 18 | +0.322 | +0.348 | +3.13 |
| country | 15 | +0.101 | **−0.793** | +3.76 |

- mean |rho| OURS = **0.212** (< within-family chance floor 0.408)
- mean |rho| PARK = **0.570** (> floor)
- delta CI (OURS−PARK) = [−0.646, +0.092] → does not beat Park
- neg_frac(OURS) = 0.0 → sign is positive, opposite the preregistered prediction

## Honest conclusion
sb1 (gpt2, n=32) showed OURS out-predicting a near-null Park with the "right sign"
— that was a **thin-sample fluke**. On a bigger model with a clean, adequately
powered measurement, it reverses: OURS is noise, Park's orthogonality baseline
is genuinely predictive. The boss's call to rerun bigger did exactly its job:
it converted a probable false-positive ("we beat the field") into a solid negative.

**Line closed as an honest dead-end.** No credible "beat the baseline" story
survives. Next queued: reflection_routing.
