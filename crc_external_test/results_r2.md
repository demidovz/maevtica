# CRC external test — round 2 (hardened) result

Run 2026-07-03, Pythia-160m vs 410m, K=12 (incl. confusable pairs), 168 s.
Raw: `results_r2.json`. All three round-1 poblazhki removed.

## Numbers (chance = 1/12 = 0.083)

| Matching method | top-1 accuracy | transfer effect (logit) |
| --- | ---: | ---: |
| **CRC, decoupled** (side-effects only, answer tokens masked) | **1.00** (std 0, 8 seeds) | 6.83 (= oracle) |
| CRC, category-tokens (round-1 style, for contrast) | 0.92 | — |
| **geometry, STRONG** (paired-activation alignment) | **0.75** (9/12) | 5.86 |
| random | 0.083 | — |

Preregistered rule on the honest (decoupled) metric → **SUPPORTED**
(CRC − strong-geometry = 0.25 ≥ 0.15).

## What changed vs round 1, and what it means

- **Round-1 concern #2 was right:** the weak geometry baseline (chance, 0.14)
  was an artifact. A *fair strong* baseline (align spaces on paired activations)
  jumps to **0.75**. Geometry is actually good — its errors are sensible
  (fruits→animals, cities→countries, clothing→animals: adjacent concepts).
- **Round-1 concern #1 is addressed:** CRC still scores **1.0 even with the
  answer-tokens masked** — it matches on the pure side-effect fingerprint, not
  on "which category tokens it boosts." So CRC's edge is not just home-field.
- **The margin shrank honestly:** 0.86 (round 1) → **0.25** (round 2). CRC still
  wins (12/12 vs 9/12), but by ~3 categories, not a landslide.
- **Small bonus finding:** with confusable categories, the decoupled fingerprint
  (1.0) beat the category-token signature (0.92) — the full causal fingerprint is
  *more* robust than "which answers light up," which is mild independent support
  for the concept's core claim (role ≠ label).

## Honest verdict

Round 2 **strengthens** CRC, it doesn't kill it. Against a fair, strong baseline
and a metric with no home-field advantage, matching mechanisms by causal role
still beats matching by (aligned) geometry, and transfers interventions to the
oracle level. That is real evidence for prediction #1 — CRC is **not** simply a
renamed activation-similarity match.

**But** still: two small same-family models, 12 broadly-separable categories,
steering only, perfect 1.0 (⇒ discrimination may still be too easy to see the
ceiling). The honest status is now:

> **CRC earns real weight — survived a fair, hardened test — but is validated
> only in a small, same-family, steering-only regime. Not yet a general concept.**

## Round 3 (what would actually stress it to failure)

1. **Cross-family** models (GPT-2 ↔ Pythia) — different tokenizers, so match on
   shared *behaviors* (judge-scored), not shared vocab. The real "across model
   instance" test.
2. **Many finer categories** (30–50, heavily confusable) to push accuracy off
   the 1.0 ceiling and find where CRC breaks.
3. **Edit/ablation transfer**, not just additive steering.
4. Compare against an **SAE-dictionary** baseline, the current interp standard.
