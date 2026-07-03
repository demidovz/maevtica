# External test of Causal Role Carrier — round 1 result

Run: 2026-07-03, Pythia-160m vs Pythia-410m, CPU, ~28 min. Raw: `results.json`.

## Numbers

| Matching method | top-1 accuracy (K=7, chance=0.14) | transfer effect (logit) |
| --- | ---: | ---: |
| **CRC** (causal-role / effect signature) | **1.00** (std 0 over 5 seeds) | **7.07** (= oracle) |
| activation-similarity (aligned geometry) | 0.14 (= chance) | 0.64 |
| random | 0.14 | — |

Preregistered rule → **SUPPORTED** (CRC − geometry = 0.86 ≫ 0.15 threshold).
Geometry collapsed most categories onto "animals"/"fruits" — it recovered the
cross-model correspondence no better than a coin.

## Why this is a GREEN LIGHT, not proof (read this)

A clean 1.0 should raise suspicion, not celebration. Three reasons the round-1
test was probably **too easy / tilted toward CRC**, and must be hardened before
the concept is trusted:

1. **Home-field advantage in the metric.** The steering vectors were built to
   promote category tokens, and the CRC "effect signature" is measured over
   those same shared-vocabulary tokens. So "match by effect signature" is close
   to "match by which category the vector boosts" — which is nearly the ground
   truth. Not fully circular (vectors come from eliciting prompts via diff-in-
   means, and transfer is a genuine cross-model measurement), but the categories
   are trivially separable in output space. **Fix:** measure causal role on a
   *different* downstream behavior than the tokens used to build the vector.

2. **The geometry baseline may be under-powered.** Directions live in different
   dims (768 vs 1024); my alignment is a simple least-squares map from token
   embeddings. Geometry failing at chance is partly a *real* property (activation
   geometry is basis/scale-dependent across models — a known fact), but I cannot
   yet separate "geometry genuinely fails" from "my baseline was too weak."
   **Fix:** give geometry its best shot (CCA / Procrustes on the steering
   directions themselves; SAE-dictionary matching as a second baseline).

3. **Easy discrimination.** 7 very distinct categories. **Fix:** use confusable /
   fine-grained categories (many animal subtypes; overlapping concepts) so
   chance is lower and matching is non-trivial; add cross-**family** models
   (GPT-2 ↔ Pythia), not just cross-scale.

## Honest verdict

Round 1 supports the *direction* of CRC prediction #1: matching internal
mechanisms across models by **causal role** beats matching by **activation
geometry** for transferring interventions. This is consistent with known
interpretability findings (function transfers across models better than raw
geometry). But the magnitude (1.0) is inflated by an easy setup. **Status:
promising tracer bullet — worth round 2, not yet a validated concept.**

Round 2 = fixes (1)+(2)+(3) above. If CRC still clearly beats a *strong*
geometry/SAE baseline on a *hard* discrimination with a *decoupled* metric →
the concept earns real weight. If it collapses to parity → reject as renamed
bundle, per `../epistemic_studio/benchmarks/benchmark_011/reports/final_report.md`.
