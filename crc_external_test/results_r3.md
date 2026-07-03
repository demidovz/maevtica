# CRC external test — round 3 (cross-family, break-it) result

Run 2026-07-03, **GPT-2 (124M) ↔ Pythia-160m** — different families, different
tokenizers. K=23 confusable categories, chance = 0.043. Decoupled metric over
108 neutral shared-string words (answer words masked). Raw: `results_r3.json`.

> ⚠️ First run of this script reported a false REJECTED: a `.norm(-1)` (order-−1
> norm) instead of `.norm(dim=-1)` scaled the steering vectors to ~zero, so
> nothing steered and everything degenerated to chance — including the oracle
> (transfer ≈ 0), which is what exposed the bug. Fixed → numbers below.

## Numbers (chance = 0.043)

| Matching method | top-1 accuracy | transfer effect (logit) |
| --- | ---: | ---: |
| **CRC** (causal role, decoupled) | **0.78** (18/23, std 0.04) | **4.11** (oracle 4.43) |
| geometry, STRONG (paired-activation alignment) | 0.17 (4/23) | −0.45 |
| random | 0.043 | — |

**Verdict: SUPPORTED** (CRC − geometry = +0.61 ≫ 0.15).

## Why this is the most meaningful round

Across three rounds of increasing rigor the story is coherent and *matches the
concept's own prediction* about where its edge should be largest:

| Round | Setup | CRC | geometry | gap |
| --- | --- | ---: | ---: | ---: |
| 1 | same family, easy, weak baseline | 1.00 | 0.14 | 0.86 *(flawed)* |
| 2 | same family, hard, **strong** baseline, decoupled | 1.00 | **0.75** | 0.25 |
| 3 | **cross family**, hard, strong baseline, decoupled | 0.78 | **0.17** | **0.61** |

- **Same family (round 2):** the two models have comparable internal geometry, so
  geometry matching is actually good (0.75). CRC only edges it.
- **Different families (round 3):** internal geometry is no longer comparable
  (different architecture/tokenizer) → geometry **collapses to near chance
  (0.17)**. But causal role — *what the mechanism does*, measured behaviorally —
  is model-agnostic, so CRC **still works (0.78)**.

That is exactly Causal Role Carrier's claim: the causal role is preserved "across
bases, model instances, and scale," while representation geometry is not. The
concept's advantage is **largest precisely where geometry has no shared basis** —
which is what we observe.

## Honest verdict on the concept

**SUPPORTED — Causal Role Carrier is not a renamed activation-similarity match.**
Geometry demonstrably fails cross-family where causal-role matching succeeds; the
matched mechanisms also transfer interventions to near-oracle strength while
geometry's picks anti-transfer. This is real, falsifiable evidence for prediction
#1, earned against a fair strong baseline and a decoupled metric, and it survived
a bug that briefly faked the opposite.

## Remaining limits (do not oversell)

- Additive **steering only** — not yet edit/ablation transfer.
- **Token-promotion / semantic-category** behaviors — not arbitrary circuits.
- Small models (124M/160M); CRC makes errors (0.78, not 1.0) on confusable cats.
- Baseline is geometry-alignment, **not an SAE dictionary** (the current interp
  standard) — that comparison is the most important next step.

**Status: a genuine, evidence-backed concept worth a real interpretability
write-up and a bigger study — no longer dismissible as a rename.** Next: SAE-
dictionary baseline + edit/ablation transfer + larger models.
