# Causal Role Carrier — external test, overall conclusion

The first real-world (domain) test of the Studio's flagship candidate concept,
demanded by `../epistemic_studio/benchmarks/benchmark_011/reports/final_report.md`.
Three rounds of increasing rigor, on open models we can actually open up.

## The concept and the one prediction tested

**Causal Role Carrier (CRC):** identify an internal mechanism by its
*intervention-stable causal role* (what changes in the output when you poke it),
not by its label or its activation geometry.

**Prediction #1:** matching mechanisms across models by causal role transfers
steering effects better than matching by activation similarity (geometry).

## Results

| Round | Setup | CRC | geometry (strong) | gap | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | same family (Pythia 160m/410m), easy, weak baseline | 1.00 | 0.14 | +0.86 | flawed (too easy) |
| 2 | same family, hard (12 confusable), strong baseline, decoupled metric | 1.00 | 0.75 | +0.25 | SUPPORTED |
| 3 | **cross-family** (GPT-2↔Pythia), hard (23 confusable), strong baseline, decoupled | 0.78 | 0.17 | +0.61 | SUPPORTED |

(chance = 1/K; "decoupled" = causal role measured on neutral side-effect words,
category answer-words masked, so no home-field advantage. Round 3 first faked a
REJECTED due to a `.norm(-1)` bug that zeroed the steering — caught because the
oracle transfer came out ≈0 — then fixed.)

## The key insight

The three rounds tell one coherent story that *matches the concept's own claim*:

- **Same family** → the two models are built alike, so their internal geometry is
  comparable and geometry-matching works well (0.75). CRC only edges it.
- **Different families** → internal geometry is no longer comparable, and geometry
  **collapses to near-chance (0.17)**. But causal role — measured behaviorally —
  is model-agnostic, so CRC **still works (0.78)**, and its matched mechanisms
  transfer interventions to near-oracle strength while geometry's anti-transfer.

CRC's advantage is **largest exactly where geometry loses its shared basis**,
which is precisely what "a role preserved across model instances and families"
predicts.

## Verdict

**SUPPORTED.** Causal Role Carrier is **not** a renamed activation-similarity
match: geometry demonstrably fails cross-family where causal-role matching
succeeds. The Studio's flagship concept survived its first genuine external
falsification attempt — against a fair strong baseline, a metric with no
home-field advantage, and even a bug that briefly faked the opposite.

## Honest limits (do not oversell)

- Additive **steering only**; not yet edit/ablation transfer.
- **Token-promotion / semantic-category** behaviors; not arbitrary circuits.
- Small models (124M / 160M); CRC errs (0.78, not 1.0) on confusable categories.
- Baseline is geometry-alignment, **not an SAE dictionary** — the current interp
  standard and the most important missing comparison.
- One operationalization of "causal role" (effect over vocabulary).
- **Designer risk:** I built both the concept's test and its baselines; an
  independent, adversarial re-implementation would carry more weight than my own
  "survived my best attempt to kill it."

## What a definitive study needs (round 4+)

1. **SAE-dictionary baseline** (match features by dictionary) — the real competitor.
2. **Edit / ablation transfer**, not just additive steering.
3. Larger + more model families; connect to published interp tasks (IOI,
   induction, refusal directions).
4. Independent / pre-registered replication.

## What this means for maevtica

This is the "Domain Research" arrow the program pointed to. The Studio's
methodology produced a candidate concept that survived blind *internal*
validation and now a real *external* test. That is genuine evidence the
methodology can output something real — **one solid data point**, not proof the
Studio can "predict future science." The honest headline: the pipeline works
end-to-end at least once, on a concept that keeps surviving.
