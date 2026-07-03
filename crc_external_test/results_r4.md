# CRC external test — round 4 (ablation + third family)

Run 2026-07-03. GPT-2 / Pythia-160m / OPT-125m — 3 families, all 3 pairs,
2 interventions (add-steer + ablate), K=24, chance=0.042. Raw: `results_r4.json`.

## Numbers

| pair | geometry | **ADD** CRC (oracle tr) | ABLATE CRC (oracle tr) |
| --- | ---: | ---: | ---: |
| gpt2 ↔ pythia-160m | 0.17 | **0.78** (4.3) | 0.39 (−0.1) |
| gpt2 ↔ opt-125m | 0.62 | **0.84** (1.3) | 0.39 (−0.1) |
| pythia-160m ↔ opt-125m | 0.62 | **0.77** (1.3) | 0.26 (−0.1) |

## Verdict — split

- **ADD / steering: SUPPORTED, and now robust across THREE families and all
  three cross-family pairs** (CRC 0.77–0.84 vs geometry 0.17–0.62). This extends
  round 3 from one pair to a family triangle. Solid.
- **ABLATE / "cut it out": INCONCLUSIVE — measurement broke.** The ORACLE
  ablation transfer is ≈ −0.1 (≈ zero) on every pair: ablating the *correct*
  direction on these probes barely moved the answer tokens. So the ablate CRC
  numbers (0.26–0.39) are noise, **not** evidence CRC is weaker on ablation.
  Caught by the oracle check (now enforced in `research_cycle/experiments/harness.py`
  as `BROKEN_MEASUREMENT`).
  - Cause: projecting the direction out on neutral/lightly-loaded probes removes
    almost nothing. Fix (round 5): ablate on strongly category-laden prompts
    where the component is actually present, and measure the drop there.

## Bottom line

The **steering** claim is now well-supported across families; the **ablation**
claim is untested (my test was too weak), not refuted. Honest status of CRC is
unchanged from round 3's conclusion, now with cross-family robustness for
steering. Round 5 = fix ablation + SAE-dictionary baseline + larger models.
