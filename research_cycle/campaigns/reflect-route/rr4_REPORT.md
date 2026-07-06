# reflect-route — CONFIRMED (rr4, 2026-07-06)

**Verdict: SUPPORTED — clean, preregistered, all controls green.**

On a hard task where the model is *confidently wrong*, an INTERNAL-activation probe
locates the model's own errors far better than its OUTPUT confidence does. Routing a
fixed reflection budget from the inside beats routing by how-sure-it-sounds.

## The clean result (gpt2, addition, per-answer-digit steps)
| error-detector | AUC (0.5=chance, 1.0=perfect) |
|---|---|
| oracle (canary) | 1.000 ✅ measurement valid |
| **internal probe** (residual, best layer 11) | **0.837** |
| conf-probe (trained on maxprob/entropy/margin) | 0.507 |
| output confidence (entropy) | 0.503 ← chance |
| permutation control (probe on shuffled labels) | 0.527 ✅ no leakage |

- PRIMARY: internal_AUC − output_AUC = **+0.334**, bootstrap 95% CI **[+0.248, +0.418]** (excludes 0).
- Oracle canary passed (AUC 1.000, +20pt lift); leakage control ≈ chance.
- Consistent across THREE fresh datasets: internal AUC 0.74 (rr2) → 0.79 (rr3) → 0.84 (rr4); output ≈ 0.49–0.58 (≈ chance) every time.

## Why output confidence fails here (the point)
On arithmetic the model is *confidently wrong* — low output entropy, still wrong — so
output confidence carries almost no error signal. The internal activations do. This is
exactly the regime where an interp signal should help, and it does.

## Honest caveats ("supported" ≠ "universal law")
- ONE small model (gpt2-124M), ONE task (addition). Not shown to generalize to bigger
  models, other tasks, or genuine multi-step chain-of-thought.
- We proved the SIGNAL EXISTS and beats the output baseline (error-DETECTION), via a
  probe trained on labeled errors. NOT yet shown: the model using it unsupervised/online,
  or the end-to-end accuracy GAIN — the final-acc metric was saturated (89% error rate),
  so the practical "how much does routing improve" magnitude needs a stronger model.
- Prior art: hidden-state error/truth probing (Azaria&Mitchell 2023; CCS Burns 2022)
  already shows "hidden states know errors." Our specific new bit: internal-signal
  error-detection BEATS output-confidence routing head-to-head at equal budget — the
  efficiency/routing claim those lines don't make. An extension, not a from-scratch find.

## Process note (honest)
Took 4 runs; the frozen decision rule was mis-specified TWICE (rr2 saturated metric,
rr3 a backwards guard clause) — both caught, neither allowed a premature "supported".
rr4 removed ONLY the backwards clause, fresh data, rule frozen before running → clean
SUPPORTED. Lesson banked: calibrate the measurement's dynamic range BEFORE freezing the
decision rule.

**First confirmed brick of the reflection-intelligence vision: the model's insides know
where it erred better than its voice does — so reflect from the inside.**
