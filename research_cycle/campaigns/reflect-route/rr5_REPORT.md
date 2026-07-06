# reflect-route DEEP (rr5) — the practical GAIN, on a competent model (2026-07-06)

**Verdict: REFUTED on the practical gain — and it reveals the real boundary.**
Fine-tuned gpt2 to partial competence (50.8% digit error, un-saturated), then measured
whether internal-signal routing raises final accuracy MORE than output-confidence routing.

## Clean measurement (un-saturated at last)
- err_rate 50.8% (target ~48%) → final-acc metric now has real dynamic range.
- Oracle canary AUC 1.000, +20pt lift ✅ · permutation control 0.475 ≈ chance ✅ · N=1713.

| arm | final acc @B=20% | detection AUC |
|---|---|---|
| oracle | 69.24 | 1.000 |
| **internal** | 66.78 | 0.904 |
| conf-probe | 66.84 | 0.870 |
| **output** (entropy) | 66.84 | 0.863 |
| random | 58.90 | 0.483 |
| no-reflection | 49.21 | — |

- PRIMARY practical GAIN internal − output = **−0.06 pts**, CI [−0.88, +0.70] (includes 0) → **REFUTED**.
- Detection edge internal − output = +0.041, CI [+0.026, +0.054] (real but tiny).

## Why — and this is the finding (regime-dependence)
Compare with rr4 (base gpt2, incompetent, *confidently wrong*):
| regime | output-conf detection | internal detection | internal advantage |
|---|---|---|---|
| base gpt2 (confidently wrong) | 0.50 (blind) | 0.84 | **HUGE (+0.33)** |
| fine-tuned gpt2 (competent) | 0.863 (sees) | 0.904 | ~zero practical (+0.04, no gain) |

Once the model actually KNOWS arithmetic, its **output confidence jumps from blind (0.50)
to strong (0.86)** — a competent model errs with genuine uncertainty, so its "voice" already
reveals its mistakes. The internal signal then adds essentially nothing to routing.

**Synthesis: looking inside beats listening to the voice EXACTLY when the model is fooling
itself (confidently wrong / doesn't know the skill), and stops helping once the model is
competent (then the voice already knows where it erred).** Internal-signal routing is a tool
for the overconfident-error regime — arguably the important one (hallucination, confident
mistakes) — not a universal free lunch. rr4 stands; rr5 maps its boundary.

## Caveats / open question
- One small model, one task; "competence" = fine-tuned on the same task.
- The payoff question for the reflection-intelligence vision: does a BIG model doing REAL
  multi-step reasoning sit in the confidently-wrong regime (internal helps) or the
  voice-already-knows regime (it doesn't)? That needs a reasoning-capable model — beyond
  the local 4GB kit for genuine chain-of-thought.

## Process (honest)
Two calibration catches before this run, both from the "calibrate before you freeze" rule:
(1) base 410m is ~95% wrong on ALL addition bands → no un-saturated regime without training;
(2) the fine-tuned model degrades 48%→74% under a prepended BOS it never trained with →
analyze without BOS. Both caught pre-run, not after a wasted campaign.
