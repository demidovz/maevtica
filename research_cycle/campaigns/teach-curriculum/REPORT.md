# Teaching the child: does looking inside tell us WHERE to teach? (2026-07-07)

**Verdict: REFUTED (the simple version) — and a clear, surprising finding.**
Being able to SEE the child's confusion (Brick-1 probe, error-detection AUC 0.84) does
NOT tell us where to teach. Restricting teaching to the confused spots HURTS; blind,
balanced practice grows the child best.

## Setup
- Child C0 = gpt2 fine-tuned to partial competence at addition 0-99 (start acc 50.7%,
  digit-err 0.49 — the un-saturated regime). One model on the RTX 3050.
- EQUAL budget: K=384 problems, 50 gentle steps (lr 1e-4). 4 teachers differ ONLY in
  WHICH problems they select from a fresh pool; 3 seeds; fixed held-out test.
- inner = top-K by an internal error-probe (resid→p wrong, trained on a separate set).
  output = top-K by the child's own output entropy (its voice hesitating).
  random = K random problems. oracle = top-K by the child's ACTUAL errors (reference).
- Probe out-of-fold AUC = 0.84 (the inner signal is real and strong).

## Result (mean digit-acc after teaching, 3 seeds)
| teacher | acc | vs C0 start |
|---|---|---|
| **random (blind)** | **52.0%** | **+1.3** |
| oracle (hardest) | 48.4% | −2.3 |
| inner (look inside) | 47.6% | −3.1 |
| output (the voice) | 47.5% | −3.3 |

- PRIMARY inner − output = +0.11 pt (2/3 seeds) → REFUTED (looking inside does NOT beat
  listening for teaching-target selection; both ≈ tied and both HURT vs random).
- inner − random = −4.4 pt, consistent across all 3 seeds. Every targeted strategy lost
  to random. oracle − random = −3.6 pt (even the perfect diagnostician hurts).

## Why (mechanism)
Restricting practice to the hard/confused subset causes **catastrophic narrowing**: the
child overfits the hard cases and forgets the broad skill (the test is all difficulties).
Balanced random practice keeps growing the whole skill. So a strong error-DETECTOR is not
a good teaching-TARGET selector — **diagnosis ≠ cure.** This also inverts the naive ZPD
guess: even teaching at the voice's hesitation-boundary (output) didn't win; broad practice did.

## Honest scope
- Toy: gpt2, addition, one child, tiny budget (50 steps), 3 seeds.
- Tests ONE teaching method: RESTRICT practice to the selected subset. It does NOT test
  INTERLEAVING (broad practice + extra emphasis on inner-flagged spots), which avoids the
  forgetting and is the fairer test of "does looking inside help teaching." → next.

**Finding for the vision: the inner error-signal earns its keep as a DETECTOR, not as a
curriculum. To teach this child, breadth beats targeting — at least when targeting means
restricting.**

---

# Fair test — INTERLEAVE (broad practice + light emphasis), 2026-07-07

REFUTED (barely), with a nuance. teach_interleave.py. Same child (start 51.6%), probe
AUC 0.85. Arms differ only in composition of the equal K-budget; run on CPU (GPU was busy
with another studio job — basketbolica video). 3 seeds.

| arm | mean acc | vs C0 |
|---|---|---|
| inner0.3 (light inside-emphasis) | 52.9% | +1.3 |
| **broad (pure breadth)** | **52.6%** | +0.9 |
| output0.3 (light voice-emphasis) | 49.7% | −1.9 |
| inner0.6 (heavy inside-emphasis) | 49.4% | −2.3 |

- PRIMARY inner0.3 − broad = **+0.36 pt (2/3 seeds)** → below +1.0 threshold → **REFUTED**.
  Light inside-emphasis merely TIES plain breadth; it does not reliably beat it.
- Nuance: at matched dose, inner-emphasis beats output-emphasis by **+3.2 pt** (per-seed
  +3.0/0.0/+6.6) — so IF you emphasize, looking INSIDE beats listening to the VOICE
  (consistent with Brick 1). But neither beats plain breadth.
- Heavy emphasis (0.6) HURTS (−2.3), same forgetting as the restrict version.

## Teaching line — closed
Two experiments (restrict + interleave): **breadth beats targeting.** Seeing exactly where
the child is tangled (probe AUC 0.85) does NOT buy a better curriculum — restricting to the
tangled spots hurts, and even gentle emphasis only ties broad practice. Diagnosis ≠ better
curriculum. The one honest positive: as an emphasis signal, INSIDE > the voice — but for
teaching this child, plain balanced practice wins.
