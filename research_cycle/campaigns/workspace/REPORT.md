# E1 — "Errors = the missing intermediate" (Brick 1 ⊕ J-space) — SUPPORTED, 2026-07-07

Local workspace program, experiment 1. Substrate: a gpt2 adder (2-digit + 2-digit) we trained
to partial competence. The unspoken intermediate for the TENS answer digit is the UNITS CARRY.

## Result (N=1000, tens-digit error 13.5%)
- **Carries are the hard intermediate:** error is +5.7pt higher when a carry is needed
  (16.5% vs 10.8%).
- **The error IS the missing carry:** of the errors on carry problems, **92%** are exactly
  "forgot the carry" (emitted digit = correct − 1, i.e. summed the tens as if no overflow).
- **The workspace knows before the voice:** an internal probe at the predicting position flags
  the tens-digit error at AUC **0.832** vs the model's own output confidence **0.598**
  (Δ+0.234, CI[+0.17,+0.30], permutation 0.55). Believable (not the AUC=1.0 trap).
- (Sanity: the carry is trivially readable from context, probe AUC 0.999 — expected, not the claim.)

## Reading
The model's arithmetic errors are the cases where the intermediate THOUGHT (the carry) went
missing, and that missing thought is visible inside — before the wrong digit is spoken — far more
clearly than in the model's own confidence. This UNIFIES Brick 1 (internal error-signal beats
output confidence in the confidently-wrong regime) with the J-space idea (unspoken intermediate):
**an error = a missing workspace intermediate, catchable early from the inside.**

## Honest caveats
- The carry-gap is modest (+5.7pt) — carries are harder but the model handles most.
- The internal probe reads ERROR beyond "is-this-a-carry-problem" (that gap alone → AUC ~0.53,
  so the 0.83 is real computation-state, not surface). Off-by-carry 0.92 is independent behavioral
  corroboration. gpt2 adder, one task — a toy, but a clean one.

---

# E3 — "Confabulation = empty workspace" — REFUTED (informative), 2026-07-07

Synthetic memory: taught gpt2 a random person→city table, HID 25/100 persons. GROUNDED = trained
(retrieval acc 0.73). CONFAB = held-out (model names a city with no grounded knowledge). All persons
same token-type → no surface confound. Person-grouped 5-fold CV.

## Result
- **The model HEDGES honestly on strangers:** output entropy grounded=0.10 vs confab=0.50 → the
  VOICE is already noticeably less sure on held-out persons. This is NOT confident confabulation.
- **No hidden signal:** internal AUC 0.826 vs output-confidence AUC 0.843, Δ**−0.017** (CI[−0.08,+0.05]
  includes 0, perm 0.478). The inside does NOT beat the voice — they carry the same information.

## Reading (sharpens the through-line)
The workspace IS emptier for confabulation, but the OUTPUT already reflects that, so it is not a
HIDDEN signal — the mouth already admits the doubt. Detection beats the voice ONLY in the
confidently-WRONG regime (E1/Brick 1); when the voice honestly hedges (E3), the inside adds nothing.
**E1 (confident-wrong arithmetic → internal WINS) and E3 (honestly-hedged confabulation → internal
TIES) together draw the line:** interp-as-detector is a lie-detector for CONFIDENT mistakes, not a
universal one. Consistent with the whole arc: deep "it knows it's making things up" reduces to the
shallow-but-real "output confidence tracks groundedness here."

## Caveat / what would flip it
A regime of CONFIDENT confabulation (low entropy but wrong) is where the inside could still win —
this small model on this task simply doesn't confidently confabulate (it hedges). The cloud plan
(bigger models, real hallucinations) is exactly where confident confabulation is expected — E3 says
test it THERE, not that the signal is absent in principle.
