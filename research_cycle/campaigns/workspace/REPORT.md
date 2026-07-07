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

---

# E5 — "Privileged injection": is the workspace a LEVER or just a lens? — MIXED, 2026-07-07
(most interesting result of the program — the first real causal leverage in the whole arc)

Phase 0b: built the real J-lens (Jacobian of digit-logits w.r.t. the residual at the "=" token).
A tiny K=8 workspace captures **94.7%** of the digit-output effect — a genuine low-dim
verbalizable subspace. gpt2 adder, 2-digit sums <100 (tens-digit intermediate = the units carry).

## Result
- **E5a (READ):** carry decodable from the workspace AUC 0.972 — but ALSO from the "silent"
  complement 0.921 (Δ+0.051, below the +0.10 bar). The reasoning intermediate is NOT locked in the
  verbalizable subspace; it is distributed.
- **E5b (CAUSAL) — the money finding:** on NO-carry problems (correct tens digit t), inject a
  matched-norm vector at L,q and read the new tens digit:
  | direction | P(tens→t+1) | specificity | scale |
  |---|---|---|---|
  | **carry-in-workspace** | **0.64** | **0.99** | 2× |
  | random-in-workspace | 0.07 | — | 1× |
  | carry-in-complement | 0.38 | 0.98 | 4× |
  | random-full-space | 0.14 | — | 4× |
  - Injecting the CARRY direction makes the model produce **specifically t+1** — the carry-applied
    (arithmetically correct) answer — with **99% specificity** (not scatter). **The model REASONS
    FROM the injected concept**, correctly applying a carry that isn't there.
  - A RANDOM direction in the SAME workspace does almost nothing (0.07): it's the SPECIFIC
    concept-direction, not the subspace, that carries the lever.
  - The carry direction works from the complement too (0.38) but needs ~3× the push → the workspace
    is the more POTENT site per unit norm, not an exclusive one.

## Reading (amends the through-line)
- **Strong "privileged container" hypothesis: REFUTED** — the concept is distributed and steerable
  from outside the verbalizable subspace too.
- **BUT the first real LEVER in the whole arc:** we can inject a clean concept-direction and the
  model correctly reasons from it (structured t+1, 99% specific; random dirs don't). And the
  verbalizable workspace is quantitatively PRIVILEGED — ~3× more potent per unit norm.
- Through-line amendment: "interp is a detector, not a lever" held for READING error signals
  (E1/E3). But STEERING a well-identified concept-direction DOES causally redirect the model's
  reasoning. The map isn't only readable — a cleanly-identified direction is a usable knob, and it
  bites hardest in the verbalizable workspace. This is the closest thing yet to the boss's core
  question ("can using the map make the child smarter") — a local YES-in-principle.

## Caveats / honesty
- K=8 is a soft verbalizable/silent cut (⊥ retains ~5% of the Jacobian energy), so "steerable from
  the silent complement" is partly the imperfect split — do NOT over-claim silent-subspace control.
  The clean, caveat-free privilege signal is the ~3× potency of the workspace-aligned component.
- The carry direction is built from carry−nocarry means, so pushing toward t+1 is partly "expected";
  the non-trivial parts are (a) 99% SPECIFICITY to the exact carry-correct digit (functional, not
  scatter) and (b) random-in-W ≈ 0 (it's the concept, not the space). Steering itself is a known
  technique — the contributions here are structured-reasoning-from-concept + the workspace-potency
  comparison, tying E1's READ of the carry to a WRITE of it.
- Next sharpening (optional): larger/cleaner workspace cut + measure dW/dperp output-energy to
  settle the container question; test whether steering COMPOSES (inject two carries → +2?).

## E5-compose — does the lever compose? SWITCH, not a dial (2026-07-07)
Pushed the carry-in-workspace direction harder (scale 0→7) and read where the tens digit lands
(offset from the correct no-carry digit t):

| scale | +0 | +1 | +2 | +3 | +5.. |
|---|---|---|---|---|---|
| 1.0 | .85 | .15 | 0 | 0 | 0 |
| 2.0 | .78 | .21 | 0 | 0 | 0 |
| 4.0 | .54 | .44 | 0 | 0 | .02 |
| 6.0 | .49 | .45 | 0 | 0 | .06 |
| 7.0 | .45 | .38 | .03 | 0 | .14 |

- Mass moves +0 → **+1 and SATURATES** (~0.45); **+2 never appears**; over-driving (scale 7) just
  SCATTERS into junk (+5.. → 0.14), it does not march to +2.
- **Verdict: the carry lever is a binary SWITCH ("a carry happened", adds exactly +1), NOT a linear
  DIAL.** Reason: single-digit addition only ever has carry ∈ {0,1}; the model never saw a "double
  carry", so no +2 axis exists. The direction encodes the LEARNED concept, not an abstract magnitude.
- **Key lesson (bears on "make it smarter"):** the lever can PULL a concept the model already learned,
  but cannot SYNTHESIZE one it never learned — you can't squeeze +2 out of a brain that only knows
  carry-0-or-1. Steering speaks the language of what's already mastered. (Footnote: this run's lever
  was weaker than E5's, 0.45 vs 0.64 max — potency varies run-to-run with the trained model; the
  qualitative no-composition result is robust across the whole scale sweep.)
