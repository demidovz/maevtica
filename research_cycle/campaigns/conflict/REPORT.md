# The "says-against-its-own-grain" conflict signal — INCONCLUSIVE (confound caught) 2026-07-07

**The flashy result did NOT survive scrutiny.** A first pass looked like a perfect internal
"conflict / unease of lying" signal (AUC 1.000). An adversarial cross-check exposed it as a
surface/binding confound. Recording honestly: no clean conflict signal established.

## Setup
gpt2, in-context fact. Model tracks the fact perfectly (honest acc 1.00). Three conditions,
residual read at the answer token:
- A honest: bound item, correct answer. B lie: bound item, wrong answer (CONFLICT). C guess:
  unbound item, wrong answer (no conflict). B-vs-C was meant to isolate conflict from surprise.

## First pass (conflict_signal.py) — looked SUPPORTED, but suspicious
- B-vs-C: internal AUC **1.000** (L9) vs surprise AUC 0.522 (Δ+0.478, perm 0.51). Surprise
  ruled out. **But AUC=1.000 is a red flag, not a trophy** → adversarial cross-check.

## Cross-check (conflict_crosscheck.py) — the confound
All three pairwise contrasts are perfectly separable (AUC≈1.0 for B-vs-C, B-vs-A, A-vs-C at
both layers) → the residual encodes the CONDITION'S SURFACE STRUCTURE (which item was queried,
whether the answer word repeats a context word), not a single "conflict" axis. Decisive tell —
where honest-A lands on the B-vs-C ("conflict") probe:
- **L6: P(A)=0.16 ≈ C (0.01)** — A groups with no-conflict (consistent with a real dissonance component).
- **L9: P(A)=1.00 ≈ B (1.00)** — A groups with the lie (probe reads BINDING-presence, not dissonance).
Layer-dependent and contradictory → the B-vs-C separation is **NOT cleanly a dissonance signal**.

## Honest verdict: INCONCLUSIVE / confounded
The apparent "unease of lying" signal is explained by surface + binding structure (B queries the
bound item, C the unbound; the answer word repeats context or not). At gpt2 scale, what looks
like "the model knows it's lying" is not separable here from mundane surface consistency-tracking.
Fits the project through-line: the "deep" reading (dissonance/self-knowledge) keeps reducing to
something shallower once you control for it.

## One faint lead (do NOT overclaim)
At the EARLIER layer (L6) honest-A separates from lie-B in a conflict-consistent direction
(A~C). A cleaner, SURFACE-MATCHED design (identical answer sentence; vary only context; control
the repeat-detection confound; read at the predicting position) might isolate a real signal — or
confirm there is none. Open.

**Method win: a striking AUC=1.000 was properly deflated by an adversarial A-cross-check before
it reached the boss. Survived ≠ proven; this one did not survive.**

---

# DEFINITIVE surface-matched redesign (conflict_clean.py) — 2026-07-07

Clean this time: TRUE/LIE are token-twins (containers W,Z swapped between "put in" and "saw
near"), same queried item, IDENTICAL answer sentence. Perm at chance (0.505); model holds the
binding (honest 0.77). And it gives a **decisive, honest answer.**

| layer | internal AUC (LIE vs TRUE) | surprise AUC | P(LIE) | P(TRUE) | P(GUESS) |
|---|---|---|---|---|---|
| 6/8/9/10 | **1.00** | 0.79 | 1.00 | 0.00 | **0.99** |

- Internal separates LIE from TRUE perfectly and **beats surprise** (Δ+0.21, CI[+0.18,+0.25]) →
  there IS a real internal signal, not just surprise.
- **But the triangulation kills the "unease" reading:** GUESS (ungrounded, no conflict) lands
  with LIE (P=0.99), NOT with TRUE. So the axis is **{LIE, GUESS} vs {TRUE}** =
  **"answer NOT confirmed by context" vs "answer confirmed"** — a CONSISTENCY / GROUNDING check,
  not a dissonance/"I'm-contradicting-what-I-know" signal. robust 0/4 for the conflict reading.

## Verdict: dissonance/unease REFUTED — but a real consistency signal found
The honest answer to "knows≠says / does it feel the unease of lying":
- **YES**, the model has a clean internal marker of whether what it's saying is CONFIRMED by
  what it was told (grounded) vs not — beyond surprise, robust across layers.
- **NO** distinct "unease of lying": a LIE (contradicting a known fact) is internally
  IDENTICAL to a GUESS (no known fact). To the model, lying and guessing are the same thing —
  "saying something unconfirmed." The imagined pang of shame isn't there; there's a bookkeeping
  flag "confirmed / not confirmed."

Textbook through-line: the deep reading (unease/self-knowledge) reduces to a shallower, real
one (grounding/consistency tracking) — shown cleanly, with the confound removed. Line CLOSED.
