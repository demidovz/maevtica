# research_cycle — the self-sustaining research loop (v1 scaffold)

Turns maevtica's validated methodology into a budgeted, running loop that shines
a "flashlight" into one meaning-space, generates concepts, **tries to kill them
against real data**, keeps survivors, stops at a budget, and reports honestly.

Vision: `wiki/projects/maevtica/self-sustaining-cycle-sketch.md` (in maestratica).
This directory is the prepared machinery for gaps #1-#3 of that sketch.

## The three built pieces

| Gap | Piece | Status |
| --- | --- | --- |
| #1 Budget cap | **`treasurer.py`** — «казначей»: turns "20% of the weekly Claude limit" into a token cap (reads `~/.config/mst/fuel.toml`, ~467M/wk → ~93M), meters the campaign, hard-stops at the cap. | ✅ works + tested |
| #2 The loop | **`cycle.workflow.js`** — 7-stage falsification loop (Explore→Generate→Attack→Test→Report) as a runnable Workflow, budget-gated, with an anti-self-fooling CONTROL stage. | ✅ ready to run (not run) |
| #3 The teeth | **`experiments/harness.py`** + **`PROTOCOL.md`** — reusable primitives for real external tests, with the oracle-sanity bug-catcher baked in; the doctrine every test follows. | ✅ library + doctrine |

## How a campaign runs (opt-in — nothing runs on its own)

```
# 1. open the campaign & size the budget (казначей)
python3 research_cycle/treasurer.py open mechinterp-2026-07 \
        --domain "mechanistic interpretability" --frac 0.20
# → cap ≈ 93,400,000 tokens (20% of 467M/week)

# 2. run the loop as a Workflow, PASSING the cap as args.capTokens
#    (Workflow tool, args: {domain, capTokens: <cap from step 1>, maxRounds})
#    the loop gates every round on budget.spent() < capTokens and self-stops.
#    the returned {survivors, killed, report} is the boss-facing result
```

> **2026-07-03 shakedown lesson (fixed):** the loop must gate on `budget.spent()`
> — NOT on `budget.total`/`budget.remaining()`, which are null unless the *turn*
> carries a `+Nk` token directive. The first shakedown relied on the latter, so
> the 3% cap silently didn't apply and the loop ran ~5 rounds / 51 agents until
> stopped by hand. Now `args.capTokens` + `budget.spent()` enforce the slice, and
> `maxRounds` (default 6) is a hard backstop. Always pass `capTokens`.

The Claude budget is spent on THINKING (explore/generate/attack/test-design/
report). The experiments themselves run on LOCAL open models
(`~/.local/state/mst/crc-venv311`, transformer_lens) — near-free vs the Claude
limit. So "20% of weekly" buys a lot of reasoning + a report, not burned on runs.

## Guardrails (the three that keep it from drifting or burning)

- **Anti-self-fooling:** every N rounds a CONTROL stage runs known-good concepts +
  planted distractors through the same bar; if the judges can't tell them apart,
  the campaign halts. Plus the oracle-sanity gate in `harness.verdict()`.
- **Anti-token-burn** (the 2026-07-03 lesson — idle Claude ate a week of fuel):
  the казначей hard-stops at the slice; the loop breaks when stress is exhausted;
  no artifact per round → it winds down, never spins idle.
- **Honest expectations:** most pans are empty. The report says so plainly; the
  value is the rare survivor + the refutation trail, not a flood of "discoveries".

## What v1 is and isn't (honest)

- **Is:** a budgeted, budget-safe loop that generates → attacks → prior-art-checks
  → externally-tests (for the shapes it knows) → reports, with controls.
- **Isn't:** autonomous domain-choice (boss seeds the domain), autonomous
  experiment-design for arbitrary predictions (novel shapes = Claude-in-loop),
  or an SAE-grade interp test suite. Those are v2 — see `experiments/PROTOCOL.md`.

## First flashlight

Mechanistic interpretability: infra exists, one survivor exists (Causal Role
Carrier — `../crc_external_test/SUMMARY.md`), and predictions are computable, so
the teeth actually bite. Prove the loop on one narrow domain before widening.
