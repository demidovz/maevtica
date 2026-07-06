# maevtica research — FINDINGS & resume point

**Last updated: 2026-07-06 (Илья).** Single source of truth for what this research
cycle has established. New session → read this first, then `RUNS.md` (run log) and
`queued/README.md` (queue). Campaign detail lives in `campaigns/<name>/*REPORT.md`.

Studio persona rule: this doc is INTERNAL (technical ok). Anything shown to the boss
goes in the 5-year-old register — see `../../maestratica/CLAUDE.md`.

---

## Confirmed bricks (toward the boss's "map + reflection loop → smarter" vision)

### Brick 1 — the model's INSIDE knows its errors, but only in one regime
`campaigns/reflect-route/` (rr4 SUPPORTED, rr5 DEEP maps the boundary).
- **rr4 (base gpt2, arithmetic):** an internal-activation probe detects the model's own
  errors far better than its output confidence — internal error-detection AUC 0.84 vs
  output 0.50 (chance). Clean, preregistered, controls green. 3 fresh datasets agree.
- **rr5 (gpt2 fine-tuned to ~50% error, un-saturated):** NO practical routing gain
  (internal − output = −0.06pt, CI incl. 0). Because once the model is COMPETENT its
  output confidence learns to know where it's wrong (detection 0.50 → 0.86).
- **Synthesis:** internal-signal error-routing beats output-confidence routing EXACTLY
  when the model is *confidently wrong* (doesn't know the skill); it adds nothing once the
  model is competent (the voice already knows). A tool for the overconfident-error /
  hallucination regime, not a free lunch.
- Prior art: hidden-state error probing (Azaria&Mitchell 2023; CCS Burns 2022). Our bit:
  head-to-head vs output-confidence routing at equal budget + the regime boundary.

### Brick 2 — the concept hierarchy is a navigable (fuzzy) TREE
`campaigns/map-tree/` (map-navigate v3 SUPPORTED).
- gpt2, 6 categories. A held-out word lands on its correct branch 67% (chance 17%, ×4);
  category directions are distinct axes (mean |cos| 0.34) → a REAL but FUZZY tree.
- You can WALK it by arithmetic: steering by (−catA + catB) flips "A {w} is a type of ___"
  to B ~100% at gentle strength, while a random push of matched size does ~nothing (0–12%).
  The SAME move works across the whole 6×6 category matrix → navigable.
- Prior art: Park 2024 (hierarchical concept geometry). Our bit: causal navigability +
  the clean-window-vs-brute-disruption distinction.

## Closed / refuted lines
- **sep-boundary** (predict where a clean concept-swap holds, beating Park's baseline):
  REFUTED on a fair 410m test — our grad-overlap predictor is noise (below chance, wrong
  sign); Park's orthogonality baseline actually works. sb1's lead was a 32-sample fluke.
- **self-distillation into weights as a RESEARCH goal:** dropped (boss call) — it's just
  re-implementing STaR/standard training; kept only as a GPU skill (`GPU.md`).

---

## Methodology lessons (hard-won — do not relearn)
1. **Calibrate the measurement BEFORE freezing the decision rule.** Bit us 3×: (a) 410m
   steering strength 10× too weak → degenerate outcome; (b) fine-tuned model degrades under
   a BOS it never trained with (48→74% err) → analyze with `prepend_bos=False`; (c) map-walk
   over-driven at oracle-MAX strength → random flips too.
2. **For DIRECTIONAL steering tests, calibrate strength to the CLEAN WINDOW** (directed move
   works AND a random push of matched size does NOT), never to max-oracle intensification —
   a strong push disrupts everything and the directed effect becomes indistinguishable from noise.
3. **Oracle / positive-control canary** on every measurement: if steering a genuine member
   by its own direction doesn't move the target logit ≥1, the run is BROKEN, not refuted.
4. **Don't move goalposts.** When a frozen rule mislabels (rr3 backwards clause), fix the
   ONE mis-specified clause, fresh data, re-freeze — never reinterpret old data in your favor.
   Report the mechanical verdict honestly even when it stings.
5. **Judge WITHIN groups**, not pooled (Simpson's paradox killed a pooled sep-boundary read).
6. **Un-saturate the metric:** a 90%+ error rate collapses final-accuracy dynamic range;
   fine-tune to ~50% or pick a difficulty band first.

## Infrastructure
- **The loop** `cycle.workflow.js` — now has an **anti-substitution guard** (2026-07-06):
  tester must declare `ran_model/n_samples/oracle/deviated_from_request(+proof)`; an
  independent VERIFIER cross-checks vs the order and VOIDS unproven deviations to
  inconclusive. Added after a tester swapped 410m→gpt2 on a false "not cached" premise.
- **Treasurer** `treasurer.py` — sizes campaigns in live `/usage` week-points (`mst-usage --json`).
- **Envs:** analysis CPU `~/.local/state/mst/crc-venv311` (transformer_lens); fine-tune GPU
  `~/.local/state/mst/gpu-venv` (cu121, RTX 3050). See `GPU.md`.
- **Running experiments:** author decisive scripts by hand in `experiments/`; launch DETACHED
  via `setsid bash -c '... > log 2>&1' &` (a plain nohup dies when the launching shell is
  killed at the 2-min tool timeout). Workflow `args` must stay < ~1.9KB (transport truncates).

---

## NEXT: connect the two bricks
**Question:** does a MAP-based signal — "this token's concept has drifted off its clean
branch" (distance/ambiguity in the tree geometry) — route reflection to the model's errors
better than output confidence, ESPECIALLY in the confidently-wrong regime where Brick 1 says
internal signals win? I.e. use the *map* (Brick 2) as the *where-to-reflect* signal (Brick 1).
Preregister; oracle canary; compare map-signal vs output-confidence vs internal-probe vs random;
run in the confidently-wrong regime (base model) AND the competent regime (fine-tuned) to see
if the map signal, unlike a raw internal probe, still helps once the model is competent.
