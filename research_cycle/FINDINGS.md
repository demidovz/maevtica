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

## Connecting the two bricks — DONE (2026-07-06): REFUTED, and it CONFIRMS the boundary
`campaigns/map-tree/connect_result.json`. Used the MAP geometry (Brick 2) as the
where-to-reflect signal for the model's own concept-mistakes (miscategorizations).
- MAP-drift signal caught miscategorizations at AUC **0.60**; the model's OUTPUT confidence
  at **0.87**. MAP − OUTPUT = −0.265 (CI excludes 0, wrong side) → **REFUTED**.
- Not surprising — it CONFIRMS Brick 1's regime boundary from a new angle: this
  categorization task is EASY for gpt2 (only 16% error = the COMPETENT regime), and there
  the voice already knows where it's wrong. Same law now from 3 angles (rr4 base / rr5
  fine-tuned / connect-categorization): internal & geometric signals beat output confidence
  ONLY when the model is confidently wrong; whenever it's competent, the voice wins.

### Hard regime tested too — REFUTED, connect line CLOSED (2026-07-06)
`campaigns/map-tree/connect_hard_result.json`. Rare/hard members (gpt2 mis-sorts 62%):
MAP AUC **0.38 (below chance!)** vs OUTPUT **0.76**, MAP−OUTPUT −0.39 (CI far from 0) → REFUTED.
Even there the voice isn't blind (entropy on errors 1.46 vs correct 1.16) and still wins.
**Closure:** the two bricks are both real, but they do NOT combine into a better error-router.
The MAP-geometry signal is a poor (even anti-correlated) predictor of the model's actual
concept-mistakes — what the model mis-sorts is NOT what's geometrically ambiguous on our axes;
its own output confidence is the better error-detector in BOTH easy and hard regimes.
Refined understanding: "inside beats voice" (Brick 1) is specific to ARITHMETIC-like
computation where the model commits *confidently* to wrong outputs; it does NOT transfer to
categorization / the map. Do not reopen without a genuinely new mechanism.

## Teaching line — does the inner signal tell us WHERE to teach? REFUTED (simple version)
`campaigns/teach-curriculum/` (teach_curriculum.py, 2026-07-07). Child = gpt2 at partial
addition competence (start 50.7%). Equal teaching budget (K=384, 50 steps); 4 teachers
differ only in WHICH problems they select. Probe error-detection AUC 0.84 (inner signal strong).
- Result (mean acc, 3 seeds): random 52.0 (+1.3) > oracle 48.4 > inner 47.6 > output 47.5.
  Every TARGETED strategy lost to random; inner−output ≈ 0 (+0.11pt) → REFUTED.
- Mechanism: restricting practice to the hard/confused subset causes catastrophic narrowing
  (forgets the broad skill). A strong error-DETECTOR is NOT a good teaching-TARGET selector —
  **diagnosis ≠ cure.** Even the voice's hesitation-boundary (naive ZPD) lost to breadth.
- Scope: tests only RESTRICT-to-subset teaching. The fairer INTERLEAVE test (broad practice +
  extra emphasis on inner-flagged spots, which avoids forgetting) is NOT yet run → the open
  question below.

### Interleave (fair test) done — teaching line CLOSED (2026-07-07)
`campaigns/teach-curriculum/interleave_result.json` (run on CPU — GPU was busy with a
basketbolica video job). Broad practice + light emphasis on inner-flagged spots vs pure broad,
equal budget. Result (start 51.6%, probe AUC 0.85): inner0.3 52.9 ≈ broad 52.6 (PRIMARY
+0.36pt, 2/3 seeds → REFUTED); heavy emphasis inner0.6 49.4 HURTS; output0.3 49.7 HURTS.
Nuance: at matched dose INSIDE beats the VOICE by +3.2pt (Brick-1 consistent) — but neither
beats plain breadth. **Closure across both experiments: breadth beats targeting; a strong
error-DETECTOR (AUC 0.85) does NOT buy a better curriculum. Diagnosis ≠ cure.** Do not reopen
the "route teaching by the inner signal" idea on this toy without a new mechanism (e.g. a real
reasoning model, or targeting COMPOSITIONAL sub-skills rather than whole problems).

## NEXT (open lines, none started)
- κ "one idea, many disguises" (Causal Quotient Feature) — loop's top untested candidate.
- Harden a night-survivor ("shape-from-data") on a bigger model.
- The regime question for Brick 1 on a REAL reasoning model (needs cloud GPU).

## Infra note (2026-07-07)
Local GPU (RTX 3050, 3.68 GiB) is SHARED across studio sessions — a basketbolica neurotrack
job saturated it (clocks clamped to 210MHz). teach_interleave has a CPU fallback
(CUDA_VISIBLE_DEVICES="" + crc-venv311, torch 2.12 cpu) with BATCHED eval/collect; ~20min for
this size. Check `nvidia-smi --query-compute-apps` before launching GPU jobs; don't kill other
sessions' processes.
