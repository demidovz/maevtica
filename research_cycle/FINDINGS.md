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

## PLANNED — CLOUD (boss-approved to schedule, needs a bigger machine) 2026-07-07
- **Hallucination detection on a REAL model.** Take Brick 1's confirmed finding (internal
  states catch confident errors the voice misses, in the confidently-wrong regime) and test it
  where it MATTERS: does an internal probe on a real reasoning model (7B+ doing QA/CoT) catch
  CONFIDENT FALSEHOODS (hallucinations) better than the model's own confidence? This both tests
  generalization of the toy law AND turns our one working tool into something useful. Needs
  cloud GPU. Boss said: keep working locally for now, but this goes in the plans. — DONE (here).
  DESIGN written: `PLAN_cloud_grounding_jlens.md` — our grounding-signal × Anthropic's J-lens
  (J-space, 2026-07-06 transformer-circuits.pub/2026/workspace). Detect a hallucination = a claim
  STATED but not GROUNDED in the workspace; head-to-head vs the model's own stated confidence
  (Brick-1). Staged: light lens on 7-13B → reimplement J-lens → scale to 70B. Honest expectation:
  a real groundedness detector, NOT "knows it's lying" (our GUESS-triangulation: lie ≈ guess internally).

## Conflict / "knows≠says" signal — INCONCLUSIVE (confound caught) 2026-07-07
`campaigns/conflict/` (conflict_signal.py + conflict_crosscheck.py, CPU).
- Warmup `emotions_tree`: feeling words (fear/anger/shame/pride/joy/love) form a CLEAN tree
  (LOO 0.51 vs chance 0.17, |cos| 0.21, shuffle-canary at chance) — concept-reading, not feeling.
- Main: gpt2 tracks an in-context fact perfectly (honest acc 1.00). First pass: internal probe
  separated conflict(B) from no-conflict(C) at AUC **1.000** while surprise was at chance (0.52)
  → looked like a clean "unease of lying" signal. **AUC=1.0 = red flag → adversarial cross-check.**
- Cross-check: ALL of A/B/C are mutually separable (surface/binding structure), and honest-A
  lands with C at L6 (conflict-like) but with B at L9 (binding-like) → the B-vs-C separation is
  NOT cleanly dissonance; it's a surface/binding confound. **INCONCLUSIVE — no clean conflict
  signal established.** Faint lead at L6 for a SURFACE-MATCHED redesign. Fits the through-line:
  the "deep" reading (self-knowledge/unease) reduces to something shallower once controlled.
- Method win: a striking AUC=1.000 was deflated by adversarial checking before reaching the boss.
- **DEFINITIVE surface-matched redesign (conflict_clean.py) — line CLOSED, clean answer:**
  TRUE/LIE are token-twins (W,Z swapped), same item, identical answer sentence; perm clean (0.51),
  model holds binding (0.77). Internal separates LIE from TRUE at AUC 1.0 BEYOND surprise
  (Δ+0.21, CI excl 0) — a REAL signal. But GUESS lands with LIE (P=0.99), not TRUE → the axis is
  {LIE,GUESS} vs {TRUE} = **"answer NOT confirmed by context" vs "confirmed"** = a CONSISTENCY /
  GROUNDING check, NOT dissonance. **Answer: the model has a clean "is what I say grounded in what
  I was told" signal (YES), but NO distinct "unease of lying" — a lie is internally identical to a
  guess.** Deep reading (unease/self-knowledge) reduces cleanly to a shallow real one (grounding).

## LOCAL WORKSPACE PROGRAM (started 2026-07-07) — plan in PLAN_local_workspace_experiments.md
Build a local J-lens on small models + a battery of original checks. Phase 0a (logit-lens) BUILT
& validated: reads "poised-to-say" (cold/animal emerge across layers); gpt2-base too weak for
multi-hop unspoken intermediates (spider→8) → use arithmetic.
- **E1 "Errors = the missing intermediate" — SUPPORTED** `campaigns/workspace/`. gpt2 adder,
  the tens-digit's unspoken intermediate = the units CARRY. Errors concentrate on carry problems
  (+5.7pt); **92% of carry-errors are exactly "forgot the carry"**; and an internal probe flags
  the error at AUC **0.83 vs output confidence 0.60** (Δ+0.23 CI excl 0, perm 0.55, believable).
  → An arithmetic error IS a missing workspace intermediate, visible inside BEFORE the wrong digit
  is spoken. Unifies Brick 1 ⊕ J-space. (Note: this is a POSITIVE for interp-as-detector, our
  through-line's one working mode — detection, in the confidently-wrong regime.)
- **E3 "Confabulation = empty workspace" — REFUTED (informative)** `campaigns/workspace/`. Synthetic
  person→city memory, 25% persons hidden. The model HEDGES honestly on strangers (output entropy
  0.10 grounded vs 0.50 confab) → NOT confident confabulation. Internal AUC 0.826 vs output-conf 0.843
  (Δ−0.017, CI incl 0, perm 0.48) → inside does NOT beat the voice. The workspace IS emptier but the
  OUTPUT already shows it → no hidden signal. **E1⊕E3 draw the line: interp beats the voice ONLY in
  the confidently-WRONG regime; when the voice honestly hedges, inside adds nothing.** Same through-line
  ("deep reduces to shallow-but-real"). Would flip only under CONFIDENT confabulation → test on cloud.
- **E5 "Privileged injection: lever or lens?" — MIXED (biggest result)** `campaigns/workspace/`. Built
  the real J-lens (Jacobian): K=8 workspace captures 94.7% of digit-output effect. READ: carry
  decodable from workspace 0.97 but ALSO complement 0.92 → not locked in the verbalizable subspace.
  CAUSAL (money): injecting the CARRY direction → model produces specifically t+1 (carry-applied)
  P=0.64, **99% specificity** ("reasons from the injected concept"); random-in-W ≈0.07 (it's the
  concept, not the space); carry-in-complement 0.38 but needs 3× push. → Strong "privileged container"
  REFUTED, but **the first real causal LEVER in the arc**: a clean concept-direction is a usable knob,
  and the workspace is ~3× more potent. **Through-line AMENDED: detection-only holds for READING error
  signals (E1/E3); but STEERING a well-identified concept-direction DOES redirect reasoning.** Closest
  yet to the boss's "use the map to make it smarter" = local yes-in-principle. Caveat: K=8 soft cut
  (⊥ keeps ~5% energy) → don't over-claim silent-subspace control; clean signal = 3× workspace potency.
- Next in program: E2 (emergence curve), E4 tip-of-tongue, E6 belief update; E5 sharpening
  (cleaner cut + does steering COMPOSE: two carries → +2?).

## SUPERSEDED / earlier direction note (2026-07-07): reading the hierarchy beyond "lying"
The boss asked: reading the concept hierarchy — can we find more than lying? shame, guilt? +
he likes the "boundary between what it KNOWS and what it can SAY". These FUSE into one clean,
cheap, falsifiable local experiment:
- **The "says-against-its-own-grain" conflict signal.** Honest framing (no mysticism): a small
  LM does not FEEL shame; but we can ask whether, when it OUTPUTS X while its internals ENCODE
  not-X, there is a measurable internal CONFLICT marker — distinct from a plain error. Three
  cases: (1) honest truth, (2) forced/known falsehood (internals say true, output says false →
  predicted conflict), (3) confident error (internals also wrong → no conflict). A signal that
  fires ONLY in case (2) is simultaneously: the KNOWS≠SAYS boundary made concrete, AND the
  closest honest mechanistic analog of the "unease of lying" (a proto-guilt/shame signal).
  Preregister; oracle canary; compare the conflict-probe across the 3 cases; control that it's
  not just detecting the falsehood token.
- Cheaper warmup / stepping stone: do emotion/evaluation words (shame, guilt, pride, fear) sit
  as clean DIRECTIONS in the same navigable tree as fruit/bird (Brick 2 machinery)? If yes,
  does the model apply an evaluation axis to ITSELF (self-reference) vs to others?

## NEXT (other open lines, none started)
- κ "one idea, many disguises" (Causal Quotient Feature) — clean but another "understand", and
  our through-line says understanding rarely converts to intervention. Lower priority.
- Harden a night-survivor ("shape-from-data") on a bigger model.

## THROUGH-LINE (name it, 2026-07-07)
Across connect / teaching / (map-as-detector): interp is a strong DIAGNOSTIC (detection, in the
confident-fool regime) but has REPEATEDLY failed to beat the model's own outputs as an
INTERVENTION (routing, teaching, lie-catching). Empirical confirmation of the VISION's original
caveat "understand ≠ improve" — 3× from 3 angles. Design future work knowing this: push the
DETECTOR where it's useful (hallucinations), or map the DETECTION boundary sharply (knows≠says).

## Infra note (2026-07-07)
Local GPU (RTX 3050, 3.68 GiB) is SHARED across studio sessions — a basketbolica neurotrack
job saturated it (clocks clamped to 210MHz). teach_interleave has a CPU fallback
(CUDA_VISIBLE_DEVICES="" + crc-venv311, torch 2.12 cpu) with BATCHED eval/collect; ~20min for
this size. Check `nvidia-smi --query-compute-apps` before launching GPU jobs; don't kill other
sessions' processes.
