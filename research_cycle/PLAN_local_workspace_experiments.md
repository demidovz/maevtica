# PLAN (local): does a SMALL model have a "workspace", and what lives in it?

Build a LOCAL J-lens (Anthropic's J-space is Claude-only; small scale is unexplored) and run a
battery of original, falsifiable checks anchored on the substrate we already own (arithmetic,
grounding, emotions). Every check: frozen rule + oracle/canary + adversarial cross-check (AUC≈1.0
is a red flag). Models: gpt2, pythia-70m/160m/410m/1.4b, our fine-tuned adder. All local (CPU/GPU).

## Phase 0 — build the tool (foundation)
A "poised-to-say" lens on a small open model, staged:
- **0a. logit-lens proxy**: unembed a mid-layer residual → token distribution = "what it's poised
  to say here". Cheap, immediate.
- **0b. J-lens proper**: Jacobian of output-token logits w.r.t. mid-layer activation, averaged over
  many contexts (corrects for downstream transforms — the Anthropic improvement). Gradients on
  gpt2 are cheap.
- Validate: reproduce a KNOWN unspoken intermediate (e.g. "the animal that spins webs has __ legs"
  → is "spider" in the lens before "8"? or an arithmetic intermediate). Oracle: swapping the
  intermediate in the lens must change the answer (causal check).

## The battery (each is a clean, original check)

### E1 — "Errors = the missing intermediate" (unifies Brick 1 ⊕ J-space) ★ recommended first
Hypothesis: our adder's arithmetic ERRORS are exactly the cases where the correct INTERMEDIATE
(e.g. the carry) never entered the workspace. Read the J-lens for the carry/partial-sum before the
answer; test whether "intermediate present in workspace" predicts correctness BETTER than output
confidence (Brick-1 head-to-head). Oracle: on solved problems the intermediate is present ≥X%.
Falsify: if presence-of-intermediate doesn't beat output confidence → errors aren't "missing
thoughts". Uses substrate we already have.

### E2 — "Workspace emergence curve" (scale, tests our through-line mechanistically)
Anthropic: on Claude the workspace is ~10% of variance, ~10-25 concepts, causally needed for
reasoning. Original check: measure the verbalizable-subspace fraction + its causal necessity
(ablate → reasoning drop) across pythia-70m→1.4b. Hypothesis: the workspace EMERGES with size —
tiny models have little/none (which would mechanistically explain our "small models are shallow,
deep reduces to shallow" through-line). Oracle: ablating the workspace must hurt multi-step tasks
more than single-step. Falsify: if even 70m has a full workspace → "shallowness" isn't about the
workspace.

### E3 — "Confabulation = empty workspace" (local seed of the cloud hallucination detector)
Ask about a MADE-UP entity ("What color is a florpine?") → the model fabricates confidently. Read
the workspace: is it "nothing grounded" (our GUESS signature) even as the output confidently
invents? Test: workspace-groundedness detects confabulation better than stated confidence. This is
the cloud plan, dry-run locally. Control (our lesson): confabulation ≈ honest "I don't know"
internally, NOT a special "lying" signature.

### E4 — "Tip of the tongue" (knows-but-can't-say — the MIRROR of hallucination) ★ original
The reverse of confabulation: cases where the answer IS in the workspace but the mouth won't emit
it (low output prob — e.g. rare word, suppressed token). Does the J-lens contain the word even
when output ranks it low? A mechanistic "tip of the tongue". Oracle: force-decode the word and
verify it was the intended one. Original: maps the OTHER half of knows≠says (we did says-not-known;
this is known-not-said).

### E5 — "Privileged injection" (causal test of workspace privilege) ★ original
Anthropic ablated the workspace (reasoning died). Ours as INJECTION: plant the SAME concept vector
(a) inside the workspace subspace vs (b) in the orthogonal 90% (out-of-workspace), matched norm.
Does in-workspace injection change the model's reasoning MORE? If yes → the workspace is causally
privileged for thought, not just readable. Ties to Brick-2 steering (we know how to inject) + the
clean-window discipline (calibrate strength). Falsify: if out-of-workspace injection is just as
potent → the workspace isn't privileged, only observable.

### E6 — "Belief update / false memory" (original, uses grounding signal)
Establish fact X; later context overrides it to X'; then query. Does the grounding signal track the
LATEST binding (updated belief) or the first (false memory)? A mechanistic belief-update probe.
Oracle: the model's answer follows the latest → grounding should too.

## Sequencing
Phase 0 (tool) → E1 (highest value, reuses our adder, unifies two bricks) → E3 (hallucination
seed, feeds the cloud plan) → E2 (scale/through-line) → E4/E5/E6 as the tool matures.

## Discipline (non-negotiable, from this whole arc)
Frozen rule before running; oracle canary each measurement; AUC≈1.0 ⇒ adversarial cross-check
before believing; surface-match to kill confounds; report the mechanical verdict even when it
stings; "survived ≠ proven".
