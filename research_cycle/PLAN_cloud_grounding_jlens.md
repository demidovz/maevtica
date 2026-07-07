# PLAN (cloud): grounding-signal × J-lens → a hallucination detector

Ties our local finding to Anthropic's J-space (2026-07-06, transformer-circuits.pub/2026/workspace).
Status: PLANNED, needs cloud GPU + an open-weights model. Boss-approved to schedule.

## The one-sentence idea
A hallucination is a claim the model STATES but does not hold as GROUNDED internally. Our local
"grounding signal" (gpt2: an internal probe cleanly separates answer-confirmed-by-context from
not-confirmed, beating surprise) + Anthropic's J-lens (reads what a model is "poised to verbalize"
/ what actually lives in its workspace) → on a real model, **detect a hallucination by checking
whether the stated fact is actually present-and-grounded in the workspace, and does that beat the
model's own stated confidence?**

## What each piece contributes
- Ours (local, done): (1) Brick 1 — internal signal beats output confidence at catching errors
  ONLY in the confidently-wrong regime; (2) grounding signal — the model internally flags
  "confirmed vs not-confirmed"; (3) the honest control lesson — a lie is internally IDENTICAL to a
  guess ("not grounded"), so the detector reads GROUNDEDNESS, not deception.
- Theirs (frontier): the J-lens (Jacobian of token-logits w.r.t. activations, averaged over many
  contexts) surfaces the ~10-25 verbalizable concepts in the workspace; ablating them kills
  multi-step reasoning. It's the sharp tool + the frontier target we lacked.

## Design (mirrors our TRUE/LIE/GUESS triad at scale)
Elicit factual claims from an open-weights model on questions with checkable answers:
- GROUNDED-TRUE: model states a fact it demonstrably knows (verified correct).
- HALLUCINATION: model confidently states a fabricated/false fact (verified wrong, high stated conf).
- GUESS/IDK: model hedges or admits not knowing.
At the token where the claim is made, read the WORKSPACE signature:
- (a) J-lens / verbalizable-lens: is the claimed entity present as a grounded top item, or absent?
- (b) our linear grounding-probe on mid-layer residual (confirmed vs not).
Primary metric — hallucination-detection AUC of the workspace/grounding signal, HEAD-TO-HEAD vs the
model's own stated confidence (Brick-1 comparison). Prediction (from our law): internal beats
stated confidence exactly for CONFIDENT hallucinations (sounds sure, ungrounded) — the money case.

## Controls (our hard-won discipline)
- Surprise/perplexity baseline; internal must beat it.
- The GUESS triangulation: predict HALLUCINATION's signature ≈ GUESS's (both "not grounded"),
  NOT a special "deception" signature. If a probe separates HALLUCINATION from GUESS, suspect a
  surface confound (as in our conflict v1). Surface-match where possible.
- Permutation canary; oracle (verified-truth labels).
- Layer robustness (our v1 flip-flopped across layers — instability was the tell).
- AUC ≈ 1.0 is a RED FLAG, not a trophy → adversarial cross-check before believing.

## Staging (cheap → expensive)
1. Light lens on a mid-size open model (7-13B, e.g. Llama-3-8B): use logit-lens / our linear
   grounding probe (no full Jacobian yet) on a factual-QA hallucination set. Establish the
   grounding-vs-confidence head-to-head. Local-ish / small cloud.
2. Reimplement a J-lens (Jacobian-of-logits averaged over contexts) on the same open model; compare
   to the light lens. Does the workspace-membership signal beat a plain probe?
3. Scale to a larger open model (70B) on cloud GPU; test whether the "small-model null" (deep=shallow)
   breaks — i.e. whether a richer workspace shows signatures a small model lacks.

## Honest expected outcome
From our through-line: likely a real GROUNDEDNESS detector (useful for hallucinations), NOT a
"model knows it's lying" signal. Either way decisive: it turns our one working tool (internal
detection in the confident-wrong regime) into something that matters (catching confident
hallucinations) and tests whether scale changes the picture.

## Prior art to cite / not rediscover
Anthropic J-space/global-workspace (2026); hidden-state truthfulness probes (Azaria&Mitchell 2023,
Burns CCS 2022); logit-lens / tuned-lens; hallucination-detection via internal states (recent).
Our fresh angle: grounding-signal head-to-head vs stated confidence, with the GUESS-triangulation
control that separates "ungrounded" from "deception".
