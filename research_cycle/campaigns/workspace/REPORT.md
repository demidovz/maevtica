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

## малыш-крошка — structure-guided vs blind compression — BLIND WINS BIG (2026-07-07)
Compressed the adder's residual at layer L, "=" position, to a k-dim bottleneck, three ways:
SMART = keep the Jacobian directions (what our lens reads as driving the digit output);
BLIND = keep the highest-variance directions (PCA, structure-blind); RANDOM = k random dirs.

| k | SMART (Jacobian) | BLIND (PCA) | RANDOM |
|---|---|---|---|
| 1 | .17 (chance) | .60 | .17 |
| 2 | .17 | **.93** | .17 |
| 8 | .31 | **.96 (=full)** | .17 |
| 16 | .91 | .96 | .17 |
| 32 | .95 | .96 | .17 |
| 128 | .96 | .96 | .30 |

Full (no compression) = 0.957. **k needed to reach 95% of full: BLIND=2, SMART=32, RANDOM=never.**

## Reading (honest, and it stings the right way)
- **Blind variance-compression (PCA) beats our structure(Jacobian)-guided compression by ~16×**
  (k=2 vs k=32). Reading the output-structure is the WRONG tool for compression — it actively hurt.
- **Why (and it's instructive):** the Jacobian directions are "what linearly maps to the answer at
  this layer" — but compression asks the UPPER layers to still COMPUTE. To compute they need the raw
  INGREDIENTS (the two digits), which live in the HIGH-VARIANCE directions. Projecting onto the
  Jacobian directions keeps the output-readout but throws away the ingredients → the upper layers
  can't recompute → chance until you keep enough (k≥16). PCA keeps the ingredients → computes fine
  at k=2. The useful "structure" for compression is the INFORMATION CONTENT (variance), and plain
  statistics find it with zero interp.
- **Sharpens E5:** the 8-dim workspace captured 94.7% of the OUTPUT-EFFECT (reading out), NOT the
  computation's ingredients. Output-readout ≠ compute-substrate. Do not conflate them.
- **Answer to the boss's idea (b):** does reading the concept-structure (our way) compress better than
  blind? NO — the opposite, decisively. This is hard-number confirmation of what we told him about
  distillation: the transfer/compression is best done by dumb gradients/statistics (PCA, standard KD),
  NOT by interp-surgery. Reinforces the through-line: interp is a DIAGNOSTIC, not a superior engineering
  lever. (Nuance for fairness: PCA is itself implicitly finding the right structure — "structure
  matters", just not the output-map structure our lens reads; plain variance already captures it.)

## минимальный мозг под дерево — capacity threshold for exact structure (2026-07-08)
How small a net stores a species tree EXACTLY (0 hallucinated edges)? Tiny MLPs memorize "node->parent"
for random rooted trees; sweep size, find the smallest with 100% exact recall. Theory: tree = log2((V-1)!)
≈ V·log2(V) bits; big LMs store ~2 bits/param -> floor ≈ bits/2.

| V | tree bits | threshold params | bits/param | min width w* |
|---|---|---|---|---|
| 32 | 113 | 308 | 0.37 | 4 |
| 64 | 290 | 460 | 0.63 | 3 |
| 128 | 709 | 1706 | 0.42 | 6 |
| 256 | 1676 | 3370 | 0.50 | 6 |

## Findings
- **Min size scales ~linearly with tree info (V·log2V), as theory predicts.** Threshold params ≈ **2× the
  tree's bit-content** → achieved **~0.5 bits/param**, i.e. ~4× BELOW the 2-bits/param ceiling of big
  optimized transformers. Tiny memorizer-MLPs are less frugal; the practical floor is ~4× the theory floor.
- **Minimal width w* ≈ log2(V)** — just enough distinct "addresses" to separate the V species (V=256→w≈6-8).
- **The boundary is a SOFT slope, not a cliff:** at V=256, recall 0.42→0.82→0.92→1.00 across a ~3× param
  range. Below threshold the net confuses a GROWING fraction of parents = graceful hallucination, not collapse.
- **Depth does NOT help storage:** 2 hidden layers ≈ 1 layer for the threshold (w*≈6 both). Capacity (total
  params) stores edges; depth would matter only for MULTI-HOP queries (ancestry "is A above B?"), a natural v2.

## Concrete answer
To hold a V-species tree exactly: **≈ 2·V·log2(V) parameters** (measured), vs ~V·log2(V)/2 theoretical floor.
E.g. 256 species → ~3.4k params; 1000 species → ~17k params (measured) / ~4-5k (theory floor). Shrink below
and it starts hallucinating edges, more as it shrinks; the edge is a slope, not a cliff; extra layers don't
buy exactness, only size does.

## как оно забывает — HOW a too-small net forgets a tree — STRUCTURED (2026-07-08)
Below the exact-storage threshold, WHICH edges break? Tiny MLPs at sub-threshold widths (V=256);
correlate edge-correctness with node DEPTH (trunk=shallow), SUBTREE SIZE, PARENT POPULARITY (hub).

| w | recall | correctness~popular-parent AUC | ~shallow AUC | ~subtree | wrong→hub (base .11) | pred in-deg (base 1.0) |
|---|---|---|---|---|---|---|
| 2 | 0.45 | **0.839** | 0.729 | 0.573 | **0.56** | 3.4 |
| 3 | 0.83 | 0.622 | 0.553 | 0.572 | 0.48 | 2.5 |
| 4 | 0.93 | 0.50 | 0.50 | 0.44 | **0.95** | 3.4 |
(perm canary ~0.49-0.52 = clean. correct-vs-wrong at w=2: subtree 6.8 vs 2.9, depth 3.8 vs 5.4.)

## Findings — forgetting is FREQUENCY-STRUCTURED, not random
- **Keeps the trunk/hubs, loses the twigs.** The strongest survival predictor is PARENT POPULARITY
  (AUC 0.84 at heavy compression) — frequent/hub links survive, rare deep twigs die. Depth (0.73) and
  subtree (0.57) follow, and are entangled with popularity (trunk nodes ARE frequent parents). Correct
  edges are shallower (3.8 vs 5.4) and bigger branches (6.8 vs 2.9).
- **Failure = collapse to the hub.** When wrong, it guesses a POPULAR node — 56% of errors land on a
  top-10% hub (vs 11% chance), rising to 95% of the residual errors near-threshold; wrong-guess parents
  are 3.4× more popular than random. "Don't know the exact parent → name the frequent one."
- **Same behaviour as confabulation (E3).** Ungrounded → fall back to the frequent/popular guess. The
  net is a lossy JPEG of the tree: it keeps the coarse hierarchy and blurs the fine detail toward hubs.

## Honesty on novelty
The underlying PRINCIPLE — frequency bias of memorization (frequent survives, rare forgotten) + undertrained
fallback to the high-prior answer — is KNOWN. What's ours: a clean demonstration on tree STRUCTURE
(trunk-survives / twig-dies), tied to the measured capacity threshold, and the connection that "how it
forgets a structure" = "confabulate toward the hub" = the same through-line thread. A nice bridge, not a
new law. (Did not run a formal lit-search; would before any novelty claim.)
