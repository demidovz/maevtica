# PREREG — Perturbation-Normalized Response Curve (PNRC) on GPT-2-small IOI

Written 2026-07-04 BEFORE running. Thresholds fixed here; they do not move.

## Concept under test

PNRC: for each head, the Δmetric of zero/mean/resample ablations is a function
of a single scalar — the measured perturbation size ε at the intervention site.
Prediction: plotting Δmetric against ε collapses the three protocols onto ONE
monotone curve (R² > 0.9 for a majority of heads); the heads where protocols
flip SIGN are exactly the heads whose fitted curve is non-monotone. If protocol
disagreement persists after conditioning on ε (same ε, different Δmetric), the
representation is falsified.

## Operationalization (fixed)

- Model: gpt2 (small), CPU, transformer_lens. Same 16 IOI prompts as
  ../rce_ioi (one template, 8 name pairs × {ABBA, BABA}, equal token length).
- Metric: LD = logit(IO) − logit(S) at final position, mean over 16 prompts.
  Δmetric(h,p,α) = LD_intervened − LD_clean (signed).
- Intervention on head h=(L,H), protocol p ∈ {zero, mean, resample},
  scale α ∈ {1/3, 2/3, 1}, at ALL positions of hook_z:
    z' = z_clean + α · (z_target_p − z_clean)
  where z_target = 0 (zero) / per-position batch mean (mean) / clean z of the
  batch rolled by 1 (resample; fixed derangement).
- ε(h,p,α) = mean over (batch, positions) of ‖z' − z_clean‖₂ of the head's
  64-dim output, divided by the head's mean clean norm ‖z_clean‖₂ over
  (batch, positions). Computed from the clean cache (exact, no extra passes).
  Note ε is linear in α, so each protocol contributes 3 points on its own ray;
  the collapse claim is about the three rays landing on one curve.
- 9 points per head: 3 protocols × 3 scales. All 144 heads.
- Effect heads (graded set, fixed rule): max over the 9 points of |Δ| ≥ 0.05
  (logit-diff units; clean LD ≈ 3). Flat heads carry no information about
  collapse and are excluded from grading (reported separately).
- Collapse fit per head: isotonic regression of Δ on ε over the pooled 9
  points, fitted both increasing and decreasing, take the better R²
  (R² = 1 − SSres/SStot). PAV implemented in numpy; ties in ε averaged.
  Head "collapses" iff R² > 0.9.
- Sign-flip heads: ∃ protocols p ≠ q with Δ(p, α=1) > +0.05 and
  Δ(q, α=1) < −0.05.
- Direct conditional-disagreement (the falsification clause, reported):
  per effect head, over all cross-protocol point pairs with
  |ε_i − ε_j| ≤ 0.1 · (ε_max − ε_min of that head's 9 points), take
  max |Δ_i − Δ_j| / (Δ_max − Δ_min of that head). Report the median over
  effect heads that have at least one such pair.

## Oracle / positive control (gate before any verdict)

- Clean mean LD > 1.0 and per-prompt argmax(IO vs S) accuracy > 90%.
- |Δ(L9H9, zero, α=1)| ≥ 0.5 (zero-ablating the name mover must move LD).
If either fails → BROKEN_MEASUREMENT, no verdict.

## Decision rule (fixed)

Let E = effect heads, C = {h ∈ E : R² > 0.9}, F = {h ∈ E : sign-flip},
NC = E \ C.

P1 (collapse onto one monotone curve for a majority):
- SUPPORTED iff |C|/|E| ≥ 0.5.
- REFUTED  iff |C|/|E| < 0.5.

P2 (sign-flip heads are EXACTLY the non-monotone heads):
- If F = NC = ∅: vacuously SUPPORTED.
- SUPPORTED iff Jaccard(F, NC) ≥ 0.75.
- REFUTED  iff Jaccard(F, NC) ≤ 0.25 (this covers the case "protocols
  disagree at matched ε without any sign flip", i.e. NC full of non-flip
  heads — exactly the falsification clause).
- Otherwise INCONCLUSIVE.

Overall: SUPPORTED iff P1 and P2 both supported; REFUTED iff either refuted;
else INCONCLUSIVE.

## Budget

One script, one run, CPU. 144 heads × 9 points = 1296 batched forwards
(~0.38 s each ≈ 9 min). No new dependencies.

## Oracle amendment (written 2026-07-04 AFTER a 1-head smoke test on 9.9,
## BEFORE the main 144-head run; decision rules above unchanged)

The original oracle (|Δ(9.9, zero, α=1)| ≥ 0.5) is confounded by self-repair:
../rce_ioi/SANITY_ADDENDUM.md established on this exact setup that the hook
mechanically applies (DLA(9.9)=0 under its own ablation) yet total LD moves
only ~0.117 because backups repair it. A positive control must test the
pipeline, not the phenomenon. Replaced by (all three must hold):

1. Clean mean LD > 1.0 and argmax accuracy > 0.9.
2. Mechanics: in a run with 9.9 zero-ablated at α=1, the cached hook_z of
   9.9 has norm < 1e-4 (the hook applies).
3. End-to-end sensitivity: max over name movers {9.9, 9.6, 10.0} × 3
   protocols at α=1 of |Δ| ≥ 0.5 (some single-head full intervention must
   move LD; the 9.9 smoke already showed resample = −1.34, so this checks
   the batch machinery, not the concept).

If any fails → BROKEN_MEASUREMENT, no verdict.
