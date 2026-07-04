# Sanity addendum — written BEFORE running sanity2.py

The preregistered oracle (zero-abl of 9.9 alone must move LD >= 0.5) failed
(0.117). Post-hoc diagnosis: that control is confounded by the phenomenon
under test — self-repair suppresses the TOTAL effect of ablating a single
name mover. A correct positive control must test the pipeline, not the
phenomenon. Fixed BEFORE running:

1. Clean DLA of L9H9 must be >= 1.0 and be the maximum-DLA head of the
   model (checks the DLA machinery).
2. Jointly zero-ablating all three name movers {9.9, 9.6, 10.0} must move
   LD by >= 1.0 (backups cannot fully repair all three; checks the
   ablation machinery end-to-end).

If BOTH pass: the measurement is declared working, and the main run's
numbers stand; the original oracle failure is reported as an oracle
mis-specification (and is itself evidence self-repair exists on 9.9).
If EITHER fails: BROKEN_MEASUREMENT stands, no verdict.

## Round 2 (final; written before sanity3.py; no further checks after this)

Result of round 1: DLA control PASSED (top-3 DLA = the three name movers,
9.9 max at 3.447). Joint zero-abl control FAILED (LD moved -0.66, i.e. UP).
Undetermined: is the ablation hook broken, or is all-position zero-ablation
of all name movers genuinely LD-increasing (off-distribution + negative-
mover interaction)? Two assumption-light checks, fixed now:

A. Mechanics: in a run with 9.9 zero-ablated, |DLA(9.9)| must be <= 0.1
   (if the hook does not apply, DLA stays ~3.4). PASS = hook applies.
B. In-distribution total effect: joint MEAN-ablation of {9.9,9.6,10.0}
   must drop LD by >= 1.0.

Decision: A and B both pass -> measurement declared WORKING; main-run
numbers stand and decide the verdict per the original PREREG decision
rules; the zero-protocol weirdness is reported as a known zero-ablation
pathology. A fails -> BROKEN_MEASUREMENT final. A passes, B fails ->
INCONCLUSIVE final (cannot separate concept failure from control failure).

## FINAL (per the rules above, no further checks were run)

A PASS (DLA(9.9)=0.0000 under its own zero-ablation — hook mechanically
applies; LD only 3.889->3.772, i.e. near-total repair, witnessed live).
B FAIL (mean-abl of all three name movers: LD drop 0.399 < 1.0).

Verdict: INCONCLUSIVE (preregistered). Honest annotation: the machinery
checks that are NOT confounded by repair all passed (clean LD 3.889,
acc 1.00, DLA recovers exactly 9.9/9.6/10.0 as top-3 unprompted, hook
verified); every total-effect control failed BECAUSE repair is strong,
which is the concept's own premise — this design has no unconfounded
end-to-end positive control. Taking the main-run numbers at face value
they REFUTE both sub-predictions: (1) RCE min pairwise Spearman 0.187 vs
raw 0.379 on the same subset (predicted >0.9 vs <0.6) — RCE is LESS
protocol-invariant than raw; raw mean-resample already agrees at 0.98,
only zero deviates; (2) closures of 9.9 and 9.6 hit the k=6 cap under all
three protocols (predicted k<=3) and include heads outside the known
backup set (11.1, 11.6), though they do heavily overlap it (10.2, 10.6,
10.10, 11.2, and negative movers 10.7, 11.10 recovered unprompted).
So: inconclusive by the letter of the gate, refuted-leaning on the numbers.
Limits: gpt2-small only, one template, 16 prompts, all-position ablation,
one closure operationalization (greedy DLA-compensation, thresh 0.1).
