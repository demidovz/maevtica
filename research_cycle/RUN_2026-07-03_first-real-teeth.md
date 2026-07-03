# First real-tester run — the teeth actually bit (2026-07-03)

Campaign `shakedown2-real`, domain: mechanistic interpretability. First run with
the tester ALLOWED TO RUN experiments (not dry). 4 concepts reached the Test
stage; each tester autonomously installed real tooling (pyvene, sae_lens), loaded
real SAEs (gpt2-small-res-jb, d_sae=24576), wired an oracle/positive-control, and
ran — 20-36 Bash calls apiece. This is the first time the full loop produced
EMPIRICAL verdicts on data, not armchair opinion.

## Verdicts (of the 4 that reached the teeth)

| concept | status | verdict | oracle | note |
| --- | --- | --- | --- | --- |
| **Reader-direction** (weight-grounded feature) | ran | **SUPPORTED** | ✅ passed | Spearman(salience,ρ)=−0.023, Pearson=0.061 — near-zero, inside SUPPORTED band, far from the 0.5 refute line. Oracle valid (ρ(top eigvec)=28.70=√λmax) — measurement real, not the .norm-bug mode. |
| **Behavioral lever** (dose-response feature) | ran | **REFUTED** | ✅ passed | SAE count=3 best=4.741 CI[3.56,5.89] vs PCA count=0 best=2.25 CI[1.56,2.93]; positive control Δ-logit 3.72–5.53 ≥ min_oracle 3.0. Honest refutation WITH a valid oracle. |
| **Functional variable** (causal-abstraction slot) | designed_not_run | inconclusive | wired, not observed | Installed pyvene+sae_lens, loaded SAE, oracle (IIA≥0.75) wired — still executing when force-finalized. |
| **Certification triple** (indexed feature) | — | aborted | — | Agent did not finish (22 Bash calls, no final output). |

**One SUPPORTED, one REFUTED — both with valid oracle checks. Differentiated,
honest verdicts from real experiments.** Caveat: single self-designed run, small
models, designer risk — "survived one test" ≠ "proven". But the machinery
demonstrably does autonomous research with controls.

## Two engineering bugs this run exposed (both fixable)

1. **Stalled before the report** again. 4 heavy experiments ran in PARALLEL on one
   CPU box → contention; two got force-finalized/aborted; the orchestrator never
   reached the Report stage (quiet 27 min → stopped by hand). Fix: run testers
   SEQUENTIALLY (concurrency 1) and/or a per-tester wall-clock cap.
2. **`maxTest:1` did not take effect** — all 4 survivors were tested, not 1. Same
   symptom as `maxRounds:1` not binding last run → the `args` passed to the
   workflow are not reaching the script's `args` global. Verify + fix arg
   threading before trusting any per-run cap; until then rely on hard code
   defaults, not passed args.

## Bottom line

The loop autonomously generated → attacked → prior-art-checked → **ran real
experiments with oracle controls → gave honest SUPPORTED/REFUTED verdicts**. The
concept works. What remains is cost/robustness plumbing: sequential heavy tests,
arg threading, and reaching the report stage cleanly.
