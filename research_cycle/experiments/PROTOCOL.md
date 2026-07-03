# Experiment protocol — the doctrine of the teeth

Every external test the cycle runs must follow this. It exists because on
2026-07-03 the CRC test twice produced a confident-but-false verdict from a
broken measurement; both were caught only by the oracle check. The protocol
turns those catches into a standing rule.

## Steps (in order, no skipping)

1. **Preregister.** Before running: write the hypothesis, the design, the
   baselines, and the DECISION RULE (what number = supported / refuted /
   inconclusive). Commit it. The threshold cannot move to fit the result.
   (Template: `../../crc_external_test/PREREGISTRATION.md`.)

2. **Build against a FAIR baseline.** The concept must beat the *strongest*
   realistic competitor, not a strawman. A weak baseline is the designer-bias
   trap in reverse — it fakes a win. (Round 1 gave geometry a weak baseline and
   "won" at 0.86; the fair strong baseline in round 2 cut the gap to 0.25.)

3. **Decouple the metric.** Measure the concept's quantity on something OTHER
   than what you built it from. For "causal role": measure the side-effect
   fingerprint on neutral words, with the answer tokens MASKED — no home field.

4. **Run, then CHECK THE ORACLE.** Compute a positive control: applying the
   *correct* / ground-truth mechanism must produce a real effect. `harness.verdict()`
   returns `BROKEN_MEASUREMENT` when |oracle| < min_oracle — a near-zero oracle
   means the pipeline is broken, NOT that the concept failed. Never report a
   verdict past a broken oracle. (This is what a `.norm(-1)` vs `.norm(dim=-1)`
   bug and a too-weak ablation both tripped.)

5. **Honest verdict.** "Survived my best attempt to kill it" ≠ "proven". State
   the regime it holds in and the limits (model sizes, intervention types,
   behaviors, single operationalization). Report failures faithfully.

6. **Persist.** Code + preregistration + numbers + verdict go in the repo with a
   backup push. A result that lives only in a chat does not exist.

## How the TESTER stage maps a prediction → an experiment (v1)

- **Shape it already knows** ("matching mechanisms across models by X transfers
  better than Y") → parameterize `harness.py` (models, behaviors, intervention,
  metric) — a short config, minutes to run. Reference impls:
  `../../crc_external_test/crc_transfer_test_r{2,3,4}.py`.
- **New but computable shape** → Claude-in-loop designs a fresh experiment
  following steps 1-6 (this is semi-automatic in v1; full autonomy is v2).
- **Not computable in-budget** → `status: designed_not_run` with the design
  written down, so a later campaign can pick it up.

## What v1 does NOT do (be honest)

- It does not invent a valid experiment for an arbitrary prediction on its own —
  novel shapes need Claude-in-loop design.
- It does not cover every interp baseline (no SAE-dictionary competitor yet).
- Small local models only (fits the studio's box); big-model claims need more.
