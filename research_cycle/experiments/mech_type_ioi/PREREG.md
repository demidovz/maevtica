# PREREG — Mechanism Type vs Circuit Token (cross-subgraph interchange, IOI)

Written BEFORE running. venv: `~/.local/state/mst/crc-venv311`. CPU only.
Date: 2026-07-04. Tester: claude (TEETH stage).

## Concept under test

"Many Circuits, One Mechanism" (arXiv:2606.06267) claims a model contains
multiple disjoint, equally-faithful subgraphs for one behavior that all
implement the SAME mechanism (a mechanism *type*, not per-subgraph tokens).
Prediction: cross-subgraph interchange accuracy for the task's key
intermediate variable exceeds 0.8, while alignments to a mismatched task's
variable stay near chance. Falsified if disjoint faithful subgraphs
systematically fail cross-alignment.

## Minimal operationalization

- Model: `gpt2` (small), CPU, transformer_lens.
- Task: IOI, 16 prompts = 8 single-token name pairs x {ABBA, BABA},
  "When {a} and {b} went to the store, {b} gave a drink to" (answer " a").
  Source (counterfactual) = role swap: "..., {a} gave a drink to" (answer
  " b"). Same names, same token length (asserted).
- Disjoint subgraphs (proxy for the paper's pruned subgraphs — name-mover
  family from the IOI paper, split into disjoint halves):
  - S1 = heads {9.6, 9.9, 9.0, 9.7}
  - S2 = heads {10.0, 10.10, 10.6, 10.2, 10.1, 11.2}
- Variable carrier: subgraph's aggregate write at END position,
  c_S(p) = sum over heads in S of hook_result[END] (resid space).
- ONE intervention type: add a delta vector at `blocks.11.hook_resid_post`,
  END position only (direct-readout path; identical insertion for all
  conditions, so comparisons are internally consistent).
- Variable subspace of subgraph S: V_S = top-8 right singular vectors of the
  16 difference vectors {c_S(base_i) - c_S(swap_i)} (uncentered SVD).
- Subspace interchange of T's write using S's alignment:
  delta_i = V_S V_S^T (c_T(swap_i) - c_T(base_i)), added at the site above.
- Full interchange of T: delta_i = c_T(swap_i) - c_T(base_i).
- Mismatched-task variable: greater-than task, "The war lasted from the year
  17{yy} to the year 17" (16 prompt pairs differing only in start-year yy,
  single-token yy asserted). V_mis = top-8 SVD of {c_S1(p_yA) - c_S1(p_yB)}
  differences at END. Random control: 8 random orthonormal dirs (seed 0).
- Metric per condition: interchange (flip) accuracy = fraction of the 16
  pairs where, after the edit on the base run, logit(" b") > logit(" a")
  (i.e. output moved to the source's answer). No-op floor ~= 1 - clean_acc.

## Conditions

1. clean accuracy (logit a > logit b, no edit)
2. ORACLE: full resid interchange, delta = resid11_post(src) - resid11_post(base)
3. S1 full interchange     (qualification: subgraph is a faithful carrier)
4. S2 full interchange     (qualification)
5. own-subspace S1 on S1   (qualification: subspace estimator adequate)
6. own-subspace S2 on S2   (qualification)
7. CROSS: S2's subspace applied on S1's write   (main)
8. CROSS: S1's subspace applied on S2's write   (main)
9/10. mismatched (greater-than) subspace on S1 / on S2   (control)
11/12. random 8-dim subspace on S1 / on S2               (control)

## DECISION RULE (fixed before running; thresholds cannot move)

- BROKEN_MEASUREMENT if clean acc < 0.9 OR oracle (cond 2) < 0.9.
- INCONCLUSIVE (regime not reproduced) if min(cond 3, cond 4) < 0.8
  — my subgraphs are then not "equally faithful" carriers and the paper's
  premise isn't instantiated; no verdict on the concept.
- INCONCLUSIVE (estimator too weak) if min(cond 5, cond 6) < 0.7 —
  cross failure can't be blamed on the concept if own-subspace fails too.
- SUPPORTED iff min(cond 7, cond 8) >= 0.8 AND max(cond 9..12) <= 0.2.
- REFUTED iff qualifications pass AND min(cond 7, cond 8) < 0.5.
- else INCONCLUSIVE (gray zone or dirty controls).

## Honest limits (stated up front)

- Head-subset proxy, not the paper's pruned full subgraphs; one model, one
  task, one insertion site (direct-readout path only — composition via
  downstream heads is not exercised); diff-in-means/SVD alignment, not DAS.
- Known triviality risk: both subgraphs are faithful for the SAME output, so
  END-write codings are pressured toward the unembedding directions of the
  answer names; a SUPPORTED here is weak evidence, a REFUTED is strong.

## Budget

No downloads (gpt2 cached). ~15 batched CPU forwards on 16 short prompts.
Target < 10 min wall clock.
