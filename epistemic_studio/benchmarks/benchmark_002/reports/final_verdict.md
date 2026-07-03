# Final Verdict

The current Studio is partially falsified as a full architecture. It does show epistemic advantages over the baselines in auditability, falsification pressure, and explicit reuse, but too much of its complexity is not yet justified.

Current Studio epistemic value estimate: 0.501.
Best simpler baseline: Baseline C - Human researcher with notes, value 0.405.

## Smallest architecture preserving at least 90% of value

Keep: append-only Research State, typed Knowledge Graph, Explorer, Engine, Synthesizer, short immutable Journal, simple priority queue, and four metrics: duplicate rate, time to falsification, concept reuse, compression-with-audit.

Remove or demote:

- Agent Evolution: remove. It can amplify noisy reputation before reputation is predictive.
- Meta Observer: demote to periodic report. Per-cycle process commentary is mostly overhead.
- Separate Counterexample Graph: remove as a separate component; keep as graph filter.
- Historian: demote to deterministic journal/state summarizer.
- Cartographer: demote to deterministic graph summarizer.
- Full Attention Market: replace with priority queue and hard budget caps.
- Dashboard: keep as audit UI, not as part of the research loop.

## Components justified

- Research State: needed because without persistent artifacts, reuse and contradiction history are unverifiable.
- Knowledge Graph: needed because compression claims require explicit links.
- Engine: needed because the null hypothesis is otherwise not attacked aggressively.
- Journal: needed because external researchers must reconstruct why decisions happened.
- Planner-lite: needed to prevent every role from consuming attention by default.

The null hypothesis is not fully proven, but it survives against several components. The Studio should become smaller before it becomes more ambitious.
