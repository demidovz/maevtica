# Epistemic Studio

Epistemic Studio is a persistent research operating system. Agents are disposable policies; the permanent object is the Research State.

This MVP freezes the existing theory and implements process infrastructure:

- append-only Research State event log plus snapshots
- typed scientific artifacts with IDs, timestamps, provenance, and links
- one evolving knowledge graph
- long-lived agent identities that exchange only artifacts
- domain plugins that contribute seed knowledge only
- research-cycle runner and static dashboard

## Run

```bash
python -m epistemic_studio.cli init --root studio_state --domain all
python -m epistemic_studio.cli run --root studio_state --cycles 5
python -m epistemic_studio.cli journal --root studio_state
python -m epistemic_studio.cli dashboard --root studio_state --out studio_state/dashboard.html
```

Open `studio_state/dashboard.html` in a browser.

## Contract

The source brief is preserved verbatim in `PROJECT_BRIEF.md`.

Agents do not use chat history. Every action is recorded as an explicit artifact event in `events.jsonl`; snapshots are derived views, not replacements.

## Evolution Program v1

The runner now treats the Studio as a research organization:

- every cycle starts with a fixed attention budget
- active agents bid in an internal research market
- Planner allocation is weighted by bid value, reputation, and domain priority
- agent records accumulate reputation, attention, generation, parent, strategy, and performance stats
- weak strategies can retire and strong strategies can clone on evolution intervals
- domain records compete for attention and can become inactive or revive
- Meta Memory records research-process patterns every 5 cycles
- Research Health Reports are emitted every 100 cycles

## Continuous Research Journal

Every completed cycle writes an immutable Markdown entry under `research_journal/`:

- `cycle_0001.md`, `cycle_0002.md`, ...
- one fixed section template per cycle
- missing journals can be backfilled with `python -m epistemic_studio.cli journal --root studio_state`
- previous journal files are never rewritten by the journal writer
- milestone reports are written under `research_journal/milestones/` at cycles 10, 50, 100, 500, and 1000

The dashboard embeds the journals and lets a researcher browse cycles chronologically with the corresponding frontier, attention allocation, agent activity, graph size, and compression metrics.

## Benchmark Research

Benchmark #001 investigates `Is an explicit world model necessary for building AGI?` as a Research State process:

```bash
python -m epistemic_studio.benchmarks.benchmark_001 --root benchmarks/benchmark_001 --cycles 30
```

The benchmark writes its own state, journals, graphs, dashboard, source brief, compression report, organizational health report, and final report under `benchmarks/benchmark_001/`.

Benchmark #002 investigates whether the Studio itself has measurable epistemic advantage over simpler workflows:

```bash
python -m epistemic_studio.benchmarks.benchmark_002 --root benchmarks/benchmark_002 --cycles 24
```

It writes baseline scores, ablation reports, failure atlas, minimal viable Studio proposal, simplification recommendations, and final verdict under `benchmarks/benchmark_002/`.

Benchmark #003 isolates organizational mechanisms behind the Studio advantage:

```bash
python -m epistemic_studio.benchmarks.benchmark_003 --root benchmarks/benchmark_003
```

It writes cross-validated mechanism rankings, synergy matrix, minimal architecture search, redundant/irreplaceable mechanism reports, confidence estimates, and a final principles-only report under `benchmarks/benchmark_003/`.

Benchmark #004 tests whether the top Studio mechanisms appear in successful real-world research organizations:

```bash
python -m epistemic_studio.benchmarks.benchmark_004 --root benchmarks/benchmark_004
```

It writes a sourced cross-case mechanism matrix, universality ranking, counterexample catalogue, historical evidence report, confidence estimates, revised theory, and final report under `benchmarks/benchmark_004/`.

Benchmark #005 attempts to falsify universality itself by generating hypothetical worlds where the candidate principles become harmful, neutral, impossible, dominated, or irrelevant:

```bash
python -m epistemic_studio.benchmarks.benchmark_005 --root benchmarks/benchmark_005
```

It writes a world catalogue, counterexample atlas, necessary-condition report, minimal assumptions report, universality phase diagram, principle dependency graph, confidence estimates, and final report under `benchmarks/benchmark_005/`.

Benchmark #006 asks whether the candidate principles can be derived from the bare requirement of cumulative reliable knowledge:

```bash
python -m epistemic_studio.benchmarks.benchmark_006 --root benchmarks/benchmark_006
```

It writes a principle dependency graph, minimal axiom proposal, necessity classification, counterexample catalogue, alternative epistemic architectures, confidence estimates, revised theory, and final report under `benchmarks/benchmark_006/`.

Benchmark #007 tests whether the five minimal axioms are jointly sufficient:

```bash
python -m epistemic_studio.benchmarks.benchmark_007 --root benchmarks/benchmark_007
```

It writes a sufficiency report, counterexample catalogue, hidden assumption report, failure taxonomy, confidence estimate, candidate representation theorem, revised minimal axiom system, and final report under `benchmarks/benchmark_007/`.

Benchmark #008 tests whether the revised theory contains genuinely new scientific content or mostly restates existing frameworks:

```bash
python -m epistemic_studio.benchmarks.benchmark_008 --root benchmarks/benchmark_008
```

It writes an equivalence map, prior-art map, novel prediction catalogue, distinguishing experiments, compression analysis, novelty confidence estimates, and final assessment under `benchmarks/benchmark_008/`.

Benchmark #009 studies the birth of major scientific concepts rather than cumulative knowledge:

```bash
python -m epistemic_studio.benchmarks.benchmark_009 --root benchmarks/benchmark_009
```

It writes concept genealogy, representation-transition atlas, pressure taxonomy, concept-value metric, historical reconstruction report, counterexample catalogue, candidate theory of concept genesis, and final report under `benchmarks/benchmark_009/`.

Benchmark #010 prospectively tests whether the Studio can identify concept pressure before a concept exists:

```bash
python -m epistemic_studio.benchmarks.benchmark_010 --root benchmarks/benchmark_010
```

It writes a concept pressure map, concept hole atlas, candidate concept catalogue, historical reconstruction comparison, adversarial review, compression estimates, predicted future concepts, confidence estimates, and final report under `benchmarks/benchmark_010/`.

Benchmark #011 validates generated concepts under blind evaluation:

```bash
python -m epistemic_studio.benchmarks.benchmark_011 --root benchmarks/benchmark_011
```

It writes a blind ranking, calibration report, false-positive analysis, false-negative analysis, concept survival curve, evaluation agreement, confidence calibration, and final report under `benchmarks/benchmark_011/`.
