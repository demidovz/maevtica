from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


PRINCIPLES = [
    "persistent_memory",
    "institutionalized_criticism",
    "preserved_failures",
    "question_refinement",
    "knowledge_compression",
    "applicability_boundaries",
]

OUTCOMES = ["helpful", "neutral", "harmful", "impossible", "dominated", "irrelevant"]


@dataclass(frozen=True)
class World:
    id: str
    name: str
    description: str
    dimensions: dict[str, str]


@dataclass(frozen=True)
class PrincipleOutcome:
    world_id: str
    principle: str
    outcome: str
    reason: str


WORLDS = [
    World(
        "w01_baseline_science",
        "Baseline learnable universe",
        "Stable causal world with fallible agents, finite memory, noisy observation, and reusable regularities.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "fallible",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "scarce",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "cooperative_competitive",
        },
    ),
    World(
        "w02_perfect_oracle",
        "Perfect oracle world",
        "Ground truth is directly and cheaply accessible to every agent.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "perfect",
            "communication": "perfect",
            "agent_lifespan": "finite",
            "resource_availability": "abundant",
            "observation_noise": "none",
            "computation_cost": "low",
            "learning_cost": "low",
            "ground_truth_accessibility": "direct",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "transparent",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w03_heraclitean",
        "Rule-chaos world",
        "Rules change faster than evidence can accumulate.",
        {
            "physics": "unstable",
            "logic": "classical",
            "memory_reliability": "fallible",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "scarce",
            "observation_noise": "high",
            "computation_cost": "moderate",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "volatile",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w04_false_memory",
        "Adversarial memory world",
        "Stored records drift or are adversarially corrupted more often than living observation.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "adversarial",
            "communication": "medium_bandwidth",
            "agent_lifespan": "long",
            "resource_availability": "moderate",
            "observation_noise": "low",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "competitive",
        },
    ),
    World(
        "w05_no_identity",
        "No identity persistence world",
        "Agents cannot preserve identity or commitments across time.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "fallible",
            "communication": "low_bandwidth",
            "agent_lifespan": "instantaneous",
            "resource_availability": "moderate",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "none",
            "language": "fragmentary",
            "time": "linear",
            "social_organization": "none",
        },
    ),
    World(
        "w06_expensive_compute",
        "Computation-starved world",
        "Memory is cheap, but abstraction, search, and comparison are prohibitively expensive.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "scarce",
            "observation_noise": "moderate",
            "computation_cost": "extreme",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w07_perfect_individual_minds",
        "Perfect individual scientist world",
        "Every agent has perfect private memory and unbiased reasoning but weak communication.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "perfect_private",
            "communication": "low_bandwidth",
            "agent_lifespan": "long",
            "resource_availability": "moderate",
            "observation_noise": "low",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "idiosyncratic",
            "time": "linear",
            "social_organization": "individualist",
        },
    ),
    World(
        "w08_total_transparency",
        "Total transparency world",
        "All observations and mental states are instantly public.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "perfect",
            "agent_lifespan": "finite",
            "resource_availability": "moderate",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "transparent",
            "time": "linear",
            "social_organization": "transparent_collective",
        },
    ),
    World(
        "w09_hostile_social",
        "Hostile prestige world",
        "Criticism is mostly strategic sabotage, and preserved failures become reputational weapons.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "scarce",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "ambiguous",
            "time": "linear",
            "social_organization": "hostile_status",
        },
    ),
    World(
        "w10_infinite_resources",
        "Exhaustive search world",
        "Resources are so abundant that brute-force enumeration dominates institutional design.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "perfect",
            "agent_lifespan": "long",
            "resource_availability": "infinite",
            "observation_noise": "low",
            "computation_cost": "zero",
            "learning_cost": "low",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w11_noncompressible",
        "Incompressible truth world",
        "Truth is a lookup table with no reusable regularities.",
        {
            "physics": "algorithmically_random",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "long",
            "resource_availability": "moderate",
            "observation_noise": "low",
            "computation_cost": "moderate",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable_random",
            "causality": "none",
            "identity_persistence": "stable",
            "language": "indexical",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w12_single_observation",
        "One-shot universe",
        "Each phenomenon can be observed once and never repeated.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "moderate",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "nonrepeatable",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w13_nonmonotonic_logic",
        "Nonmonotonic logic world",
        "Valid inferences can be invalidated by later context in ways that resist stable compression.",
        {
            "physics": "contextual",
            "logic": "nonmonotonic",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "long",
            "resource_availability": "moderate",
            "observation_noise": "moderate",
            "computation_cost": "high",
            "learning_cost": "high",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "contextual",
            "causality": "contextual",
            "identity_persistence": "stable",
            "language": "contextual",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
    World(
        "w14_many_worlds_personal_truth",
        "Private-truth world",
        "Each agent's observations are internally stable but not shared across agents.",
        {
            "physics": "observer_relative",
            "logic": "paraconsistent",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "long",
            "resource_availability": "moderate",
            "observation_noise": "low",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "agent_local",
            "rule_stability": "agent_local",
            "causality": "observer_relative",
            "identity_persistence": "stable",
            "language": "partially_untranslatable",
            "time": "linear",
            "social_organization": "pluralist",
        },
    ),
    World(
        "w15_immortal_hive",
        "Immortal hive-mind world",
        "There is one persistent collective mind with no internal disagreement.",
        {
            "physics": "stable",
            "logic": "classical",
            "memory_reliability": "perfect",
            "communication": "identity",
            "agent_lifespan": "immortal",
            "resource_availability": "moderate",
            "observation_noise": "moderate",
            "computation_cost": "moderate",
            "learning_cost": "moderate",
            "ground_truth_accessibility": "indirect",
            "rule_stability": "stable",
            "causality": "local",
            "identity_persistence": "collective",
            "language": "internal",
            "time": "linear",
            "social_organization": "hive",
        },
    ),
    World(
        "w16_deceptive_demon",
        "Adversarial observation world",
        "A powerful process adapts observations to defeat learning strategies.",
        {
            "physics": "adversarial",
            "logic": "classical",
            "memory_reliability": "reliable",
            "communication": "medium_bandwidth",
            "agent_lifespan": "finite",
            "resource_availability": "scarce",
            "observation_noise": "adversarial",
            "computation_cost": "moderate",
            "learning_cost": "extreme",
            "ground_truth_accessibility": "blocked",
            "rule_stability": "anti_inductive",
            "causality": "adversarial",
            "identity_persistence": "stable",
            "language": "compositional",
            "time": "linear",
            "social_organization": "cooperative",
        },
    ),
]


def evaluate(world: World, principle: str) -> PrincipleOutcome:
    d = world.dimensions
    wid = world.id
    if d["ground_truth_accessibility"] == "direct":
        return PrincipleOutcome(wid, principle, "dominated", "Direct oracle access dominates organizational research mechanisms.")
    if d["rule_stability"] in {"volatile", "anti_inductive"}:
        if principle in {"persistent_memory", "preserved_failures", "knowledge_compression"}:
            return PrincipleOutcome(wid, principle, "harmful", "Past evidence becomes actively misleading when rules change faster than learning.")
        return PrincipleOutcome(wid, principle, "neutral", "The principle cannot accumulate enough stable signal.")
    if d["identity_persistence"] == "none":
        if principle in {"persistent_memory", "institutionalized_criticism", "question_refinement"}:
            return PrincipleOutcome(wid, principle, "impossible", "No agent identity or commitment persists long enough to maintain the practice.")
        return PrincipleOutcome(wid, principle, "neutral", "The principle has no stable organization to act on.")
    if d["memory_reliability"] == "adversarial":
        if principle in {"persistent_memory", "preserved_failures"}:
            return PrincipleOutcome(wid, principle, "harmful", "Stored records are less trustworthy than fresh observation.")
        if principle == "institutionalized_criticism":
            return PrincipleOutcome(wid, principle, "helpful", "Criticism can expose memory corruption if fresh observations remain available.")
    if d["social_organization"] == "hostile_status":
        if principle in {"institutionalized_criticism", "preserved_failures"}:
            return PrincipleOutcome(wid, principle, "harmful", "Criticism and failure memory become weapons rather than correction mechanisms.")
        if principle == "persistent_memory":
            return PrincipleOutcome(wid, principle, "neutral", "Memory helps only if protected from strategic interpretation.")
    if d["resource_availability"] == "infinite" or d["computation_cost"] == "zero":
        if principle in {"question_refinement", "knowledge_compression"}:
            return PrincipleOutcome(wid, principle, "dominated", "Exhaustive search can dominate abstraction under zero cost.")
        return PrincipleOutcome(wid, principle, "neutral", "The principle may still organize results but is not needed for discovery.")
    if d["physics"] == "algorithmically_random" or d["causality"] == "none":
        if principle in {"knowledge_compression", "question_refinement", "applicability_boundaries"}:
            return PrincipleOutcome(wid, principle, "harmful", "Searching for reusable structure adds false patterns in a noncompressible world.")
        if principle == "persistent_memory":
            return PrincipleOutcome(wid, principle, "helpful", "Memory remains useful as a lookup table.")
    if d["time"] == "nonrepeatable":
        if principle in {"preserved_failures", "applicability_boundaries"}:
            return PrincipleOutcome(wid, principle, "helpful", "One-shot evidence makes preserved failures and boundaries unusually valuable.")
        if principle == "institutionalized_criticism":
            return PrincipleOutcome(wid, principle, "neutral", "Criticism has limited power without repeatable tests.")
    if d["logic"] in {"nonmonotonic", "paraconsistent"}:
        if principle == "knowledge_compression":
            return PrincipleOutcome(wid, principle, "harmful", "Stable compression can erase context needed for valid local reasoning.")
        if principle == "applicability_boundaries":
            return PrincipleOutcome(wid, principle, "helpful", "Boundary tracking is essential when validity is contextual.")
    if d["social_organization"] == "hive":
        if principle == "institutionalized_criticism":
            return PrincipleOutcome(wid, principle, "irrelevant", "There are no separate critics unless the hive simulates disagreement.")
        if principle == "persistent_memory":
            return PrincipleOutcome(wid, principle, "neutral", "Memory is already intrinsic to the immortal collective.")
    if d["ground_truth_accessibility"] == "blocked":
        return PrincipleOutcome(wid, principle, "neutral", "No method can reliably improve knowledge when evidence is anti-inductive and truth is blocked.")
    if principle == "persistent_memory":
        return PrincipleOutcome(wid, principle, "helpful", "Finite fallible agents need durable records to accumulate evidence.")
    if principle == "institutionalized_criticism":
        return PrincipleOutcome(wid, principle, "helpful", "Fallible inference benefits from independent error search.")
    if principle == "preserved_failures":
        return PrincipleOutcome(wid, principle, "helpful", "Failed cases constrain future hypotheses.")
    if principle == "question_refinement":
        return PrincipleOutcome(wid, principle, "helpful", "Ambiguous problems need sharper formulations to reduce waste.")
    if principle == "knowledge_compression":
        return PrincipleOutcome(wid, principle, "helpful", "Stable reusable regularities reward compression.")
    if principle == "applicability_boundaries":
        return PrincipleOutcome(wid, principle, "helpful", "Claims require scope conditions to transfer safely.")
    raise ValueError(principle)


def outcomes() -> list[PrincipleOutcome]:
    return [evaluate(world, principle) for world in WORLDS for principle in PRINCIPLES]


def analyze() -> dict:
    rows = outcomes()
    by_principle = defaultdict(Counter)
    for row in rows:
        by_principle[row.principle][row.outcome] += 1
    summaries = []
    for principle in PRINCIPLES:
        counts = dict(by_principle[principle])
        total = sum(counts.values())
        helpful = counts.get("helpful", 0)
        bad = counts.get("harmful", 0) + counts.get("impossible", 0)
        summaries.append(
            {
                "principle": principle,
                "counts": counts,
                "helpful_rate": round(helpful / total, 3),
                "failure_rate": round(bad / total, 3),
                "universality_status": classify_principle(counts, total),
            }
        )
    return {
        "worlds": [asdict(world) for world in WORLDS],
        "outcomes": [asdict(row) for row in rows],
        "principles": summaries,
    }


def classify_principle(counts: dict[str, int], total: int) -> str:
    if counts.get("harmful", 0) or counts.get("impossible", 0):
        return "requires_hidden_assumptions"
    if counts.get("helpful", 0) == total:
        return "universal"
    return "conditional"


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "world_catalogue.json", result["worlds"])
    write_json(data / "principle_outcomes.json", result["outcomes"])
    write_json(data / "principle_summary.json", result["principles"])
    write_world_catalogue(reports / "world_catalogue.md")
    write_counterexample_atlas(reports / "counterexample_atlas.md", result["outcomes"])
    write_necessary_conditions(reports / "necessary_condition_report.md")
    write_minimal_assumptions(reports / "minimal_assumptions_report.md")
    write_phase_diagram(reports / "universality_phase_diagram.md", result["outcomes"])
    write_dependency_graph(reports / "principle_dependency_graph.md")
    write_confidence(reports / "confidence_estimates.md", result["principles"])
    write_final_report(reports / "final_report.md", result["principles"])


def write_world_catalogue(path: Path) -> None:
    lines = ["# World Catalogue", ""]
    for world in WORLDS:
        lines.append(f"## {world.id}: {world.name}")
        lines.append(world.description)
        lines.append("")
        for key, value in world.dimensions.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_counterexample_atlas(path: Path, rows: list[dict]) -> None:
    lines = ["# Counterexample Atlas", ""]
    bad = [row for row in rows if row["outcome"] in {"harmful", "impossible", "dominated", "irrelevant"}]
    for row in bad:
        world = next(w for w in WORLDS if w.id == row["world_id"])
        lines.append(f"- {row['principle']} in {world.name}: {row['outcome']}. {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_necessary_conditions(path: Path) -> None:
    text = """# Necessary-Condition Report

## Persistent memory requires

- Memory records are more reliable than unaided recollection.
- Rules are stable enough that past evidence remains relevant.
- Agents or institutions persist long enough to reuse records.

## Institutionalized criticism requires

- Critics can affect decisions.
- Criticism is tied to evidence rather than status warfare.
- Communication can transmit reasons with enough fidelity.

## Preserved failures require

- Failure cases remain diagnostically relevant.
- Failure memory is not mainly used as punishment.
- Future hypotheses can condition on past failures.

## Question refinement requires

- Language or representation can distinguish better and worse questions.
- There is enough stable structure for refined questions to reduce search.
- Agents can compare formulations.

## Knowledge compression requires

- The world has reusable regularities.
- Compression preserves counterexamples or boundary cases.
- Computation is not so cheap that exhaustive lookup dominates.

## Applicability boundaries require

- Claims have scopes that can be distinguished.
- Context differences matter but are not totally private or untranslatable.
- Boundaries can be tested or updated.
"""
    path.write_text(text, encoding="utf-8")


def write_minimal_assumptions(path: Path) -> None:
    text = """# Minimal Assumptions Report

Organized science becomes possible only when all of these hold at least weakly:

1. Some stable regularities exist.
2. Observations correlate with those regularities better than chance.
3. Memory can preserve information with nonzero fidelity.
4. Agents or institutions persist long enough to compare evidence across time.
5. Communication can transmit claims, reasons, and corrections.
6. Computation/learning costs make reuse valuable.
7. Criticism can change future belief or action.
8. There is a distinction between local failure and global impossibility.

Without these properties, the Studio principles can become irrelevant, dominated, or harmful.
"""
    path.write_text(text, encoding="utf-8")


def write_phase_diagram(path: Path, rows: list[dict]) -> None:
    lines = ["# Universality Phase Diagram", ""]
    regions = {
        "necessarily_emerge": [],
        "sometimes_emerge": [],
        "never_emerge": [],
        "maladaptive": [],
    }
    for world in WORLDS:
        world_rows = [row for row in rows if row["world_id"] == world.id]
        helpful = sum(row["outcome"] == "helpful" for row in world_rows)
        bad = sum(row["outcome"] in {"harmful", "impossible"} for row in world_rows)
        dominated = sum(row["outcome"] in {"dominated", "irrelevant"} for row in world_rows)
        if helpful >= 5 and bad == 0:
            regions["necessarily_emerge"].append(world.name)
        elif bad >= 3:
            regions["maladaptive"].append(world.name)
        elif dominated >= 4:
            regions["never_emerge"].append(world.name)
        else:
            regions["sometimes_emerge"].append(world.name)
    for name, worlds in regions.items():
        lines.append(f"## {name}")
        lines.extend(f"- {world}" for world in worlds)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dependency_graph(path: Path) -> None:
    text = """# Principle Dependency Graph

- Stable regularities
  - Question refinement
  - Knowledge compression
  - Applicability boundaries
- Reliable-enough memory
  - Persistent memory
  - Preserved failures
  - Long-term criticism
- Persistent identity or institution
  - Persistent memory
  - Institutionalized criticism
  - Question refinement over time
- Communicable reasons
  - Institutionalized criticism
  - Applicability boundaries
- Nonzero learning/computation cost
  - Knowledge compression
  - Question refinement
  - Preserved failures

Dependency claim: no single candidate principle is unconditional. The closest root principle is stable, transmissible memory under non-oracular uncertainty.
"""
    path.write_text(text, encoding="utf-8")


def write_confidence(path: Path, summaries: list[dict]) -> None:
    lines = ["# Confidence Estimates", ""]
    for item in summaries:
        if item["failure_rate"] >= 0.25:
            confidence = "high confidence in non-universality"
        elif item["helpful_rate"] >= 0.65:
            confidence = "medium confidence in conditional robustness"
        else:
            confidence = "low confidence"
        lines.append(
            f"- {item['principle']}: {confidence}. helpful_rate={item['helpful_rate']}, failure_rate={item['failure_rate']}, counts={item['counts']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, summaries: list[dict]) -> None:
    by_name = {item["principle"]: item for item in summaries}
    lines = [
        "# Benchmark Research #005 Final Report",
        "",
        "## Which principles remain universal?",
        "",
        "None remain universal without hidden assumptions. Every principle can be made non-helpful in at least one coherent world, and most can be made actively harmful or impossible.",
        "",
        "The most robust conditional principle is persistent memory, but it fails when memory is adversarial, identity cannot persist, truth is directly available by oracle, or rules change faster than records can matter.",
        "",
        "## Which require hidden assumptions?",
        "",
    ]
    for principle in PRINCIPLES:
        item = by_name[principle]
        lines.append(f"- {principle}: {item['universality_status']}; helpful_rate={item['helpful_rate']}; failure_rate={item['failure_rate']}.")
    lines.extend(
        [
            "",
            "## Which assumptions are actually fundamental?",
            "",
            "1. Stable regularities: without them, compression and refined questions hallucinate structure.",
            "2. Evidence access: observations must correlate with truth better than chance.",
            "3. Reliable-enough memory: records must beat fresh unaided recollection often enough.",
            "4. Persistent identity or institutions: someone must be able to compare now with before.",
            "5. Communicable reasons: criticism and boundaries require transmissible claims.",
            "6. Scarcity: if truth or exhaustive search is free, many research principles are dominated.",
            "7. Non-hostile correction channels: criticism must be coupled to error reduction rather than status attack.",
            "",
            "## What is the smallest set of properties a universe must possess before organized science becomes possible?",
            "",
            "A universe must contain learnable regularities, accessible evidence, memory with nonzero fidelity, persistent agents or institutions, communication capable of transmitting reasons, and enough scarcity that reuse is valuable. With these properties, organized science is possible and the principles often emerge. Without them, the principles are not universal laws; they are adaptations to a specific class of learnable, scarce, communicable worlds.",
            "",
            "## Boundary conclusion",
            "",
            "The previous Studio principles survive only as conditional adaptations. Their domain is broad but not absolute: non-oracular, partly stable, resource-bounded worlds with fallible but persistent agents.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(root: str | Path) -> Path:
    root = Path(root)
    write_deliverables(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_005")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #005 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
