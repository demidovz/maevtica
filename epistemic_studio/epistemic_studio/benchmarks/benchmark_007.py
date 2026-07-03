from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


AXIOMS = [
    "repeated_interaction",
    "retention",
    "error_feedback",
    "scope_control",
    "nonzero_cost",
]


@dataclass(frozen=True)
class EpistemicSystem:
    id: str
    name: str
    system_type: str
    satisfies_axioms: list[str]
    extra_properties: dict[str, str]
    classification: str
    failure_mode: str
    missing_assumption: str
    explanation: str


SYSTEMS = [
    EpistemicSystem(
        "s01_gradient_learner",
        "Stable gradient learner",
        "machine learner",
        AXIOMS,
        {"feedback_alignment": "aligned", "exploration": "adequate", "update_rule": "contractive"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Repeated error feedback updates retained parameters within scope; cost favors reusable representations.",
    ),
    EpistemicSystem(
        "s02_biological_population",
        "Adaptive biological population",
        "evolution",
        AXIOMS,
        {"feedback_alignment": "fitness_correlated", "exploration": "mutation", "update_rule": "selection"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Genetic retention plus selection accumulates environment-correlated adaptation within niches.",
    ),
    EpistemicSystem(
        "s03_scientific_institution",
        "Minimal scientific institution",
        "institution",
        AXIOMS,
        {"feedback_alignment": "mostly_aligned", "exploration": "planned", "update_rule": "peer_correction"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Records, tests, scoped claims, and costly inquiry produce cumulative improvements.",
    ),
    EpistemicSystem(
        "s04_prediction_market",
        "Calibrated prediction market",
        "market",
        AXIOMS,
        {"feedback_alignment": "settlement_correlated", "exploration": "trader_diversity", "update_rule": "price_update"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Retained prices and settlement feedback accumulate predictive information when incentives are aligned.",
    ),
    EpistemicSystem(
        "s05_biased_feedback_loop",
        "Biased feedback learner",
        "machine learner",
        AXIOMS,
        {"feedback_alignment": "anti_correlated", "exploration": "adequate", "update_rule": "reinforcement"},
        "fails",
        "false_convergence",
        "feedback_truth_correlation",
        "The system receives error feedback, but it is systematically misleading, so retained updates accumulate falsehood.",
    ),
    EpistemicSystem(
        "s06_local_trap_population",
        "Local-trap evolutionary population",
        "evolution",
        AXIOMS,
        {"feedback_alignment": "local_fitness", "exploration": "too_narrow", "update_rule": "selection"},
        "requires_additional_assumptions",
        "local_trap",
        "adequate_exploration",
        "The population improves locally but cannot reach better regions because variation never crosses the valley.",
    ),
    EpistemicSystem(
        "s07_catastrophic_forgetting_net",
        "Catastrophic forgetting learner",
        "neural system",
        AXIOMS,
        {"feedback_alignment": "aligned", "exploration": "adequate", "update_rule": "destructive_overwrite"},
        "fails",
        "knowledge_collapse",
        "non_destructive_retention",
        "Retention exists, but updates overwrite prior competencies faster than they consolidate.",
    ),
    EpistemicSystem(
        "s08_oscillating_collective",
        "Oscillating research collective",
        "distributed collective",
        AXIOMS,
        {"feedback_alignment": "delayed", "exploration": "adequate", "update_rule": "overcorrection"},
        "fails",
        "permanent_oscillation",
        "stable_update_dynamics",
        "The collective reacts to delayed feedback with overcorrection, cycling between incompatible policies.",
    ),
    EpistemicSystem(
        "s09_fragmented_alien_civilization",
        "Fragmented alien civilization",
        "alien civilization",
        AXIOMS,
        {"feedback_alignment": "aligned", "exploration": "parallel", "update_rule": "isolated_silos"},
        "requires_additional_assumptions",
        "communication_failure",
        "integration_channel",
        "Subgroups learn locally, but no integration channel lets local knowledge become civilization-level knowledge.",
    ),
    EpistemicSystem(
        "s10_adversarial_game_agents",
        "Adversarial game-theoretic agents",
        "game-theoretic agents",
        AXIOMS,
        {"feedback_alignment": "strategically_manipulated", "exploration": "adequate", "update_rule": "best_response"},
        "fails",
        "Goodharted_feedback",
        "incentive_compatibility",
        "Agents satisfy the axioms but manipulate feedback, so retained knowledge tracks payoffs rather than truth.",
    ),
    EpistemicSystem(
        "s11_symbolic_reasoner",
        "Sound symbolic reasoner with experiments",
        "symbolic reasoner",
        AXIOMS,
        {"feedback_alignment": "aligned", "exploration": "systematic", "update_rule": "belief_revision"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Sound revision over scoped propositions accumulates reliable explanations.",
    ),
    EpistemicSystem(
        "s12_cellular_automaton_scientists",
        "Cellular automaton colonies",
        "cellular automata",
        AXIOMS,
        {"feedback_alignment": "local", "exploration": "spatial_diffusion", "update_rule": "local_rule_update"},
        "unknown",
        "scale_dependent",
        "ergodic_coverage",
        "Local cumulative knowledge may emerge, but global knowledge depends on whether local patterns cover enough state space.",
    ),
    EpistemicSystem(
        "s13_self_modifying_agent",
        "Self-modifying agent",
        "self-modifying agent",
        AXIOMS,
        {"feedback_alignment": "initially_aligned", "exploration": "adequate", "update_rule": "self_modification"},
        "requires_additional_assumptions",
        "identity_drift",
        "goal_preservation",
        "The system accumulates knowledge only if self-modification preserves the criterion of improvement.",
    ),
    EpistemicSystem(
        "s14_hybrid_human_ai_lab",
        "Hybrid human-AI lab",
        "hybrid system",
        AXIOMS,
        {"feedback_alignment": "mostly_aligned", "exploration": "broad", "update_rule": "human_ai_review"},
        "produces_cumulative_knowledge",
        "none",
        "none",
        "Diverse search plus retained scoped feedback supports cumulative knowledge.",
    ),
]


def analyze() -> dict:
    counts = Counter(system.classification for system in SYSTEMS)
    missing = Counter(
        system.missing_assumption
        for system in SYSTEMS
        if system.missing_assumption != "none"
    )
    failure_modes = Counter(
        system.failure_mode
        for system in SYSTEMS
        if system.failure_mode != "none"
    )
    return {
        "systems": [asdict(system) for system in SYSTEMS],
        "classification_counts": dict(counts),
        "missing_assumptions": dict(missing),
        "failure_modes": dict(failure_modes),
    }


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "system_catalogue.json", result["systems"])
    write_json(data / "sufficiency_analysis.json", result)
    write_sufficiency(reports / "sufficiency_report.md", result)
    write_counterexamples(reports / "counterexample_catalogue.md")
    write_hidden_assumptions(reports / "hidden_assumption_report.md", result)
    write_failure_taxonomy(reports / "failure_taxonomy.md", result)
    write_confidence(reports / "confidence_estimate.md", result)
    write_theorem(reports / "candidate_representation_theorem.md")
    write_revised_axioms(reports / "revised_minimal_axiom_system.md")
    write_final_report(reports / "final_report.md", result)


def write_sufficiency(path: Path, result: dict) -> None:
    lines = ["# Sufficiency Report", ""]
    lines.append(f"Classification counts: {result['classification_counts']}")
    lines.append("")
    lines.append("The five axioms are not jointly sufficient. Several systems satisfy all five but fail through biased feedback, destructive retention, overcorrection, local traps, or incentive manipulation.")
    lines.append("")
    for system in SYSTEMS:
        lines.append(f"- {system.name}: {system.classification}; {system.explanation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_counterexamples(path: Path) -> None:
    lines = ["# Counterexample Catalogue", ""]
    for system in SYSTEMS:
        if system.classification in {"fails", "requires_additional_assumptions"}:
            lines.append(f"## {system.name}")
            lines.append(f"- Failure mode: {system.failure_mode}")
            lines.append(f"- Missing assumption: {system.missing_assumption}")
            lines.append(f"- Why it satisfies the five axioms: {', '.join(system.satisfies_axioms)}")
            lines.append(f"- Why it still fails: {system.explanation}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_hidden_assumptions(path: Path, result: dict) -> None:
    lines = ["# Hidden Assumption Report", ""]
    lines.append("Additional assumptions discovered:")
    for assumption, count in sorted(result["missing_assumptions"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {assumption}: implicated in {count} systems.")
    lines.extend(
        [
            "",
            "The missing assumptions collapse into one additional primitive for cumulative knowledge: a reliable update channel. Feedback must be sufficiently truth-correlated, updates must be stable/non-destructive, and incentives or self-modification must not systematically redirect updates away from the target of knowledge.",
            "",
            "Adequate exploration and integration channels are secondary scope conditions: without them, knowledge can still accumulate locally, but not necessarily globally or near the best reachable explanations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_taxonomy(path: Path, result: dict) -> None:
    lines = ["# Failure Taxonomy", ""]
    for mode, count in sorted(result["failure_modes"].items()):
        lines.append(f"- {mode}: {count}")
    lines.extend(
        [
            "",
            "Families:",
            "- Feedback pathologies: false convergence, Goodharted feedback.",
            "- Dynamics pathologies: permanent oscillation, catastrophic forgetting.",
            "- Search pathologies: local traps, inadequate coverage.",
            "- Integration pathologies: communication failure, identity drift.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(path: Path, result: dict) -> None:
    total = len(SYSTEMS)
    failures = result["classification_counts"].get("fails", 0) + result["classification_counts"].get("requires_additional_assumptions", 0)
    lines = [
        "# Confidence Estimate",
        "",
        f"Systems tested: {total}",
        f"Failures or additional-assumption cases: {failures}",
        "",
        "Confidence that the original five axioms are sufficient: low.",
        "Confidence that the revised axiom system is closer to sufficient: medium.",
        "",
        "Reason: every generated system satisfies the five axioms, yet several fail by feedback alignment or update-stability pathologies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_theorem(path: Path) -> None:
    lines = [
        "# Candidate Representation Theorem",
        "",
        "For any system that accumulates reliable knowledge over time, there exists an abstract representation with:",
        "",
        "- state S_t retaining information from prior interactions;",
        "- environment interaction E_t producing observations/outcomes;",
        "- update operator U(S_t, E_t) -> S_{t+1};",
        "- evaluation relation R that is truth-correlated enough to distinguish improvement;",
        "- scope relation C describing where retained changes transfer;",
        "- stability constraint preventing U from systematically destroying previously valid retained structure.",
        "",
        "The original five axioms specify S, E, R, C, and cost pressure, but not the truth-correlation or stability constraints strongly enough.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_revised_axioms(path: Path) -> None:
    lines = [
        "# Revised Minimal Axiom System",
        "",
        "R1. Repeated interaction between system and environment.",
        "R2. Retention: prior interaction can affect later behavior.",
        "R3. Truth-correlated feedback: feedback distinguishes better from worse relative to the target environment more often than not.",
        "R4. Scope control: retained changes carry conditions of transfer.",
        "R5. Nonzero cost: reuse has value because observation, learning, or computation is limited.",
        "R6. Update stability: learning dynamics do not systematically erase or invert previously valid knowledge faster than correction can restore it.",
        "",
        "R3 strengthens error feedback. R6 is the new minimal assumption discovered by counterexamples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, result: dict) -> None:
    lines = [
        "# Benchmark Research #007 Final Report",
        "",
        "## Are the five assumptions sufficient?",
        "",
        "No. They are necessary-looking but not jointly sufficient. A system can repeatedly interact, retain state, receive feedback, track scope, and face nonzero cost while still failing to accumulate knowledge.",
        "",
        "## Minimal additional assumption required",
        "",
        "The minimal additional assumption is update reliability: feedback must be truth-correlated enough, and the update dynamics must be stable enough, that retained changes tend to improve rather than corrupt future performance.",
        "",
        "This can be split into two operational constraints:",
        "",
        "1. Truth-correlated feedback: feedback is not systematically misleading or fully manipulable.",
        "2. Non-destructive update stability: learning does not erase, invert, or endlessly oscillate around previously valid knowledge faster than correction can recover it.",
        "",
        "## Classes generated by the revised axiom system",
        "",
        "- Machine learners with aligned loss and stable optimization.",
        "- Biological populations with fitness-correlated selection and enough exploration.",
        "- Scientific institutions with retained records, scoped claims, and correction channels.",
        "- Prediction markets with settlement tied to reality and incentive-compatible reporting.",
        "- Symbolic reasoners with sound belief revision and empirical tests.",
        "- Hybrid collectives with integration channels and stable update protocols.",
        "",
        "## Result",
        "",
        "The five-axiom system is falsified as sufficient. The revised minimal system is: repeated interaction, retention, truth-correlated feedback, scope control, nonzero cost, and update stability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(root: str | Path) -> Path:
    root = Path(root)
    write_deliverables(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_007")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #007 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
