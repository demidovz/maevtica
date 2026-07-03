from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PRINCIPLES = [
    "persistent_memory",
    "institutionalized_criticism",
    "preserved_failures",
    "question_refinement",
    "knowledge_compression",
    "applicability_boundaries",
]


@dataclass(frozen=True)
class PrimitiveRequirement:
    id: str
    name: str
    derivation: str


@dataclass(frozen=True)
class PrincipleDerivation:
    principle: str
    classification: str
    derived_from: list[str]
    replaceable_by: list[str]
    lost_if_removed: str
    derivation: str


@dataclass(frozen=True)
class AlternativeArchitecture:
    id: str
    name: str
    violates: list[str]
    preserves_function: list[str]
    verdict: str


PRIMITIVES = [
    PrimitiveRequirement(
        "A1",
        "Temporal retention",
        "If performance must improve over time, information from earlier interactions must affect later interactions.",
    ),
    PrimitiveRequirement(
        "A2",
        "Error-sensitive update",
        "If predictions can be wrong, the system needs a way for mismatch to change future behavior.",
    ),
    PrimitiveRequirement(
        "A3",
        "Discriminable alternatives",
        "Improvement requires distinguishing at least two possible policies, models, or explanations.",
    ),
    PrimitiveRequirement(
        "A4",
        "Scope tracking",
        "A learned claim can improve future performance only where its conditions of transfer are not completely lost.",
    ),
    PrimitiveRequirement(
        "A5",
        "Resource-bounded reuse",
        "When learning or computation is costly, successful structure must be reused rather than rediscovered from scratch.",
    ),
    PrimitiveRequirement(
        "A6",
        "Representation revision",
        "If the current question or representation causes avoidable error, cumulative improvement needs a way to change it.",
    ),
]


DERIVATIONS = [
    PrincipleDerivation(
        "persistent_memory",
        "necessary_function",
        ["A1"],
        ["biological weights", "distributed traces", "versioned records", "cultural transmission"],
        "Earlier evidence cannot constrain later belief; cumulative knowledge collapses into repeated first attempts.",
        "From A1, some state must persist. A literal research database is replaceable, but retention is necessary.",
    ),
    PrincipleDerivation(
        "institutionalized_criticism",
        "emergent_consequence",
        ["A2", "A3"],
        ["adversarial testing", "prediction markets", "selection pressure", "independent replication"],
        "Errors can persist without a channel that forces alternatives to compete against evidence.",
        "Criticism is one social implementation of error-sensitive update over discriminable alternatives.",
    ),
    PrincipleDerivation(
        "preserved_failures",
        "emergent_consequence",
        ["A1", "A2", "A4"],
        ["negative gradients", "test failures", "replication failures", "selection against bad variants"],
        "The system repeats known bad regions or overgeneralizes beyond defeated scope.",
        "Failures must affect future updates, but they need not be preserved as human-readable cases.",
    ),
    PrincipleDerivation(
        "question_refinement",
        "emergent_consequence",
        ["A3", "A6"],
        ["feature learning", "model class revision", "ontology update", "problem decomposition"],
        "The system remains trapped in a representation that may make improvement impossible or inefficient.",
        "When alternatives are represented badly, improvement requires changing the question or representation.",
    ),
    PrincipleDerivation(
        "knowledge_compression",
        "contingent_but_expected_under_scarcity",
        ["A5"],
        ["cached policies", "abstractions", "compiled heuristics", "generalized models"],
        "Without reusable structure, the system may still learn by lookup if resources are abundant, but loses efficiency under scarcity.",
        "Compression follows from resource-bounded reuse, not from cumulative knowledge alone.",
    ),
    PrincipleDerivation(
        "applicability_boundaries",
        "necessary_function",
        ["A4"],
        ["context tags", "confidence intervals", "domain restrictions", "policy guards"],
        "The system transfers claims into contexts where they do not hold, corrupting cumulative knowledge.",
        "Any reliable cumulative learner must encode, infer, or test where learned structure applies.",
    ),
]


ALTERNATIVES = [
    AlternativeArchitecture(
        "alt01_perfect_memory_agent",
        "Perfect-memory individual learner",
        ["institutionalized_criticism"],
        ["persistent_memory", "preserved_failures", "applicability_boundaries"],
        "Can accumulate knowledge without social criticism if internal error update is unbiased.",
    ),
    AlternativeArchitecture(
        "alt02_hive_mind",
        "Collective hive mind",
        ["institutionalized_criticism", "question_refinement as discourse"],
        ["persistent_memory", "knowledge_compression", "applicability_boundaries"],
        "Social mechanisms collapse into internal cognitive functions; criticism is replaceable by internal model comparison.",
    ),
    AlternativeArchitecture(
        "alt03_non_symbolic_learner",
        "Non-symbolic gradient learner",
        ["journal-like memory", "explicit preserved failures", "verbal question refinement"],
        ["temporal retention", "negative error signal", "compression", "scope control"],
        "Violates human-readable forms but preserves primitive functions through weights and gradients.",
    ),
    AlternativeArchitecture(
        "alt04_evolution_only",
        "Evolution-only civilization",
        ["persistent explicit memory", "institutionalized criticism", "question refinement"],
        ["retention in genomes", "selection against failures", "implicit applicability through niches"],
        "Cumulative adaptation occurs slowly without explicit science; it satisfies weaker knowledge accumulation but not explanatory science.",
    ),
    AlternativeArchitecture(
        "alt05_quantum_broadcast",
        "Perfect quantum communication collective",
        ["append-only records", "journal", "role separation"],
        ["shared state retention", "instant comparison", "scope tags"],
        "Communication changes implementations, not primitive requirements.",
    ),
    AlternativeArchitecture(
        "alt06_bruteforce_oracle",
        "Brute-force oracle search",
        ["knowledge_compression", "question_refinement"],
        ["retention", "scope control"],
        "If exhaustive search and truth access are free, compression and refinement become convenient but unnecessary.",
    ),
]


def analyze() -> dict:
    classification_counts = {}
    for item in DERIVATIONS:
        classification_counts[item.classification] = classification_counts.get(item.classification, 0) + 1
    dependency_edges = []
    for item in DERIVATIONS:
        for primitive in item.derived_from:
            dependency_edges.append({"from": primitive, "to": item.principle})
    return {
        "primitives": [asdict(item) for item in PRIMITIVES],
        "derivations": [asdict(item) for item in DERIVATIONS],
        "alternative_architectures": [asdict(item) for item in ALTERNATIVES],
        "classification_counts": classification_counts,
        "dependency_edges": dependency_edges,
    }


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "derivation_model.json", result)
    write_dependency_graph(reports / "principle_dependency_graph.md", result)
    write_minimal_axioms(reports / "minimal_axiom_proposal.md")
    write_necessity(reports / "necessity_classification.md")
    write_counterexamples(reports / "counterexample_catalogue.md")
    write_alternatives(reports / "alternative_epistemic_architectures.md")
    write_confidence(reports / "universality_confidence_estimates.md")
    write_revised_theory(reports / "revised_theory_of_cumulative_knowledge.md")
    write_final_report(reports / "final_report.md")


def write_dependency_graph(path: Path, result: dict) -> None:
    lines = ["# Principle Dependency Graph", ""]
    for primitive in PRIMITIVES:
        lines.append(f"- {primitive.id}: {primitive.name}")
        for edge in result["dependency_edges"]:
            if edge["from"] == primitive.id:
                lines.append(f"  - derives {edge['to']}")
    lines.extend(
        [
            "",
            "Reduction: persistent memory and applicability boundaries are necessary functions. Criticism, preserved failures, and question refinement emerge when error correction is distributed or representational. Compression follows only under scarcity and reusable regularities.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minimal_axioms(path: Path) -> None:
    lines = [
        "# Minimal Axiom Proposal",
        "",
        "A0. There is an environment and at least one agent or institution interacting repeatedly with it.",
        "A1. Some information from earlier interactions can affect later interactions.",
        "A2. Performance feedback can distinguish better from worse future behavior.",
        "A3. Learned changes have non-universal scope.",
        "A4. Learning, computation, or observation has nonzero cost.",
        "",
        "Derived from these:",
        "- A1 gives retention / memory.",
        "- A2 gives error correction / criticism-like mechanisms.",
        "- A3 gives applicability boundaries.",
        "- A4 plus repeated structure gives compression pressure.",
        "- A2 plus bad representations gives question refinement.",
        "",
        "This axiom set is smaller than the six candidate principles because it describes functions, not institutions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_necessity(path: Path) -> None:
    lines = ["# Necessity Classification", ""]
    for item in DERIVATIONS:
        lines.append(f"## {item.principle}")
        lines.append(f"Classification: {item.classification}")
        lines.append(f"Derived from: {', '.join(item.derived_from)}")
        lines.append(f"Replaceable by: {', '.join(item.replaceable_by)}")
        lines.append(f"Lost if removed: {item.lost_if_removed}")
        lines.append(f"Derivation: {item.derivation}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_counterexamples(path: Path) -> None:
    lines = ["# Counterexample Catalogue", ""]
    for architecture in ALTERNATIVES:
        lines.append(f"## {architecture.name}")
        lines.append(f"- Violates: {', '.join(architecture.violates)}")
        lines.append(f"- Preserves function: {', '.join(architecture.preserves_function)}")
        lines.append(f"- Verdict: {architecture.verdict}")
        lines.append("")
    lines.extend(
        [
            "Strongest counterexample: non-symbolic gradient learners can accumulate predictive knowledge without journals, explicit hypotheses, or verbal criticism. They do not refute retention, error correction, or scope control; they refute the necessity of Studio-shaped implementations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_alternatives(path: Path) -> None:
    lines = ["# Alternative Epistemic Architectures", ""]
    for architecture in ALTERNATIVES:
        lines.append(f"- {architecture.name}: {architecture.verdict}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(path: Path) -> None:
    lines = [
        "# Universality Confidence Estimates",
        "",
        "- Temporal retention / memory function: high. It follows almost directly from improvement over time.",
        "- Scope control / applicability boundaries: high. Transfer without scope control can degrade performance.",
        "- Error correction / criticism function: high. Some mechanism must make error matter, but social criticism is replaceable.",
        "- Preserved failures: medium. Failures must influence updates, but need not be stored as explicit cases.",
        "- Question refinement: medium. Necessary when representations are revisable and imperfect, not in all learners.",
        "- Knowledge compression: medium-low as necessity. It is strongly favored under scarcity, but dominated by free exhaustive search.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_revised_theory(path: Path) -> None:
    lines = [
        "# Revised Theory of Cumulative Knowledge",
        "",
        "The six candidate principles reduce to four primitive functions:",
        "",
        "1. Retention: past interaction must affect future interaction.",
        "2. Error-sensitive update: mismatch must change future behavior.",
        "3. Scope control: learned changes must carry conditions of valid transfer.",
        "4. Resource-bounded reuse: costly learning favors reusable structure.",
        "",
        "Question refinement is representation-level error correction. Preserved failures are memory plus error signal. Institutionalized criticism is socialized error-sensitive update. Knowledge compression is resource-bounded reuse under regularity. Persistent memory and applicability boundaries are closest to necessary functions, though their implementations vary radically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path) -> None:
    lines = [
        "# Benchmark Research #006 Final Report",
        "",
        "## Which principles are truly necessary?",
        "",
        "No named Studio principle is necessary as an implementation. Two underlying functions are necessary: temporal retention and scope control. A system cannot accumulate knowledge if earlier interaction cannot affect later interaction, and it cannot reliably transfer knowledge if learned changes have no conditions of applicability.",
        "",
        "## Which are merely convenient implementations?",
        "",
        "- Institutionalized criticism is a social implementation of error-sensitive update.",
        "- Preserved failures are one implementation of negative evidence retention.",
        "- Question refinement is one implementation of representation revision.",
        "- Knowledge compression is an efficiency strategy under scarcity, not a logical necessity.",
        "",
        "## Which emerge automatically from more primitive assumptions?",
        "",
        "- Persistent memory emerges from temporal retention.",
        "- Applicability boundaries emerge from non-universal scope.",
        "- Criticism emerges when error correction is distributed across agents.",
        "- Preserved failures emerge when error cases must influence future updates.",
        "- Question refinement emerges when current representations cause avoidable error.",
        "- Compression emerges when reusable regularities exist and learning/computation is costly.",
        "",
        "## What is the smallest known set of assumptions from which cumulative science can arise?",
        "",
        "1. Repeated interaction between agent/institution and environment.",
        "2. Retention: earlier information can influence later behavior.",
        "3. Error feedback: some outcomes distinguish better from worse models or policies.",
        "4. Scope: learned changes are not automatically valid everywhere.",
        "5. Nonzero cost: reuse is valuable because observation, learning, or computation is limited.",
        "",
        "From these, the previous principles are derivable as common organizational solutions. The theory is therefore reduced: effective science does not require the six principles as primitives; it requires retention, error-sensitive update, scope control, and resource-bounded reuse.",
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
    parser.add_argument("--root", default="benchmarks/benchmark_006")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #006 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
