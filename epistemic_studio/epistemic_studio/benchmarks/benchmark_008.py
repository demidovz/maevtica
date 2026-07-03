from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    id: str
    tradition: str
    title: str
    url: str
    relevance: str


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    prediction: str
    closest_equivalents: list[str]
    partial_overlaps: list[str]
    genuine_difference: str
    novelty_class: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class Experiment:
    id: str
    target_claims: list[str]
    design: str
    distinguishes_from: list[str]
    decision_rule: str
    expected_value: float


SOURCES = [
    Source(
        "shannon_1948",
        "information_theory",
        "Shannon, A Mathematical Theory of Communication",
        "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
        "Uncertainty, information, noise, channel limits, and coding efficiency.",
    ),
    Source(
        "mdl_grunwald",
        "information_theory",
        "Grunwald, A Tutorial Introduction to the Minimum Description Length Principle",
        "https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf",
        "Compression as inductive inference and model selection.",
    ),
    Source(
        "pac_valiant",
        "statistical_learning_theory",
        "Valiant PAC learning overview",
        "https://en.wikipedia.org/wiki/Probably_approximately_correct_learning",
        "Learnability under samples, error bounds, and computational cost.",
    ),
    Source(
        "vc_blumer",
        "statistical_learning_theory",
        "Blumer et al., Learnability and the Vapnik-Chervonenkis Dimension",
        "https://www.cis.upenn.edu/~danroth/Teaching/CS446-17/Papers/p929-blumer.pdf",
        "Finite capacity as a condition for uniform learnability.",
    ),
    Source(
        "natural_selection_sep",
        "evolutionary_theory",
        "Stanford Encyclopedia of Philosophy, Natural Selection",
        "https://plato.stanford.edu/entries/natural-selection/",
        "Heritable variation, differential reproduction, selection, and adaptation over generations.",
    ),
    Source(
        "wiener_cybernetics",
        "cybernetics",
        "Wiener, Cybernetics: Control and Communication in the Animal and the Machine",
        "https://direct.mit.edu/books/oa-monograph/4581/Cybernetics-or-Control-and-Communication-in-the",
        "Feedback, control, communication, noise, and self-regulation.",
    ),
    Source(
        "kalman_filter",
        "control_theory",
        "Kalman filtering overview",
        "https://en.wikipedia.org/wiki/Kalman_filter",
        "Recursive state estimation from noisy measurements using prediction and update.",
    ),
    Source(
        "bayesian_sep",
        "bayesian_epistemology",
        "Stanford Encyclopedia of Philosophy, Bayesian Epistemology",
        "https://plato.stanford.edu/entries/epistemology-bayesian/",
        "Conditionalization and evidential updating of credences.",
    ),
    Source(
        "popper_refutations",
        "philosophy_of_science",
        "Popper, Conjectures and Refutations",
        "https://philpapers.org/rec/POPCAR-5",
        "Scientific growth through conjectures controlled by criticism and attempted refutation.",
    ),
    Source(
        "kuhn_revolutions",
        "philosophy_of_science",
        "Kuhn, The Structure of Scientific Revolutions",
        "https://www.lri.fr/~mbl/Stanford/CS477/papers/Kuhn-SSR-2ndEd.pdf",
        "Anomalies, crisis, question changes, and paradigm shifts.",
    ),
    Source(
        "sutton_barto",
        "reinforcement_learning",
        "Sutton and Barto, Reinforcement Learning: An Introduction",
        "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf",
        "Repeated interaction, reward/error feedback, value updating, and control.",
    ),
    Source(
        "active_inference",
        "active_inference",
        "Free energy principle and active inference overview",
        "https://en.wikipedia.org/wiki/Free_energy_principle",
        "Prediction-error/free-energy minimization linking perception and action.",
    ),
    Source(
        "cap_theorem",
        "distributed_systems",
        "CAP theorem overview",
        "https://en.wikipedia.org/wiki/CAP_theorem",
        "Distributed trade-offs among consistency, availability, and partition tolerance.",
    ),
    Source(
        "nonaka_seci",
        "knowledge_management",
        "Nonaka and Takeuchi SECI model overview",
        "https://ascnhighered.org/ASCN/change_theories/collection/seci.html",
        "Organizational conversion between tacit and explicit knowledge.",
    ),
    Source(
        "argyris_schon",
        "organizational_learning",
        "Argyris and Schon single-loop and double-loop learning overview",
        "https://www.open.edu/openlearn/mod/oucontent/view.php?id=135424&section=4.2",
        "Organizations correct errors and revise governing assumptions.",
    ),
]


CLAIMS = [
    Claim(
        "c01_repeated_interaction",
        "Cumulative knowledge requires repeated interaction between system and environment.",
        "One-shot systems cannot accumulate reliable knowledge over time except by inheriting another system's retained state.",
        ["sutton_barto", "wiener_cybernetics", "kalman_filter", "pac_valiant"],
        ["bayesian_sep", "active_inference"],
        "The current theory states the condition at a higher abstraction level across biological, institutional, computational, and organizational systems.",
        "already_known",
        0.96,
        "Repeated sampling, feedback, recursive estimation, and interaction are core assumptions in existing learning, control, and cybernetic frameworks.",
    ),
    Claim(
        "c02_retention",
        "Cumulative knowledge requires retention: prior interaction must affect later behavior.",
        "Systems without durable state cannot improve across cycles at the system level.",
        ["kalman_filter", "bayesian_sep", "sutton_barto", "nonaka_seci"],
        ["cap_theorem", "shannon_1948"],
        "The theory treats memory as substrate-neutral: parameters, records, genes, prices, practices, and institutions can all instantiate retention.",
        "improved_formulation",
        0.78,
        "The core idea is old, but the substrate-neutral organizational framing usefully prevents confusing human-readable archives with memory itself.",
    ),
    Claim(
        "c03_truth_correlated_feedback",
        "Feedback must be truth-correlated rather than merely present.",
        "A system can have repeated interaction, retention, and feedback while converging to falsehood if feedback is biased, adversarial, or Goodharted.",
        ["wiener_cybernetics", "sutton_barto", "bayesian_sep", "kalman_filter"],
        ["popper_refutations", "active_inference"],
        "The theory packages feedback alignment as a primitive axiom for any cumulative knowledge system, not only as a loss/reward design issue.",
        "stronger_formulation",
        0.72,
        "Misleading signals and bad rewards are well known; making truth-correlation a cross-domain axiom is a useful unification, not a clear discovery.",
    ),
    Claim(
        "c04_scope_control",
        "Reliable knowledge requires scope control: retained claims must carry conditions of transfer.",
        "Systems that ignore scope will overgeneralize, transfer failures across domains, and mistake local adaptation for global knowledge.",
        ["pac_valiant", "vc_blumer", "kuhn_revolutions"],
        ["sutton_barto", "bayesian_sep", "cap_theorem"],
        "The theory makes applicability boundaries a first-class requirement rather than a downstream caveat.",
        "stronger_formulation",
        0.68,
        "Domain restrictions, hypothesis classes, external validity, and paradigm-relative puzzles are established; the novelty is the operational insistence that scope must be retained with knowledge artifacts.",
    ),
    Claim(
        "c05_nonzero_cost",
        "Nonzero cost makes compression, reuse, and prioritization epistemically relevant.",
        "When observation, computation, or communication is costless, compression pressure becomes optional rather than necessary.",
        ["shannon_1948", "mdl_grunwald", "pac_valiant", "cap_theorem"],
        ["sutton_barto", "nonaka_seci"],
        "The theory links cost pressure to organizational attention and research strategy, not just coding length or sample complexity.",
        "improved_formulation",
        0.74,
        "The cost-compression connection is classic; applying it uniformly to research organizations is a useful reformulation.",
    ),
    Claim(
        "c06_update_stability",
        "Cumulative knowledge requires update stability: learning must not destroy, invert, or endlessly oscillate around valid retained knowledge faster than correction can recover it.",
        "Some systems satisfying interaction, retention, feedback, scope, and cost still fail through catastrophic forgetting, oscillation, or destructive self-modification.",
        ["kalman_filter", "sutton_barto", "cap_theorem", "argyris_schon", "natural_selection_sep"],
        ["bayesian_sep", "nonaka_seci"],
        "The theory elevates stability of updates from an engineering property to a minimal axiom for cumulative knowledge.",
        "potentially_novel",
        0.55,
        "Every ingredient has prior art, but the exact necessity claim across all epistemic substrates appears less directly covered.",
    ),
    Claim(
        "c07_representation_theorem",
        "Any cumulative knowledge system can be represented as retained state, environment interaction, update operator, truth-correlated evaluation, scope relation, cost pressure, and stability constraint.",
        "The same abstract schema should classify machine learners, evolution, institutions, markets, collectives, and symbolic reasoners.",
        ["kalman_filter", "sutton_barto", "bayesian_sep", "wiener_cybernetics"],
        ["active_inference", "nonaka_seci", "cap_theorem"],
        "The schema is intentionally minimal and cross-substrate; it avoids assuming probabilities, symbols, rewards, agents, or explicit theories.",
        "potentially_novel",
        0.48,
        "This is mostly a synthesis. It becomes novel only if it predicts failures outside the home domains of existing frameworks better than those frameworks do.",
    ),
    Claim(
        "c08_preserved_failures",
        "Preserved failures and counterexamples improve cumulative knowledge by preventing rediscovery and stabilizing criticism.",
        "Organizations that preserve failed hypotheses should reduce duplicate exploration and improve time-to-falsification compared with equally active organizations that do not.",
        ["popper_refutations", "kuhn_revolutions", "argyris_schon", "nonaka_seci"],
        ["cap_theorem", "bayesian_sep"],
        "The theory treats failures as reusable state, not just historical narrative or local lessons learned.",
        "improved_formulation",
        0.64,
        "Failed hypotheses, anomalies, and lessons learned are well established. The stronger artifact-level prediction is testable but not obviously unprecedented.",
    ),
    Claim(
        "c09_question_refinement_compression",
        "Compression through question refinement is a central route of cumulative understanding.",
        "Research progress often deletes ill-posed questions rather than simply answering them.",
        ["kuhn_revolutions", "mdl_grunwald", "argyris_schon", "popper_refutations"],
        ["active_inference", "bayesian_sep"],
        "The theory operationalizes question disappearance as a compression event in a research state.",
        "improved_formulation",
        0.66,
        "The intellectual content overlaps Kuhn, MDL, and double-loop learning; the improved part is measurement and artifact tracking.",
    ),
]


EXPERIMENTS = [
    Experiment(
        "e01_cross_substrate_failure_prediction",
        ["c06_update_stability", "c07_representation_theorem"],
        "Build a blinded benchmark of learning systems, institutions, markets, and evolutionary simulations. Code only the six abstract conditions, then predict which systems accumulate knowledge over long horizons. Compare against PAC-only, RL-only, Bayesian-only, and organizational-learning-only feature sets.",
        ["PAC learning", "reinforcement learning", "Bayesian epistemology", "organizational learning"],
        "Novelty survives only if the six-condition model predicts cross-substrate failures significantly better than each domain framework without adding domain-specific features.",
        0.88,
    ),
    Experiment(
        "e02_preserved_failure_ablation",
        ["c08_preserved_failures"],
        "Run matched research teams or LLM research collectives on multi-stage tasks with and without structured preserved failures. Hold total notes, time, and participants constant.",
        ["Popperian criticism", "ordinary lab notes", "knowledge management"],
        "The claim gains novelty if structured failure artifacts reduce duplicate hypotheses and time-to-falsification beyond ordinary notes and post-hoc summaries.",
        0.74,
    ),
    Experiment(
        "e03_scope_annotation_transfer",
        ["c04_scope_control"],
        "Compare model repositories, scientific claims, or organizational playbooks with explicit scope annotations against equivalent repositories without scope metadata across out-of-domain transfer tasks.",
        ["statistical learning theory", "external validity methods", "RL generalization"],
        "The Studio formulation adds value if scope metadata predicts transfer failure earlier than standard validation metrics alone.",
        0.69,
    ),
    Experiment(
        "e04_compression_predicts_future_reuse",
        ["c05_nonzero_cost", "c09_question_refinement_compression"],
        "Track research projects over time and measure whether question-refinement compression predicts later reuse, fewer duplicates, and faster falsification after controlling for publication count, citation count, and model accuracy.",
        ["MDL", "bibliometrics", "organizational learning"],
        "Novelty survives if compression-through-question-refinement predicts future efficiency where existing complexity or impact measures do not.",
        0.71,
    ),
]


def analyze() -> dict:
    class_counts = Counter(claim.novelty_class for claim in CLAIMS)
    tradition_hits: dict[str, int] = defaultdict(int)
    for claim in CLAIMS:
        for source_id in claim.closest_equivalents + claim.partial_overlaps:
            source = source_by_id(source_id)
            if source:
                tradition_hits[source.tradition] += 1

    potentially_novel = [
        claim
        for claim in CLAIMS
        if claim.novelty_class in {"potentially_novel", "clearly_novel"}
    ]
    weighted_novelty = sum(
        claim.confidence
        for claim in CLAIMS
        if claim.novelty_class in {"stronger_formulation", "potentially_novel", "clearly_novel"}
    ) / len(CLAIMS)
    rediscovery_rate = sum(
        1
        for claim in CLAIMS
        if claim.novelty_class in {"already_known", "equivalent"}
    ) / len(CLAIMS)

    return {
        "claims": [asdict(claim) for claim in CLAIMS],
        "sources": [asdict(source) for source in SOURCES],
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "classification_counts": dict(class_counts),
        "tradition_hits": dict(sorted(tradition_hits.items())),
        "potentially_novel_claims": [asdict(claim) for claim in potentially_novel],
        "weighted_novelty_score": round(weighted_novelty, 3),
        "rediscovery_rate": round(rediscovery_rate, 3),
        "final_assessment": {
            "rediscoveries": [
                "Repeated interaction",
                "Basic retention",
                "Feedback-driven correction",
                "Compression under cost",
            ],
            "improved_formulations": [
                "Substrate-neutral retention",
                "Truth-correlated feedback as a cross-domain axiom",
                "Applicability boundaries as retained scope metadata",
                "Failures as reusable research state",
                "Question deletion as measurable compression",
            ],
            "genuinely_new": [
                "No claim is clearly novel on current evidence.",
                "The best candidate is update stability as a universal condition for cumulative knowledge.",
                "The second candidate is the six-condition cross-substrate representation theorem.",
            ],
            "highest_value_experiment": "e01_cross_substrate_failure_prediction",
        },
    }


def source_by_id(source_id: str) -> Source | None:
    return next((source for source in SOURCES if source.id == source_id), None)


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "sources.json", result["sources"])
    write_json(data / "claim_analysis.json", result["claims"])
    write_json(data / "experiments.json", result["experiments"])
    write_json(data / "novelty_summary.json", result)
    write_equivalence_map(reports / "equivalence_map.md")
    write_prior_art_map(reports / "prior_art_map.md")
    write_prediction_catalogue(reports / "novel_prediction_catalogue.md")
    write_distinguishing_experiments(reports / "distinguishing_experiments.md")
    write_compression_analysis(reports / "compression_analysis.md", result)
    write_confidence(reports / "novelty_confidence_estimates.md", result)
    write_final_assessment(reports / "final_assessment.md", result)


def write_equivalence_map(path: Path) -> None:
    lines = ["# Equivalence Map", ""]
    for claim in CLAIMS:
        lines.append(f"## {claim.id}")
        lines.append(claim.text)
        lines.append("")
        lines.append(f"- Novelty class: {claim.novelty_class}")
        lines.append("- Closest equivalents:")
        for source_id in claim.closest_equivalents:
            source = source_by_id(source_id)
            if source:
                lines.append(f"  - {source.tradition}: [{source.title}]({source.url})")
        lines.append("- Partial overlaps:")
        for source_id in claim.partial_overlaps:
            source = source_by_id(source_id)
            if source:
                lines.append(f"  - {source.tradition}: [{source.title}]({source.url})")
        lines.append(f"- Genuine difference: {claim.genuine_difference}")
        lines.append(f"- Assessment: {claim.reason}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prior_art_map(path: Path) -> None:
    by_tradition: dict[str, list[Source]] = defaultdict(list)
    for source in SOURCES:
        by_tradition[source.tradition].append(source)

    lines = ["# Prior-Art Map", ""]
    lines.append("The prior-art search weakens broad originality claims. Nearly every primitive in the current theory has a close ancestor.")
    lines.append("")
    for tradition, sources in sorted(by_tradition.items()):
        lines.append(f"## {tradition}")
        for source in sources:
            lines.append(f"- [{source.title}]({source.url}): {source.relevance}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prediction_catalogue(path: Path) -> None:
    lines = ["# Novel Prediction Catalogue", ""]
    lines.append("| Claim | Prediction | Classification | Confidence |")
    lines.append("| --- | --- | --- | --- |")
    for claim in CLAIMS:
        lines.append(f"| {claim.id} | {claim.prediction} | {claim.novelty_class} | {claim.confidence:.2f} |")
    lines.extend(
        [
            "",
            "No prediction is classified as clearly novel. The strongest candidates are update stability and the cross-substrate representation theorem because their exact universal framing is less directly stated in the comparison traditions.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_distinguishing_experiments(path: Path) -> None:
    lines = ["# Distinguishing Experiments", ""]
    for experiment in sorted(EXPERIMENTS, key=lambda item: item.expected_value, reverse=True):
        lines.append(f"## {experiment.id}")
        lines.append(f"- Target claims: {', '.join(experiment.target_claims)}")
        lines.append(f"- Design: {experiment.design}")
        lines.append(f"- Existing alternatives: {', '.join(experiment.distinguishes_from)}")
        lines.append(f"- Decision rule: {experiment.decision_rule}")
        lines.append(f"- Expected novelty value: {experiment.expected_value:.2f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_compression_analysis(path: Path, result: dict) -> None:
    lines = [
        "# Compression Analysis",
        "",
        f"Rediscovery rate: {result['rediscovery_rate']}",
        f"Weighted novelty score: {result['weighted_novelty_score']}",
        "",
        "The current theory compresses prior frameworks by replacing domain-specific terms with six substrate-neutral functions: repeated interaction, retention, truth-correlated feedback, scope control, nonzero cost, and update stability.",
        "",
        "This is genuine compression only if the abstraction predicts failures across domains without importing each domain's vocabulary. Otherwise it is mostly a renaming of learning theory, cybernetics, Bayesian updating, MDL, and organizational learning.",
        "",
        "Current judgment: partial compression. The theory is not mathematically deeper than its strongest ancestors, but it is operationally simpler as a checklist for knowledge-producing systems.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(path: Path, result: dict) -> None:
    lines = ["# Novelty Confidence Estimates", ""]
    lines.append(f"Classification counts: {result['classification_counts']}")
    lines.append("")
    for claim in sorted(CLAIMS, key=lambda item: item.confidence):
        lines.append(f"- {claim.id}: {claim.novelty_class}; confidence {claim.confidence:.2f}. {claim.reason}")
    lines.extend(
        [
            "",
            "Overall confidence:",
            "- Clearly novel content: low.",
            "- Improved formulation: medium.",
            "- Useful compression of prior frameworks: medium-low until tested on cross-substrate prediction.",
            "- Risk of relabeling known theories: high.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_assessment(path: Path, result: dict) -> None:
    final = result["final_assessment"]
    lines = [
        "# Benchmark Research #008 Final Assessment",
        "",
        "## Which parts are rediscoveries?",
        "",
        "Repeated interaction, retention, feedback-driven correction, and compression under cost are rediscoveries. They are already central in cybernetics, control theory, Bayesian epistemology, reinforcement learning, information theory, MDL, and statistical learning theory.",
        "",
        "## Which parts are improved formulations?",
        "",
        "The improved formulations are substrate-neutral retention, truth-correlated feedback as a general epistemic requirement, explicit scope control, preserved failures as reusable state, and question refinement as measurable compression. These do not defeat prior art, but they package scattered ideas into a compact operational test.",
        "",
        "## Which parts appear genuinely new?",
        "",
        "No part is clearly novel on current evidence. The smallest plausible novel core is the claim that update stability is a universal condition for cumulative knowledge, together with the six-condition representation theorem. Both remain only potentially novel because each ingredient has strong prior art.",
        "",
        "## Highest-probability novelty experiment",
        "",
        f"The highest-value experiment is {final['highest_value_experiment']}: run a blinded cross-substrate failure-prediction benchmark comparing the six-condition theory against PAC-only, RL-only, Bayesian-only, control-only, and organizational-learning-only baselines. If the six-condition model predicts long-horizon knowledge failure across machines, institutions, markets, evolution, and collectives better than those alternatives without domain-specific features, genuine novelty becomes plausible. If it does not, the theory should be classified as a useful synthesis rather than a new theory.",
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
    parser.add_argument("--root", default="benchmarks/benchmark_008")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #008 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
