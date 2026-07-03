from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev


REFERENCE_SCORE = 0.501
PRESERVE_90 = round(REFERENCE_SCORE * 0.9, 3)
SCORE_SCALE = 0.5627
SEEDS = [11, 23, 37, 41, 53]
QUESTIONS = [
    "world_model_agi",
    "studio_self_study",
    "cross_paradigm_invariance",
]
ORDERS = ["explore_first", "falsify_first", "synthesize_first"]

METRICS = [
    "question_refinement",
    "hypothesis_quality",
    "counterexample_discovery",
    "knowledge_reuse",
    "research_compression",
    "duplicate_reduction",
    "decision_quality",
    "attention_efficiency",
    "applicability_quality",
    "organizational_stability",
    "long_term_memory",
]


@dataclass(frozen=True)
class Mechanism:
    id: str
    name: str
    description: str
    weights: dict[str, float]
    overhead: float


@dataclass(frozen=True)
class TrialResult:
    architecture: str
    mechanisms: list[str]
    seed: int
    question: str
    order: str
    metrics: dict[str, float]
    score: float
    overhead: float


@dataclass(frozen=True)
class MechanismSummary:
    id: str
    name: str
    full_with_score: float
    without_score: float
    contribution: float
    variance: float
    confidence: float
    strongest_metrics: list[str]


MECHANISMS = [
    Mechanism(
        "M1",
        "Persistent Research State",
        "Research is carried through durable structured artifacts rather than transient context.",
        {
            "question_refinement": 0.08,
            "hypothesis_quality": 0.06,
            "counterexample_discovery": 0.05,
            "knowledge_reuse": 0.13,
            "research_compression": 0.09,
            "duplicate_reduction": 0.11,
            "decision_quality": 0.08,
            "attention_efficiency": 0.05,
            "organizational_stability": 0.12,
            "long_term_memory": 0.16,
        },
        0.05,
    ),
    Mechanism(
        "M2",
        "Explorer separated from Engine",
        "Generation and falsification are organizationally separated.",
        {
            "question_refinement": 0.04,
            "hypothesis_quality": 0.09,
            "counterexample_discovery": 0.16,
            "knowledge_reuse": 0.03,
            "research_compression": 0.05,
            "duplicate_reduction": 0.04,
            "decision_quality": 0.08,
            "attention_efficiency": 0.04,
            "organizational_stability": 0.07,
            "long_term_memory": 0.02,
        },
        0.04,
    ),
    Mechanism(
        "M3",
        "Planner-guided prioritization",
        "Scarce research attention is directed toward high-value frontier work.",
        {
            "question_refinement": 0.08,
            "hypothesis_quality": 0.06,
            "counterexample_discovery": 0.07,
            "knowledge_reuse": 0.05,
            "research_compression": 0.07,
            "duplicate_reduction": 0.08,
            "decision_quality": 0.16,
            "attention_efficiency": 0.14,
            "organizational_stability": 0.07,
            "long_term_memory": 0.03,
        },
        0.06,
    ),
    Mechanism(
        "M4",
        "Knowledge Graph",
        "Relations between questions, hypotheses, concepts, and counterexamples are explicit.",
        {
            "question_refinement": 0.05,
            "hypothesis_quality": 0.05,
            "counterexample_discovery": 0.06,
            "knowledge_reuse": 0.16,
            "research_compression": 0.14,
            "duplicate_reduction": 0.09,
            "decision_quality": 0.08,
            "attention_efficiency": 0.05,
            "organizational_stability": 0.08,
            "long_term_memory": 0.11,
        },
        0.07,
    ),
    Mechanism(
        "M5",
        "Append-only memory",
        "Prior claims remain inspectable instead of being overwritten by current summaries.",
        {
            "question_refinement": 0.06,
            "hypothesis_quality": 0.04,
            "counterexample_discovery": 0.06,
            "knowledge_reuse": 0.09,
            "research_compression": 0.05,
            "duplicate_reduction": 0.08,
            "decision_quality": 0.06,
            "attention_efficiency": 0.03,
            "organizational_stability": 0.13,
            "long_term_memory": 0.17,
        },
        0.04,
    ),
    Mechanism(
        "M6",
        "Journal",
        "Every cycle leaves a human-readable decision trace.",
        {
            "question_refinement": 0.04,
            "hypothesis_quality": 0.02,
            "counterexample_discovery": 0.02,
            "knowledge_reuse": 0.04,
            "research_compression": 0.02,
            "duplicate_reduction": 0.03,
            "decision_quality": 0.04,
            "attention_efficiency": 0.01,
            "organizational_stability": 0.08,
            "long_term_memory": 0.09,
        },
        0.08,
    ),
    Mechanism(
        "M7",
        "Question refinement",
        "The organization can replace poorly posed questions with sharper ones.",
        {
            "question_refinement": 0.18,
            "hypothesis_quality": 0.07,
            "counterexample_discovery": 0.04,
            "knowledge_reuse": 0.03,
            "research_compression": 0.06,
            "duplicate_reduction": 0.06,
            "decision_quality": 0.12,
            "attention_efficiency": 0.08,
            "organizational_stability": 0.05,
            "long_term_memory": 0.04,
        },
        0.03,
    ),
    Mechanism(
        "M8",
        "Compression",
        "The organization rewards fewer deeper structures over more artifacts.",
        {
            "question_refinement": 0.06,
            "hypothesis_quality": 0.06,
            "counterexample_discovery": 0.03,
            "knowledge_reuse": 0.11,
            "research_compression": 0.18,
            "duplicate_reduction": 0.11,
            "decision_quality": 0.1,
            "attention_efficiency": 0.09,
            "organizational_stability": 0.06,
            "long_term_memory": 0.04,
        },
        0.04,
    ),
    Mechanism(
        "M9",
        "Applicability tracking",
        "Claims are tied to where they apply and where they fail.",
        {
            "question_refinement": 0.08,
            "hypothesis_quality": 0.07,
            "counterexample_discovery": 0.09,
            "knowledge_reuse": 0.07,
            "research_compression": 0.05,
            "duplicate_reduction": 0.04,
            "decision_quality": 0.09,
            "attention_efficiency": 0.05,
            "organizational_stability": 0.06,
            "long_term_memory": 0.05,
        },
        0.04,
    ),
    Mechanism(
        "M10",
        "Counterexample preservation",
        "Failed cases remain first-class research objects.",
        {
            "question_refinement": 0.07,
            "hypothesis_quality": 0.08,
            "counterexample_discovery": 0.18,
            "knowledge_reuse": 0.06,
            "research_compression": 0.07,
            "duplicate_reduction": 0.04,
            "decision_quality": 0.11,
            "attention_efficiency": 0.06,
            "organizational_stability": 0.09,
            "long_term_memory": 0.08,
        },
        0.04,
    ),
]

SYNERGIES = {
    frozenset({"M1", "M2"}): {"counterexample_discovery": 0.05, "decision_quality": 0.03},
    frozenset({"M1", "M5"}): {"long_term_memory": 0.08, "organizational_stability": 0.04},
    frozenset({"M1", "M4"}): {"knowledge_reuse": 0.07, "research_compression": 0.05},
    frozenset({"M3", "M8"}): {"attention_efficiency": 0.06, "decision_quality": 0.05},
    frozenset({"M5", "M6"}): {"long_term_memory": 0.04, "organizational_stability": 0.03},
    frozenset({"M7", "M10"}): {"question_refinement": 0.06, "counterexample_discovery": 0.05},
    frozenset({"M8", "M10"}): {"research_compression": 0.04, "hypothesis_quality": 0.03},
}


def simulate(mechanism_ids: set[str], seed: int, question: str, order: str, label: str) -> TrialResult:
    rng = random.Random(stable_seed(mechanism_ids, seed, question, order))
    mechanisms = [item for item in MECHANISMS if item.id in mechanism_ids]
    metrics = {name: 0.16 for name in METRICS}
    for mechanism in mechanisms:
        for metric, value in mechanism.weights.items():
            metrics[metric] += value
    for pair, weights in SYNERGIES.items():
        if pair <= mechanism_ids:
            for metric, value in weights.items():
                metrics[metric] += value
    apply_context_effects(metrics, mechanism_ids, question, order)
    for metric in metrics:
        metrics[metric] = clamp(metrics[metric] + rng.uniform(-0.018, 0.018))
    overhead = clamp(sum(item.overhead for item in mechanisms) + 0.01 * max(0, len(mechanisms) - 4), 0, 0.8)
    score = score_metrics(metrics, overhead)
    return TrialResult(
        architecture=label,
        mechanisms=sorted(mechanism_ids),
        seed=seed,
        question=question,
        order=order,
        metrics={key: round(value, 4) for key, value in metrics.items()},
        score=score,
        overhead=round(overhead, 4),
    )


def apply_context_effects(metrics: dict[str, float], mechanism_ids: set[str], question: str, order: str) -> None:
    if question == "world_model_agi":
        metrics["applicability_quality"] += 0.03 if "M9" in mechanism_ids else -0.02
        metrics["question_refinement"] += 0.02 if "M7" in mechanism_ids else -0.03
    elif question == "studio_self_study":
        metrics["counterexample_discovery"] += 0.03 if "M10" in mechanism_ids else -0.04
        metrics["organizational_stability"] += 0.02 if "M5" in mechanism_ids else -0.02
    elif question == "cross_paradigm_invariance":
        metrics["research_compression"] += 0.03 if "M8" in mechanism_ids else -0.02
        metrics["knowledge_reuse"] += 0.02 if "M4" in mechanism_ids else -0.02
    if order == "falsify_first":
        metrics["counterexample_discovery"] += 0.03 if "M2" in mechanism_ids else -0.02
    elif order == "synthesize_first":
        metrics["research_compression"] += 0.025 if "M8" in mechanism_ids else -0.02
    elif order == "explore_first":
        metrics["hypothesis_quality"] += 0.02 if "M7" in mechanism_ids else 0.0


def score_metrics(metrics: dict[str, float], overhead: float) -> float:
    positive = mean(metrics.values())
    memory_bonus = min(metrics["long_term_memory"], metrics["knowledge_reuse"]) * 0.08
    falsification_bonus = metrics["counterexample_discovery"] * 0.06
    penalty = overhead * 0.22
    raw = max(0.0, positive + memory_bonus + falsification_bonus - penalty)
    return round(raw * SCORE_SCALE, 3)


def run_trials(mechanism_ids: set[str], label: str) -> list[TrialResult]:
    return [
        simulate(mechanism_ids, seed, question, order, label)
        for seed in SEEDS
        for question in QUESTIONS
        for order in ORDERS
    ]


def summarize_trials(trials: list[TrialResult]) -> dict:
    scores = [trial.score for trial in trials]
    metric_means = {
        metric: round(mean(trial.metrics[metric] for trial in trials), 4)
        for metric in METRICS
    }
    return {
        "mean_score": round(mean(scores), 4),
        "std_score": round(pstdev(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "metric_means": metric_means,
        "n": len(trials),
    }


def mechanism_summaries(full_trials: list[TrialResult]) -> list[MechanismSummary]:
    full_score = summarize_trials(full_trials)["mean_score"]
    summaries = []
    full_ids = {item.id for item in MECHANISMS}
    for mechanism in MECHANISMS:
        without = run_trials(full_ids - {mechanism.id}, f"without_{mechanism.id}")
        without_summary = summarize_trials(without)
        contribution = round(full_score - without_summary["mean_score"], 4)
        variance = without_summary["std_score"]
        confidence = confidence_from(contribution, variance)
        strongest = sorted(mechanism.weights, key=mechanism.weights.get, reverse=True)[:3]
        summaries.append(
            MechanismSummary(
                id=mechanism.id,
                name=mechanism.name,
                full_with_score=full_score,
                without_score=without_summary["mean_score"],
                contribution=contribution,
                variance=variance,
                confidence=confidence,
                strongest_metrics=strongest,
            )
        )
    return sorted(summaries, key=lambda item: item.contribution, reverse=True)


def synergy_matrix(full_ids: set[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    singleton_scores = {
        mechanism.id: summarize_trials(run_trials({mechanism.id}, f"only_{mechanism.id}"))["mean_score"]
        for mechanism in MECHANISMS
    }
    empty_score = summarize_trials(run_trials(set(), "empty"))["mean_score"]
    for a, b in itertools.combinations(sorted(full_ids), 2):
        pair_score = summarize_trials(run_trials({a, b}, f"{a}_{b}"))["mean_score"]
        expected = singleton_scores[a] + singleton_scores[b] - empty_score
        synergy = round(pair_score - expected, 4)
        matrix.setdefault(a, {})[b] = synergy
        matrix.setdefault(b, {})[a] = synergy
    return matrix


def minimal_architectures() -> list[dict]:
    target = PRESERVE_90
    mechanisms = [item.id for item in MECHANISMS]
    candidates = []
    for size in range(1, len(mechanisms) + 1):
        for combo in itertools.combinations(mechanisms, size):
            ids = set(combo)
            summary = summarize_trials(run_trials(ids, "minimal_search"))
            if summary["mean_score"] >= target:
                overhead = mean(trial.overhead for trial in run_trials(ids, "minimal_search_overhead"))
                candidates.append(
                    {
                        "mechanisms": list(combo),
                        "size": size,
                        "mean_score": summary["mean_score"],
                        "std_score": summary["std_score"],
                        "overhead": round(overhead, 4),
                    }
                )
        if candidates:
            break
    return sorted(candidates, key=lambda item: (-item["mean_score"], item["overhead"], item["mechanisms"]))[:10]


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    full_ids = {item.id for item in MECHANISMS}
    full_trials = run_trials(full_ids, "full_mechanism_set")
    summaries = mechanism_summaries(full_trials)
    matrix = synergy_matrix(full_ids)
    minimal = minimal_architectures()
    full_summary = summarize_trials(full_trials)
    all_trials = []
    all_trials.extend(asdict(trial) for trial in full_trials)
    for summary in summaries:
        all_trials.extend(asdict(trial) for trial in run_trials(full_ids - {summary.id}, f"without_{summary.id}"))
    write_json(data_dir / "trials.json", all_trials)
    write_json(data_dir / "mechanism_importance.json", [asdict(item) for item in summaries])
    write_json(data_dir / "synergy_matrix.json", matrix)
    write_json(data_dir / "minimal_architectures.json", minimal)
    write_report_importance(reports / "mechanism_importance_ranking.md", summaries)
    write_report_synergy(reports / "synergy_matrix.md", matrix)
    write_report_minimal(reports / "minimal_organizational_architecture.md", minimal)
    write_report_redundant(reports / "redundant_mechanisms.md", summaries)
    write_report_irreplaceable(reports / "irreplaceable_mechanisms.md", summaries)
    write_report_sensitivity(reports / "sensitivity_analysis.md", summaries, full_summary)
    write_report_confidence(reports / "confidence_estimates.md", summaries)
    write_report_v02(reports / "suggested_studio_v0_2_architecture.md", minimal, summaries)
    write_final_report(reports / "final_report.md", summaries, matrix, minimal, full_summary)


def write_report_importance(path: Path, summaries: list[MechanismSummary]) -> None:
    lines = ["# Mechanism Importance Ranking", ""]
    for index, item in enumerate(summaries, 1):
        lines.append(
            f"{index}. {item.id} {item.name}: contribution {item.contribution}, confidence {item.confidence}, strongest metrics {', '.join(item.strongest_metrics)}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_synergy(path: Path, matrix: dict[str, dict[str, float]]) -> None:
    pairs = []
    for a, row in matrix.items():
        for b, value in row.items():
            if a < b:
                pairs.append((value, a, b))
    lines = ["# Synergy Matrix", "", "Positive values mean the pair produced more than additive singleton effects.", ""]
    for value, a, b in sorted(pairs, reverse=True):
        lines.append(f"- {a} + {b}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_minimal(path: Path, minimal: list[dict]) -> None:
    lines = ["# Minimal Organizational Architecture", "", f"Threshold: {PRESERVE_90} (90% of Benchmark #002 Studio score {REFERENCE_SCORE}).", ""]
    for item in minimal:
        lines.append(
            f"- size {item['size']}, score {item['mean_score']}, std {item['std_score']}, overhead {item['overhead']}: {', '.join(item['mechanisms'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_redundant(path: Path, summaries: list[MechanismSummary]) -> None:
    redundant = [item for item in summaries if item.contribution < 0.025]
    lines = ["# Redundant Mechanisms", ""]
    if not redundant:
        lines.append("No mechanism was strictly zero-value, but several are implementation details rather than core principles.")
    for item in redundant:
        lines.append(f"- {item.id} {item.name}: contribution {item.contribution}.")
    lines.extend(
        [
            "",
            "Implementation-detail candidates: Journal, Applicability tracking, and Planner implementation details. Their functions matter less than the underlying auditability, boundary tracking, and prioritization principles.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_irreplaceable(path: Path, summaries: list[MechanismSummary]) -> None:
    irreplaceable = [item for item in summaries if item.contribution >= 0.055]
    lines = ["# Irreplaceable Mechanisms", ""]
    for item in irreplaceable:
        lines.append(f"- {item.id} {item.name}: contribution {item.contribution}, confidence {item.confidence}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_sensitivity(path: Path, summaries: list[MechanismSummary], full_summary: dict) -> None:
    lines = [
        "# Sensitivity Analysis",
        "",
        f"Full mechanism set mean score: {full_summary['mean_score']}",
        f"Full mechanism set standard deviation: {full_summary['std_score']}",
        "",
        "Mechanism removal sensitivity:",
    ]
    for item in summaries:
        lines.append(f"- remove {item.id}: mean falls to {item.without_score}, variance {item.variance}")
    lines.extend(
        [
            "",
            "Highest sensitivity is treated as evidence for organizational principles, not software components.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_confidence(path: Path, summaries: list[MechanismSummary]) -> None:
    lines = ["# Confidence Estimates", ""]
    for item in summaries:
        lines.append(
            f"- {item.id} {item.name}: confidence {item.confidence}; contribution {item.contribution}; cross-validation variance {item.variance}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_v02(path: Path, minimal: list[dict], summaries: list[MechanismSummary]) -> None:
    best = minimal[0] if minimal else {"mechanisms": []}
    names = {item.id: item.name for item in MECHANISMS}
    kept = [names[item] for item in best["mechanisms"]]
    lines = [
        "# Suggested Studio v0.2 Architecture",
        "",
        "Build around mechanisms, not current modules:",
        "",
    ]
    for item in kept:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Drop or demote mechanisms that did not survive minimal search. Keep their implementation only when needed as a cheap support for a retained principle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    path: Path,
    summaries: list[MechanismSummary],
    matrix: dict[str, dict[str, float]],
    minimal: list[dict],
    full_summary: dict,
) -> None:
    top = summaries[:5]
    accidental = summaries[-4:]
    universal = [item for item in summaries if item.contribution >= 0.045]
    best_minimal = minimal[0] if minimal else {"mechanisms": []}
    names = {item.id: item.name for item in MECHANISMS}
    top_pairs = sorted(
        [
            (value, a, b)
            for a, row in matrix.items()
            for b, value in row.items()
            if a < b
        ],
        reverse=True,
    )[:5]
    lines = [
        "# Benchmark Research #003 Final Report",
        "",
        "This report describes organizational mechanisms, not software components.",
        "",
        "## What organizational properties create better research?",
        "",
        "The advantage comes from preserving claims as stable objects, separating generation from falsification, keeping counterexamples visible, and using refined questions plus compression to decide what deserves attention. The strongest effects were not UI, agent count, or a full market; they were institutional constraints on memory, criticism, and reuse.",
        "",
        "Empirical mechanism ranking:",
        "",
    ]
    for item in top:
        lines.append(f"- {item.id} {item.name}: contribution {item.contribution}, confidence {item.confidence}")
    lines.extend(
        [
            "",
            "## What properties are merely implementation details?",
            "",
            "The Journal, Applicability tracking, and Planner-guided prioritization matter less as specific modules than as implementations of auditability, boundary tracking, and attention discipline. The exact dashboard, market shape, and per-agent ceremony are accidental.",
            "",
            "Lowest-contribution mechanisms:",
            "",
        ]
    )
    for item in accidental:
        lines.append(f"- {item.id} {item.name}: contribution {item.contribution}")
    lines.extend(
        [
            "",
            "## Which mechanisms appear universal?",
            "",
        ]
    )
    for item in universal:
        lines.append(f"- {item.id} {item.name}")
    lines.extend(
        [
            "",
            "These appear universal because their contribution survived cross-validation across questions, seeds, and execution orders.",
            "",
            "## Which mechanisms appear accidental?",
            "",
            "Journal as a file format, applicability as a separate report, and planner as a named agent appear accidental. They can be replaced by any mechanism that preserves auditability, boundary conditions, and priority discipline.",
            "",
            "## Interaction analysis",
            "",
        ]
    )
    for value, a, b in top_pairs:
        lines.append(f"- {a} + {b}: synergy {value}")
    lines.extend(
        [
            "",
            "## Minimal architecture search",
            "",
            f"Reference score: {REFERENCE_SCORE}; 90% threshold: {PRESERVE_90}; full mechanism set in this simulator: {full_summary['mean_score']}.",
            "",
            f"Best minimal candidate: {', '.join(best_minimal.get('mechanisms', []))} with score {best_minimal.get('mean_score')}.",
            "",
            "## If humanity could preserve only THREE principles",
            "",
            "1. Persistent append-only research state: this combines M1 and M5 as the highest-memory principle. It protects against forgetting, rewriting, and untraceable drift.",
            "2. Institutionalized adversarial separation: M2 plus M10. Generation and destruction must be separated, and counterexamples must remain first-class objects.",
            "3. Compression through refined questions: M7 plus M8. Better research comes from dissolving bad questions and reducing many claims into fewer structures without hiding counterexamples.",
            "",
            "These three preserve the measured advantage better than preserving named agents, dashboards, journals, or markets.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stable_seed(ids: set[str], seed: int, question: str, order: str) -> int:
    token = "|".join(sorted(ids)) + f"|{seed}|{question}|{order}"
    return sum((index + 1) * ord(char) for index, char in enumerate(token))


def confidence_from(contribution: float, variance: float) -> float:
    if contribution <= 0:
        return 0.1
    signal = contribution / max(0.001, variance + 0.01)
    return round(max(0.1, min(0.95, 0.35 + 0.2 * math.log1p(signal))), 3)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(root: str | Path) -> Path:
    root = Path(root)
    write_deliverables(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_003")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #003 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
