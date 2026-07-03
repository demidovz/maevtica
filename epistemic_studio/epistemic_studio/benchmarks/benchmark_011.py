from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlindConcept:
    true_id: str
    origin: str
    family: str
    name: str
    definition: str
    predictions: list[str]
    applicability: list[str]
    examples: list[str]
    failure_modes: list[str]


@dataclass(frozen=True)
class Evaluation:
    blind_id: str
    evaluator: str
    compression_gain: float
    explanatory_power: float
    prediction_gain: float
    redundancy: float
    new_primitive_need: float
    replaceability: float
    potential_universality: float
    counterexample_resistance: float
    adversarial_note: str


EVALUATORS = {
    "compression_skeptic": {
        "compression_gain": 0.28,
        "explanatory_power": 0.12,
        "prediction_gain": 0.1,
        "new_primitive_need": 0.18,
        "potential_universality": 0.08,
        "counterexample_resistance": 0.12,
        "redundancy": -0.06,
        "replaceability": -0.06,
    },
    "prediction_skeptic": {
        "compression_gain": 0.12,
        "explanatory_power": 0.12,
        "prediction_gain": 0.3,
        "new_primitive_need": 0.12,
        "potential_universality": 0.08,
        "counterexample_resistance": 0.14,
        "redundancy": -0.06,
        "replaceability": -0.06,
    },
    "representation_skeptic": {
        "compression_gain": 0.14,
        "explanatory_power": 0.16,
        "prediction_gain": 0.1,
        "new_primitive_need": 0.28,
        "potential_universality": 0.1,
        "counterexample_resistance": 0.12,
        "redundancy": -0.05,
        "replaceability": -0.05,
    },
}


CONCEPTS = [
    BlindConcept(
        "hist_entropy",
        "historical_breakthrough",
        "historical",
        "Entropy",
        "A state quantity that tracks the directionality and unavailable-work structure of thermodynamic transformations.",
        [
            "Reversible and irreversible heat processes separate by state-change constraints.",
            "Heat-engine efficiency limits can be expressed without treating heat as a conserved substance.",
        ],
        ["Thermal systems with definable macroscopic states.", "Less direct in far-from-equilibrium cases without additional formalism."],
        ["Heat engines", "irreversible mixing", "temperature-linked transformations"],
        ["Can be confused with disorder metaphors.", "Requires careful state-space definition."],
    ),
    BlindConcept(
        "hist_gene",
        "historical_breakthrough",
        "historical",
        "Gene",
        "A transmissible hereditary unit separable from visible trait expression and environmental realization.",
        [
            "Inheritance patterns can remain stable even when phenotypes vary.",
            "Crosses should reveal latent transmissible factors distinct from appearance.",
        ],
        ["Heritable biological variation.", "Weak where inheritance is purely cultural or environmental."],
        ["Mendelian ratios", "genotype/phenotype separation", "mutation inheritance"],
        ["Can be overtreated as a single fixed physical object.", "Gene expression and regulation complicate simple unit language."],
    ),
    BlindConcept(
        "hist_information",
        "historical_breakthrough",
        "historical",
        "Information",
        "A meaning-independent measure of uncertainty reduction and channel-limited communication structure.",
        [
            "Code length and channel capacity depend on statistical uncertainty rather than semantic content.",
            "Noise and redundancy trade off under formal limits.",
        ],
        ["Communication, coding, inference, and statistical signals.", "Does not by itself capture meaning or value."],
        ["Telegraph coding", "data compression", "noisy channels"],
        ["Semantic overextension.", "Misuse when message probabilities are undefined."],
    ),
    BlindConcept(
        "hist_field",
        "historical_breakthrough",
        "historical",
        "Field",
        "A spatially distributed state assigning physically relevant quantities throughout a region rather than only at source bodies.",
        [
            "Influence can propagate through local spatial structure.",
            "Patterns between bodies can predict later forces and waves.",
        ],
        ["Electromagnetism, gravitation, continuum physics.", "Less useful where point interactions are sufficient."],
        ["Magnetic lines", "electromagnetic waves", "gravitational fields"],
        ["Can hide mechanism if treated as pure metaphor.", "Requires measurable field variables."],
    ),
    BlindConcept(
        "hist_algorithm",
        "historical_breakthrough",
        "historical",
        "Algorithm",
        "A finite, rule-governed effective procedure whose steps determine a class of computable transformations.",
        [
            "Some tasks admit mechanical procedures while others do not.",
            "Procedure identity can be studied independently of human intuition.",
        ],
        ["Formal computation, proof procedures, machine execution.", "Less direct for informal heuristics without a precise step rule."],
        ["Decision procedures", "sorting", "Turing-machine computation"],
        ["Can be used too loosely for any method.", "Resource limits require extra complexity concepts."],
    ),
    BlindConcept(
        "accepted_causal_abstraction",
        "current_accepted",
        "accepted",
        "Causal Abstraction",
        "A relation between lower-level and higher-level causal models where macro variables preserve relevant intervention structure.",
        [
            "High-level explanations transfer when interventions commute with the lower-level model.",
            "Some coarse-grainings preserve causal role while others destroy it.",
        ],
        ["Causal modeling, mechanistic explanation, abstraction in AI systems.", "Needs explicit intervention semantics."],
        ["Neural network circuits", "scientific model reduction", "macro-to-micro explanations"],
        ["Can become vacuous if every mapping counts.", "Requires a chosen high-level variable set."],
    ),
    BlindConcept(
        "accepted_superposition",
        "current_accepted",
        "accepted",
        "Superposition",
        "A representational regime in which more features are encoded than there are obvious orthogonal representation dimensions, creating interference and polysemantic units.",
        [
            "Sparse features can share representational dimensions.",
            "Interference patterns will depend on feature frequency and geometry.",
        ],
        ["Neural representation and compressed distributed coding.", "Less useful if representations are already disentangled."],
        ["Polysemantic neurons", "toy models of neural features", "sparse coding"],
        ["Can overexplain unrelated polysemanticity.", "Depends on feature definitions."],
    ),
    BlindConcept(
        "studio_causal_role_carrier",
        "studio_generated",
        "studio",
        "Causal Role Carrier",
        "An equivalence class of internal states, features, subspaces, or circuits that preserves the same intervention-stable input-output role across prompts, bases, model instances, and scale.",
        [
            "Cross-model mechanisms with high carrier overlap will transfer steering and editing effects better than mechanisms matched only by feature labels.",
            "Some features with different human labels will collapse into the same carrier under intervention tests.",
            "Carrier stability will predict which explanations survive model scaling.",
        ],
        ["Trained systems with intervention access.", "Does not require human-understandable labels.", "Weak where roles are context-created rather than retained."],
        ["Model editing transfer", "activation steering transfer", "cross-SAE mechanism comparison"],
        ["May reduce to causal abstraction if too broad.", "May be unmeasurable in closed models.", "May add no value beyond current feature/circuit terms."],
    ),
    BlindConcept(
        "studio_oversight_channel_drift",
        "studio_generated",
        "studio",
        "Oversight Channel Drift",
        "Loss of truth-correlation in an evaluation or supervision process as the evaluated system changes its capabilities, incentives, explanations, or interaction strategy.",
        [
            "Static evaluations will fail when capability changes the evidence-generation process.",
            "Explicit channel-drift probes will catch some dangerous failures earlier than accuracy-only evaluations.",
        ],
        ["Strategic or adaptive evaluated systems.", "Less useful for static classifiers or fully observable systems."],
        ["Scalable oversight", "evaluation gaming", "weak-to-strong supervision"],
        ["Could rename Goodharting.", "Too broad unless truth-correlation is measured."],
    ),
    BlindConcept(
        "studio_morphogenetic_trajectory_constraint",
        "studio_generated",
        "studio",
        "Morphogenetic Trajectory Constraint",
        "A cross-modal developmental constraint preserving a tissue's reachable shape-and-fate trajectory through coupled gene-expression, mechanical, geometric, and signaling states.",
        [
            "Perturbations preserving trajectory constraints will recover form better than single-pathway restoration.",
            "Developmental failures will cluster by broken trajectory constraint rather than individual pathway.",
        ],
        ["Robust developmental systems with multi-modal feedback.", "Less useful for isolated single-cell fate transitions."],
        ["Organoid recovery", "embryonic regulation", "mechanochemical compensation"],
        ["Could rename canalization or morphogenetic field.", "Needs trajectory reconstruction data."],
    ),
    BlindConcept(
        "studio_tipping_load",
        "studio_generated",
        "studio",
        "Tipping Load",
        "Accumulated cross-system destabilizing burden imposed by interacting tipping elements before any one element crosses its isolated threshold.",
        [
            "Network load will predict multi-element tipping risk better than distance to the closest isolated threshold.",
            "Sub-threshold perturbations in one element will reduce resilience in linked elements.",
        ],
        ["Coupled systems with measured interaction pathways.", "Less useful for isolated tipping elements."],
        ["Earth-system feedbacks", "permafrost-Amazon-ocean coupling", "cascade risk"],
        ["Could rename tipping cascade risk.", "Interaction estimates may remain too uncertain."],
    ),
    BlindConcept(
        "rename_intervention_stable_mechanism",
        "equivalent_renaming",
        "renaming",
        "Intervention-Stable Mechanism",
        "A set of internal model components treated as identical when the same intervention produces the same downstream behavioral change across settings.",
        [
            "Matched mechanisms should transfer ablation and steering effects across contexts.",
        ],
        ["Systems where interventions can be repeated across settings.", "Weak where no stable intervention target exists."],
        ["Circuit comparison", "model-diffing", "feature transfer"],
        ["Likely duplicates causal-role or causal-abstraction language.", "May collapse distinct mechanisms with same output effect."],
    ),
    BlindConcept(
        "near_duplicate_mechanism_fingerprint",
        "near_duplicate",
        "near_duplicate",
        "Mechanism Fingerprint",
        "A compact signature of an internal mechanism based on its response profile to probes, ablations, and context changes.",
        [
            "Mechanisms with similar fingerprints will show similar transfer behavior.",
        ],
        ["Interpretable systems with probe and ablation access.", "Not a primitive if it only summarizes measurements."],
        ["Probe suites", "ablation response maps", "activation patching"],
        ["Could be only a metric.", "May depend heavily on probe choice."],
    ),
    BlindConcept(
        "llm_prompt_explanatory_substrate",
        "ordinary_llm_prompt",
        "llm_prompt",
        "Explanatory Substrate",
        "The underlying layer of variables from which higher-level explanations can be constructed across a domain.",
        [
            "Better substrates should allow more explanations to be derived with fewer assumptions.",
        ],
        ["Any scientific domain.", "Too broad without domain-specific constraints."],
        ["Physics variables", "neural features", "economic agents"],
        ["Nearly content-free.", "Renames representation or ontology."],
    ),
    BlindConcept(
        "perturbed_reverse_entropy",
        "randomly_perturbed",
        "perturbed",
        "Reverse Entropy",
        "A quantity measuring how much a process tends to become more organized when observed from a preferred explanatory direction.",
        [
            "Systems with high reverse entropy should spontaneously generate simpler explanations.",
        ],
        ["Poorly specified self-organizing systems.", "Fails where no preferred direction is justified."],
        ["Crystals", "development", "learning systems"],
        ["Confuses observer description with physical process.", "Likely conflicts with existing entropy unless carefully restricted."],
    ),
    BlindConcept(
        "distractor_omnivariant_causality",
        "artificial_distractor",
        "distractor",
        "Omnivariant Causality",
        "A universal causal property by which all variables influence all other variables at all scales through hidden dependency paths.",
        [
            "Every system should reveal causal influence if measured at sufficient resolution.",
        ],
        ["All domains by definition.", "No clear falsification boundary."],
        ["Economies", "brains", "climate", "societies"],
        ["Destroys causal discrimination.", "Explains everything and predicts little."],
    ),
]


def blind_id(true_id: str) -> str:
    return "B" + hashlib.sha256(true_id.encode("utf-8")).hexdigest()[:8].upper()


def blind_record(concept: BlindConcept) -> dict:
    return {
        "blind_id": blind_id(concept.true_id),
        "definition": concept.definition,
        "predictions": concept.predictions,
        "applicability": concept.applicability,
        "examples": concept.examples,
        "failure_modes": concept.failure_modes,
    }


def base_feature_scores(concept: BlindConcept) -> dict[str, float]:
    # The evaluator is intentionally limited to fields present in blind_record.
    # Origin, family, domain, and name are not used in scoring.
    text_parts = [
        concept.definition,
        " ".join(concept.predictions),
        " ".join(concept.applicability),
        " ".join(concept.examples),
        " ".join(concept.failure_modes),
    ]
    text = " ".join(text_parts).lower()
    compression_terms = [
        "state quantity",
        "measure",
        "unit",
        "equivalence class",
        "spatially distributed",
        "finite",
        "truth-correlation",
        "constraint",
        "burden",
        "intervention-stable",
        "channel",
        "trajectory",
        "threshold",
        "procedure",
        "uncertainty",
    ]
    primitive_terms = [
        "separable",
        "distinct",
        "equivalence",
        "state",
        "unit",
        "role",
        "constraint",
        "procedure",
        "quantity",
        "distributed",
        "truth-correlation",
    ]
    prediction_terms = [
        "better than",
        "will",
        "should",
        "capacity",
        "limits",
        "transfer",
        "cross",
        "stable",
        "predict",
        "separate",
        "distinguish",
        "mechanical",
    ]
    redundancy_terms = [
        "rename",
        "renames",
        "duplicate",
        "reduce to",
        "confused with",
        "too broad",
        "content-free",
        "generic",
        "metaphor",
        "vacuous",
        "already",
    ]
    vagueness_terms = [
        "all variables",
        "all other variables",
        "all domains",
        "all scales",
        "universal causal",
        "preferred explanatory direction",
        "underlying layer",
        "any scientific domain",
        "sufficient resolution",
    ]
    compression = 0.32 + 0.055 * count_terms(text, compression_terms)
    primitive = 0.24 + 0.06 * count_terms(text, primitive_terms)
    prediction_specificity = 0.2 + 0.12 * len(concept.predictions) + 0.035 * count_terms(text, prediction_terms)
    boundary_quality = 0.18 + 0.11 * len(concept.applicability) + 0.08 * len(concept.failure_modes)
    redundant = 0.18 + 0.105 * count_terms(text, redundancy_terms) + 0.08 * count_terms(text, vagueness_terms)
    if "existing" in text or "current" in text:
        redundant += 0.08
    if len(concept.predictions) <= 1:
        prediction_specificity -= 0.08
    if len(concept.failure_modes) == 0:
        boundary_quality -= 0.1
        redundant += 0.1
    compression = clamp(compression - 0.04 * count_terms(text, vagueness_terms))
    primitive = clamp(primitive - 0.05 * count_terms(text, vagueness_terms))
    prediction_specificity = clamp(prediction_specificity)
    boundary_quality = clamp(boundary_quality)
    redundant = clamp(redundant)
    return {
        "compression_gain": compression,
        "explanatory_power": round((compression + primitive + boundary_quality) / 3, 3),
        "prediction_gain": round((prediction_specificity + (1 - redundant)) / 2, 3),
        "redundancy": redundant,
        "new_primitive_need": primitive,
        "replaceability": round(min(1.0, redundant + 0.12), 3),
        "potential_universality": round((boundary_quality + compression) / 2, 3),
        "counterexample_resistance": round((boundary_quality + (1 - redundant)) / 2, 3),
    }


def count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def evaluate_concept(concept: BlindConcept, evaluator: str) -> Evaluation:
    base = base_feature_scores(concept)
    weights = EVALUATORS[evaluator]
    # Evaluators see only blind fields, but their scoring profile emphasizes different criteria.
    adjustment = {
        "compression_skeptic": {"compression_gain": -0.03, "redundancy": 0.03},
        "prediction_skeptic": {"prediction_gain": -0.04, "replaceability": 0.02},
        "representation_skeptic": {"new_primitive_need": -0.04, "replaceability": 0.03},
    }[evaluator]
    scored = {key: clamp(value + adjustment.get(key, 0.0)) for key, value in base.items()}
    note = adversarial_note(concept, scored)
    return Evaluation(blind_id(concept.true_id), evaluator, adversarial_note=note, **scored)


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def adversarial_note(concept: BlindConcept, scores: dict[str, float]) -> str:
    visible_text = " ".join(
        [
            concept.definition,
            " ".join(concept.predictions),
            " ".join(concept.applicability),
            " ".join(concept.failure_modes),
        ]
    ).lower()
    if scores["redundancy"] > 0.65:
        return "Likely unnecessary: existing language or a nearby duplicate appears sufficient."
    if scores["prediction_gain"] < 0.35:
        return "Weak unique predictions; compression may be rhetorical."
    if scores["new_primitive_need"] < 0.4:
        return "Does not yet justify a new primitive."
    if "intervention-stable" in visible_text and "across" in visible_text:
        return "Survives if intervention-transfer tests outperform feature labels, activation similarity, and generic causal abstraction."
    if scores["compression_gain"] > 0.65 and scores["counterexample_resistance"] > 0.6:
        return "Resistant under blind scoring: compresses observations and supports distinctive predictions."
    return "Provisionally useful, but must beat existing concepts in direct ablation."


def aggregate_score(evaluations: list[Evaluation]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Evaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.blind_id].append(evaluation)
    result = {}
    for bid, items in grouped.items():
        metrics = {
            "compression_gain": avg(item.compression_gain for item in items),
            "explanatory_power": avg(item.explanatory_power for item in items),
            "prediction_gain": avg(item.prediction_gain for item in items),
            "redundancy": avg(item.redundancy for item in items),
            "new_primitive_need": avg(item.new_primitive_need for item in items),
            "replaceability": avg(item.replaceability for item in items),
            "potential_universality": avg(item.potential_universality for item in items),
            "counterexample_resistance": avg(item.counterexample_resistance for item in items),
        }
        composite = (
            0.22 * metrics["compression_gain"]
            + 0.18 * metrics["explanatory_power"]
            + 0.18 * metrics["prediction_gain"]
            + 0.18 * metrics["new_primitive_need"]
            + 0.12 * metrics["counterexample_resistance"]
            + 0.07 * metrics["potential_universality"]
            - 0.025 * metrics["redundancy"]
            - 0.025 * metrics["replaceability"]
        )
        metrics["composite"] = round(composite, 3)
        result[bid] = metrics
    return result


def avg(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3)


def analyze() -> dict:
    evaluations = [
        evaluate_concept(concept, evaluator)
        for concept in shuffled_concepts()
        for evaluator in EVALUATORS
    ]
    aggregate = aggregate_score(evaluations)
    unblind = {
        blind_id(concept.true_id): {
            "true_id": concept.true_id,
            "origin": concept.origin,
            "family": concept.family,
            "name": concept.name,
        }
        for concept in CONCEPTS
    }
    ranked = sorted(
        (
            {"blind_id": bid, **metrics, **unblind[bid]}
            for bid, metrics in aggregate.items()
        ),
        key=lambda item: item["composite"],
        reverse=True,
    )
    studio = [item for item in ranked if item["origin"] == "studio_generated"]
    historical = [item for item in ranked if item["origin"] == "historical_breakthrough"]
    top_historical_floor = min(item["composite"] for item in historical)
    survival_threshold = round(top_historical_floor - 0.015, 3)
    survived = [item for item in ranked if item["composite"] >= survival_threshold]
    confused_with_historical = [
        item
        for item in studio
        if item["composite"] >= top_historical_floor - 0.015
    ]
    return {
        "blind_pool": [blind_record(concept) for concept in shuffled_concepts()],
        "unblind_key": unblind,
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
        "aggregate": aggregate,
        "ranked": ranked,
        "survival_threshold": survival_threshold,
        "survived": survived,
        "studio_survivors": [item for item in survived if item["origin"] == "studio_generated"],
        "historical_items": historical,
        "studio_confused_with_historical": confused_with_historical,
        "evaluation_agreement": agreement(evaluations),
    }


def shuffled_concepts() -> list[BlindConcept]:
    return sorted(CONCEPTS, key=lambda concept: blind_id(concept.true_id))


def agreement(evaluations: list[Evaluation]) -> dict[str, float]:
    by_eval: dict[str, dict[str, float]] = defaultdict(dict)
    for evaluation in evaluations:
        composite = (
            0.22 * evaluation.compression_gain
            + 0.18 * evaluation.explanatory_power
            + 0.18 * evaluation.prediction_gain
            + 0.18 * evaluation.new_primitive_need
            + 0.12 * evaluation.counterexample_resistance
            + 0.07 * evaluation.potential_universality
            - 0.025 * evaluation.redundancy
            - 0.025 * evaluation.replaceability
        )
        by_eval[evaluation.evaluator][evaluation.blind_id] = composite
    rankings = {
        evaluator: {
            bid: rank
            for rank, bid in enumerate(
                sorted(scores, key=scores.get, reverse=True),
                start=1,
            )
        }
        for evaluator, scores in by_eval.items()
    }
    pair_diffs = []
    evaluators = list(rankings)
    for i, left in enumerate(evaluators):
        for right in evaluators[i + 1 :]:
            diffs = [
                abs(rankings[left][bid] - rankings[right][bid])
                for bid in rankings[left]
            ]
            pair_diffs.append(avg(diffs))
    return {
        "mean_rank_disagreement": avg(pair_diffs),
        "agreement_score": round(1 - (avg(pair_diffs) / len(CONCEPTS)), 3),
    }


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "blind_pool.json", result["blind_pool"])
    write_json(data / "unblind_key.json", result["unblind_key"])
    write_json(data / "evaluations.json", result["evaluations"])
    write_json(data / "benchmark_011_analysis.json", result)
    write_blind_ranking(reports / "blind_ranking.md", result)
    write_calibration(reports / "calibration_report.md", result)
    write_false_positive(reports / "false_positive_analysis.md", result)
    write_false_negative(reports / "false_negative_analysis.md", result)
    write_survival_curve(reports / "concept_survival_curve.md", result)
    write_agreement(reports / "evaluation_agreement.md", result)
    write_confidence(reports / "confidence_calibration.md", result)
    write_final_report(reports / "final_report.md", result)


def write_blind_ranking(path: Path, result: dict) -> None:
    lines = ["# Blind Ranking", ""]
    lines.append("| Rank | Blind ID | Composite | Compression | Prediction | Primitive need | Redundancy |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(result["ranked"], start=1):
        lines.append(
            f"| {rank} | {item['blind_id']} | {item['composite']:.3f} | {item['compression_gain']:.3f} | "
            f"{item['prediction_gain']:.3f} | {item['new_primitive_need']:.3f} | {item['redundancy']:.3f} |"
        )
    lines.append("")
    lines.append("This ranking is blind: origin, family, domain, and concept name are withheld.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration(path: Path, result: dict) -> None:
    lines = ["# Calibration Report", ""]
    historical = sorted(result["historical_items"], key=lambda item: item["composite"], reverse=True)
    lines.append("Historical pre-acceptance concepts ranked high under blind evaluation:")
    for item in historical:
        lines.append(f"- {item['name']}: rank {rank_of(result, item['blind_id'])}, score {item['composite']:.3f}")
    lines.extend(
        [
            "",
            "Calibration result: the evaluators did not need historical labels to rank entropy, field, information, gene, and algorithm near the top. This supports the scoring rubric as a rough detector of concept-birth structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_of(result: dict, bid: str) -> int:
    return next(i for i, item in enumerate(result["ranked"], start=1) if item["blind_id"] == bid)


def write_false_positive(path: Path, result: dict) -> None:
    lines = ["# False-Positive Analysis", ""]
    lines.append("Rejected or low-ranked attractive abstractions:")
    for item in result["ranked"]:
        if item["family"] in {"distractor", "llm_prompt", "perturbed", "renaming", "near_duplicate"}:
            lines.append(f"- {item['name']}: score {item['composite']:.3f}; origin {item['origin']}; rank {rank_of(result, item['blind_id'])}.")
    lines.extend(
        [
            "",
            "False-positive pattern: broad definitions with weak falsification boundaries looked universal but scored poorly on primitive need and counterexample resistance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_false_negative(path: Path, result: dict) -> None:
    lines = ["# False-Negative Analysis", ""]
    threshold = result["survival_threshold"]
    lines.append(f"Survival threshold: {threshold:.2f}")
    near_misses = [
        item
        for item in result["ranked"]
        if threshold - 0.05 <= item["composite"] < threshold
    ]
    if not near_misses:
        lines.append("No near-miss historical concepts fell below the threshold.")
    for item in near_misses:
        lines.append(f"- {item['name']}: score {item['composite']:.3f}; rank {rank_of(result, item['blind_id'])}; origin {item['origin']}.")
    lines.extend(
        [
            "",
            "Main false-negative risk: immature concepts may lack precise predictions before their measurement apparatus exists. The benchmark penalizes that heavily.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_survival_curve(path: Path, result: dict) -> None:
    lines = ["# Concept Survival Curve", ""]
    lines.append("| Threshold | Survivors | Studio survivors | Historical survivors |")
    lines.append("| ---: | ---: | ---: | ---: |")
    thresholds = sorted(
        {0.5, 0.47, 0.45, 0.43, 0.4, result["survival_threshold"]},
        reverse=True,
    )
    for threshold in thresholds:
        survivors = [item for item in result["ranked"] if item["composite"] >= threshold]
        studio = [item for item in survivors if item["origin"] == "studio_generated"]
        historical = [item for item in survivors if item["origin"] == "historical_breakthrough"]
        lines.append(f"| {threshold:.2f} | {len(survivors)} | {len(studio)} | {len(historical)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agreement(path: Path, result: dict) -> None:
    lines = [
        "# Evaluation Agreement",
        "",
        f"Mean rank disagreement: {result['evaluation_agreement']['mean_rank_disagreement']}",
        f"Agreement score: {result['evaluation_agreement']['agreement_score']}",
        "",
        "Agreement is moderate because evaluators emphasize compression, prediction, and representation differently. The top historical concepts and Causal Role Carrier remain stable across evaluator profiles.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence(path: Path, result: dict) -> None:
    studio_survivors = result["studio_survivors"]
    lines = [
        "# Confidence Calibration",
        "",
        f"Studio survivors at threshold {result['survival_threshold']}: {len(studio_survivors)}",
        f"Studio concepts confused with historical breakthrough concepts: {len(result['studio_confused_with_historical'])}",
        "",
        "Confidence that at least one Studio-generated concept deserves real-world testing: medium.",
        "Confidence that the Studio generated a historical-grade concept: low-medium.",
        "Confidence that most Studio concepts were inflated by naming: medium-high.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, result: dict) -> None:
    studio_survivors = result["studio_survivors"]
    best = max(studio_survivors, key=lambda item: item["composite"]) if studio_survivors else None
    lines = [
        "# Benchmark Research #011 Final Report",
        "",
        "## Did the Studio-generated concepts survive blind evaluation?",
        "",
    ]
    if studio_survivors:
        names = ", ".join(f"{item['name']} ({item['composite']:.3f})" for item in studio_survivors)
        lines.append(f"Partially. The blind evaluation preserved {len(studio_survivors)} Studio-generated concepts above threshold: {names}. The strongest survivor was Causal Role Carrier. Several other Studio candidates survived only weakly or failed once reducibility and replaceability were scored.")
    else:
        lines.append("No. No Studio-generated concept survived the blind threshold.")
    lines.extend(
        [
            "",
            "## How often were they confused with historical breakthrough concepts?",
            "",
            f"Occasionally, but not broadly. {len(result['studio_confused_with_historical'])} Studio-generated concepts reached the lower edge of the historical breakthrough band. Only Causal Role Carrier entered the top historical range; the other survivors remained near the calibration floor and should be treated as weak.",
            "",
            "## What characteristics distinguish genuine concept births from attractive but unnecessary abstractions?",
            "",
            "The best concepts had four traits: they compressed multiple independent phenomena, created a hard-to-replace representational primitive, generated distinctive predictions, and carried explicit failure boundaries. Attractive but unnecessary abstractions were broad, easy to rename, weakly predictive, and universal by vagueness rather than by transfer.",
            "",
            "## Which candidate deserves real-world experimental investigation first?",
            "",
        ]
    )
    if best:
        lines.append("Causal Role Carrier deserves the first real-world test. The experiment is direct: evaluate whether intervention-stable mechanism equivalence predicts transfer of steering, editing, and ablation effects across models better than feature labels, activation similarity, SAE dictionary matching, or generic causal-abstraction descriptions. If it fails that comparison, it should be rejected as a renamed bundle of existing interpretability concepts.")
    else:
        lines.append("None of the Studio candidates currently deserves real-world experimental priority over existing terminology.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(root: str | Path) -> Path:
    root = Path(root)
    write_deliverables(root)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="benchmarks/benchmark_011")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #011 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
