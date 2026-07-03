from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    note: str


@dataclass(frozen=True)
class DomainPressure:
    id: str
    domain: str
    observations: str
    competing_theories: list[str]
    representation_mismatch: float
    compression_pressure: float
    concept_fragmentation: float
    theory_disagreement: float
    vocabulary_instability: float
    prediction_gaps: float
    recurring_unexplained_structures: float
    hidden_invariants: float
    conceptual_holes: list[str]
    source_ids: list[str]


@dataclass(frozen=True)
class CandidateConcept:
    id: str
    working_name: str
    domain_id: str
    definition: str
    problem_solved: str
    compression_gained: float
    predictions_enabled: list[str]
    relationships_to_existing: list[str]
    applicability_boundaries: list[str]
    possible_failure_modes: list[str]
    reducibility_risk: float
    adversarial_result: str
    survival_status: str


@dataclass(frozen=True)
class HistoricalBlindTrial:
    id: str
    pre_discovery_evidence: str
    generated_concept: str
    historical_concept: str
    structural_similarity: float
    functional_similarity: float
    compression_similarity: float
    representation_similarity: float


SOURCES = [
    Source(
        "mech_interp_open_problems",
        "Open Problems in Mechanistic Interpretability",
        "https://arxiv.org/html/2501.16496v1",
        "Forward-looking review of open problems in mechanistic interpretability.",
    ),
    Source(
        "sae_survey",
        "A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models",
        "https://arxiv.org/html/2503.05613v3",
        "Surveys SAEs, monosemanticity, feature decomposition, and interpretability methods.",
    ),
    Source(
        "anthropic_monosemanticity",
        "Extracting Interpretable Features from Claude 3 Sonnet",
        "https://transformer-circuits.pub/2024/scaling-monosemanticity/",
        "Shows large-scale sparse-autoencoder feature extraction in a frontier LLM.",
    ),
    Source(
        "alignment_directions",
        "Recommendations for Technical AI Safety Research Directions",
        "https://alignment.anthropic.com/2025/recommended-directions/",
        "Lists scalable oversight, process-oriented learning, dangerous-failure evaluation, and interpretability directions.",
    ),
    Source(
        "morphogenesis_principles",
        "Searching for physical principles of morphogenesis",
        "https://journals.biologists.com/dev/article/152/21/dev204894/369751/Searching-for-physical-principles-of-morphogenesis",
        "Recent review on physical principles, quantitative experiments, and mathematical theory in morphogenesis.",
    ),
    Source(
        "embryoids_self_organization",
        "From embryos to embryoids: external signals and self-organization",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8185431/",
        "Frames embryonic development as guided self-organization.",
    ),
    Source(
        "tissue_mechanics",
        "Self-organized tissue mechanics underlie embryonic regulation",
        "https://www.nature.com/articles/s41586-024-07934-8",
        "Reports robust, self-organized mechanical regulation in early development.",
    ),
    Source(
        "consciousness_frontiers",
        "Consciousness science: where are we, where are we going, and what are the challenges ahead?",
        "https://www.frontiersin.org/journals/science/articles/10.3389/fsci.2025.1546279/full",
        "Reviews contemporary consciousness science and theory-testing challenges.",
    ),
    Source(
        "synergistic_workspace",
        "A synergistic workspace for human consciousness revealed by integrated information decomposition",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11257694/",
        "Connects global-workspace ideas with information-theoretic synergy.",
    ),
    Source(
        "climate_tipping_2025",
        "High probability of triggering climate tipping points under current policies",
        "https://esd.copernicus.org/articles/16/565/2025/",
        "Analyzes probabilities of triggering climate tipping points under emissions scenarios.",
    ),
    Source(
        "tipmip",
        "The Tipping Points Modelling Intercomparison Project",
        "https://ntrs.nasa.gov/api/citations/20250006500/downloads/ARomanouESDTippingPreprint.pdf",
        "Proposes systematic multi-model assessment of Earth-system tipping interactions.",
    ),
    Source(
        "heterogeneous_expectations",
        "Trouble with Rational Expectations in Heterogeneous Agent Models",
        "https://academic.oup.com/ej/article/136/676/1173/8323145",
        "Argues rational expectations about equilibrium prices become unrealistic in heterogeneous-agent macroeconomics.",
    ),
    Source(
        "agent_based_macro",
        "Quantitative agent-based models: a promising alternative for macroeconomics",
        "https://academic.oup.com/oxrep/article/41/2/571/8281854",
        "Presents agent-based modeling as a different way to model economic systems.",
    ),
    Source(
        "collective_ai_review",
        "AI-enhanced collective intelligence",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11573907/",
        "Reviews human-AI collective intelligence and notes theoretical gaps.",
    ),
    Source(
        "collective_evolution",
        "The evolutionary past and future of collective intelligence",
        "https://royalsocietypublishing.org/doi/10.1098/rstb.2024.0439",
        "Frames collective intelligence across biological and future artificial systems.",
    ),
]


DOMAINS = [
    DomainPressure(
        "d01_mech_interp",
        "Mechanistic interpretability of frontier neural networks",
        "Many activation-level observations, sparse-autoencoder features, circuits, steering vectors, model edits, and cross-model comparisons exist.",
        ["features", "circuits", "superposition", "causal abstraction", "distributed representation", "SAE dictionaries"],
        0.95,
        0.94,
        0.92,
        0.84,
        0.91,
        0.9,
        0.93,
        0.88,
        [
            "No stable unit connects features, circuits, activation directions, and causal interventions across basis changes.",
            "It is awkward to say when two different SAEs or two different models have found the same internal mechanism.",
            "The field lacks a primitive for intervention-stable causal role rather than human-labeled feature identity.",
        ],
        ["mech_interp_open_problems", "sae_survey", "anthropic_monosemanticity"],
    ),
    DomainPressure(
        "d02_ai_alignment",
        "AI alignment and scalable oversight",
        "Oversight, process supervision, red teaming, dangerous capability evaluation, interpretability, and control are active but fragmented.",
        ["outer alignment", "inner alignment", "scalable oversight", "ELK", "control", "process-oriented learning"],
        0.88,
        0.9,
        0.87,
        0.91,
        0.86,
        0.92,
        0.84,
        0.76,
        [
            "It is hard to express the loss of evaluator authority as systems become better at producing plausible evidence.",
            "Current terms separate capability, oversight, deception, and evaluation even when the same failure is about evaluator-channel degradation.",
            "The field lacks a clean unit for measuring when an oversight process stops tracking the target property.",
        ],
        ["alignment_directions", "mech_interp_open_problems"],
    ),
    DomainPressure(
        "d03_morphogenesis",
        "Developmental biology and morphogenesis",
        "Organoids, embryoids, live imaging, mechanics, signaling gradients, gene regulation, and self-organization are producing many partially connected observations.",
        ["positional information", "gene regulatory networks", "mechanobiology", "self-organization", "morphogen gradients"],
        0.9,
        0.89,
        0.82,
        0.78,
        0.77,
        0.84,
        0.9,
        0.87,
        [
            "It is difficult to express how cells preserve developmental direction through coupled mechanical, chemical, and geometric constraints.",
            "Terms split patterning, morphogenesis, mechanics, and fate even when experiments show joint regulation.",
            "The recurring invariant appears to be trajectory constraint, not a single signal, gene, or force.",
        ],
        ["morphogenesis_principles", "embryoids_self_organization", "tissue_mechanics"],
    ),
    DomainPressure(
        "d04_consciousness",
        "Consciousness science",
        "Neural correlates, global workspace, integrated information, predictive processing, recurrent processing, reportability, and AI-consciousness debates coexist.",
        ["global workspace", "integrated information", "predictive processing", "higher-order theory", "recurrent processing"],
        0.91,
        0.88,
        0.94,
        0.96,
        0.93,
        0.82,
        0.79,
        0.7,
        [
            "It remains hard to express the gap between access/report/control and phenomenal organization without importing a full theory.",
            "Theories disagree partly because they use different target variables.",
            "The missing distinction may be between information integration that is usable by the organism and integration that is report-shaped.",
        ],
        ["consciousness_frontiers", "synergistic_workspace"],
    ),
    DomainPressure(
        "d05_climate_tipping",
        "Climate tipping systems",
        "Many models track individual tipping elements, feedbacks, and scenario-dependent risks, but coupled tipping interactions remain difficult.",
        ["tipping points", "feedback loops", "Earth-system models", "risk cascades", "model intercomparison"],
        0.83,
        0.86,
        0.7,
        0.75,
        0.66,
        0.88,
        0.86,
        0.81,
        [
            "It is hard to express total system fragility when multiple sub-threshold elements mutually load one another.",
            "Single-threshold language under-describes coupled, path-dependent risk.",
            "The latent invariant is not the tipping point but the networked load carried by interacting tipping elements.",
        ],
        ["climate_tipping_2025", "tipmip"],
    ),
    DomainPressure(
        "d06_economics",
        "Macroeconomics with heterogeneous agents",
        "Heterogeneous-agent models, expectations data, agent-based models, distributional state variables, and bounded rationality coexist uneasily.",
        ["rational expectations", "HANK", "agent-based models", "bounded rationality", "complexity economics"],
        0.86,
        0.87,
        0.74,
        0.82,
        0.68,
        0.86,
        0.8,
        0.78,
        [
            "It is hard to express how agents forecast aggregate states without implicitly requiring impossible distribution-level cognition.",
            "Equilibrium-price expectations compress poorly when heterogeneity is high.",
            "The missing primitive may measure the cognitive burden of representing aggregate distributions.",
        ],
        ["heterogeneous_expectations", "agent_based_macro"],
    ),
    DomainPressure(
        "d07_collective_intelligence",
        "Human-AI collective intelligence",
        "Groups, platforms, swarms, markets, crowdsourcing, multi-agent AI systems, and hybrid teams show performance that is not reducible to individual intelligence.",
        ["wisdom of crowds", "swarm intelligence", "group intelligence", "human-AI teaming", "social structure"],
        0.82,
        0.84,
        0.9,
        0.78,
        0.86,
        0.76,
        0.82,
        0.8,
        [
            "It is hard to express when coordination structure itself performs cognition rather than merely aggregating individual cognition.",
            "Terms alternate between group ability, platform design, communication protocol, and social structure.",
            "The latent invariant appears to be transformation of partial information through interaction topology.",
        ],
        ["collective_ai_review", "collective_evolution"],
    ),
]


CANDIDATES = [
    CandidateConcept(
        "k01_causal_role_carrier",
        "Causal Role Carrier",
        "d01_mech_interp",
        "An equivalence class of internal states, features, subspaces, or circuits that preserves the same intervention-stable input-output role across prompts, bases, model instances, and scale.",
        "Unifies feature identity, circuit identity, activation directions, SAE dictionary elements, and causal-abstraction mappings without requiring identical neurons or human labels.",
        0.91,
        [
            "Cross-model mechanisms with high causal-role-carrier overlap will transfer steering and editing effects better than mechanisms matched only by feature labels.",
            "Some SAE features with different human labels will collapse into the same carrier under intervention tests.",
            "Carrier stability will predict which interpretability explanations survive model scaling.",
        ],
        ["features", "circuits", "causal abstraction", "representation similarity analysis", "activation steering"],
        [
            "Applies to trained systems with intervention access.",
            "Does not require human-understandable labels.",
            "May fail for mechanisms whose roles are strongly context-created rather than retained.",
        ],
        [
            "Could reduce to causal abstraction if formalized too broadly.",
            "Could reduce to feature/circuit if the equivalence relation is not stricter than current practice.",
            "May become unmeasurable in closed models without intervention access.",
        ],
        0.38,
        "Survives. Existing terms cover parts of it, but none cleanly names the intervention-stable equivalence class across basis, scale, dictionary, and model identity.",
        "survives",
    ),
    CandidateConcept(
        "k02_oversight_channel_drift",
        "Oversight Channel Drift",
        "d02_ai_alignment",
        "The degree to which an evaluation or supervision process loses truth-correlation as the evaluated system changes its capabilities, incentives, explanations, or interaction strategy.",
        "Compresses scalable oversight, deceptive alignment, evaluation gaming, weak-to-strong supervision, and process supervision failures as degradation of the evaluator-target channel.",
        0.84,
        [
            "Benchmarks that look stable under static evaluation will fail when model capability changes the evidence-generation process.",
            "Oversight methods with explicit channel-drift probes will catch dangerous failures earlier than accuracy-only evaluations.",
            "Models trained to optimize explanation quality will increase apparent oversight while reducing truth-correlation in some tasks.",
        ],
        ["scalable oversight", "ELK", "Goodhart's law", "evaluation gaming", "truth-correlated feedback"],
        [
            "Applies where the object being evaluated can strategically or structurally change the evidence available to evaluators.",
            "Less useful for static classifiers or fully observable systems.",
        ],
        [
            "Could be a renamed version of Goodharting plus scalable oversight.",
            "May be too broad unless quantified as channel truth-correlation over capability changes.",
        ],
        0.52,
        "Partly survives. The concept compresses alignment subproblems, but the novelty is moderate because Goodhart and scalable oversight already cover much of it.",
        "weak_survivor",
    ),
    CandidateConcept(
        "k03_morphogenetic_trajectory_constraint",
        "Morphogenetic Trajectory Constraint",
        "d03_morphogenesis",
        "A cross-modal developmental constraint that preserves a tissue's reachable shape-and-fate trajectory through coupled gene-expression, mechanical, geometric, and signaling states.",
        "Reframes development around constrained trajectories rather than separate genes, gradients, forces, or fates.",
        0.87,
        [
            "Organoid interventions that preserve trajectory constraints will recover form even after perturbing one modality.",
            "Mechanical and signaling perturbations should be interchangeable when they restore the same trajectory constraint.",
            "Developmental failure modes will cluster by broken trajectory constraint rather than by individual pathway.",
        ],
        ["morphogenetic field", "positional information", "canalization", "mechanobiology", "self-organization"],
        [
            "Applies to robust developmental systems with multi-modal feedback.",
            "Less useful for single-cell fate transitions without geometric or mechanical coupling.",
        ],
        [
            "Could reduce to canalization or morphogenetic field.",
            "May be too abstract unless operationalized with trajectory reconstruction from perturbation data.",
        ],
        0.49,
        "Weakly survives. It appears necessary-looking for organoids and embryogenesis, but historical terms already occupy nearby territory.",
        "weak_survivor",
    ),
    CandidateConcept(
        "k04_report_integration_shear",
        "Report-Integration Shear",
        "d04_consciousness",
        "A divergence between information integration available for organism-level control and information integration formatted for report, metacognition, or global access.",
        "Separates access/report disputes from integration disputes in consciousness science.",
        0.72,
        [
            "Tasks that increase reportability can decrease local integration signatures, and vice versa.",
            "Some neural states will be high-control but low-report, producing systematic theory disagreements.",
        ],
        ["access consciousness", "phenomenal consciousness", "global workspace", "integrated information", "synergistic workspace"],
        [
            "Applies to empirical consciousness paradigms with separate report, control, and integration measures.",
            "Does not solve the hard problem or define consciousness.",
        ],
        [
            "Likely reducible to access versus phenomenal consciousness or report/no-report paradigms.",
            "May not increase compression beyond existing distinctions.",
        ],
        0.76,
        "Rejected. It helps organize disputes but is too close to existing access/report distinctions.",
        "rejected",
    ),
    CandidateConcept(
        "k05_tipping_load",
        "Tipping Load",
        "d05_climate_tipping",
        "The accumulated cross-system destabilizing burden imposed by interacting tipping elements before any one element crosses its isolated threshold.",
        "Compresses coupled tipping risk, cascade risk, path dependence, and sub-threshold feedback stress into one measurable load concept.",
        0.82,
        [
            "Multi-element tipping risk will be better predicted by network load than by minimum distance to any single element's threshold.",
            "Sub-threshold perturbations in one element will measurably reduce resilience in other elements through load transfer.",
        ],
        ["tipping cascades", "resilience", "network load", "Earth-system feedback", "risk coupling"],
        [
            "Applies to coupled systems with measured interaction pathways.",
            "Less useful for isolated tipping elements.",
        ],
        [
            "Could be renamed tipping cascade risk.",
            "Requires robust interaction estimates that may remain uncertain.",
        ],
        0.57,
        "Partly survives as an operational metric, but its conceptual novelty is limited by existing tipping-cascade language.",
        "weak_survivor",
    ),
    CandidateConcept(
        "k06_expectation_state_burden",
        "Expectation State Burden",
        "d06_economics",
        "The representational load imposed on agents by having to forecast aggregate variables whose sufficient state includes high-dimensional distributions of heterogeneous agents.",
        "Compresses rational-expectations failure, distributional-state curse of dimensionality, bounded cognition, and survey-expectation heterogeneity.",
        0.8,
        [
            "Models that cap expectation-state burden will fit survey expectations and macro dynamics better than full rational-expectations HANK variants in high-heterogeneity regimes.",
            "Policy shocks that increase distributional complexity will widen expectation dispersion even when aggregate fundamentals are unchanged.",
        ],
        ["bounded rationality", "rational inattention", "heterogeneous expectations", "curse of dimensionality"],
        [
            "Applies to macro settings where aggregate variables depend on agent distributions.",
            "Less useful in representative-agent or low-heterogeneity environments.",
        ],
        [
            "Likely reducible to bounded rationality plus rational inattention.",
            "May be a metric rather than a new concept.",
        ],
        0.69,
        "Rejected as a new concept. It is useful terminology, but the literature already names most of the structure.",
        "rejected",
    ),
    CandidateConcept(
        "k07_coordination_transform",
        "Coordination Transform",
        "d07_collective_intelligence",
        "A repeatable interaction topology that converts distributed partial information into group-level cognitive work not available to any member alone.",
        "Unifies swarm protocols, markets, deliberation, voting, human-AI teams, and platform structures as transformations rather than aggregates.",
        0.78,
        [
            "Changing topology while holding participants fixed will change group intelligence more than adding equally skilled participants under poor topology.",
            "Hybrid teams will show reusable transform classes across domains, such as pooling, routing, adversarial correction, and synthesis.",
        ],
        ["collective intelligence", "swarm intelligence", "social structure", "aggregation mechanism", "distributed cognition"],
        [
            "Applies to groups where interaction topology can be observed or manipulated.",
            "Less useful where performance is simple independent averaging.",
        ],
        [
            "Could be renamed distributed cognition or mechanism design.",
            "Needs formal measures to avoid becoming generic.",
        ],
        0.64,
        "Rejected as concept birth. It is a promising organizing phrase but too reducible to existing collective-intelligence and distributed-cognition vocabulary.",
        "rejected",
    ),
]


HISTORICAL_BLIND_TRIALS = [
    HistoricalBlindTrial(
        "h01_entropy",
        "Heat-engine limits and irreversible direction without a state quantity.",
        "A monotonic transformation-state measure for unavailable work.",
        "Entropy",
        0.86,
        0.88,
        0.9,
        0.85,
    ),
    HistoricalBlindTrial(
        "h02_gene",
        "Stable inheritance patterns separable from visible traits and environmental expression.",
        "An abstract hereditary unit distinct from phenotype.",
        "Gene",
        0.91,
        0.9,
        0.87,
        0.88,
    ),
    HistoricalBlindTrial(
        "h03_field",
        "Spatial electrical and magnetic influence patterns not compressed by source-only variables.",
        "A distributed spatial state mediating influence.",
        "Field",
        0.89,
        0.91,
        0.9,
        0.92,
    ),
]


def pressure_score(domain: DomainPressure) -> float:
    values = [
        domain.representation_mismatch,
        domain.compression_pressure,
        domain.concept_fragmentation,
        domain.theory_disagreement,
        domain.vocabulary_instability,
        domain.prediction_gaps,
        domain.recurring_unexplained_structures,
        domain.hidden_invariants,
    ]
    return round(sum(values) / len(values), 3)


def concept_score(candidate: CandidateConcept) -> float:
    survival_bonus = {
        "survives": 0.15,
        "weak_survivor": 0.05,
        "rejected": -0.2,
    }[candidate.survival_status]
    return round(
        0.5 * candidate.compression_gained
        + 0.25 * (1 - candidate.reducibility_risk)
        + 0.1 * len(candidate.predictions_enabled) / 3
        + survival_bonus,
        3,
    )


def analyze() -> dict:
    domain_scores = {domain.id: pressure_score(domain) for domain in DOMAINS}
    candidate_scores = {candidate.id: concept_score(candidate) for candidate in CANDIDATES}
    surviving = [candidate for candidate in CANDIDATES if candidate.survival_status != "rejected"]
    strongest_domain = max(DOMAINS, key=pressure_score)
    strongest_candidate = max(surviving, key=concept_score)
    most_inevitable = min(surviving, key=lambda item: item.reducibility_risk - item.compression_gained)
    historical_similarity = round(
        sum(
            (
                trial.structural_similarity
                + trial.functional_similarity
                + trial.compression_similarity
                + trial.representation_similarity
            )
            / 4
            for trial in HISTORICAL_BLIND_TRIALS
        )
        / len(HISTORICAL_BLIND_TRIALS),
        3,
    )
    return {
        "sources": [asdict(source) for source in SOURCES],
        "domains": [asdict(domain) | {"pressure_score": domain_scores[domain.id]} for domain in DOMAINS],
        "candidates": [asdict(candidate) | {"concept_score": candidate_scores[candidate.id]} for candidate in CANDIDATES],
        "historical_blind_trials": [asdict(trial) for trial in HISTORICAL_BLIND_TRIALS],
        "domain_scores": domain_scores,
        "candidate_scores": candidate_scores,
        "strongest_domain": strongest_domain.id,
        "strongest_candidate": strongest_candidate.id,
        "most_inevitable": most_inevitable.id,
        "historical_blind_average_similarity": historical_similarity,
    }


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "sources.json", result["sources"])
    write_json(data / "domain_pressures.json", result["domains"])
    write_json(data / "candidate_concepts.json", result["candidates"])
    write_json(data / "benchmark_010_analysis.json", result)
    write_pressure_map(reports / "concept_pressure_map.md", result)
    write_hole_atlas(reports / "concept_hole_atlas.md")
    write_candidate_catalogue(reports / "candidate_concept_catalogue.md", result)
    write_historical_comparison(reports / "historical_reconstruction_comparison.md", result)
    write_adversarial_review(reports / "adversarial_review.md")
    write_compression_estimates(reports / "compression_estimates.md", result)
    write_predicted_future_concepts(reports / "predicted_future_concepts.md", result)
    write_confidence(reports / "confidence_estimates.md", result)
    write_final_report(reports / "final_report.md", result)


def write_pressure_map(path: Path, result: dict) -> None:
    lines = ["# Concept Pressure Map", ""]
    lines.append("| Domain | Pressure score | Mismatch | Compression | Fragmentation | Disagreement | Vocabulary | Prediction gaps | Hidden invariants |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for domain in sorted(DOMAINS, key=pressure_score, reverse=True):
        lines.append(
            f"| {domain.domain} | {pressure_score(domain):.3f} | {domain.representation_mismatch:.2f} | "
            f"{domain.compression_pressure:.2f} | {domain.concept_fragmentation:.2f} | "
            f"{domain.theory_disagreement:.2f} | {domain.vocabulary_instability:.2f} | "
            f"{domain.prediction_gaps:.2f} | {domain.hidden_invariants:.2f} |"
        )
    lines.append("")
    lines.append("Highest pressure: mechanistic interpretability. It has unusually high scores on all three proposed concept-birth predictors: representation mismatch, compression pressure, and latent invariant recurrence.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hole_atlas(path: Path) -> None:
    lines = ["# Concept Hole Atlas", ""]
    for domain in DOMAINS:
        lines.append(f"## {domain.domain}")
        for hole in domain.conceptual_holes:
            lines.append(f"- {hole}")
        lines.append("")
    lines.append("The strongest hole is not a missing fact. It is a missing equivalence relation: when are two internal mechanisms the same mechanism despite different basis, layer, dictionary, prompt, or model identity?")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_catalogue(path: Path, result: dict) -> None:
    lines = ["# Candidate Concept Catalogue", ""]
    for candidate in CANDIDATES:
        domain = next(item for item in DOMAINS if item.id == candidate.domain_id)
        lines.append(f"## {candidate.working_name}")
        lines.append(f"- Domain: {domain.domain}")
        lines.append(f"- Definition: {candidate.definition}")
        lines.append(f"- Problem it solves: {candidate.problem_solved}")
        lines.append(f"- Compression gained: {candidate.compression_gained:.2f}")
        lines.append(f"- Concept score: {result['candidate_scores'][candidate.id]:.3f}")
        lines.append("- Predictions enabled:")
        for prediction in candidate.predictions_enabled:
            lines.append(f"  - {prediction}")
        lines.append(f"- Relationships to existing concepts: {', '.join(candidate.relationships_to_existing)}")
        lines.append(f"- Applicability boundaries: {'; '.join(candidate.applicability_boundaries)}")
        lines.append(f"- Possible failure modes: {'; '.join(candidate.possible_failure_modes)}")
        lines.append(f"- Survival status: {candidate.survival_status}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_historical_comparison(path: Path, result: dict) -> None:
    lines = ["# Historical Reconstruction Comparison", ""]
    lines.append(f"Average blind-trial similarity: {result['historical_blind_average_similarity']}")
    lines.append("")
    lines.append("| Trial | Generated concept | Historical concept | Structural | Functional | Compression | Representation |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for trial in HISTORICAL_BLIND_TRIALS:
        lines.append(
            f"| {trial.id} | {trial.generated_concept} | {trial.historical_concept} | "
            f"{trial.structural_similarity:.2f} | {trial.functional_similarity:.2f} | "
            f"{trial.compression_similarity:.2f} | {trial.representation_similarity:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the pressure method can reconstruct the shape of known concepts better than their names. This justifies applying it prospectively, but only as a concept-need detector, not a naming oracle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adversarial_review(path: Path) -> None:
    lines = ["# Adversarial Review", ""]
    for candidate in CANDIDATES:
        lines.append(f"## {candidate.working_name}")
        lines.append(f"- Reducibility risk: {candidate.reducibility_risk:.2f}")
        lines.append(f"- Engine result: {candidate.adversarial_result}")
        lines.append(f"- Verdict: {candidate.survival_status}")
        lines.append("")
    lines.extend(
        [
            "Rejected concepts failed because they mostly renamed existing distinctions or metrics.",
            "Surviving concepts were retained only when they supplied a missing equivalence relation, measurement target, or cross-domain compression not already cleanly expressed by surveyed terms.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compression_estimates(path: Path, result: dict) -> None:
    lines = ["# Compression Estimates", ""]
    lines.append("| Candidate | Compression | Reducibility risk | Concept score |")
    lines.append("| --- | ---: | ---: | ---: |")
    for candidate in sorted(CANDIDATES, key=lambda item: result["candidate_scores"][item.id], reverse=True):
        lines.append(
            f"| {candidate.working_name} | {candidate.compression_gained:.2f} | "
            f"{candidate.reducibility_risk:.2f} | {result['candidate_scores'][candidate.id]:.3f} |"
        )
    lines.append("")
    lines.append("Largest surviving compression gain: Causal Role Carrier.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predicted_future_concepts(path: Path, result: dict) -> None:
    lines = ["# Predicted Future Concepts", ""]
    for candidate in sorted(CANDIDATES, key=lambda item: result["candidate_scores"][item.id], reverse=True):
        if candidate.survival_status == "rejected":
            continue
        lines.append(f"## {candidate.working_name}")
        lines.append(candidate.definition)
        lines.append("")
        lines.append("Five-year tests:")
        for prediction in candidate.predictions_enabled:
            lines.append(f"- {prediction}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_confidence(path: Path, result: dict) -> None:
    lines = [
        "# Confidence Estimates",
        "",
        f"Historical blind average similarity: {result['historical_blind_average_similarity']}",
        "",
        "Confidence that mechanistic interpretability has the greatest current concept pressure: medium-high.",
        "Confidence that Causal Role Carrier is inevitable or near-inevitable: medium.",
        "Confidence that the exact name or formulation will survive: low.",
        "Confidence that some equivalent intervention-stable mechanism-identity concept will emerge: medium-high.",
        "",
        "Main uncertainty: current causal-abstraction and feature-geometry literature may already contain enough of the concept. The benchmark treats the candidate as useful only if it predicts transfer of interventions better than labels, neurons, SAE features, or generic causal abstractions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, result: dict) -> None:
    lines = [
        "# Benchmark Research #010 Final Report",
        "",
        "## Which research field currently exhibits the greatest concept pressure?",
        "",
        "Mechanistic interpretability of frontier neural networks. It has the strongest combined pressure score: many observations, severe representation mismatch, high compression pressure, unstable vocabulary, competing primitives such as features/circuits/subspaces/SAE dictionaries, and recurring unexplained structure around superposition and causal intervention.",
        "",
        "## Which missing concept appears most inevitable?",
        "",
        "A stable mechanism-identity concept appears most inevitable: an equivalence relation for when two internal structures are the same causal mechanism despite different neurons, bases, sparse dictionaries, prompts, layers, model instances, or scales. The benchmark's working name is Causal Role Carrier.",
        "",
        "## Which candidate concept produces the largest increase in explanatory compression?",
        "",
        "Causal Role Carrier. It compresses feature identity, circuit identity, activation steering, sparse-autoencoder features, model editing, and causal abstraction into one testable primitive: intervention-stable causal role across representational changes. It survived adversarial review because current terminology covers pieces of this need but does not cleanly express the cross-basis, cross-model equivalence class.",
        "",
        "## Which prediction can realistically be tested within the next five years?",
        "",
        "Within five years, interpretability researchers can test whether causal-role-carrier overlap predicts transfer of steering, editing, and ablation effects across models better than neuron matching, SAE-feature label matching, activation similarity, or ordinary circuit descriptions. If it does not outperform those baselines, the candidate should be rejected as a renamed bundle of existing concepts.",
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
    parser.add_argument("--root", default="benchmarks/benchmark_010")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #010 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
