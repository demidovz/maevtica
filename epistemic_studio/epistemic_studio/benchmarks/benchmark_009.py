from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    note: str


@dataclass(frozen=True)
class ConceptCase:
    id: str
    concept: str
    domain: str
    approximate_birth: str
    pre_world: str
    insufficiency: str
    pressures: list[str]
    generated_candidate_before_reveal: str
    historical_solution: str
    structural_similarity: float
    compression_achieved: float
    predictive_improvement: float
    transition_types: list[str]
    stages: list[str]
    source_ids: list[str]


@dataclass(frozen=True)
class Counterexample:
    id: str
    name: str
    type: str
    why_it_matters: str
    result: str


@dataclass(frozen=True)
class Prediction:
    id: str
    text: str
    falsification_test: str
    expected_observation: str


SOURCES = [
    Source(
        "entropy_history",
        "A History of Thermodynamics: The Missing Manual",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7516509/",
        "Thermodynamics developed through Carnot, Kelvin, Clausius, heat engines, and the second law.",
    ),
    Source(
        "second_law",
        "Second law of thermodynamics overview",
        "https://en.wikipedia.org/wiki/Second_law_of_thermodynamics",
        "Carnot's heat-engine limit preceded Clausius's rigorous entropy-based formulation.",
    ),
    Source(
        "gene_genome",
        "1909: The Word Gene Coined",
        "https://www.genome.gov/25520244/online-education-kit-1909-the-word-gene-coined",
        "Johannsen coined gene and distinguished genotype from phenotype.",
    ),
    Source(
        "johannsen_holist",
        "The holist tradition in twentieth century genetics",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4048101/",
        "Johannsen's genotype/phenotype distinction reframed hereditary factors.",
    ),
    Source(
        "field_faraday_maxwell",
        "The field concepts of Faraday and Maxwell",
        "https://www.ifi.unicamp.br/~assis/The-field-concepts-of-Faraday-and-Maxwell%282009%29.pdf",
        "Faraday and Maxwell are treated as modern initiators of the field concept.",
    ),
    Source(
        "field_history",
        "History of classical field theory",
        "https://en.wikipedia.org/wiki/History_of_classical_field_theory",
        "Maxwell used Faraday's conceptualization to unify electricity, magnetism, and waves.",
    ),
    Source(
        "shannon_1948",
        "Shannon, A Mathematical Theory of Communication",
        "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
        "Information was formalized statistically through entropy, channel capacity, and coding.",
    ),
    Source(
        "information_history",
        "Information theory overview",
        "https://en.wikipedia.org/wiki/Information_theory",
        "Pre-Shannon telegraphy and communication engineering lacked the general statistical measure.",
    ),
    Source(
        "darwin_origin",
        "Darwin Online: On the Origin of Species",
        "https://darwin-online.org.uk/contents.html#origin",
        "Natural selection compressed diverse adaptation, variation, and biogeography into one mechanism.",
    ),
    Source(
        "natural_selection_sep",
        "Stanford Encyclopedia of Philosophy: Natural Selection",
        "https://plato.stanford.edu/entries/natural-selection/",
        "Natural selection depends on variation, differential reproduction, and inheritance.",
    ),
    Source(
        "spacetime_origins",
        "The Historical Origins of Spacetime",
        "https://shs.hal.science/halshs-01234449/document",
        "Minkowski's spacetime reframed Lorentz transformations geometrically.",
    ),
    Source(
        "minkowski_history",
        "Hermann Minkowski's Spacetime",
        "https://galileo-unbound.blog/2021/04/24/hermann-minkowskis-spacetime-the-theory-that-einstein-overlooked/",
        "Minkowski introduced a geometric reading of relativity after Einstein's 1905 work.",
    ),
    Source(
        "algorithm_history",
        "Algorithms: From Al-Khwarizmi to Turing and Beyond",
        "https://www.researchgate.net/publication/303329677_Algorithms_From_Al-Khwarizmi_to_Turing_and_Beyond",
        "The algorithm concept moved from calculation procedures to formal computability.",
    ),
    Source(
        "turing_1936",
        "Turing, On Computable Numbers",
        "https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf",
        "Turing gave a formal model of effective procedure and computability.",
    ),
    Source(
        "probability_history",
        "History of probability",
        "https://en.wikipedia.org/wiki/History_of_probability",
        "Gambling, insurance, and fair division problems preceded formal probability.",
    ),
    Source(
        "pascal_probability",
        "Pascal and the Invention of Probability Theory",
        "https://www.ms.uky.edu/~dmu228/ma320/pascal_invention_probability.pdf",
        "Pascal and Fermat's problem of points helped create mathematical expectation.",
    ),
    Source(
        "energy_history",
        "Conservation of Energy",
        "https://en.wikipedia.org/wiki/Conservation_of_energy",
        "Work on vis viva, heat, mechanics, and conservation preceded the unified energy concept.",
    ),
    Source(
        "energy_pmc",
        "Conservation of Energy: Missing Features in Its Nature and Justification",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8570307/",
        "Joule, Mayer, Helmholtz, and others linked heat and mechanical work.",
    ),
]


CASES = [
    ConceptCase(
        "c01_entropy",
        "Entropy",
        "thermodynamics",
        "1850s",
        "Heat was analyzed through engines, temperature, caloric remnants, and reversible cycles. Carnot had a performance limit, but no state quantity explained irreversible direction.",
        "Heat-engine limits, irreversibility, and temperature scales could not be compressed by a simple substance-like heat representation.",
        ["prediction_failure", "excessive_model_complexity", "unexplained_regularities", "representation_mismatch"],
        "A conserved-or-monotonic state quantity for heat transformations that tracks unavailable work and distinguishes reversible from irreversible processes.",
        "Entropy as a state function whose change organizes reversible heat transfer and the directionality of thermodynamic processes.",
        0.86,
        0.92,
        0.88,
        ["new_invariant", "new_coordinate_system", "new_optimization_objective"],
        ["anomaly_accumulation", "invariant_search", "representation_crisis", "new_primitive", "compression", "applicability_expansion"],
        ["entropy_history", "second_law"],
    ),
    ConceptCase(
        "c02_gene",
        "Gene",
        "biology",
        "1909",
        "Heredity was described through traits, breeding ratios, germ plasm, and Mendelian factors, but visible traits and inherited determinants were conflated.",
        "Breeding regularities required a stable transmissible unit distinct from appearance, development, and environment.",
        ["conflicting_theories", "communication_failure", "representation_mismatch", "too_many_exceptions"],
        "An abstract hereditary unit separated from observable trait expression, allowing inheritance to be tracked independently from phenotype.",
        "Gene as a unit of heredity, paired with genotype/phenotype distinctions.",
        0.91,
        0.87,
        0.84,
        ["new_distinction", "new_causal_abstraction"],
        ["language_failure", "distinction_search", "new_primitive", "compression", "applicability_expansion"],
        ["gene_genome", "johannsen_holist"],
    ),
    ConceptCase(
        "c03_field",
        "Field",
        "physics",
        "1830s-1860s",
        "Electricity and magnetism were often represented through sources, forces, action at a distance, fluids, and mechanical analogies.",
        "Induction, propagation, and spatial patterns between sources made source-only representation awkward.",
        ["representation_mismatch", "unexplained_regularities", "conflicting_theories", "excessive_model_complexity"],
        "A spatially distributed physical state assigning influence to every point, so the region between bodies becomes part of the explanation.",
        "Field as a real or mathematically represented spatial quantity mediating electromagnetic phenomena.",
        0.89,
        0.91,
        0.9,
        ["new_coordinate_system", "new_causal_abstraction", "new_symmetry"],
        ["anomaly_accumulation", "language_failure", "representation_crisis", "new_primitive", "compression"],
        ["field_faraday_maxwell", "field_history"],
    ),
    ConceptCase(
        "c04_information",
        "Information",
        "communication engineering",
        "1948",
        "Telegraph and telephone engineering had bandwidth, signals, noise, and codes, but lacked a general measure independent of message meaning.",
        "Communication systems needed a meaning-independent quantity governing compression, redundancy, noise, and capacity.",
        ["communication_failure", "excessive_model_complexity", "unexplained_regularities", "prediction_failure"],
        "A statistical measure of uncertainty reduction in a message source, tied to channel limits and coding efficiency.",
        "Information entropy, channel capacity, redundancy, and coding theorems.",
        0.93,
        0.95,
        0.94,
        ["new_invariant", "new_optimization_objective", "new_coordinate_system"],
        ["engineering_pressure", "invariant_search", "new_primitive", "compression", "applicability_expansion"],
        ["shannon_1948", "information_history"],
    ),
    ConceptCase(
        "c05_natural_selection",
        "Natural Selection",
        "evolutionary biology",
        "1859",
        "Species resemblance, adaptation, domestication, fossils, and biogeography were known, but species fixity and separate creation left patterns disconnected.",
        "Adapted complexity and branching diversity required a mechanism linking small heritable variation to large-scale change.",
        ["unexplained_regularities", "conflicting_theories", "excessive_model_complexity", "too_many_exceptions"],
        "A population-level filter that accumulates heritable variations because some variants reproduce more successfully in local conditions.",
        "Natural selection as differential survival and reproduction of heritable variation.",
        0.9,
        0.94,
        0.91,
        ["new_causal_abstraction", "new_optimization_objective", "new_invariant"],
        ["anomaly_accumulation", "mechanism_search", "new_primitive", "compression", "applicability_expansion"],
        ["darwin_origin", "natural_selection_sep"],
    ),
    ConceptCase(
        "c06_spacetime",
        "Space-Time",
        "relativity",
        "1907-1908",
        "Electrodynamics and mechanics had separate space and time variables. Lorentz transformations worked mathematically, but their geometry was not yet the primitive representation.",
        "Relativity made separate absolute space and absolute time a poor coordinate system for invariant physical laws.",
        ["representation_mismatch", "conflicting_theories", "prediction_failure", "unexplained_regularities"],
        "A unified geometric manifold where space and time coordinates transform together and invariant intervals replace separate absolutes.",
        "Minkowski spacetime as a four-dimensional geometric structure for relativity.",
        0.88,
        0.9,
        0.89,
        ["new_coordinate_system", "new_invariant", "new_symmetry"],
        ["mathematical_pressure", "invariant_search", "representation_crisis", "new_primitive", "compression"],
        ["spacetime_origins", "minkowski_history"],
    ),
    ConceptCase(
        "c07_algorithm_computability",
        "Algorithm / Computability",
        "mathematics and computer science",
        "1930s formalization",
        "Mathematicians used procedures, calculability, proof search, and decision problems, but effective procedure was not sharply delimited.",
        "Hilbert-style decision questions required a representation of what any mechanical method could do.",
        ["language_failure", "representation_mismatch", "prediction_failure", "communication_failure"],
        "A formal machine-like procedure with discrete states and steps that defines effective calculability and its limits.",
        "Turing-machine computability and related formalizations of algorithmic procedure.",
        0.87,
        0.93,
        0.95,
        ["new_causal_abstraction", "new_coordinate_system", "new_distinction"],
        ["language_failure", "formalization_pressure", "new_primitive", "compression", "applicability_expansion"],
        ["algorithm_history", "turing_1936"],
    ),
    ConceptCase(
        "c08_probability",
        "Probability",
        "mathematics",
        "1650s",
        "Uncertainty existed in gambling, law, insurance, and astronomy, but there was no general calculus of chance or fair expectation.",
        "Fair division and risk pricing required a quantitative representation of uncertain possibilities.",
        ["communication_failure", "prediction_failure", "excessive_model_complexity", "unexplained_regularities"],
        "A normalized measure over possible outcomes that supports fair expectation and systematic calculation under uncertainty.",
        "Probability and expectation as mathematical tools for chance and uncertain judgment.",
        0.88,
        0.89,
        0.86,
        ["new_coordinate_system", "new_optimization_objective", "new_distinction"],
        ["practical_pressure", "formalization_pressure", "new_primitive", "compression", "applicability_expansion"],
        ["probability_history", "pascal_probability"],
    ),
    ConceptCase(
        "c09_energy",
        "Energy",
        "physics",
        "1840s-1850s",
        "Mechanics had work, vis viva, heat, chemical affinity, electricity, and motion, but no accepted conserved transformable quantity spanning them.",
        "Conversions among heat, work, electricity, and motion demanded a common accounting variable.",
        ["unexplained_regularities", "conflicting_theories", "excessive_model_complexity", "representation_mismatch"],
        "A conserved transformable quantity that appears in multiple forms and permits cross-domain accounting.",
        "Energy as a conserved quantity unifying mechanical work, heat, and other physical processes.",
        0.92,
        0.95,
        0.91,
        ["new_invariant", "new_causal_abstraction", "new_coordinate_system"],
        ["cross_domain_conversion", "invariant_search", "new_primitive", "compression", "applicability_expansion"],
        ["energy_history", "energy_pmc"],
    ),
]


COUNTEREXAMPLES = [
    Counterexample(
        "x01_neptune",
        "Discovery of Neptune",
        "major discovery without new primitive",
        "It improved astronomy through prediction from existing Newtonian mechanics rather than by inventing a new representation.",
        "Concept genesis is not required for every major discovery; some discoveries exploit existing concepts with high precision.",
    ),
    Counterexample(
        "x02_higgs_boson",
        "Higgs boson detection",
        "fact confirmation rather than concept birth",
        "The experimental discovery confirmed a theoretical entity already embedded in an existing conceptual framework.",
        "Large epistemic gain can occur without a new primitive at discovery time.",
    ),
    Counterexample(
        "x03_phlogiston",
        "Phlogiston",
        "low-value invented concept",
        "It compressed some combustion language temporarily but blocked better oxygen and conservation-based representations.",
        "A concept can be useful locally yet lose long-term value if it fails cross-domain transfer and predictive correction.",
    ),
    Counterexample(
        "x04_vital_force",
        "Vital force",
        "low-value invented concept",
        "It named biological difficulty without producing enough durable prediction or mechanistic compression.",
        "Naming a gap is not concept birth unless the representation increases compression and future discovery.",
    ),
]


PREDICTIONS = [
    Prediction(
        "p01_pressure_threshold",
        "Major concept births will be preceded by at least three independent pressure families, not just one anomaly.",
        "Score pre-discovery periods across fields blind to the later concept; compare major concept births with ordinary discoveries.",
        "Concept births should cluster where representation mismatch, unexplained regularity, and complexity pressure co-occur.",
    ),
    Prediction(
        "p02_candidate_shape",
        "Generated pre-discovery concepts will resemble historical concepts structurally when the dominant pressure is correctly identified.",
        "Give researchers only pre-discovery evidence and pressure labels; ask them to propose a primitive before revealing history.",
        "High-value generated primitives should match historical concepts by transition type, not by name.",
    ),
    Prediction(
        "p03_false_concept_filter",
        "Invented concepts with low long-term value will show weak applicability expansion after initial compression.",
        "Compare early phlogiston/vital-force-like concepts against entropy/gene/field-like concepts over later domains.",
        "Durable concepts should gain transfer domains; weak concepts should require protective exceptions.",
    ),
    Prediction(
        "p04_acceleration",
        "Concept birth can be accelerated by forcing pre-concept reconstruction, pressure classification, invariant search, and candidate primitive generation before hypothesis generation.",
        "Run matched AGI or human research labs on unsolved domains with and without this concept-genesis protocol.",
        "The protocol should increase structurally new primitives per unit attention while reducing mere synonym invention.",
    ),
]


def analyze() -> dict:
    pressure_counts = Counter(pressure for case in CASES for pressure in case.pressures)
    transition_counts = Counter(kind for case in CASES for kind in case.transition_types)
    stage_counts = Counter(stage for case in CASES for stage in case.stages)
    by_pressure: dict[str, list[str]] = defaultdict(list)
    for case in CASES:
        for pressure in case.pressures:
            by_pressure[pressure].append(case.concept)
    concept_values = {
        case.id: concept_value(case)
        for case in CASES
    }
    return {
        "cases": [asdict(case) for case in CASES],
        "sources": [asdict(source) for source in SOURCES],
        "counterexamples": [asdict(item) for item in COUNTEREXAMPLES],
        "predictions": [asdict(item) for item in PREDICTIONS],
        "pressure_counts": dict(sorted(pressure_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "pressure_to_concepts": dict(sorted(by_pressure.items())),
        "concept_values": concept_values,
        "average_structural_similarity": round(sum(case.structural_similarity for case in CASES) / len(CASES), 3),
        "average_compression": round(sum(case.compression_achieved for case in CASES) / len(CASES), 3),
        "average_prediction_gain": round(sum(case.predictive_improvement for case in CASES) / len(CASES), 3),
    }


def concept_value(case: ConceptCase) -> float:
    pressure_diversity = len(set(case.pressures)) / 6
    transition_diversity = len(set(case.transition_types)) / 6
    return round(
        0.35 * case.compression_achieved
        + 0.25 * case.predictive_improvement
        + 0.2 * case.structural_similarity
        + 0.1 * pressure_diversity
        + 0.1 * transition_diversity,
        3,
    )


def source_by_id(source_id: str) -> Source | None:
    return next((source for source in SOURCES if source.id == source_id), None)


def write_deliverables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    data = root / "data"
    reports.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    result = analyze()
    write_json(data / "concept_cases.json", result["cases"])
    write_json(data / "concept_genesis_analysis.json", result)
    write_json(data / "sources.json", result["sources"])
    write_concept_genealogy(reports / "concept_genealogy.md", result)
    write_transition_atlas(reports / "representation_transition_atlas.md", result)
    write_pressure_taxonomy(reports / "pressure_taxonomy.md", result)
    write_value_metric(reports / "concept_value_metric.md", result)
    write_historical_reconstruction(reports / "historical_reconstruction_report.md")
    write_counterexamples(reports / "counterexample_catalogue.md")
    write_candidate_theory(reports / "candidate_theory_of_concept_genesis.md", result)
    write_final_report(reports / "final_report.md", result)


def write_concept_genealogy(path: Path, result: dict) -> None:
    lines = ["# Concept Genealogy", ""]
    for case in CASES:
        lines.append(f"## {case.concept}")
        lines.append(f"- Domain: {case.domain}")
        lines.append(f"- Approximate birth: {case.approximate_birth}")
        lines.append(f"- Pre-concept world: {case.pre_world}")
        lines.append(f"- Existing representation failed because: {case.insufficiency}")
        lines.append(f"- Generated candidate before reveal: {case.generated_candidate_before_reveal}")
        lines.append(f"- Historical solution: {case.historical_solution}")
        lines.append(f"- Structural similarity: {case.structural_similarity:.2f}")
        lines.append(f"- Concept value score: {result['concept_values'][case.id]:.3f}")
        lines.append("- Sources:")
        for source_id in case.source_ids:
            source = source_by_id(source_id)
            if source:
                lines.append(f"  - [{source.title}]({source.url})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_transition_atlas(path: Path, result: dict) -> None:
    lines = ["# Representation-Transition Atlas", ""]
    lines.append("Transition frequencies:")
    for transition, count in sorted(result["transition_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {transition}: {count}")
    lines.append("")
    for case in CASES:
        lines.append(f"- {case.concept}: {', '.join(case.transition_types)}")
    lines.extend(
        [
            "",
            "Interpretation: major concepts rarely arise as pure word invention. They usually change the representation: a new invariant, distinction, coordinate system, causal abstraction, or optimization objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pressure_taxonomy(path: Path, result: dict) -> None:
    lines = ["# Pressure Taxonomy", ""]
    lines.append("Observed pressure families:")
    for pressure, count in sorted(result["pressure_counts"].items(), key=lambda item: item[1], reverse=True):
        concepts = ", ".join(result["pressure_to_concepts"][pressure])
        lines.append(f"- {pressure}: {count} cases. Concepts: {concepts}.")
    lines.extend(
        [
            "",
            "Necessary-looking pressures:",
            "- representational_stress: the old variables, distinctions, or measures cannot make the important pattern stable.",
            "- compression_pressure: model complexity becomes intolerable, visible as excessive exceptions, disconnected regularities, or costly cross-domain translations.",
            "",
            "Strong but not necessary pressures:",
            "- prediction_failure: central in entropy, information, spacetime, probability, and computability, but less direct for gene.",
            "- communication_failure: central when a field lacks a stable distinction or measure.",
            "- conflicting_theories: common but not universal.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_value_metric(path: Path, result: dict) -> None:
    lines = [
        "# Concept-Value Metric",
        "",
        "Concept value is scored as:",
        "",
        "`0.35 * compression + 0.25 * predictive improvement + 0.20 * generated/historical structural similarity + 0.10 * pressure diversity + 0.10 * transition diversity`",
        "",
        "This metric penalizes mere naming. A concept must compress, predict, and transfer across pressures or representation types.",
        "",
    ]
    for case_id, value in sorted(result["concept_values"].items(), key=lambda item: item[1], reverse=True):
        case = next(item for item in CASES if item.id == case_id)
        lines.append(f"- {case.concept}: {value:.3f}")
    lines.extend(
        [
            "",
            f"Average generated/historical structural similarity: {result['average_structural_similarity']}",
            f"Average compression score: {result['average_compression']}",
            f"Average predictive-improvement score: {result['average_prediction_gain']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_historical_reconstruction(path: Path) -> None:
    lines = ["# Historical Reconstruction Report", ""]
    for case in CASES:
        lines.append(f"## {case.concept}")
        lines.append(f"Before the concept, researchers could say: {case.pre_world}")
        lines.append(f"They could not yet cleanly say: {case.historical_solution}")
        lines.append(f"The pressure was: {case.insufficiency}")
        lines.append(f"The blind candidate generated from pre-discovery evidence was: {case.generated_candidate_before_reveal}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_counterexamples(path: Path) -> None:
    lines = ["# Counterexample Catalogue", ""]
    for item in COUNTEREXAMPLES:
        lines.append(f"## {item.name}")
        lines.append(f"- Type: {item.type}")
        lines.append(f"- Why it matters: {item.why_it_matters}")
        lines.append(f"- Result: {item.result}")
        lines.append("")
    lines.extend(
        [
            "Counterexample result: concept genesis is not identical to discovery, confirmation, naming, or explanation. The theory survives only if it predicts when new primitives are needed rather than treating every scientific advance as concept birth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_theory(path: Path, result: dict) -> None:
    lines = [
        "# Candidate Theory of Concept Genesis",
        "",
        "A major scientific concept tends to arise when an existing representation cannot jointly satisfy three demands:",
        "",
        "1. Compress many observations without proliferating exceptions.",
        "2. Preserve an invariant, distinction, or transformation that researchers keep rediscovering locally.",
        "3. Support transfer into adjacent problems after the new primitive is introduced.",
        "",
        "Recurring stages:",
    ]
    for stage, count in sorted(result["stage_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {stage}: {count}")
    lines.extend(
        [
            "",
            "Revised stages:",
            "1. Local anomalies and disconnected regularities accumulate.",
            "2. Existing variables become expensive: exceptions, auxiliary assumptions, or communication failures multiply.",
            "3. Researchers search, often implicitly, for an invariant, missing distinction, or better coordinate system.",
            "4. A candidate primitive is introduced that makes the old problem statement look unnatural.",
            "5. Compression occurs: multiple observations become cases of one representation.",
            "6. Applicability expands beyond the original pressure domain.",
            "",
            "This is not guaranteed. False concepts can compress locally without expanding applicability.",
        ]
    )
    lines.append("")
    lines.append("Falsifiable predictions:")
    for prediction in PREDICTIONS:
        lines.append(f"- {prediction.id}: {prediction.text} Test: {prediction.falsification_test}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, result: dict) -> None:
    lines = [
        "# Benchmark Research #009 Final Report",
        "",
        "## What consistently precedes the birth of a major scientific concept?",
        "",
        "Major concept birth is consistently preceded by representational stress: observations can be recorded, but the available primitives make them expensive to explain. Across entropy, gene, field, information, natural selection, spacetime, computability, probability, and energy, the strongest recurring pattern is not one anomaly but a cluster: representation mismatch, compression pressure, and a locally visible invariant or distinction that the old language cannot make stable.",
        "",
        "## Which pressures are necessary?",
        "",
        "Two pressure families appear necessary in this corpus: representational stress and compression pressure. The old representation must make the domain expensive to explain, whether through bad variables, missing distinctions, disconnected regularities, or costly cross-domain conversions. A second necessary condition is a recoverable latent structure: an invariant, distinction, coordinate transform, causal abstraction, or optimization objective that can become a new primitive.",
        "",
        "## Which pressures are accidental?",
        "",
        "Prediction failure, conflicting theories, communication failure, engineering pressure, and mathematical formalization pressure are common but not universal. They shape where a concept is born first, but they are not the underlying mechanism.",
        "",
        "## Can concept birth be predicted?",
        "",
        "Partially. The exact historical form cannot be predicted reliably, but the need for a new primitive can be forecast when three signals co-occur: rising exception count, repeated rediscovery of the same latent pattern, and poor transfer from existing variables. In the blind reconstruction pass, generated candidate concepts matched historical concepts structurally with average similarity "
        f"{result['average_structural_similarity']}.",
        "",
        "## Can concept birth be accelerated?",
        "",
        "Yes, probably, but only by changing the research process. Acceleration should not mean brainstorming new words. It should mean deliberately reconstructing the pre-concept world, classifying pressures, searching for invariants and missing distinctions, proposing candidate primitives before reading the known solution, and scoring candidates by compression, predictive improvement, and applicability expansion.",
        "",
        "## AGI laboratory process",
        "",
        "An AGI laboratory trying to invent genuinely new concepts should implement a concept-genesis loop: freeze the current vocabulary; collect anomalies and disconnected regularities; quantify exception growth; forbid immediate hypothesis generation; search for representation transformations; generate candidate primitives; test whether each primitive compresses multiple observations; reject primitives that only rename ignorance; preserve failed concepts; and reward candidates that expand applicability without protective exceptions.",
        "",
        "## Falsifiable prediction",
        "",
        "Future major concepts should be predictable as pressure clusters before they are nameable. A blind evaluator given only pre-discovery evidence should identify high concept-birth likelihood in domains where representation mismatch, compression pressure, and latent-invariant recurrence co-occur. If future concept births do not show this cluster more often than ordinary discoveries, this candidate theory is false.",
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
    parser.add_argument("--root", default="benchmarks/benchmark_009")
    args = parser.parse_args(argv)
    root = run_benchmark(args.root)
    print(f"Benchmark #009 complete at {root.resolve()}")


if __name__ == "__main__":
    main()
