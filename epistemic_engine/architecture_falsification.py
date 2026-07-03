from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import (
    adjusted_rand_index,
    cluster_count,
    dbscan,
    kmeans,
    pca,
    standardize,
)


CAPABILITIES = (
    "genesis",
    "admissibility",
    "uncertainty",
    "abstraction",
    "substitutability",
    "transfer",
    "representation_translation",
    "generalization",
    "stable_structures",
    "experimentally_distinguishable_predictions",
)

REPRESENTATION_CLASSES = ("C1", "C2", "C3", "C4", "C5")

PARADIGMS = (
    "graph_system",
    "hypergraph",
    "cellular_automata",
    "symbolic_system",
    "probabilistic_system",
    "differentiable_system",
    "rewriting_system",
    "sat_based",
    "random_interaction",
)

PRIMITIVES = (
    "recurrence",
    "prediction",
    "merge",
    "compression",
    "graph_rewrite",
    "stochastic_memory",
    "attention",
    "constraint_propagation",
    "rewriting_system",
    "cellular_automata",
    "hypergraph_dynamics",
    "probabilistic_grammar",
    "sat_constraints",
    "finite_automata",
    "symbolic_rules",
    "differentiable_objective",
    "belief_revision",
    "counterexample_search",
    "program_synthesis",
    "type_unification",
    "message_passing",
    "latent_state",
    "entropy_regularization",
    "memory_decay",
    "schema_alignment",
    "local_replacement",
    "invariant_detection",
    "causal_probe",
    "operator_composition",
    "noise_filter",
)

VARIABLE_FEATURES = (
    "birth",
    "content",
    "ambiguity",
    "transform",
    "role",
    "memory",
    "prediction",
    "constraint",
    "compression",
    "stochasticity",
)


PRIMITIVE_EFFECTS: dict[str, dict[str, float]] = {
    "recurrence": {"genesis": 0.45, "stable_structures": 0.45, "memory": 0.7},
    "prediction": {"experimentally_distinguishable_predictions": 0.75, "generalization": 0.35, "prediction": 0.9},
    "merge": {"abstraction": 0.7, "genesis": 0.35, "birth": 0.6},
    "compression": {"abstraction": 0.45, "generalization": 0.45, "compression": 0.9},
    "graph_rewrite": {"representation_translation": 0.55, "genesis": 0.3, "transform": 0.8},
    "stochastic_memory": {"uncertainty": 0.65, "transfer": 0.2, "stochasticity": 0.8},
    "attention": {"admissibility": 0.35, "transfer": 0.35, "role": 0.5},
    "constraint_propagation": {"admissibility": 0.7, "stable_structures": 0.45, "constraint": 0.9},
    "rewriting_system": {"representation_translation": 0.55, "substitutability": 0.35, "transform": 0.75},
    "cellular_automata": {"genesis": 0.5, "stable_structures": 0.35, "birth": 0.6},
    "hypergraph_dynamics": {"abstraction": 0.45, "representation_translation": 0.35, "role": 0.45},
    "probabilistic_grammar": {"uncertainty": 0.6, "generalization": 0.45, "content": 0.45},
    "sat_constraints": {"admissibility": 0.8, "experimentally_distinguishable_predictions": 0.35, "constraint": 0.9},
    "finite_automata": {"stable_structures": 0.6, "transfer": 0.25, "role": 0.35},
    "symbolic_rules": {"admissibility": 0.45, "experimentally_distinguishable_predictions": 0.45, "content": 0.75},
    "differentiable_objective": {"generalization": 0.45, "transfer": 0.45, "compression": 0.35},
    "belief_revision": {"uncertainty": 0.55, "admissibility": 0.35, "ambiguity": 0.65},
    "counterexample_search": {"experimentally_distinguishable_predictions": 0.6, "admissibility": 0.45, "ambiguity": 0.45},
    "program_synthesis": {"genesis": 0.4, "representation_translation": 0.45, "birth": 0.45},
    "type_unification": {"substitutability": 0.7, "representation_translation": 0.35, "role": 0.7},
    "message_passing": {"transfer": 0.35, "stable_structures": 0.25, "transform": 0.4},
    "latent_state": {"uncertainty": 0.45, "generalization": 0.35, "ambiguity": 0.6},
    "entropy_regularization": {"uncertainty": 0.35, "compression": 0.3, "stochasticity": 0.5},
    "memory_decay": {"stable_structures": -0.2, "transfer": 0.25, "memory": 0.45},
    "schema_alignment": {"representation_translation": 0.65, "substitutability": 0.45, "transform": 0.55},
    "local_replacement": {"substitutability": 0.65, "stable_structures": 0.25, "role": 0.7},
    "invariant_detection": {"stable_structures": 0.55, "generalization": 0.45, "transform": 0.45},
    "causal_probe": {"experimentally_distinguishable_predictions": 0.55, "transfer": 0.35, "content": 0.35},
    "operator_composition": {"abstraction": 0.45, "representation_translation": 0.35, "birth": 0.35},
    "noise_filter": {"uncertainty": 0.3, "stable_structures": 0.35, "constraint": 0.25},
}

SYNERGIES = (
    ({"merge", "compression"}, {"abstraction": 0.35, "generalization": 0.25}),
    ({"prediction", "counterexample_search"}, {"experimentally_distinguishable_predictions": 0.35, "admissibility": 0.2}),
    ({"probabilistic_grammar", "belief_revision"}, {"uncertainty": 0.35, "ambiguity": 0.35}),
    ({"rewriting_system", "schema_alignment"}, {"representation_translation": 0.35, "transform": 0.35}),
    ({"type_unification", "local_replacement"}, {"substitutability": 0.35, "role": 0.35}),
    ({"recurrence", "invariant_detection"}, {"stable_structures": 0.35, "transfer": 0.2}),
    ({"sat_constraints", "constraint_propagation"}, {"admissibility": 0.35, "constraint": 0.35}),
    ({"hypergraph_dynamics", "operator_composition"}, {"abstraction": 0.3, "representation_translation": 0.2}),
    ({"cellular_automata", "noise_filter"}, {"stable_structures": 0.2, "genesis": 0.2}),
)

CLASS_PROTOTYPES = {
    "C1": {"birth": 1.0, "memory": 0.3, "prediction": 0.1},
    "C2": {"content": 0.8, "constraint": 0.8, "prediction": 0.2},
    "C3": {"ambiguity": 1.0, "stochasticity": 0.5, "memory": 0.2},
    "C4": {"transform": 1.0, "compression": 0.3, "role": 0.2},
    "C5": {"role": 1.0, "content": 0.2, "transform": 0.2},
}

PARADIGM_BIAS = {
    "graph_system": ("graph_rewrite", "message_passing", "invariant_detection"),
    "hypergraph": ("hypergraph_dynamics", "operator_composition", "merge"),
    "cellular_automata": ("cellular_automata", "recurrence", "noise_filter"),
    "symbolic_system": ("symbolic_rules", "type_unification", "program_synthesis"),
    "probabilistic_system": ("probabilistic_grammar", "belief_revision", "stochastic_memory"),
    "differentiable_system": ("differentiable_objective", "prediction", "compression"),
    "rewriting_system": ("rewriting_system", "graph_rewrite", "schema_alignment"),
    "sat_based": ("sat_constraints", "constraint_propagation", "counterexample_search"),
    "random_interaction": ("stochastic_memory", "attention", "memory_decay"),
}


@dataclass(frozen=True)
class Architecture:
    architecture_id: str
    paradigm: str
    primitives: tuple[str, ...]
    variables: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Evaluation:
    architecture_id: str
    paradigm: str
    primitives: tuple[str, ...]
    variables: tuple[tuple[float, ...], ...]
    primitive_count: int
    capability_scores: dict[str, float]
    e_star_score: float
    e_star_complete: bool
    class_mapping: dict[str, list[int]]
    mapping_ambiguity: float
    correspondence_score: float
    counterexample_score: float
    proof_complexity: float
    representation_complexity: float
    translation_complexity: float
    compression: float
    reuse: float
    hierarchy_depth: float
    transfer_quality: float


def generate_architecture(index: int, rng: random.Random, paradigm: str | None = None) -> Architecture:
    paradigm = paradigm or rng.choice(PARADIGMS)
    min_size = 2
    max_size = 8
    size = rng.randint(min_size, max_size)
    biased = list(PARADIGM_BIAS[paradigm])
    pool = list(PRIMITIVES)
    primitives: set[str] = set(rng.sample(biased, rng.randint(1, min(len(biased), size))))
    while len(primitives) < size:
        primitive = rng.choice(pool)
        primitives.add(primitive)
    variables = tuple(_generate_variables(tuple(sorted(primitives)), rng))
    return Architecture(f"a{index:06d}", paradigm, tuple(sorted(primitives)), variables)


def evaluate(architecture: Architecture) -> Evaluation:
    capability = {name: 0.0 for name in CAPABILITIES}
    variable_profile = _mean_vector(architecture.variables)
    for primitive in architecture.primitives:
        for key, value in PRIMITIVE_EFFECTS[primitive].items():
            if key in capability:
                capability[key] += value
    for required, effects in SYNERGIES:
        if required <= set(architecture.primitives):
            for key, value in effects.items():
                if key in capability:
                    capability[key] += value

    capability["genesis"] += 0.25 * variable_profile["birth"]
    capability["admissibility"] += 0.2 * variable_profile["constraint"] + 0.2 * variable_profile["content"]
    capability["uncertainty"] += 0.25 * variable_profile["ambiguity"] + 0.15 * variable_profile["stochasticity"]
    capability["abstraction"] += 0.2 * variable_profile["compression"] + 0.15 * variable_profile["birth"]
    capability["substitutability"] += 0.25 * variable_profile["role"]
    capability["transfer"] += 0.2 * variable_profile["transform"] + 0.15 * variable_profile["memory"]
    capability["representation_translation"] += 0.25 * variable_profile["transform"]
    capability["generalization"] += 0.2 * variable_profile["compression"] + 0.15 * variable_profile["prediction"]
    capability["stable_structures"] += 0.2 * variable_profile["memory"] + 0.15 * variable_profile["constraint"]
    capability["experimentally_distinguishable_predictions"] += 0.25 * variable_profile["prediction"]
    capability = {key: _squash(value) for key, value in capability.items()}

    e_star_score = min(capability.values())
    e_star_complete = e_star_score >= 0.34 and sum(score >= 0.45 for score in capability.values()) >= 7
    class_mapping, ambiguity, correspondence = map_to_classes(architecture.variables)
    primitive_count = len(architecture.primitives)
    proof_complexity = primitive_count * (1.0 + ambiguity) + 0.2 * len(architecture.variables)
    representation_complexity = len(architecture.variables) * _variable_entropy(architecture.variables)
    translation_complexity = 1.0 + 3.0 * (1.0 - correspondence) + 0.2 * primitive_count
    compression = capability["abstraction"] * (1.0 / (1.0 + 0.08 * primitive_count))
    reuse = 0.5 * capability["substitutability"] + 0.5 * capability["transfer"]
    hierarchy_depth = 1.0 + 4.0 * capability["abstraction"] * capability["stable_structures"]
    transfer_quality = capability["transfer"] * capability["generalization"]
    counterexample_score = e_star_score * (1.0 - correspondence) * (1.0 - min(0.8, ambiguity))
    return Evaluation(
        architecture.architecture_id,
        architecture.paradigm,
        architecture.primitives,
        architecture.variables,
        primitive_count,
        capability,
        e_star_score,
        e_star_complete,
        class_mapping,
        ambiguity,
        correspondence,
        counterexample_score,
        proof_complexity,
        representation_complexity,
        translation_complexity,
        compression,
        reuse,
        hierarchy_depth,
        transfer_quality,
    )


def map_to_classes(variables: tuple[tuple[float, ...], ...]) -> tuple[dict[str, list[int]], float, float]:
    mapping: dict[str, list[int]] = {}
    all_best_scores: list[float] = []
    ambiguity_scores: list[float] = []
    for class_name, prototype in CLASS_PROTOTYPES.items():
        scored = []
        for index, variable in enumerate(variables):
            vector = dict(zip(VARIABLE_FEATURES, variable))
            scored.append((_cosine(vector, prototype), index))
        scored.sort(reverse=True)
        best_score = scored[0][0] if scored else 0.0
        close = [index for score, index in scored if score >= max(0.72, best_score - 0.04)]
        mapping[class_name] = close
        all_best_scores.append(best_score)
        ambiguity_scores.append(max(0, len(close) - 1) / max(1, len(variables)))
    correspondence = sum(all_best_scores) / len(all_best_scores) if all_best_scores else 0.0
    ambiguity = sum(ambiguity_scores) / len(ambiguity_scores) if ambiguity_scores else 0.0
    return mapping, ambiguity, correspondence


def random_catalogue(count: int, seed: int) -> list[Evaluation]:
    rng = random.Random(seed)
    return [evaluate(generate_architecture(index, rng)) for index in range(count)]


def evolutionary_counterexample_search(population: int, generations: int, seed: int) -> list[Evaluation]:
    rng = random.Random(seed)
    current = [generate_architecture(index, rng) for index in range(population)]
    archive: list[Evaluation] = []
    next_id = population
    for _ in range(generations):
        evaluated = sorted((evaluate(item) for item in current), key=lambda item: -item.counterexample_score)
        archive.extend(evaluated[: max(5, population // 8)])
        parents = evaluated[: max(4, population // 4)]
        next_generation: list[Architecture] = []
        for _ in range(population):
            parent = rng.choice(parents)
            next_generation.append(_mutate(parent, next_id, rng))
            next_id += 1
        current = next_generation
    return sorted(archive, key=lambda item: -item.counterexample_score)[:50]


def representation_discovery(evaluations: list[Evaluation]) -> dict[str, object]:
    rows = []
    owner = []
    for evaluation in evaluations:
        for index, variable in enumerate(_evaluation_variables(evaluation)):
            rows.append(variable)
            owner.append((evaluation.architecture_id, index))
    sampled = False
    if len(rows) > 2500:
        rng = random.Random(991)
        indices = sorted(rng.sample(range(len(rows)), 2500))
        rows = [rows[index] for index in indices]
        owner = [owner[index] for index in indices]
        sampled = True
    if len(rows) < 8:
        return {"variable_count": len(rows), "cluster_count": 0, "clusters": {}}
    values, _, _ = standardize(np.array(rows, dtype=float))
    labels = dbscan(values, eps=1.7, min_samples=5)
    clusters: dict[str, dict[str, object]] = {}
    for label in sorted(set(labels)):
        if label < 0:
            continue
        members = values[np.array(labels) == label]
        mean = members.mean(axis=0)
        ranked = sorted(zip(VARIABLE_FEATURES, mean), key=lambda item: -abs(item[1]))[:4]
        clusters[str(label)] = {
            "size": labels.count(label),
            "description": [f"{feature}:{score:.2f}z" for feature, score in ranked],
        }
    return {
        "variable_count": len(rows),
        "sampled": sampled,
        "cluster_count": cluster_count(labels),
        "noise_count": labels.count(-1),
        "clusters": clusters,
        "note": "Clusters are discovered from variable features only; C1-C5 labels are not used.",
    }


def universality_statistics(evaluations: list[Evaluation]) -> dict[str, object]:
    successful = [item for item in evaluations if item.e_star_complete]
    if not successful:
        return {"successful_count": 0, "confidence": 0.0, "by_paradigm": {}}
    by_paradigm: dict[str, list[Evaluation]] = {}
    for item in successful:
        by_paradigm.setdefault(item.paradigm, []).append(item)
    by_paradigm_stats = {
        paradigm: {
            "count": len(items),
            "mean_correspondence": _mean([item.correspondence_score for item in items]),
            "mean_ambiguity": _mean([item.mapping_ambiguity for item in items]),
        }
        for paradigm, items in sorted(by_paradigm.items())
    }
    high_correspondence = sum(1 for item in successful if item.correspondence_score >= 0.72)
    confidence = _wilson_lower_bound(high_correspondence, len(successful))
    return {
        "successful_count": len(successful),
        "high_correspondence_successes": high_correspondence,
        "empirical_confidence_lower_bound": confidence,
        "mean_correspondence": _mean([item.correspondence_score for item in successful]),
        "mean_ambiguity": _mean([item.mapping_ambiguity for item in successful]),
        "by_paradigm": by_paradigm_stats,
    }


def pareto_frontier(evaluations: list[Evaluation]) -> list[Evaluation]:
    successful = [item for item in evaluations if item.e_star_complete]
    if len(successful) > 450:
        successful = sorted(
            successful,
            key=lambda item: (
                -item.e_star_score,
                item.primitive_count,
                item.proof_complexity,
                item.translation_complexity,
                -item.transfer_quality,
            ),
        )[:450]
    frontier = []
    for candidate in successful:
        dominated = False
        for other in successful:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (-item.e_star_score, item.primitive_count, item.translation_complexity))


def write_outputs(out_dir: Path, catalogue: list[Evaluation], evolved: list[Evaluation]) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_evaluations = catalogue + evolved
    _write_catalogue(out_dir / "architecture_catalogue.csv", all_evaluations)
    _write_capabilities(out_dir / "capability_matrix.csv", all_evaluations)
    counterexamples = sorted(
        [
        item for item in all_evaluations
        if item.e_star_complete and item.correspondence_score < 0.62
        ],
        key=lambda item: -item.counterexample_score,
    )
    near_counterexamples = sorted(
        [item for item in all_evaluations if item.e_star_score >= 0.45],
        key=lambda item: -item.counterexample_score,
    )[:25]
    frontier = pareto_frontier(all_evaluations)
    discovery = representation_discovery(all_evaluations)
    universality = universality_statistics(all_evaluations)
    _write_mapping_report(out_dir / "mapping_report.csv", all_evaluations)
    (out_dir / "counterexamples.json").write_text(
        json.dumps([_summary_row(item) for item in counterexamples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "pareto_frontier.json").write_text(
        json.dumps([_summary_row(item) for item in frontier], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "universality_statistics.json").write_text(
        json.dumps(universality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "representation_emergence_report.json").write_text(
        json.dumps(discovery, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "architectures": len(all_evaluations),
        "random_architectures": len(catalogue),
        "evolved_architectures": len(evolved),
        "e_star_complete": sum(1 for item in all_evaluations if item.e_star_complete),
        "e_star_completeness_rule": "min capability >= 0.34 and at least 7 capabilities >= 0.45",
        "counterexamples": len(counterexamples),
        "best_counterexample_candidates": [_summary_row(item) for item in near_counterexamples[:10]],
        "genuine_counterexamples": [_summary_row(item) for item in counterexamples[:25]],
        "representation_emergence": discovery,
        "universality_statistics": universality,
        "pareto_frontier": [_summary_row(item) for item in frontier[:50]],
        "negative_results": negative_results(all_evaluations, counterexamples, universality),
    }
    (out_dir / "architecture_falsification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def negative_results(evaluations: list[Evaluation], counterexamples: list[Evaluation], universality: dict[str, object]) -> list[str]:
    results = []
    if not counterexamples:
        results.append("No E*-complete architecture below correspondence threshold was found in this run.")
    successful = [item for item in evaluations if item.e_star_complete]
    if successful and universality.get("mean_correspondence", 0.0) < 0.72:
        results.append("Successful architectures do not strongly converge to C1-C5 under the current mapping metric.")
    if not successful:
        results.append("Search did not find E*-complete architectures; capability thresholds or generator coverage may be too strict.")
    weak_paradigms = [
        paradigm
        for paradigm, stats in universality.get("by_paradigm", {}).items()
        if stats["count"] == 0
    ]
    if weak_paradigms:
        results.append("Some paradigms produced no successful architecture: " + ", ".join(weak_paradigms))
    return results


def _generate_variables(primitives: tuple[str, ...], rng: random.Random) -> list[tuple[float, ...]]:
    count = rng.randint(3, 10)
    primitive_profiles = [PRIMITIVE_EFFECTS[primitive] for primitive in primitives]
    variables = []
    for _ in range(count):
        vector = {feature: rng.random() * 0.2 for feature in VARIABLE_FEATURES}
        for profile in rng.sample(primitive_profiles, rng.randint(1, min(3, len(primitive_profiles)))):
            for key, value in profile.items():
                if key in vector:
                    vector[key] += value * rng.uniform(0.45, 1.0)
        variables.append(tuple(min(1.0, vector[feature]) for feature in VARIABLE_FEATURES))
    return variables


def _mutate(parent: Evaluation, architecture_id: int, rng: random.Random) -> Architecture:
    primitives = set(parent.primitives)
    if rng.random() < 0.45 and len(primitives) < 9:
        primitives.add(rng.choice(PRIMITIVES))
    if rng.random() < 0.35 and len(primitives) > 2:
        primitives.remove(rng.choice(tuple(sorted(primitives))))
    if rng.random() < 0.35:
        primitives.add(rng.choice(PARADIGM_BIAS[parent.paradigm]))
    variables = [list(variable) for variable in _evaluation_variables(parent)]
    if rng.random() < 0.35 and len(variables) < 12:
        variables.append(list(_generate_variables(tuple(primitives), rng)[0]))
    if rng.random() < 0.25 and len(variables) > 3:
        del variables[rng.randrange(len(variables))]
    for variable in variables:
        if rng.random() < 0.6:
            index = rng.randrange(len(VARIABLE_FEATURES))
            variable[index] = min(1.0, max(0.0, variable[index] + rng.uniform(-0.18, 0.18)))
    return Architecture(f"e{architecture_id:06d}", parent.paradigm, tuple(sorted(primitives)), tuple(tuple(v) for v in variables))


def _evaluation_variables(evaluation: Evaluation) -> tuple[tuple[float, ...], ...]:
    return evaluation.variables


def _write_catalogue(path: Path, evaluations: list[Evaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "architecture_id",
            "paradigm",
            "primitives",
            "primitive_count",
            "e_star_score",
            "e_star_complete",
            "correspondence_score",
            "mapping_ambiguity",
            "counterexample_score",
            "proof_complexity",
            "representation_complexity",
            "translation_complexity",
            "compression",
            "reuse",
            "hierarchy_depth",
            "transfer_quality",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in evaluations:
            row = asdict(item)
            row["primitives"] = " ".join(item.primitives)
            writer.writerow({key: row[key] for key in fieldnames})


def _write_capabilities(path: Path, evaluations: list[Evaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["architecture_id", "paradigm", *CAPABILITIES])
        for item in evaluations:
            writer.writerow([item.architecture_id, item.paradigm, *(item.capability_scores[name] for name in CAPABILITIES)])


def _write_mapping_report(path: Path, evaluations: list[Evaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "architecture_id",
                "paradigm",
                "e_star_complete",
                "correspondence_score",
                "mapping_ambiguity",
                *REPRESENTATION_CLASSES,
            ]
        )
        for item in evaluations:
            writer.writerow(
                [
                    item.architecture_id,
                    item.paradigm,
                    item.e_star_complete,
                    item.correspondence_score,
                    item.mapping_ambiguity,
                    *(json.dumps(item.class_mapping[class_name]) for class_name in REPRESENTATION_CLASSES),
                ]
            )


def _summary_row(item: Evaluation) -> dict[str, object]:
    return {
        "architecture_id": item.architecture_id,
        "paradigm": item.paradigm,
        "primitives": item.primitives,
        "e_star_score": item.e_star_score,
        "correspondence_score": item.correspondence_score,
        "mapping_ambiguity": item.mapping_ambiguity,
        "counterexample_score": item.counterexample_score,
        "primitive_count": item.primitive_count,
    }


def _dominates(left: Evaluation, right: Evaluation) -> bool:
    left_values = (
        -left.primitive_count,
        -left.proof_complexity,
        -left.representation_complexity,
        -left.translation_complexity,
        left.compression,
        left.reuse,
        left.hierarchy_depth,
        left.transfer_quality,
        left.e_star_score,
    )
    right_values = (
        -right.primitive_count,
        -right.proof_complexity,
        -right.representation_complexity,
        -right.translation_complexity,
        right.compression,
        right.reuse,
        right.hierarchy_depth,
        right.transfer_quality,
        right.e_star_score,
    )
    return all(l >= r for l, r in zip(left_values, right_values)) and any(l > r for l, r in zip(left_values, right_values))


def _mean_vector(variables: tuple[tuple[float, ...], ...]) -> dict[str, float]:
    if not variables:
        return {feature: 0.0 for feature in VARIABLE_FEATURES}
    return {
        feature: sum(variable[index] for variable in variables) / len(variables)
        for index, feature in enumerate(VARIABLE_FEATURES)
    }


def _variable_entropy(variables: tuple[tuple[float, ...], ...]) -> float:
    if not variables:
        return 0.0
    arr = np.array(variables, dtype=float)
    return float(np.mean(np.std(arr, axis=0))) + 1.0


def _cosine(vector: dict[str, float], prototype: dict[str, float]) -> float:
    keys = VARIABLE_FEATURES
    left = np.array([vector.get(key, 0.0) for key in keys])
    right = np.array([prototype.get(key, 0.0) for key in keys])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _squash(value: float) -> float:
    return max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, value))))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)
