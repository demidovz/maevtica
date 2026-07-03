from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import adjusted_rand_index, cluster_count, dbscan, kmeans, standardize
from epistemic_engine.architecture_falsification import (
    Architecture,
    Evaluation,
    PARADIGMS,
    PRIMITIVES,
    VARIABLE_FEATURES,
    evaluate,
    evolutionary_counterexample_search,
    random_catalogue,
)


BEHAVIOR_DIMENSIONS = (
    "perturbation_sensitivity",
    "propagation",
    "fixed_point_strength",
    "abstraction_growth",
    "hierarchy_emergence",
    "transfer",
    "uncertainty_handling",
    "reconstruction",
    "convergence",
    "stability",
    "compositionality",
    "substitution_response",
)

ROLE_NAMES = {
    "perturbation_sensitivity": "fragility",
    "propagation": "propagation",
    "fixed_point_strength": "stabilization",
    "abstraction_growth": "compression",
    "hierarchy_emergence": "stratification",
    "transfer": "transport",
    "uncertainty_handling": "uncertainty",
    "reconstruction": "reconstruction",
    "convergence": "coordination",
    "stability": "persistence",
    "compositionality": "composition",
    "substitution_response": "replacement",
}


@dataclass(frozen=True)
class SemanticRecord:
    architecture_id: str
    paradigm: str
    e_star_complete: bool
    fingerprint: tuple[float, ...]
    canonical_signature: str
    role_multiset: tuple[int, ...]
    edge_signature: tuple[tuple[int, int], ...]
    role_graph_distance_to_lab: float
    weakest_lab_correspondence: str
    equivalence_class: str


@dataclass(frozen=True)
class OperatorRoleSample:
    architecture_id: str
    operator_index: int
    primitive: str
    contribution: tuple[float, ...]
    role: int


def build_architectures(random_count: int, population: int, generations: int, seed: int) -> list[Evaluation]:
    return random_catalogue(random_count, seed) + evolutionary_counterexample_search(population, generations, seed + 1)


def behavioral_fingerprint(evaluation: Evaluation, *, include_perturbation: bool = True) -> tuple[float, ...]:
    perturbation = perturbation_sensitivity(evaluation) if include_perturbation else 0.0
    capability = evaluation.capability_scores
    propagation = _clip(0.5 * capability["transfer"] + 0.3 * capability["representation_translation"] + 0.2 * evaluation.reuse)
    fixed_point = _clip(0.7 * capability["stable_structures"] + 0.3 / (1.0 + perturbation))
    abstraction = _clip(capability["abstraction"])
    hierarchy = _clip((evaluation.hierarchy_depth - 1.0) / 4.0)
    transfer = _clip(evaluation.transfer_quality)
    uncertainty = _clip(capability["uncertainty"])
    reconstruction = _clip(0.5 * capability["experimentally_distinguishable_predictions"] + 0.5 * capability["representation_translation"])
    convergence = _clip(1.0 / (1.0 + evaluation.proof_complexity / 10.0) + 0.25 * fixed_point)
    stability = _clip(0.6 * capability["stable_structures"] + 0.4 * (1.0 - perturbation))
    compositionality = _clip(0.45 * capability["abstraction"] + 0.35 * capability["representation_translation"] + 0.2 * evaluation.compression)
    substitution = _clip(capability["substitutability"])
    return (
        perturbation,
        propagation,
        fixed_point,
        abstraction,
        hierarchy,
        transfer,
        uncertainty,
        reconstruction,
        convergence,
        stability,
        compositionality,
        substitution,
    )


def perturbation_sensitivity(evaluation: Evaluation) -> float:
    if not evaluation.primitives:
        return 0.0
    base = _behavior_seed_vector(evaluation)
    deltas = []
    for primitive in evaluation.primitives:
        reduced = tuple(item for item in evaluation.primitives if item != primitive)
        if not reduced:
            continue
        arch = Architecture(
            architecture_id=evaluation.architecture_id + "_drop",
            paradigm=evaluation.paradigm,
            primitives=reduced,
            variables=evaluation.variables,
        )
        delta = np.linalg.norm(base - _behavior_seed_vector(evaluate(arch))) / math.sqrt(len(base))
        deltas.append(float(delta))
    return _clip(sum(deltas) / len(deltas) if deltas else 0.0)


def operator_contribution_samples(evaluations: list[Evaluation]) -> tuple[list[OperatorRoleSample], dict[int, str]]:
    raw_samples: list[tuple[str, int, str, tuple[float, ...]]] = []
    for evaluation in evaluations:
        base = np.array(behavioral_fingerprint(evaluation, include_perturbation=False), dtype=float)
        for index, primitive in enumerate(evaluation.primitives):
            reduced = tuple(item for i, item in enumerate(evaluation.primitives) if i != index)
            if not reduced:
                continue
            perturbed = evaluate(
                Architecture(
                    architecture_id=evaluation.architecture_id + "_op",
                    paradigm=evaluation.paradigm,
                    primitives=reduced,
                    variables=evaluation.variables,
                )
            )
            contribution = tuple((base - np.array(behavioral_fingerprint(perturbed, include_perturbation=False), dtype=float)).tolist())
            raw_samples.append((evaluation.architecture_id, index, primitive, contribution))
    values = np.array([sample[3] for sample in raw_samples], dtype=float)
    if len(values) < 8:
        return [], {}
    scaled, _, _ = standardize(values)
    labels = best_role_labels(scaled)
    samples = [
        OperatorRoleSample(architecture_id, index, primitive, contribution, int(role))
        for (architecture_id, index, primitive, contribution), role in zip(raw_samples, labels)
    ]
    role_names = infer_role_names(samples)
    return samples, role_names


def best_role_labels(values: np.ndarray) -> list[int]:
    candidates: list[tuple[float, list[int]]] = []
    for eps in (1.2, 1.5, 1.8, 2.1):
        labels = dbscan(values, eps=eps, min_samples=6)
        count = cluster_count(labels)
        if 2 <= count <= 12:
            noise = labels.count(-1) / len(labels)
            candidates.append((count - noise, labels))
    for k in range(3, 10):
        labels = kmeans(values, k, seed=31)
        candidates.append((k * 0.8, labels))
    return max(candidates, key=lambda item: item[0])[1] if candidates else [0] * len(values)


def infer_role_names(samples: list[OperatorRoleSample]) -> dict[int, str]:
    by_role: dict[int, list[OperatorRoleSample]] = {}
    for sample in samples:
        if sample.role < 0:
            continue
        by_role.setdefault(sample.role, []).append(sample)
    names: dict[int, str] = {}
    used: set[str] = set()
    for role, members in sorted(by_role.items()):
        mean = np.array([member.contribution for member in members], dtype=float).mean(axis=0)
        index = int(np.argmax(np.abs(mean)))
        base_name = ROLE_NAMES[BEHAVIOR_DIMENSIONS[index]]
        name = base_name
        suffix = 2
        while name in used:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used.add(name)
        names[role] = name
    return names


def canonicalize(evaluations: list[Evaluation], lab_reference: Evaluation) -> tuple[list[SemanticRecord], list[OperatorRoleSample], dict[int, str]]:
    samples, role_names = operator_contribution_samples(evaluations + [lab_reference])
    by_arch: dict[str, list[OperatorRoleSample]] = {}
    for sample in samples:
        by_arch.setdefault(sample.architecture_id, []).append(sample)
    lab_graph = role_graph(lab_reference, by_arch.get(lab_reference.architecture_id, []))
    records = []
    for evaluation in evaluations:
        graph = role_graph(evaluation, by_arch.get(evaluation.architecture_id, []))
        fingerprint = behavioral_fingerprint(evaluation)
        distance = graph_distance(graph, lab_graph)
        correspondence = weakest_correspondence(distance)
        signature = graph_signature(graph)
        records.append(
            SemanticRecord(
                architecture_id=evaluation.architecture_id,
                paradigm=evaluation.paradigm,
                e_star_complete=evaluation.e_star_complete,
                fingerprint=fingerprint,
                canonical_signature=signature,
                role_multiset=tuple(sorted(graph["roles"])),
                edge_signature=tuple(sorted(graph["edges"])),
                role_graph_distance_to_lab=distance,
                weakest_lab_correspondence=correspondence,
                equivalence_class="",
            )
        )
    classes = assign_equivalence_classes(records)
    return [
        SemanticRecord(
            record.architecture_id,
            record.paradigm,
            record.e_star_complete,
            record.fingerprint,
            record.canonical_signature,
            record.role_multiset,
            record.edge_signature,
            record.role_graph_distance_to_lab,
            record.weakest_lab_correspondence,
            classes[record.canonical_signature],
        )
        for record in records
    ], samples, role_names


def role_graph(evaluation: Evaluation, samples: list[OperatorRoleSample]) -> dict[str, object]:
    roles = [sample.role for sample in samples if sample.role >= 0]
    edges: set[tuple[int, int]] = set()
    ordered = sorted(samples, key=lambda item: item.operator_index)
    for left, right in zip(ordered, ordered[1:]):
        if left.role >= 0 and right.role >= 0 and left.role != right.role:
            edges.add((left.role, right.role))
    for left in ordered:
        for right in ordered:
            if left.operator_index >= right.operator_index or left.role < 0 or right.role < 0:
                continue
            similarity = _positive_dot(left.contribution, right.contribution)
            if similarity > 0.18 and left.role != right.role:
                edges.add((left.role, right.role))
    return {
        "roles": tuple(sorted(roles)),
        "edges": tuple(sorted(edges)),
        "fingerprint": behavioral_fingerprint(evaluation),
    }


def graph_signature(graph: dict[str, object]) -> str:
    role_counts: dict[int, int] = {}
    for role in graph["roles"]:
        role_counts[int(role)] = role_counts.get(int(role), 0) + 1
    roles = ";".join(f"{role}:{count}" for role, count in sorted(role_counts.items()))
    edges = ";".join(f"{left}>{right}" for left, right in graph["edges"])
    return f"R[{roles}]E[{edges}]"


def graph_distance(left: dict[str, object], right: dict[str, object]) -> float:
    left_roles = set(left["roles"])
    right_roles = set(right["roles"])
    left_edges = set(left["edges"])
    right_edges = set(right["edges"])
    role_jaccard = _jaccard_distance(left_roles, right_roles)
    edge_jaccard = _jaccard_distance(left_edges, right_edges)
    fp_delta = np.linalg.norm(np.array(left["fingerprint"]) - np.array(right["fingerprint"])) / math.sqrt(len(BEHAVIOR_DIMENSIONS))
    return float(0.35 * role_jaccard + 0.35 * edge_jaccard + 0.30 * fp_delta)


def weakest_correspondence(distance: float) -> str:
    if distance <= 0.08:
        return "exact_isomorphism"
    if distance <= 0.16:
        return "bisimulation"
    if distance <= 0.26:
        return "simulation_relation"
    if distance <= 0.38:
        return "quotient_or_refinement_morphism"
    if distance <= 0.52:
        return "weak_behavioral_equivalence"
    return "failed"


def assign_equivalence_classes(records: list[SemanticRecord]) -> dict[str, str]:
    signatures = sorted({record.canonical_signature for record in records})
    return {signature: f"canon_{index:03d}" for index, signature in enumerate(signatures)}


def role_necessity(evaluations: list[Evaluation], records: list[SemanticRecord]) -> dict[str, object]:
    successful = [record for record in records if record.e_star_complete]
    all_roles = sorted({role for record in successful for role in record.role_multiset})
    necessity: dict[str, object] = {}
    for role in all_roles:
        without = [record for record in successful if role not in record.role_multiset]
        necessity[str(role)] = {
            "successful_without_role": len(without),
            "empirical_necessity": 1.0 - len(without) / len(successful) if successful else 0.0,
            "counterexample_architecture": without[0].architecture_id if without else None,
        }
    return necessity


def hypothesis_support(records: list[SemanticRecord]) -> dict[str, float]:
    successful = [record for record in records if record.e_star_complete]
    if not successful:
        return {"H0_no_universal_architecture": 1.0, "H1_several_incompatible_architectures": 0.0, "H2_one_canonical_architecture": 0.0}
    class_counts: dict[str, int] = {}
    for record in successful:
        class_counts[record.equivalence_class] = class_counts.get(record.equivalence_class, 0) + 1
    largest = max(class_counts.values()) / len(successful)
    class_count = len(class_counts)
    mean_lab_distance = sum(record.role_graph_distance_to_lab for record in successful) / len(successful)
    h2 = _clip(largest * (1.0 - mean_lab_distance))
    h1 = _clip((1.0 - largest) * min(1.0, class_count / 6.0))
    h0 = _clip(mean_lab_distance * 0.55 + (1.0 if class_count == len(successful) else 0.0) * 0.15)
    total = h0 + h1 + h2
    if total == 0:
        return {"H0_no_universal_architecture": 0.0, "H1_several_incompatible_architectures": 0.0, "H2_one_canonical_architecture": 0.0}
    return {
        "H0_no_universal_architecture": h0 / total,
        "H1_several_incompatible_architectures": h1 / total,
        "H2_one_canonical_architecture": h2 / total,
    }


def semantic_canonicalization_report(
    random_count: int,
    population: int,
    generations: int,
    seed: int,
    out_dir: Path,
    analysis_limit: int = 650,
) -> dict[str, object]:
    evaluations = build_architectures(random_count, population, generations, seed)
    analyzed = select_analysis_sample(evaluations, analysis_limit, seed)
    lab_reference = make_lab_reference()
    records, samples, role_names = canonicalize(analyzed, lab_reference)
    summary = {
        "architectures": len(evaluations),
        "analyzed_architectures": len(analyzed),
        "analysis_sampled": len(analyzed) < len(evaluations),
        "generated_e_star_complete": sum(1 for item in evaluations if item.e_star_complete),
        "e_star_complete": sum(1 for item in analyzed if item.e_star_complete),
        "role_count": len(role_names),
        "role_vocabulary": {str(role): name for role, name in sorted(role_names.items())},
        "canonical_class_count": len({record.equivalence_class for record in records}),
        "successful_canonical_class_count": len({record.equivalence_class for record in records if record.e_star_complete}),
        "role_necessity": role_necessity(evaluations, records),
        "hypothesis_support": hypothesis_support(records),
        "semantic_isomorphism": semantic_isomorphism_summary(records),
        "genuine_counterexamples": genuine_counterexamples(evaluations, records),
    }
    write_semantic_outputs(out_dir, analyzed, records, samples, role_names, summary)
    return summary


def select_analysis_sample(evaluations: list[Evaluation], limit: int, seed: int) -> list[Evaluation]:
    if len(evaluations) <= limit:
        return evaluations
    rng = random.Random(seed + 902)
    successful = [item for item in evaluations if item.e_star_complete]
    failed = [item for item in evaluations if not item.e_star_complete]
    selected = list(successful[: min(len(successful), limit // 2)])
    remaining = limit - len(selected)
    if remaining > 0 and failed:
        selected.extend(rng.sample(failed, min(remaining, len(failed))))
    if len(selected) < limit:
        leftovers = [item for item in evaluations if item not in selected]
        selected.extend(rng.sample(leftovers, min(limit - len(selected), len(leftovers))))
    selected.sort(key=lambda item: item.architecture_id)
    return selected


def semantic_isomorphism_summary(records: list[SemanticRecord]) -> dict[str, object]:
    successful = [record for record in records if record.e_star_complete]
    counts: dict[str, int] = {}
    for record in successful:
        counts[record.weakest_lab_correspondence] = counts.get(record.weakest_lab_correspondence, 0) + 1
    return {
        "successful_count": len(successful),
        "weakest_correspondence_counts": counts,
        "failed_count": counts.get("failed", 0),
        "mean_lab_distance": sum(record.role_graph_distance_to_lab for record in successful) / len(successful) if successful else 0.0,
    }


def genuine_counterexamples(evaluations: list[Evaluation], records: list[SemanticRecord]) -> list[dict[str, object]]:
    by_id = {evaluation.architecture_id: evaluation for evaluation in evaluations}
    result = []
    for record in sorted(records, key=lambda item: -item.role_graph_distance_to_lab):
        if not record.e_star_complete or record.weakest_lab_correspondence != "failed":
            continue
        evaluation = by_id[record.architecture_id]
        result.append(
            {
                "architecture_id": record.architecture_id,
                "paradigm": record.paradigm,
                "e_star_score": evaluation.e_star_score,
                "lab_distance": record.role_graph_distance_to_lab,
                "canonical_class": record.equivalence_class,
                "primitives": evaluation.primitives,
            }
        )
    return result[:30]


def make_lab_reference() -> Evaluation:
    variables = (
        (0.82, 0.25, 0.18, 0.42, 0.20, 0.75, 0.30, 0.20, 0.25, 0.10),
        (0.20, 0.80, 0.25, 0.25, 0.35, 0.20, 0.55, 0.75, 0.20, 0.10),
        (0.15, 0.30, 0.85, 0.20, 0.25, 0.55, 0.25, 0.20, 0.10, 0.70),
        (0.35, 0.25, 0.20, 0.85, 0.35, 0.30, 0.20, 0.25, 0.45, 0.10),
        (0.25, 0.30, 0.20, 0.35, 0.85, 0.20, 0.30, 0.20, 0.20, 0.10),
    )
    return evaluate(
        Architecture(
            "lab_reference",
            "reference",
            (
                "recurrence",
                "prediction",
                "merge",
                "compression",
                "belief_revision",
                "graph_rewrite",
                "schema_alignment",
                "local_replacement",
            ),
            variables,
        )
    )


def write_semantic_outputs(
    out_dir: Path,
    evaluations: list[Evaluation],
    records: list[SemanticRecord],
    samples: list[OperatorRoleSample],
    role_names: dict[int, str],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_fingerprints(out_dir / "behavioral_fingerprints.csv", records)
    _write_roles(out_dir / "operator_role_samples.csv", samples, role_names)
    _write_catalogue(out_dir / "canonical_architecture_catalogue.csv", records)
    (out_dir / "canonical_role_graph.json").write_text(
        json.dumps(canonical_role_graph(records, role_names), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "role_discovery_report.json").write_text(
        json.dumps(role_discovery_report(samples, role_names), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "semantic_isomorphism_report.json").write_text(
        json.dumps(summary["semantic_isomorphism"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "hypothesis_comparison.json").write_text(
        json.dumps(summary["hypothesis_support"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "genuine_counterexamples.json").write_text(
        json.dumps(summary["genuine_counterexamples"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "semantic_canonicalization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def canonical_role_graph(records: list[SemanticRecord], role_names: dict[int, str]) -> dict[str, object]:
    successful = [record for record in records if record.e_star_complete]
    role_counts: dict[int, int] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    for record in successful:
        for role in record.role_multiset:
            role_counts[role] = role_counts.get(role, 0) + 1
        for edge in record.edge_signature:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    threshold = max(1, int(0.2 * len(successful)))
    return {
        "nodes": [
            {"role": role, "name": role_names.get(role, f"role_{role}"), "support": count}
            for role, count in sorted(role_counts.items())
        ],
        "edges": [
            {"source": left, "target": right, "support": count}
            for (left, right), count in sorted(edge_counts.items())
            if count >= threshold
        ],
        "support_threshold": threshold,
        "successful_architectures": len(successful),
    }


def role_discovery_report(samples: list[OperatorRoleSample], role_names: dict[int, str]) -> dict[str, object]:
    by_role: dict[int, list[OperatorRoleSample]] = {}
    for sample in samples:
        if sample.role >= 0:
            by_role.setdefault(sample.role, []).append(sample)
    return {
        "role_count": len(role_names),
        "roles": {
            str(role): {
                "name": role_names.get(role, f"role_{role}"),
                "samples": len(members),
                "dominant_behavior_dimensions": _dominant_dimensions(members),
            }
            for role, members in sorted(by_role.items())
        },
    }


def _write_fingerprints(path: Path, records: list[SemanticRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["architecture_id", "paradigm", "e_star_complete", *BEHAVIOR_DIMENSIONS, "canonical_class", "lab_distance", "weakest_lab_correspondence"])
        for record in records:
            writer.writerow([
                record.architecture_id,
                record.paradigm,
                record.e_star_complete,
                *record.fingerprint,
                record.equivalence_class,
                record.role_graph_distance_to_lab,
                record.weakest_lab_correspondence,
            ])


def _write_roles(path: Path, samples: list[OperatorRoleSample], role_names: dict[int, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["architecture_id", "operator_index", "role", "role_name", *BEHAVIOR_DIMENSIONS])
        for sample in samples:
            writer.writerow([sample.architecture_id, sample.operator_index, sample.role, role_names.get(sample.role, "noise"), *sample.contribution])


def _write_catalogue(path: Path, records: list[SemanticRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["architecture_id", "paradigm", "e_star_complete", "canonical_class", "canonical_signature", "lab_distance", "weakest_lab_correspondence"])
        for record in records:
            writer.writerow([
                record.architecture_id,
                record.paradigm,
                record.e_star_complete,
                record.equivalence_class,
                record.canonical_signature,
                record.role_graph_distance_to_lab,
                record.weakest_lab_correspondence,
            ])


def _dominant_dimensions(samples: list[OperatorRoleSample]) -> list[str]:
    mean = np.array([sample.contribution for sample in samples], dtype=float).mean(axis=0)
    ranked = sorted(zip(BEHAVIOR_DIMENSIONS, mean), key=lambda item: -abs(item[1]))[:4]
    return [f"{name}:{value:.3f}" for name, value in ranked]


def _behavior_seed_vector(evaluation: Evaluation) -> np.ndarray:
    return np.array(
        [
            evaluation.capability_scores["transfer"],
            evaluation.capability_scores["representation_translation"],
            evaluation.capability_scores["stable_structures"],
            evaluation.capability_scores["abstraction"],
            evaluation.capability_scores["uncertainty"],
            evaluation.capability_scores["substitutability"],
            evaluation.compression,
            evaluation.reuse,
            evaluation.hierarchy_depth / 5.0,
            evaluation.transfer_quality,
        ],
        dtype=float,
    )


def _positive_dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_arr = np.array(left, dtype=float)
    right_arr = np.array(right, dtype=float)
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    if denom == 0:
        return 0.0
    return max(0.0, float(np.dot(left_arr, right_arr) / denom))


def _jaccard_distance(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
