from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import adjusted_rand_index, cluster_count, dbscan, kmeans, pca, standardize
from epistemic_engine.architecture_falsification import (
    Architecture,
    Evaluation,
    PARADIGM_BIAS,
    PARADIGMS,
    PRIMITIVES,
    VARIABLE_FEATURES,
    evaluate,
)
from epistemic_engine.semantic_canonicalization import BEHAVIOR_DIMENSIONS, behavioral_fingerprint


PARAMETERS = (
    "memory_capacity",
    "stochasticity",
    "environmental_regularity",
    "compositionality",
    "interaction_locality",
    "constraint_density",
    "graph_connectivity",
    "prediction_horizon",
    "abstraction_budget",
    "communication_bandwidth",
    "feedback_strength",
    "hierarchy_depth",
    "computational_budget",
)

ORDER_PARAMETERS = (
    "abstraction_depth",
    "transfer_efficiency",
    "behavioral_entropy",
    "compression_ratio",
    "fixed_point_density",
    "stability",
    "uncertainty_persistence",
    "reuse",
    "hierarchy_growth",
    "canonical_role_count_proxy",
    "response_susceptibility",
    "coordination",
)


PARAMETER_PRIMITIVES = {
    "memory_capacity": ("recurrence", "stochastic_memory", "memory_decay"),
    "stochasticity": ("stochastic_memory", "probabilistic_grammar", "entropy_regularization"),
    "environmental_regularity": ("invariant_detection", "noise_filter", "finite_automata"),
    "compositionality": ("merge", "operator_composition", "hypergraph_dynamics"),
    "interaction_locality": ("attention", "message_passing", "cellular_automata"),
    "constraint_density": ("constraint_propagation", "sat_constraints", "symbolic_rules"),
    "graph_connectivity": ("graph_rewrite", "message_passing", "schema_alignment"),
    "prediction_horizon": ("prediction", "causal_probe", "counterexample_search"),
    "abstraction_budget": ("compression", "merge", "type_unification"),
    "communication_bandwidth": ("message_passing", "schema_alignment", "rewriting_system"),
    "feedback_strength": ("belief_revision", "counterexample_search", "recurrence"),
    "hierarchy_depth": ("operator_composition", "program_synthesis", "hypergraph_dynamics"),
    "computational_budget": ("differentiable_objective", "program_synthesis", "finite_automata"),
}


@dataclass(frozen=True)
class PhasePoint:
    point_id: str
    paradigm: str
    parameters: dict[str, float]
    evaluation: Evaluation
    fingerprint: tuple[float, ...]
    order_parameters: dict[str, float]
    phase: int


def generate_phase_points(count: int, seed: int) -> list[PhasePoint]:
    rng = random.Random(seed)
    points: list[PhasePoint] = []
    for index in range(count):
        paradigm = PARADIGMS[index % len(PARADIGMS)]
        parameters = sample_parameters(index, count, rng)
        architecture = architecture_from_parameters(f"p{index:06d}", paradigm, parameters, rng)
        evaluation = evaluate(architecture)
        fingerprint = behavioral_fingerprint(evaluation)
        orders = order_parameters(evaluation, fingerprint)
        points.append(PhasePoint(architecture.architecture_id, paradigm, parameters, evaluation, fingerprint, orders, -1))
    phases = discover_phases(points)
    return [
        PhasePoint(point.point_id, point.paradigm, point.parameters, point.evaluation, point.fingerprint, point.order_parameters, phase)
        for point, phase in zip(points, phases)
    ]


def sample_parameters(index: int, count: int, rng: random.Random) -> dict[str, float]:
    # Stratified low-discrepancy-ish sweep: each coordinate is continuous, but
    # shifted differently to avoid a small preset grid.
    params: dict[str, float] = {}
    base = (index + 0.5) / count
    for offset, parameter in enumerate(PARAMETERS):
        irrational = (math.sqrt(offset + 2) % 1.0)
        jitter = rng.uniform(-0.025, 0.025)
        value = (base + irrational * index * 0.173 + jitter) % 1.0
        params[parameter] = max(0.0, min(1.0, value))
    return params


def architecture_from_parameters(architecture_id: str, paradigm: str, parameters: dict[str, float], rng: random.Random) -> Architecture:
    weighted: dict[str, float] = {primitive: 0.04 for primitive in PRIMITIVES}
    for primitive in PARADIGM_BIAS[paradigm]:
        weighted[primitive] += 0.28
    for parameter, value in parameters.items():
        for primitive in PARAMETER_PRIMITIVES[parameter]:
            weighted[primitive] += value * 0.38

    primitive_count = 2 + int(parameters["computational_budget"] * 4) + int(parameters["graph_connectivity"] * 2) + int(parameters["abstraction_budget"] * 2)
    primitive_count = max(2, min(10, primitive_count))
    primitives = tuple(sorted(weighted_sample_without_replacement(weighted, primitive_count, rng)))

    variable_count = 3 + int(parameters["memory_capacity"] * 4) + int(parameters["hierarchy_depth"] * 3) + int(parameters["computational_budget"] * 2)
    variables = []
    for _ in range(max(3, min(12, variable_count))):
        values = {
            "birth": 0.18 + 0.42 * parameters["compositionality"] + 0.25 * parameters["abstraction_budget"],
            "content": 0.15 + 0.55 * parameters["constraint_density"] + 0.20 * parameters["prediction_horizon"],
            "ambiguity": 0.12 + 0.65 * parameters["stochasticity"] + 0.18 * (1 - parameters["environmental_regularity"]),
            "transform": 0.12 + 0.50 * parameters["graph_connectivity"] + 0.35 * parameters["communication_bandwidth"],
            "role": 0.12 + 0.45 * parameters["interaction_locality"] + 0.35 * parameters["feedback_strength"],
            "memory": 0.10 + 0.72 * parameters["memory_capacity"],
            "prediction": 0.10 + 0.72 * parameters["prediction_horizon"],
            "constraint": 0.10 + 0.75 * parameters["constraint_density"],
            "compression": 0.10 + 0.60 * parameters["abstraction_budget"] + 0.22 * parameters["compositionality"],
            "stochasticity": 0.08 + 0.82 * parameters["stochasticity"],
        }
        noisy = tuple(max(0.0, min(1.0, values[feature] + rng.uniform(-0.08, 0.08))) for feature in VARIABLE_FEATURES)
        variables.append(noisy)
    return Architecture(architecture_id, paradigm, primitives, tuple(variables))


def weighted_sample_without_replacement(weights: dict[str, float], count: int, rng: random.Random) -> list[str]:
    pool = dict(weights)
    result = []
    for _ in range(min(count, len(pool))):
        total = sum(pool.values())
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = next(iter(pool))
        for item, weight in sorted(pool.items()):
            cumulative += weight
            if cumulative >= threshold:
                chosen = item
                break
        result.append(chosen)
        del pool[chosen]
    return result


def order_parameters(evaluation: Evaluation, fingerprint: tuple[float, ...]) -> dict[str, float]:
    fp = dict(zip(BEHAVIOR_DIMENSIONS, fingerprint))
    entropy = behavioral_entropy(fingerprint)
    return {
        "abstraction_depth": evaluation.hierarchy_depth,
        "transfer_efficiency": evaluation.transfer_quality,
        "behavioral_entropy": entropy,
        "compression_ratio": evaluation.compression,
        "fixed_point_density": fp["fixed_point_strength"],
        "stability": fp["stability"],
        "uncertainty_persistence": fp["uncertainty_handling"] * (1.0 - fp["convergence"] * 0.3),
        "reuse": evaluation.reuse,
        "hierarchy_growth": fp["hierarchy_emergence"],
        "canonical_role_count_proxy": 1.0 + 8.0 * entropy,
        "response_susceptibility": fp["perturbation_sensitivity"],
        "coordination": fp["convergence"],
    }


def behavioral_entropy(fingerprint: tuple[float, ...]) -> float:
    total = sum(abs(value) for value in fingerprint)
    if total == 0:
        return 0.0
    probs = [abs(value) / total for value in fingerprint if value > 0]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    return entropy / math.log(len(fingerprint), 2)


def discover_phases(points: list[PhasePoint]) -> list[int]:
    values = phase_matrix(points)
    scaled, _, _ = standardize(values)
    candidates: list[tuple[float, list[int]]] = []
    for eps in (1.25, 1.55, 1.85, 2.15, 2.45):
        labels = dbscan(scaled, eps=eps, min_samples=6)
        count = cluster_count(labels)
        if 2 <= count <= 12:
            candidates.append((count - labels.count(-1) / len(labels), labels))
    for k in range(3, 9):
        labels = kmeans(scaled, k, seed=71)
        candidates.append((0.75 * k, labels))
    return max(candidates, key=lambda item: item[0])[1] if candidates else [0] * len(points)


def phase_matrix(points: list[PhasePoint]) -> np.ndarray:
    return np.array(
        [
            [*point.fingerprint, *(point.order_parameters[name] for name in ORDER_PARAMETERS)]
            for point in points
        ],
        dtype=float,
    )


def transition_atlas(points: list[PhasePoint], neighbor_count: int = 6) -> dict[str, object]:
    param_values = np.array([[point.parameters[name] for name in PARAMETERS] for point in points], dtype=float)
    behavior_values = phase_matrix(points)
    scaled_behavior, _, _ = standardize(behavior_values)
    transitions = []
    for index, point in enumerate(points):
        distances = np.linalg.norm(param_values - param_values[index], axis=1)
        neighbors = np.argsort(distances)[1 : neighbor_count + 1]
        for neighbor in neighbors:
            if distances[neighbor] == 0:
                continue
            behavior_delta = float(np.linalg.norm(scaled_behavior[index] - scaled_behavior[neighbor]))
            gradient = behavior_delta / float(distances[neighbor])
            phase_changed = point.phase != points[neighbor].phase
            transitions.append(
                {
                    "left": point.point_id,
                    "right": points[neighbor].point_id,
                    "left_phase": point.phase,
                    "right_phase": points[neighbor].phase,
                    "parameter_distance": float(distances[neighbor]),
                    "behavior_delta": behavior_delta,
                    "gradient": gradient,
                    "phase_changed": phase_changed,
                    "midpoint": {name: (point.parameters[name] + points[neighbor].parameters[name]) / 2 for name in PARAMETERS},
                }
            )
    transitions.sort(key=lambda item: -item["gradient"])
    gradients = [item["gradient"] for item in transitions]
    critical_threshold = float(np.quantile(gradients, 0.92)) if gradients else 0.0
    critical = [item for item in transitions if item["gradient"] >= critical_threshold and item["phase_changed"]]
    return {
        "neighbor_edges": len(transitions),
        "critical_threshold": critical_threshold,
        "critical_edges": critical[:120],
        "phase_change_rate": sum(1 for item in transitions if item["phase_changed"]) / len(transitions) if transitions else 0.0,
        "metastable_edges": [item for item in transitions if item["gradient"] >= critical_threshold and not item["phase_changed"]][:80],
    }


def order_parameter_report(points: list[PhasePoint]) -> dict[str, object]:
    phase_labels = [point.phase for point in points]
    values = {name: [point.order_parameters[name] for point in points] for name in ORDER_PARAMETERS}
    scores = {}
    for name, series in values.items():
        scores[name] = phase_separation_score(series, phase_labels)
    discovered = sorted(scores.items(), key=lambda item: -item[1])
    return {
        "ranked_order_parameters": [{"name": name, "phase_separation": score} for name, score in discovered],
        "top_order_parameters": [name for name, _ in discovered[:5]],
    }


def phase_separation_score(values: list[float], labels: list[int]) -> float:
    global_mean = sum(values) / len(values)
    between = 0.0
    within = 0.0
    for label in sorted(set(labels)):
        members = [value for value, item_label in zip(values, labels) if item_label == label]
        if not members:
            continue
        mean = sum(members) / len(members)
        between += len(members) * (mean - global_mean) ** 2
        within += sum((value - mean) ** 2 for value in members)
    return between / (between + within + 1e-9)


def scaling_laws(points: list[PhasePoint], atlas: dict[str, object]) -> dict[str, object]:
    critical_edges = atlas["critical_edges"]
    if not critical_edges:
        return {"fits": {}, "note": "No critical edges detected."}
    by_id = {point.point_id: point for point in points}
    fits = {}
    for order_name in ORDER_PARAMETERS:
        xs = []
        ys = []
        for edge in critical_edges:
            left = by_id[edge["left"]]
            right = by_id[edge["right"]]
            x = max(1e-6, edge["parameter_distance"])
            y = abs(left.order_parameters[order_name] - right.order_parameters[order_name])
            if y > 1e-6:
                xs.append(math.log(x))
                ys.append(math.log(y))
        if len(xs) < 5:
            continue
        slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
        pred = slope * np.array(xs) + intercept
        ss_res = float(np.sum((np.array(ys) - pred) ** 2))
        ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
        fits[order_name] = {"exponent": float(slope), "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0}
    return {"fits": fits, "critical_edge_count": len(critical_edges)}


def universality_classes(points: list[PhasePoint], atlas: dict[str, object]) -> dict[str, object]:
    by_phase: dict[int, list[PhasePoint]] = {}
    for point in points:
        by_phase.setdefault(point.phase, []).append(point)
    catalogue = {}
    for phase, members in sorted(by_phase.items()):
        paradigms = sorted({point.paradigm for point in members})
        mean_orders = {
            name: sum(point.order_parameters[name] for point in members) / len(members)
            for name in ORDER_PARAMETERS
        }
        catalogue[str(phase)] = {
            "size": len(members),
            "paradigms": paradigms,
            "paradigm_count": len(paradigms),
            "mean_order_parameters": mean_orders,
            "implementation_invariance": len(paradigms) / len(PARADIGMS),
        }
    return catalogue


def latent_coordinate_analysis(points: list[PhasePoint]) -> dict[str, object]:
    values = phase_matrix(points)
    scaled, _, _ = standardize(values)
    coords, explained, components = pca(scaled, components=6)
    cumulative = np.cumsum(np.array(explained))
    intrinsic = int(np.searchsorted(cumulative, 0.85) + 1) if len(cumulative) else 0
    feature_names = (*BEHAVIOR_DIMENSIONS, *ORDER_PARAMETERS)
    return {
        "explained_variance": explained,
        "intrinsic_dimension_85pct": intrinsic,
        "top_coordinates": {
            f"pc{index + 1}": [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in sorted(zip(feature_names, component), key=lambda item: -abs(item[1]))[:8]
            ]
            for index, component in enumerate(components[:3])
        },
        "coords": coords[:, :2].tolist(),
    }


def topology_report(points: list[PhasePoint], atlas: dict[str, object]) -> dict[str, object]:
    phase_edges: set[tuple[int, int]] = set()
    for edge in atlas["critical_edges"]:
        left = int(edge["left_phase"])
        right = int(edge["right_phase"])
        if left >= 0 and right >= 0 and left != right:
            phase_edges.add(tuple(sorted((left, right))))
    phases = sorted({point.phase for point in points if point.phase >= 0})
    adjacency = {str(phase): sorted({b if a == phase else a for a, b in phase_edges if a == phase or b == phase}) for phase in phases}
    components = connected_components(phases, phase_edges)
    assigned = sum(1 for point in points if point.phase >= 0)
    return {
        "phase_count": len(phases),
        "assigned_fraction": assigned / len(points),
        "noise_fraction": 1.0 - assigned / len(points),
        "phase_adjacency": adjacency,
        "connected_components": components,
        "disconnected_component_count": len(components),
        "forbidden_transitions_proxy": [
            [left, right]
            for i, left in enumerate(phases)
            for right in phases[i + 1 :]
            if tuple(sorted((left, right))) not in phase_edges
        ][:100],
    }


def connected_components(phases: list[int], edges: set[tuple[int, int]]) -> list[list[int]]:
    unseen = set(phases)
    result = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            neighbors = {b if a == current else a for a, b in edges if a == current or b == current}
            for neighbor in neighbors & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        result.append(sorted(component))
    return result


def confidence_estimates(points: list[PhasePoint], seed: int) -> dict[str, object]:
    rng = random.Random(seed + 404)
    labels = [point.phase for point in points]
    values = phase_matrix(points)
    scores = []
    for trial in range(20):
        indices = sorted(rng.sample(range(len(points)), max(12, int(0.65 * len(points)))))
        sub_values, _, _ = standardize(values[indices])
        sub_labels = kmeans(sub_values, max(2, cluster_count(labels)), seed=seed + trial)
        scores.append(adjusted_rand_index([labels[index] for index in indices], sub_labels))
    return {
        "bootstrap_phase_ari": sum(scores) / len(scores) if scores else 0.0,
        "phase_count": cluster_count(labels),
    }


def hypothesis_result(points: list[PhasePoint], topology: dict[str, object], confidence: dict[str, object]) -> dict[str, float]:
    phase_count = int(topology["phase_count"])
    stability = float(confidence["bootstrap_phase_ari"])
    components = int(topology["disconnected_component_count"])
    noise_fraction = float(topology.get("noise_fraction", 0.0))
    if phase_count <= 1 or stability < 0.25 or noise_fraction > 0.5:
        h_continuous = 0.72 + 0.2 * noise_fraction
    else:
        h_continuous = max(0.05, 0.55 * (1.0 - stability))
    h_finite = min(0.9, stability * min(1.0, phase_count / 3.0) * (1.0 - noise_fraction) * (1.0 if phase_count < 12 else 0.65))
    h_fragmented = min(0.9, 0.25 + 0.12 * max(0, components - 1) + 0.2 * (1.0 - stability))
    total = h_continuous + h_finite + h_fragmented
    return {
        "finite_robust_phase_structure": h_finite / total,
        "essentially_continuous_space": h_continuous / total,
        "fragmented_or_underresolved_phase_structure": h_fragmented / total,
    }


def phase_transition_report(count: int, seed: int, out_dir: Path) -> dict[str, object]:
    points = generate_phase_points(count, seed)
    atlas = transition_atlas(points)
    order_report = order_parameter_report(points)
    scaling = scaling_laws(points, atlas)
    classes = universality_classes(points, atlas)
    latent = latent_coordinate_analysis(points)
    topology = topology_report(points, atlas)
    confidence = confidence_estimates(points, seed)
    hypotheses = hypothesis_result(points, topology, confidence)
    summary = {
        "points": len(points),
        "phase_count": topology["phase_count"],
        "noise_fraction": topology["noise_fraction"],
        "critical_edges": len(atlas["critical_edges"]),
        "phase_change_rate": atlas["phase_change_rate"],
        "top_order_parameters": order_report["top_order_parameters"],
        "intrinsic_dimension_85pct": latent["intrinsic_dimension_85pct"],
        "confidence_estimates": confidence,
        "hypothesis_support": hypotheses,
    }
    write_outputs(out_dir, points, atlas, order_report, scaling, classes, latent, topology, confidence, hypotheses, summary)
    return summary


def write_outputs(
    out_dir: Path,
    points: list[PhasePoint],
    atlas: dict[str, object],
    order_report: dict[str, object],
    scaling: dict[str, object],
    classes: dict[str, object],
    latent: dict[str, object],
    topology: dict[str, object],
    confidence: dict[str, object],
    hypotheses: dict[str, float],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_phase_points(out_dir / "empirical_phase_diagram.csv", points, latent["coords"])
    (out_dir / "discovered_order_parameters.json").write_text(json.dumps(order_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "transition_atlas.json").write_text(json.dumps(atlas, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "scaling_law_report.json").write_text(json.dumps(scaling, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "universality_class_catalogue.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    latent_public = {key: value for key, value in latent.items() if key != "coords"}
    (out_dir / "latent_coordinate_analysis.json").write_text(json.dumps(latent_public, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "topology_report.json").write_text(json.dumps(topology, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "confidence_estimates.json").write_text(json.dumps(confidence, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "hypothesis_comparison.json").write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "epistemic_phase_transition_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_phase_points(path: Path, points: list[PhasePoint], coords: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "point_id",
            "paradigm",
            "phase",
            "e_star_score",
            "e_star_complete",
            "pc1",
            "pc2",
            *PARAMETERS,
            *BEHAVIOR_DIMENSIONS,
            *ORDER_PARAMETERS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point, coord in zip(points, coords):
            row = {
                "point_id": point.point_id,
                "paradigm": point.paradigm,
                "phase": point.phase,
                "e_star_score": point.evaluation.e_star_score,
                "e_star_complete": point.evaluation.e_star_complete,
                "pc1": coord[0],
                "pc2": coord[1],
            }
            row.update(point.parameters)
            row.update(dict(zip(BEHAVIOR_DIMENSIONS, point.fingerprint)))
            row.update(point.order_parameters)
            writer.writerow(row)
