from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import adjusted_rand_index, cluster_count, dbscan, kmeans, pca, standardize
from epistemic_engine.architecture_falsification import PARADIGMS, evaluate
from epistemic_engine.epistemic_phase_transition import (
    ORDER_PARAMETERS,
    PARAMETERS,
    architecture_from_parameters,
    behavioral_entropy,
    order_parameters,
    sample_parameters,
)
from epistemic_engine.semantic_canonicalization import BEHAVIOR_DIMENSIONS, behavioral_fingerprint


UPDATE_RULES = (
    "merge_driven",
    "prediction_driven",
    "constraint_driven",
    "mdl_like",
    "information_bottleneck_like",
    "graph_rewriting",
    "sat_propagation",
    "stochastic_interaction",
    "differentiable_optimization",
    "random_interaction",
)

WORLDS = ("regular", "noisy", "compositional", "volatile")

OBSERVABLES = (
    *BEHAVIOR_DIMENSIONS,
    *(f"order_{name}" for name in ORDER_PARAMETERS),
    "e_star_score",
    "representation_complexity",
    "translation_complexity",
)


@dataclass(frozen=True)
class FlowState:
    trajectory_id: str
    step: int
    update_rule: str
    world: str
    paradigm: str
    parameters: dict[str, float]
    observables: dict[str, float]


def generate_trajectories(initial_conditions: int, steps: int, seed: int) -> list[FlowState]:
    rng = random.Random(seed)
    rows: list[FlowState] = []
    for index in range(initial_conditions):
        paradigm = PARADIGMS[index % len(PARADIGMS)]
        base_params = sample_parameters(index, initial_conditions, rng)
        for update_rule in UPDATE_RULES:
            for world in WORLDS:
                params = dict(base_params)
                trajectory_id = f"tr_{index:04d}_{update_rule}_{world}"
                for step in range(steps):
                    observables = observe_state(trajectory_id, step, paradigm, params, seed)
                    rows.append(FlowState(trajectory_id, step, update_rule, world, paradigm, dict(params), observables))
                    params = update_parameters(params, update_rule, world, step, rng)
    return rows


def observe_state(trajectory_id: str, step: int, paradigm: str, parameters: dict[str, float], seed: int) -> dict[str, float]:
    stable_seed = seed + step * 1009 + sum((index + 1) * ord(char) for index, char in enumerate(trajectory_id))
    rng = random.Random(stable_seed)
    architecture = architecture_from_parameters(f"{trajectory_id}_{step}", paradigm, parameters, rng)
    evaluation = evaluate(architecture)
    fingerprint = behavioral_fingerprint(evaluation, include_perturbation=False)
    orders = order_parameters(evaluation, fingerprint)
    observables = dict(zip(BEHAVIOR_DIMENSIONS, fingerprint))
    observables.update({f"order_{key}": value for key, value in orders.items()})
    observables["e_star_score"] = evaluation.e_star_score
    observables["representation_complexity"] = evaluation.representation_complexity
    observables["translation_complexity"] = evaluation.translation_complexity
    return observables


def update_parameters(params: dict[str, float], update_rule: str, world: str, step: int, rng: random.Random) -> dict[str, float]:
    delta = {name: 0.0 for name in PARAMETERS}
    if update_rule == "merge_driven":
        delta["compositionality"] += 0.045
        delta["abstraction_budget"] += 0.035
        delta["hierarchy_depth"] += 0.025
        delta["interaction_locality"] -= 0.015
    elif update_rule == "prediction_driven":
        delta["prediction_horizon"] += 0.045
        delta["feedback_strength"] += 0.035
        delta["environmental_regularity"] += 0.020
        delta["stochasticity"] -= 0.020
    elif update_rule == "constraint_driven":
        delta["constraint_density"] += 0.050
        delta["environmental_regularity"] += 0.030
        delta["stochasticity"] -= 0.025
        delta["interaction_locality"] += 0.010
    elif update_rule == "mdl_like":
        delta["abstraction_budget"] += 0.030
        delta["computational_budget"] -= 0.020
        delta["memory_capacity"] -= 0.010
        delta["compositionality"] += 0.015
    elif update_rule == "information_bottleneck_like":
        delta["abstraction_budget"] += 0.035
        delta["communication_bandwidth"] -= 0.020
        delta["stochasticity"] -= 0.010
        delta["transfer_bias"] = 0.0  # ignored marker keeps rule explicit
    elif update_rule == "graph_rewriting":
        delta["graph_connectivity"] += 0.045
        delta["communication_bandwidth"] += 0.025
        delta["hierarchy_depth"] += 0.015
    elif update_rule == "sat_propagation":
        delta["constraint_density"] += 0.060
        delta["computational_budget"] += 0.015
        delta["stochasticity"] -= 0.030
    elif update_rule == "stochastic_interaction":
        delta["stochasticity"] += 0.045
        delta["memory_capacity"] += 0.020
        delta["feedback_strength"] += rng.uniform(-0.025, 0.025)
    elif update_rule == "differentiable_optimization":
        delta["prediction_horizon"] += 0.030
        delta["abstraction_budget"] += 0.025
        delta["feedback_strength"] += 0.030
        delta["computational_budget"] += 0.020
    elif update_rule == "random_interaction":
        for name in PARAMETERS:
            delta[name] += rng.uniform(-0.025, 0.025)

    if world == "regular":
        delta["environmental_regularity"] += 0.020
        delta["stochasticity"] -= 0.010
    elif world == "noisy":
        delta["stochasticity"] += 0.030
        delta["environmental_regularity"] -= 0.025
    elif world == "compositional":
        delta["compositionality"] += 0.030
        delta["hierarchy_depth"] += 0.020
    elif world == "volatile":
        delta["feedback_strength"] += 0.020 * math.sin(step / 2.0)
        delta["stochasticity"] += 0.020
        delta["memory_capacity"] -= 0.010

    next_params = {}
    for name in PARAMETERS:
        noise = 0.006 * rng.uniform(-1.0, 1.0)
        next_params[name] = max(0.0, min(1.0, params[name] + delta.get(name, 0.0) + noise))
    return next_params


def observable_matrix(states: list[FlowState]) -> np.ndarray:
    return np.array([[state.observables[name] for name in OBSERVABLES] for state in states], dtype=float)


def latent_manifold(states: list[FlowState]) -> dict[str, object]:
    values = observable_matrix(states)
    scaled, _, _ = standardize(values)
    coords, explained, components = pca(scaled, components=6)
    cumulative = np.cumsum(np.array(explained))
    intrinsic = int(np.searchsorted(cumulative, 0.85) + 1) if len(cumulative) else 0
    return {
        "coords": coords,
        "explained_variance": explained,
        "intrinsic_dimension_85pct": intrinsic,
        "top_coordinates": {
            f"pc{index + 1}": [
                {"observable": observable, "weight": float(weight)}
                for observable, weight in sorted(zip(OBSERVABLES, component), key=lambda item: -abs(item[1]))[:8]
            ]
            for index, component in enumerate(components[:3])
        },
    }


def vector_field(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    index = {(state.trajectory_id, state.step): i for i, state in enumerate(states)}
    vectors = []
    rows = []
    by_rule: dict[str, list[float]] = {}
    by_world: dict[str, list[float]] = {}
    for i, state in enumerate(states):
        j = index.get((state.trajectory_id, state.step + 1))
        if j is None:
            continue
        vec = coords[j, :3] - coords[i, :3]
        speed = float(np.linalg.norm(vec))
        vectors.append(vec)
        by_rule.setdefault(state.update_rule, []).append(speed)
        by_world.setdefault(state.world, []).append(speed)
        rows.append(
            {
                "trajectory_id": state.trajectory_id,
                "step": state.step,
                "update_rule": state.update_rule,
                "world": state.world,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]) if coords.shape[1] > 2 else 0.0,
                "dx": float(vec[0]),
                "dy": float(vec[1]),
                "dz": float(vec[2]) if len(vec) > 2 else 0.0,
                "speed": speed,
            }
        )
    vector_array = np.array(vectors, dtype=float) if vectors else np.zeros((0, 3))
    return {
        "rows": rows,
        "mean_speed": float(np.mean(np.linalg.norm(vector_array, axis=1))) if len(vector_array) else 0.0,
        "speed_by_rule": {key: _mean(values) for key, values in by_rule.items()},
        "speed_by_world": {key: _mean(values) for key, values in by_world.items()},
        "divergence_proxy": divergence_proxy(rows),
        "rotational_proxy": rotational_proxy(states, coords),
        "curvature_proxy": curvature_proxy(states, coords),
    }


def divergence_proxy(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    values = []
    for row in rows:
        position = np.array([row["x"], row["y"], row["z"]], dtype=float)
        velocity = np.array([row["dx"], row["dy"], row["dz"]], dtype=float)
        denom = np.linalg.norm(position) * np.linalg.norm(velocity)
        if denom > 0:
            values.append(float(np.dot(position, velocity) / denom))
    return _mean(values)


def rotational_proxy(states: list[FlowState], coords: np.ndarray) -> float:
    values = []
    by_traj = group_indices_by_trajectory(states)
    for indices in by_traj.values():
        for a, b, c in zip(indices, indices[1:], indices[2:]):
            v1 = coords[b, :2] - coords[a, :2]
            v2 = coords[c, :2] - coords[b, :2]
            values.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    return _mean([abs(value) for value in values])


def curvature_proxy(states: list[FlowState], coords: np.ndarray) -> float:
    angles = []
    by_traj = group_indices_by_trajectory(states)
    for indices in by_traj.values():
        for a, b, c in zip(indices, indices[1:], indices[2:]):
            v1 = coords[b, :3] - coords[a, :3]
            v2 = coords[c, :3] - coords[b, :3]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                angles.append(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))))
    return _mean(angles)


def group_indices_by_trajectory(states: list[FlowState]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        grouped.setdefault(state.trajectory_id, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda item: states[item].step)
    return grouped


def attractor_atlas(states: list[FlowState], coords: np.ndarray, field: dict[str, object]) -> dict[str, object]:
    by_traj = group_indices_by_trajectory(states)
    endpoint_indices = [indices[-1] for indices in by_traj.values()]
    endpoint_values = coords[endpoint_indices, :3]
    scaled, _, _ = standardize(endpoint_values)
    labels = dbscan(scaled, eps=0.9, min_samples=4)
    speeds_by_key = {(row["trajectory_id"], row["step"]): row["speed"] for row in field["rows"]}
    fixed = []
    cycles = []
    slow = []
    for trajectory_id, indices in by_traj.items():
        final = indices[-1]
        final_state = states[final]
        final_speed = speeds_by_key.get((trajectory_id, final_state.step - 1), 0.0)
        if final_speed < 0.08:
            fixed.append(trajectory_id)
        path = coords[indices, :3]
        if len(path) > 6:
            distances = np.linalg.norm(path[-1] - path[:-3], axis=1)
            if float(np.min(distances)) < 0.18:
                cycles.append(trajectory_id)
        step_speeds = [speeds_by_key.get((trajectory_id, states[index].step), 0.0) for index in indices[:-1]]
        if step_speeds and sum(1 for speed in step_speeds if speed < 0.12) / len(step_speeds) > 0.45:
            slow.append(trajectory_id)
    return {
        "endpoint_attractor_count": cluster_count(labels),
        "endpoint_noise_count": labels.count(-1),
        "stable_fixed_point_candidates": len(fixed),
        "limit_cycle_candidates": len(cycles),
        "slow_manifold_candidates": len(slow),
        "attractor_sizes": {str(label): labels.count(label) for label in sorted(set(labels)) if label >= 0},
        "examples": {
            "fixed": fixed[:20],
            "cycles": cycles[:20],
            "slow": slow[:20],
        },
    }


def coordinate_independence(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    values = observable_matrix(states)
    scaled, _, _ = standardize(values)
    embeddings = {
        "pca": coords[:, :3],
        "random_projection": random_projection(scaled, 3, seed=81),
        "autoencoder_linear_proxy": np.tanh(coords[:, :3]),
    }
    base_speeds = trajectory_speeds(states, embeddings["pca"])
    report = {}
    for name, embedding in embeddings.items():
        speeds = trajectory_speeds(states, embedding)
        report[name] = {
            "speed_correlation_with_pca": correlation(base_speeds, speeds),
            "mean_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
        }
    sample_size = min(500, len(states))
    sample_indices = np.linspace(0, len(states) - 1, sample_size, dtype=int)
    sample_scaled = scaled[sample_indices]
    sample_pca = coords[sample_indices, :3]
    for name, embedding in {
        "diffusion_map_proxy": diffusion_map(sample_scaled, 3),
        "spectral_embedding_proxy": spectral_embedding(sample_scaled, 3),
    }.items():
        report[name] = {
            "sampled": True,
            "sample_size": int(sample_size),
            "distance_correlation_with_pca": distance_correlation(sample_pca, embedding),
        }
    return report


def trajectory_speeds(states: list[FlowState], coords: np.ndarray) -> np.ndarray:
    values = []
    for indices in group_indices_by_trajectory(states).values():
        for left, right in zip(indices, indices[1:]):
            values.append(float(np.linalg.norm(coords[right] - coords[left])))
    return np.array(values, dtype=float)


def random_projection(values: np.ndarray, components: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(values.shape[1], components))
    return values @ projection / math.sqrt(values.shape[1])


def diffusion_map(values: np.ndarray, components: int) -> np.ndarray:
    sample = values
    distances = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2)
    epsilon = float(np.median(distances)) or 1.0
    kernel = np.exp(-distances / epsilon)
    kernel = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-9)
    eigvals, eigvecs = np.linalg.eig(kernel)
    order = np.argsort(-np.real(eigvals))
    return np.real(eigvecs[:, order[1 : components + 1]])


def spectral_embedding(values: np.ndarray, components: int) -> np.ndarray:
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
    sigma = float(np.median(distances)) or 1.0
    affinity = np.exp(-distances / sigma)
    degree = np.diag(affinity.sum(axis=1))
    laplacian = degree - affinity
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    return eigvecs[:, order[1 : components + 1]]


def invariant_report(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    values = observable_matrix(states)
    deltas = []
    levels = []
    for indices in group_indices_by_trajectory(states).values():
        for left, right in zip(indices, indices[1:]):
            deltas.append(values[right] - values[left])
            levels.append(values[left])
    delta_array = np.array(deltas, dtype=float)
    _, _, vt = np.linalg.svd(delta_array, full_matrices=False)
    candidates = []
    for vector in vt[-5:]:
        projected_delta = delta_array @ vector
        projected_level = np.array(levels) @ vector
        candidates.append(
            {
                "drift_std": float(np.std(projected_delta)),
                "drift_mean_abs": float(np.mean(np.abs(projected_delta))),
                "level_std": float(np.std(projected_level)),
                "terms": [
                    {"observable": observable, "weight": float(weight)}
                    for observable, weight in sorted(zip(OBSERVABLES, vector), key=lambda item: -abs(item[1]))[:8]
                ],
            }
        )
    monotone = []
    for index, observable in enumerate(OBSERVABLES):
        signs = []
        for indices in group_indices_by_trajectory(states).values():
            series = values[indices, index]
            diffs = np.diff(series)
            if len(diffs):
                signs.append(max(np.mean(diffs >= 0), np.mean(diffs <= 0)))
        monotone.append({"observable": observable, "monotonicity": _mean([float(x) for x in signs])})
    return {
        "approx_conserved_quantities": sorted(candidates, key=lambda item: item["drift_mean_abs"])[:5],
        "monotone_candidates": sorted(monotone, key=lambda item: -item["monotonicity"])[:10],
    }


def empirical_law(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    xs = []
    ys = []
    for indices in group_indices_by_trajectory(states).values():
        for left, right in zip(indices, indices[1:]):
            xs.append(np.r_[coords[left, :4], 1.0])
            ys.append(coords[right, :4] - coords[left, :4])
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    smoothness = 1.0 / (1.0 + float(np.linalg.norm(coef[:-1])))
    return {
        "form": "x_next - x = A x + b in first 4 PCA coordinates",
        "predictive_r2": r2,
        "smoothness_proxy": smoothness,
        "stability_spectral_radius": float(max(abs(np.linalg.eigvals(np.eye(4) + coef[:-1, :].T)))),
        "coefficients": coef.tolist(),
    }


def meta_dynamics_report(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    world_vectors: dict[str, list[np.ndarray]] = {}
    rule_vectors: dict[str, list[np.ndarray]] = {}
    for indices in group_indices_by_trajectory(states).values():
        for left, right in zip(indices, indices[1:]):
            vec = coords[right, :3] - coords[left, :3]
            world_vectors.setdefault(states[left].world, []).append(vec)
            rule_vectors.setdefault(states[left].update_rule, []).append(vec)
    world_means = {key: np.mean(values, axis=0) for key, values in world_vectors.items()}
    rule_means = {key: np.mean(values, axis=0) for key, values in rule_vectors.items()}
    return {
        "world_vector_distances": pairwise_distances(world_means),
        "rule_vector_distances": pairwise_distances(rule_means),
        "mean_world_distance": _mean(list(pairwise_distances(world_means).values())),
        "mean_rule_distance": _mean(list(pairwise_distances(rule_means).values())),
    }


def pairwise_distances(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    keys = sorted(vectors)
    result = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            result[f"{left}|{right}"] = float(np.linalg.norm(vectors[left] - vectors[right]))
    return result


def flow_topology(states: list[FlowState], coords: np.ndarray) -> dict[str, object]:
    endpoint_coords = []
    endpoint_labels = []
    for trajectory_id, indices in group_indices_by_trajectory(states).items():
        endpoint_coords.append(coords[indices[-1], :3])
        endpoint_labels.append(trajectory_id)
    endpoint_coords = np.array(endpoint_coords, dtype=float)
    scaled, _, _ = standardize(endpoint_coords)
    labels = dbscan(scaled, eps=0.9, min_samples=4)
    return {
        "basin_count": cluster_count(labels),
        "basin_noise_fraction": labels.count(-1) / len(labels) if labels else 0.0,
        "basin_sizes": {str(label): labels.count(label) for label in sorted(set(labels)) if label >= 0},
    }


def confidence_estimates(states: list[FlowState], coords: np.ndarray, seed: int) -> dict[str, object]:
    rng = random.Random(seed + 777)
    indices = list(group_indices_by_trajectory(states).values())
    base_speeds = trajectory_speeds(states, coords[:, :3])
    scores = []
    for trial in range(16):
        keep = set(rng.sample(range(len(indices)), max(3, int(0.65 * len(indices)))))
        selected_ids = {list(group_indices_by_trajectory(states))[i] for i in keep}
        selected_states = [state for state in states if state.trajectory_id in selected_ids]
        selected_coords = coords[[i for i, state in enumerate(states) if state.trajectory_id in selected_ids], :3]
        speeds = trajectory_speeds(selected_states, selected_coords)
        if len(speeds) and len(base_speeds):
            scores.append(float(np.mean(speeds)))
    return {
        "speed_bootstrap_mean": _mean(scores),
        "speed_bootstrap_std": float(np.std(scores)) if scores else 0.0,
    }


def hypothesis_support(field: dict[str, object], law: dict[str, object], coord: dict[str, object], meta: dict[str, object], attractors: dict[str, object]) -> dict[str, float]:
    coord_scores = []
    for item in coord.values():
        if "speed_correlation_with_pca" in item and not math.isnan(item["speed_correlation_with_pca"]):
            coord_scores.append(item["speed_correlation_with_pca"])
        if "distance_correlation_with_pca" in item and not math.isnan(item["distance_correlation_with_pca"]):
            coord_scores.append(item["distance_correlation_with_pca"])
    coord_invariance = _mean(coord_scores)
    predictive = max(0.0, law["predictive_r2"])
    rule_distance = meta["mean_rule_distance"]
    world_distance = meta["mean_world_distance"]
    attractor_signal = min(1.0, attractors["endpoint_attractor_count"] / 8.0)
    coherent = max(0.0, 0.35 * coord_invariance + 0.35 * predictive + 0.15 * attractor_signal + 0.15 * (1.0 / (1.0 + rule_distance)))
    dependent = max(0.0, 0.45 * (1.0 - predictive) + 0.25 * (1.0 - coord_invariance) + 0.20 * min(1.0, rule_distance) + 0.10 * min(1.0, world_distance))
    total = coherent + dependent
    if total == 0:
        return {"no_coherent_flow": 0.5, "stable_vector_field": 0.5}
    return {"no_coherent_flow": dependent / total, "stable_vector_field": coherent / total}


def flow_report(initial_conditions: int, steps: int, seed: int, out_dir: Path) -> dict[str, object]:
    states = generate_trajectories(initial_conditions, steps, seed)
    latent = latent_manifold(states)
    coords = np.array(latent["coords"], dtype=float)
    field = vector_field(states, coords)
    attractors = attractor_atlas(states, coords, field)
    coord = coordinate_independence(states, coords)
    invariants = invariant_report(states, coords)
    law = empirical_law(states, coords)
    meta = meta_dynamics_report(states, coords)
    topology = flow_topology(states, coords)
    confidence = confidence_estimates(states, coords, seed)
    hypotheses = hypothesis_support(field, law, coord, meta, attractors)
    coord_scores = [
        score
        for item in coord.values()
        for score in (
            ([item["speed_correlation_with_pca"]] if "speed_correlation_with_pca" in item else [])
            + ([item["distance_correlation_with_pca"]] if "distance_correlation_with_pca" in item else [])
        )
        if not math.isnan(score)
    ]
    summary = {
        "states": len(states),
        "trajectories": len(group_indices_by_trajectory(states)),
        "intrinsic_dimension_85pct": latent["intrinsic_dimension_85pct"],
        "mean_speed": field["mean_speed"],
        "endpoint_attractor_count": attractors["endpoint_attractor_count"],
        "limit_cycle_candidates": attractors["limit_cycle_candidates"],
        "empirical_law_r2": law["predictive_r2"],
        "coordinate_invariance_mean": _mean(coord_scores),
        "hypothesis_support": hypotheses,
    }
    write_outputs(out_dir, states, coords, latent, field, attractors, coord, invariants, law, meta, topology, confidence, summary)
    return summary


def write_outputs(
    out_dir: Path,
    states: list[FlowState],
    coords: np.ndarray,
    latent: dict[str, object],
    field: dict[str, object],
    attractors: dict[str, object],
    coord: dict[str, object],
    invariants: dict[str, object],
    law: dict[str, object],
    meta: dict[str, object],
    topology: dict[str, object],
    confidence: dict[str, object],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory_catalogue(out_dir / "trajectory_catalogue.csv", states, coords)
    write_vector_field(out_dir / "reconstructed_vector_field.csv", field["rows"])
    latent_public = {key: value for key, value in latent.items() if key != "coords"}
    (out_dir / "latent_manifold.json").write_text(json.dumps(latent_public, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "attractor_atlas.json").write_text(json.dumps(attractors, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "coordinate_independence_analysis.json").write_text(json.dumps(coord, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "invariant_report.json").write_text(json.dumps(invariants, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "empirical_dynamical_law.json").write_text(json.dumps(law, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta_dynamics_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "flow_topology.json").write_text(json.dumps(topology, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "confidence_estimates.json").write_text(json.dumps(confidence, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "epistemic_flow_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_trajectory_catalogue(path: Path, states: list[FlowState], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["trajectory_id", "step", "update_rule", "world", "paradigm", "x", "y", "z", *PARAMETERS, *OBSERVABLES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state, coord in zip(states, coords):
            row = {
                "trajectory_id": state.trajectory_id,
                "step": state.step,
                "update_rule": state.update_rule,
                "world": state.world,
                "paradigm": state.paradigm,
                "x": coord[0],
                "y": coord[1],
                "z": coord[2] if len(coord) > 2 else 0.0,
            }
            row.update(state.parameters)
            row.update(state.observables)
            writer.writerow(row)


def write_vector_field(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["trajectory_id", "step", "update_rule", "world", "x", "y", "z", "dx", "dy", "dz", "speed"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    left_dist = []
    right_dist = []
    for index in range(len(left)):
        for j in range(index + 1, len(left)):
            left_dist.append(float(np.linalg.norm(left[index] - left[j])))
            right_dist.append(float(np.linalg.norm(right[index] - right[j])))
    return correlation(np.array(left_dist, dtype=float), np.array(right_dist, dtype=float))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
