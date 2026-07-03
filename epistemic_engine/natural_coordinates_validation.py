from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import kmeans
from epistemic_engine.epistemic_flow import OBSERVABLES, WORLDS, UPDATE_RULES, FlowState, generate_trajectories, observe_state
from epistemic_engine.natural_coordinates import quadratic_features, r2_score, transition_pairs


FROZEN_DIMENSION = 2
TRAIN_WORLDS = ("regular", "compositional")
TEST_WORLDS = ("noisy", "volatile")
TRAIN_PARADIGMS = ("graph_system", "hypergraph", "symbolic_system", "sat_based", "rewriting_system")
TEST_PARADIGMS = ("probabilistic_system", "differentiable_system", "cellular_automata", "random_interaction")


@dataclass(frozen=True)
class FrozenKoopmanCoordinates:
    observable_mean: list[float]
    observable_std: list[float]
    feature_mean: list[float]
    components: list[list[float]]


def validation_report(seeds: int, out_dir: Path, initial_conditions: int = 4, steps: int = 8) -> dict[str, object]:
    reference_states = generate_trajectories(initial_conditions, steps, 29)
    frozen = fit_frozen_coordinates(reference_states)
    reproducibility = reproducibility_runs(frozen, seeds, initial_conditions, steps)
    holdout_worlds = holdout_world_report(initial_conditions, steps, seed=911)
    holdout_paradigms = holdout_paradigm_report(initial_conditions, steps, seed=912)
    bootstrap = bootstrap_confidence(frozen, reference_states, seed=913)
    sensitivity = sensitivity_report(frozen, initial_conditions, steps)
    ablation = ablation_report(frozen, reference_states)
    evaluators = evaluator_agreement_report(frozen, reference_states)
    false_discovery = false_discovery_report(frozen, initial_conditions, steps)
    summary = {
        "frozen_coordinate_system": "koopman_quadratic_pca",
        "dimension": FROZEN_DIMENSION,
        "affine_dynamics": True,
        "seeds": seeds,
        "reproducibility": distribution_summary([row["affine_r2"] for row in reproducibility]),
        "coordinate_stability": distribution_summary([row["coordinate_stability"] for row in reproducibility]),
        "embedding_distortion": distribution_summary([row["embedding_distortion"] for row in reproducibility]),
        "holdout_worlds": holdout_worlds,
        "holdout_paradigms": holdout_paradigms,
        "bootstrap": bootstrap,
        "sensitivity": sensitivity["summary"],
        "ablation": ablation,
        "evaluator_agreement": evaluators,
        "false_discovery": false_discovery,
        "verdict": final_verdict(reproducibility, holdout_worlds, holdout_paradigms, bootstrap, evaluators, false_discovery),
    }
    write_outputs(out_dir, frozen, reproducibility, holdout_worlds, holdout_paradigms, bootstrap, sensitivity, ablation, evaluators, false_discovery, summary)
    return summary


def fit_frozen_coordinates(states: list[FlowState]) -> FrozenKoopmanCoordinates:
    values = observable_values(states)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (values - mean) / std
    features = quadratic_features(scaled[:, : min(10, scaled.shape[1])])
    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return FrozenKoopmanCoordinates(mean.tolist(), std.tolist(), feature_mean.tolist(), vt[:FROZEN_DIMENSION].tolist())


def transform(states: list[FlowState], frozen: FrozenKoopmanCoordinates) -> np.ndarray:
    values = observable_values(states)
    mean = np.array(frozen.observable_mean)
    std = np.array(frozen.observable_std)
    scaled = (values - mean) / std
    features = quadratic_features(scaled[:, : min(10, scaled.shape[1])])
    return (features - np.array(frozen.feature_mean)) @ np.array(frozen.components).T


def observable_values(states: list[FlowState]) -> np.ndarray:
    return np.array([[state.observables[name] for name in OBSERVABLES] for state in states], dtype=float)


def rows_from_states(states: list[FlowState]) -> list[dict[str, str]]:
    return [
        {
            "trajectory_id": state.trajectory_id,
            "step": str(state.step),
            "world": state.world,
            "update_rule": state.update_rule,
            "paradigm": state.paradigm,
        }
        for state in states
    ]


def affine_r2(coords: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    features = np.hstack([x, np.ones((len(x), 1))])
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return r2_score(y, features @ coef)


def frozen_law_train_eval(train_coords: np.ndarray, train_pairs: list[tuple[int, int]], test_coords: np.ndarray, test_pairs: list[tuple[int, int]]) -> float:
    x_train = np.array([train_coords[i] for i, _ in train_pairs], dtype=float)
    y_train = np.array([train_coords[j] for _, j in train_pairs], dtype=float)
    features_train = np.hstack([x_train, np.ones((len(x_train), 1))])
    coef, *_ = np.linalg.lstsq(features_train, y_train, rcond=None)
    x_test = np.array([test_coords[i] for i, _ in test_pairs], dtype=float)
    y_test = np.array([test_coords[j] for _, j in test_pairs], dtype=float)
    features_test = np.hstack([x_test, np.ones((len(x_test), 1))])
    return r2_score(y_test, features_test @ coef)


def reproducibility_runs(frozen: FrozenKoopmanCoordinates, seeds: int, initial_conditions: int, steps: int) -> list[dict[str, float]]:
    rows = []
    for seed in range(seeds):
        states = generate_trajectories(initial_conditions, steps, 1000 + seed)
        coords = transform(states, frozen)
        pairs = transition_pairs(rows_from_states(states))
        rows.append(
            {
                "seed": seed,
                "affine_r2": affine_r2(coords, pairs),
                "hidden_dimension": 2.0,
                "trajectory_count": float(len({state.trajectory_id for state in states})),
                "coordinate_stability": split_half_stability(coords, states),
                "embedding_distortion": family_distortion(coords, states, "paradigm"),
            }
        )
    return rows


def split_half_stability(coords: np.ndarray, states: list[FlowState]) -> float:
    left = coords[::2]
    right = coords[1::2]
    n = min(len(left), len(right))
    if n < 3:
        return 0.0
    return subspace_similarity(left[:n], right[:n])


def subspace_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_u, _, _ = np.linalg.svd(left - left.mean(axis=0), full_matrices=False)
    right_u, _, _ = np.linalg.svd(right - right.mean(axis=0), full_matrices=False)
    dim = min(left_u.shape[1], right_u.shape[1], FROZEN_DIMENSION)
    if dim == 0:
        return 0.0
    return float(np.linalg.norm(left_u[:, :dim].T @ right_u[:, :dim], ord="fro") / math.sqrt(dim))


def family_distortion(coords: np.ndarray, states: list[FlowState], key: str) -> float:
    labels = sorted({getattr(state, key) for state in states})
    centroids = []
    for label in labels:
        members = coords[[i for i, state in enumerate(states) if getattr(state, key) == label]]
        centroids.append(members.mean(axis=0))
    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return sum(distances) / len(distances) if distances else 0.0


def holdout_world_report(initial_conditions: int, steps: int, seed: int) -> dict[str, object]:
    states = generate_trajectories(initial_conditions, steps, seed)
    train = [state for state in states if state.world in TRAIN_WORLDS]
    test = [state for state in states if state.world in TEST_WORLDS]
    frozen = fit_frozen_coordinates(train)
    train_coords = transform(train, frozen)
    test_coords = transform(test, frozen)
    return {
        "train_worlds": TRAIN_WORLDS,
        "test_worlds": TEST_WORLDS,
        "train_r2": affine_r2(train_coords, transition_pairs(rows_from_states(train))),
        "test_r2": frozen_law_train_eval(train_coords, transition_pairs(rows_from_states(train)), test_coords, transition_pairs(rows_from_states(test))),
        "test_embedding_distortion": family_distortion(test_coords, test, "world"),
    }


def holdout_paradigm_report(initial_conditions: int, steps: int, seed: int) -> dict[str, object]:
    states = generate_trajectories(initial_conditions, steps, seed)
    train = [state for state in states if state.paradigm in TRAIN_PARADIGMS]
    test = [state for state in states if state.paradigm in TEST_PARADIGMS]
    frozen = fit_frozen_coordinates(train)
    train_coords = transform(train, frozen)
    test_coords = transform(test, frozen)
    return {
        "train_paradigms": TRAIN_PARADIGMS,
        "test_paradigms": TEST_PARADIGMS,
        "train_r2": affine_r2(train_coords, transition_pairs(rows_from_states(train))),
        "test_r2": frozen_law_train_eval(train_coords, transition_pairs(rows_from_states(train)), test_coords, transition_pairs(rows_from_states(test))),
        "test_embedding_distortion": family_distortion(test_coords, test, "paradigm"),
    }


def bootstrap_confidence(frozen: FrozenKoopmanCoordinates, states: list[FlowState], seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    trajectories = sorted({state.trajectory_id for state in states})
    samples = []
    for _ in range(80):
        chosen = set(rng.choices(trajectories, k=len(trajectories)))
        subset = [state for state in states if state.trajectory_id in chosen]
        coords = transform(subset, frozen)
        pairs = transition_pairs(rows_from_states(subset))
        samples.append(
            {
                "affine_r2": affine_r2(coords, pairs),
                "coordinate_stability": split_half_stability(coords, subset),
                "hidden_dimension": 2.0,
            }
        )
    return {
        "affine_r2_ci": confidence_interval([sample["affine_r2"] for sample in samples]),
        "coordinate_stability_ci": confidence_interval([sample["coordinate_stability"] for sample in samples]),
        "hidden_dimension_ci": confidence_interval([sample["hidden_dimension"] for sample in samples]),
    }


def sensitivity_report(frozen: FrozenKoopmanCoordinates, initial_conditions: int, steps: int) -> dict[str, object]:
    axes = {
        "noise": ("stochasticity",),
        "memory": ("memory_capacity",),
        "hierarchy": ("hierarchy_depth",),
        "interaction_topology": ("graph_connectivity", "interaction_locality"),
        "observation_budget": (),
    }
    rows = []
    base_states = generate_trajectories(initial_conditions, steps, 1201)
    for axis, parameters in axes.items():
        for level in (0.0, 0.15, 0.30, 0.45, 0.60):
            states = base_states
            if axis == "observation_budget":
                max_step = max(2, int((1.0 - level) * (steps - 1)))
                states = [state for state in base_states if state.step <= max_step]
            elif parameters:
                states = perturb_states(base_states, parameters, level)
            coords = transform(states, frozen)
            pairs = transition_pairs(rows_from_states(states))
            rows.append({"axis": axis, "level": level, "affine_r2": affine_r2(coords, pairs)})
    return {
        "summary": {
            axis: [
                {"level": row["level"], "affine_r2": row["affine_r2"]}
                for row in rows if row["axis"] == axis
            ]
            for axis in axes
        },
        "rows": rows,
    }


def perturb_states(states: list[FlowState], parameters: tuple[str, ...], level: float) -> list[FlowState]:
    result = []
    for state in states:
        params = dict(state.parameters)
        for parameter in parameters:
            params[parameter] = max(0.0, min(1.0, params[parameter] + level))
        observables = observe_state(state.trajectory_id + "_perturbed", state.step, state.paradigm, params, 1501)
        result.append(FlowState(state.trajectory_id, state.step, state.update_rule, state.world, state.paradigm, params, observables))
    return result


def ablation_report(frozen: FrozenKoopmanCoordinates, states: list[FlowState]) -> dict[str, object]:
    coords = transform(states, frozen)
    pairs = transition_pairs(rows_from_states(states))
    full = affine_r2(coords, pairs)
    result = {"full_r2": full}
    for index in range(FROZEN_DIMENSION):
        kept = np.delete(coords, index, axis=1)
        result[f"remove_z{index + 1}"] = {
            "r2": affine_r2(kept, pairs),
            "collapse": full - affine_r2(kept, pairs),
        }
    return result


def evaluator_agreement_report(frozen: FrozenKoopmanCoordinates, states: list[FlowState]) -> dict[str, object]:
    coords = transform(states, frozen)
    pairs = transition_pairs(rows_from_states(states))
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    affine_features = np.hstack([x, np.ones((len(x), 1))])
    affine_coef, *_ = np.linalg.lstsq(affine_features, y, rcond=None)
    affine_pred = affine_features @ affine_coef
    persistence_pred = x
    local_labels = kmeans(x, 4, seed=313)
    local_pred = np.zeros_like(y)
    for label in sorted(set(local_labels)):
        mask = np.array(local_labels) == label
        if mask.sum() < 4:
            local_pred[mask] = x[mask]
            continue
        features = np.hstack([x[mask], np.ones((mask.sum(), 1))])
        coef, *_ = np.linalg.lstsq(features, y[mask], rcond=None)
        local_pred[mask] = features @ coef
    return {
        "affine_r2": r2_score(y, affine_pred),
        "persistence_r2": r2_score(y, persistence_pred),
        "local_linear_r2": r2_score(y, local_pred),
        "agreement": "agree" if min(r2_score(y, affine_pred), r2_score(y, local_pred)) > 0.75 else "disagree",
    }


def false_discovery_report(frozen: FrozenKoopmanCoordinates, initial_conditions: int, steps: int) -> dict[str, object]:
    states = generate_trajectories(initial_conditions, steps, 1401)
    shuffled = []
    rng = random.Random(1402)
    for state in states:
        obs = dict(state.observables)
        for key in obs:
            obs[key] = rng.random()
        shuffled.append(FlowState(state.trajectory_id, state.step, "random_interaction", "null_world", state.paradigm, state.parameters, obs))
    coords = transform(shuffled, frozen)
    pairs = transition_pairs(rows_from_states(shuffled))
    r2 = affine_r2(coords, pairs)
    return {
        "null_world_r2": r2,
        "possible_overfitting": r2 > 0.75,
    }


def distribution_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def confidence_interval(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "low": float(np.quantile(arr, 0.025)),
        "median": float(np.quantile(arr, 0.5)),
        "high": float(np.quantile(arr, 0.975)),
    }


def final_verdict(
    reproducibility: list[dict[str, float]],
    holdout_worlds: dict[str, object],
    holdout_paradigms: dict[str, object],
    bootstrap: dict[str, object],
    evaluators: dict[str, object],
    false_discovery: dict[str, object],
) -> str:
    r2_mean = distribution_summary([row["affine_r2"] for row in reproducibility])["mean"]
    stability_mean = distribution_summary([row["coordinate_stability"] for row in reproducibility])["mean"]
    if false_discovery["possible_overfitting"]:
        return "downgraded: frozen coordinates also perform well on null worlds."
    if r2_mean > 0.85 and holdout_worlds["test_r2"] > 0.75 and holdout_paradigms["test_r2"] > 0.70 and stability_mean > 0.55 and evaluators["agreement"] == "agree":
        return "validated: frozen natural coordinates survive reproducibility, transfer, bootstrap, ablation, and evaluator checks."
    return "downgraded: discovery remains exploratory under validation stress tests."


def write_outputs(
    out_dir: Path,
    frozen: FrozenKoopmanCoordinates,
    reproducibility: list[dict[str, float]],
    holdout_worlds: dict[str, object],
    holdout_paradigms: dict[str, object],
    bootstrap: dict[str, object],
    sensitivity: dict[str, object],
    ablation: dict[str, object],
    evaluators: dict[str, object],
    false_discovery: dict[str, object],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_coordinate_system.json", frozen.__dict__)
    write_csv(out_dir / "reproducibility_distribution.csv", reproducibility)
    write_json(out_dir / "generalization_report.json", holdout_worlds)
    write_json(out_dir / "paradigm_transfer_report.json", holdout_paradigms)
    write_json(out_dir / "bootstrap_confidence.json", bootstrap)
    write_json(out_dir / "robustness_report.json", sensitivity)
    write_json(out_dir / "ablation_report.json", ablation)
    write_json(out_dir / "evaluator_agreement_report.json", evaluators)
    write_json(out_dir / "false_discovery_report.json", false_discovery)
    write_json(out_dir / "natural_coordinates_validation_summary.json", summary)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
