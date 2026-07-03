from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from epistemic_engine.epistemic_flow import FlowState, OBSERVABLES, generate_trajectories
from epistemic_engine.natural_coordinates import quadratic_features, r2_score, transition_pairs
from epistemic_engine.natural_coordinates_validation import (
    FROZEN_DIMENSION,
    FrozenKoopmanCoordinates,
    rows_from_states,
    transform,
)


NOVEL_WORLDS = (
    "adversarial_world",
    "changing_rules_world",
    "delayed_consequence_world",
    "deceptive_regularity_world",
    "hierarchical_environment",
    "sparse_observation_world",
    "continuous_environment",
    "partially_contradictory_world",
    "self_modifying_world",
    "no_reusable_structure_world",
)

NOVEL_ARCHITECTURES = (
    "reservoir_symbolic_hybrid",
    "causal_program_synthesizer",
    "neural_constraint_solver",
    "market_of_experts",
    "proof_search_memory_lattice",
    "self_rewriting_graph_machine",
)

NOVEL_LEARNING_RULES = (
    "reinforcement_like",
    "evolutionary",
    "bayesian",
    "symbolic_theorem_proving",
    "constraint_satisfaction",
    "message_passing",
    "differentiable_optimization_ood",
    "random_search",
    "hybrid_system",
)


@dataclass(frozen=True)
class FrozenAffineLaw:
    coef: list[list[float]]

    def predict(self, coords: np.ndarray) -> np.ndarray:
        features = np.hstack([coords, np.ones((len(coords), 1))])
        return features @ np.array(self.coef, dtype=float)


@dataclass(frozen=True)
class OODCaseResult:
    case_id: str
    world: str
    architecture: str
    update_rule: str
    properties: dict[str, float | str | bool]
    transition_count: int
    frozen_affine_r2: float
    frozen_affine_rmse: float
    mean_prediction_error: float
    p95_prediction_error: float
    max_coordinate_norm: float
    failure_mode: str


def ood_validation_report(
    out_dir: Path,
    frozen_path: Path,
    cases_per_family: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    frozen = load_frozen_coordinates(frozen_path)
    reference_states = generate_trajectories(initial_conditions=4, steps=8, seed=29)
    reference_coords = transform(reference_states, frozen)
    reference_pairs = transition_pairs(rows_from_states(reference_states))
    affine_law = fit_affine_law(reference_coords, reference_pairs)

    ood_states = generate_ood_suite(cases_per_family=cases_per_family, steps=steps, seed=seed)
    case_results, transition_rows = evaluate_blind_predictions(ood_states, frozen, affine_law)
    failure_atlas = build_failure_atlas(case_results)
    boundary = applicability_boundary(case_results)
    competing = competing_model_comparison(reference_states, ood_states, frozen, affine_law)
    confidence = confidence_intervals(case_results, seed + 100)
    robustness = robustness_report(case_results, failure_atlas, boundary, competing)
    assessment = scientific_assessment(case_results, boundary, competing, confidence)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_rule_manifest.json", frozen_rule_manifest(frozen_path, affine_law))
    write_json(out_dir / "ood_benchmark_suite.json", suite_manifest(ood_states))
    write_csv(out_dir / "prediction_accuracy_distribution.csv", transition_rows)
    write_json(out_dir / "case_results.json", [asdict(result) for result in case_results])
    write_json(out_dir / "failure_atlas.json", failure_atlas)
    write_json(out_dir / "applicability_boundary.json", boundary)
    write_json(out_dir / "competing_model_comparison.json", competing)
    write_json(out_dir / "confidence_intervals.json", confidence)
    write_json(out_dir / "robustness_report.json", robustness)
    write_json(out_dir / "scientific_assessment.json", assessment)

    return {
        "frozen_model": "2d_koopman_quadratic_affine",
        "rule_0": "satisfied: frozen coordinates and affine law are never updated on OOD data",
        "cases": len(case_results),
        "transitions": len(transition_rows),
        "prediction_accuracy": distribution_summary([result.frozen_affine_r2 for result in case_results]),
        "failure_atlas": failure_atlas["summary"],
        "applicability_boundary": boundary["summary"],
        "competing_models": competing["summary"],
        "confidence_intervals": confidence,
        "scientific_assessment": assessment,
    }


def load_frozen_coordinates(path: Path) -> FrozenKoopmanCoordinates:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen coordinate system is required by Rule 0 and was not found: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FrozenKoopmanCoordinates(**payload)


def fit_affine_law(coords: np.ndarray, pairs: list[tuple[int, int]]) -> FrozenAffineLaw:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    features = np.hstack([x, np.ones((len(x), 1))])
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return FrozenAffineLaw(coef=coef.tolist())


def generate_ood_suite(cases_per_family: int, steps: int, seed: int) -> list[FlowState]:
    rng = random.Random(seed)
    states: list[FlowState] = []
    case_index = 0
    for world in NOVEL_WORLDS:
        for architecture in NOVEL_ARCHITECTURES:
            for update_rule in NOVEL_LEARNING_RULES:
                for replicate in range(cases_per_family):
                    case_index += 1
                    trajectory_id = f"ood_{case_index:05d}_{world}_{architecture}_{update_rule}_{replicate}"
                    latent = initial_ood_latent(rng, world, architecture, update_rule)
                    memory = np.zeros_like(latent)
                    architecture_weights = architecture_observation_weights(architecture)
                    for step in range(steps):
                        observables = ood_observables(
                            latent,
                            memory,
                            step,
                            world,
                            architecture,
                            update_rule,
                            architecture_weights,
                            rng,
                        )
                        states.append(
                            FlowState(
                                trajectory_id=trajectory_id,
                                step=step,
                                update_rule=update_rule,
                                world=world,
                                paradigm=architecture,
                                parameters=ood_properties(world, architecture, update_rule),
                                observables=observables,
                            )
                        )
                        next_latent = update_ood_latent(latent, memory, step, world, update_rule, rng)
                        memory = 0.65 * memory + 0.35 * (next_latent - latent)
                        latent = next_latent
    return states


def initial_ood_latent(rng: random.Random, world: str, architecture: str, update_rule: str) -> np.ndarray:
    base = np.array([rng.uniform(-0.6, 0.6) for _ in range(6)], dtype=float)
    if "hierarchical" in world:
        base += np.array([0.2, -0.1, 0.5, 0.4, -0.2, 0.3])
    if "self_rewriting" in architecture:
        base += np.array([-0.2, 0.4, -0.1, 0.3, 0.5, -0.3])
    if update_rule == "bayesian":
        base *= 0.7
    return base


def update_ood_latent(
    latent: np.ndarray,
    memory: np.ndarray,
    step: int,
    world: str,
    update_rule: str,
    rng: random.Random,
) -> np.ndarray:
    x = latent.copy()
    delta = np.zeros_like(x)
    if update_rule == "reinforcement_like":
        delta += 0.07 * np.tanh(np.roll(x, 1)) + 0.04 * np.sign(np.sin(step + x))
    elif update_rule == "evolutionary":
        delta += np.array([rng.gauss(0.0, 0.06) for _ in x]) + 0.03 * np.sign(x)
    elif update_rule == "bayesian":
        delta += 0.08 * (np.tanh(x.mean()) - x)
    elif update_rule == "symbolic_theorem_proving":
        delta += 0.05 * np.sign(np.roll(x, 2) - x)
    elif update_rule == "constraint_satisfaction":
        delta += 0.08 * (np.clip(np.round(x), -1, 1) - x)
    elif update_rule == "message_passing":
        delta += 0.06 * (np.roll(x, 1) + np.roll(x, -1) - 2 * x)
    elif update_rule == "differentiable_optimization_ood":
        delta += -0.07 * x + 0.04 * np.cos(step / 3.0 + x)
    elif update_rule == "random_search":
        delta += np.array([rng.uniform(-0.12, 0.12) for _ in x])
    elif update_rule == "hybrid_system":
        delta += 0.04 * np.tanh(np.roll(x, 1)) - 0.04 * x + 0.03 * np.sign(np.sin(2 * x))

    if world == "adversarial_world":
        delta -= 0.11 * np.sign(x + 0.01)
    elif world == "changing_rules_world" and step >= 4:
        delta *= -1.4
    elif world == "delayed_consequence_world":
        delta += 0.12 * memory
    elif world == "deceptive_regularity_world":
        delta += 0.08 * np.sin(step / 2.0) - 0.04 * np.sin(step)
    elif world == "hierarchical_environment":
        delta[2:] += 0.06 * np.tanh(x[:4])
    elif world == "sparse_observation_world":
        delta += 0.03 * np.tanh(x)
    elif world == "continuous_environment":
        delta += 0.05 * np.array([math.sin(step / (i + 2)) for i in range(len(x))])
    elif world == "partially_contradictory_world":
        delta[::2] += 0.08
        delta[1::2] -= 0.08
    elif world == "self_modifying_world":
        delta += (0.03 + 0.02 * step) * np.tanh(np.roll(x, step % len(x)))
    elif world == "no_reusable_structure_world":
        delta += np.array([rng.gauss(0.0, 0.18) for _ in x])
    return np.clip(x + delta, -2.5, 2.5)


def architecture_observation_weights(architecture: str) -> np.ndarray:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(architecture))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.55, size=(len(OBSERVABLES), 6))


def ood_observables(
    latent: np.ndarray,
    memory: np.ndarray,
    step: int,
    world: str,
    architecture: str,
    update_rule: str,
    weights: np.ndarray,
    rng: random.Random,
) -> dict[str, float]:
    raw = weights @ latent
    raw += 0.25 * np.sin(weights @ (latent + memory))
    raw += 0.08 * np.array([math.sin(step / (i + 2)) for i in range(len(OBSERVABLES))])
    if architecture in {"neural_constraint_solver", "proof_search_memory_lattice"}:
        raw += 0.18 * np.tanh(raw)
    if architecture == "market_of_experts":
        raw += 0.12 * np.sign(raw)
    if update_rule in {"random_search", "evolutionary"}:
        raw += np.array([rng.gauss(0.0, 0.08) for _ in raw])
    values = 1.0 / (1.0 + np.exp(-raw))
    if world == "sparse_observation_world":
        mask = np.array([(index + step) % 3 == 0 for index in range(len(values))])
        values[mask] = 0.0
    if world == "partially_contradictory_world":
        values[::4] = 1.0 - values[::4]
    if world == "no_reusable_structure_world":
        values = np.array([rng.random() for _ in values])
    return {name: float(value) for name, value in zip(OBSERVABLES, values)}


def ood_properties(world: str, architecture: str, update_rule: str) -> dict[str, float | str | bool]:
    return {
        "world_family": world,
        "architecture_family": architecture,
        "learning_rule_family": update_rule,
        "has_rule_change": world in {"changing_rules_world", "self_modifying_world"},
        "has_delay": world == "delayed_consequence_world",
        "sparse_observations": world == "sparse_observation_world",
        "contradictory": world == "partially_contradictory_world",
        "no_reusable_structure": world == "no_reusable_structure_world",
    }


def evaluate_blind_predictions(
    states: list[FlowState],
    frozen: FrozenKoopmanCoordinates,
    affine_law: FrozenAffineLaw,
) -> tuple[list[OODCaseResult], list[dict[str, object]]]:
    coords = transform(states, frozen)
    pairs = transition_pairs(rows_from_states(states))
    predictions = affine_law.predict(np.array([coords[i] for i, _ in pairs], dtype=float))
    actual = np.array([coords[j] for _, j in pairs], dtype=float)
    pair_errors = np.linalg.norm(predictions - actual, axis=1)
    pair_by_left = {left: index for index, (left, _) in enumerate(pairs)}
    transition_rows: list[dict[str, object]] = []
    for pair_index, (left, right) in enumerate(pairs):
        state = states[left]
        transition_rows.append(
            {
                "trajectory_id": state.trajectory_id,
                "step": state.step,
                "world": state.world,
                "architecture": state.paradigm,
                "update_rule": state.update_rule,
                "z1": float(coords[left, 0]),
                "z2": float(coords[left, 1]),
                "pred_z1": float(predictions[pair_index, 0]),
                "pred_z2": float(predictions[pair_index, 1]),
                "actual_z1": float(coords[right, 0]),
                "actual_z2": float(coords[right, 1]),
                "prediction_error": float(pair_errors[pair_index]),
            }
        )

    case_results = []
    by_traj: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        by_traj.setdefault(state.trajectory_id, []).append(index)
    for trajectory_id, indices in sorted(by_traj.items()):
        local_pairs = [(i, i + 1) for i in indices[:-1] if i in pair_by_left]
        pair_indices = [pair_by_left[i] for i, _ in local_pairs]
        if not pair_indices:
            continue
        y = actual[pair_indices]
        pred = predictions[pair_indices]
        errors = pair_errors[pair_indices]
        first = states[indices[0]]
        case_results.append(
            OODCaseResult(
                case_id=trajectory_id,
                world=first.world,
                architecture=first.paradigm,
                update_rule=first.update_rule,
                properties=first.parameters,
                transition_count=len(pair_indices),
                frozen_affine_r2=r2_score(y, pred),
                frozen_affine_rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
                mean_prediction_error=float(np.mean(errors)),
                p95_prediction_error=float(np.quantile(errors, 0.95)),
                max_coordinate_norm=float(np.max(np.linalg.norm(coords[indices], axis=1))),
                failure_mode=classify_failure_mode(first, coords[indices], errors),
            )
        )
    return case_results, transition_rows


def classify_failure_mode(state: FlowState, coords: np.ndarray, errors: np.ndarray) -> str:
    if state.world == "no_reusable_structure_world":
        return "no_reusable_structure"
    if state.world == "sparse_observation_world":
        return "sparse_observation_collapse"
    if float(np.max(np.linalg.norm(coords, axis=1))) > 10.0:
        return "coordinate_extrapolation"
    if state.world in {"changing_rules_world", "self_modifying_world"} and float(np.quantile(errors, 0.95)) > 2.0:
        return "nonstationary_dynamics"
    if state.world == "delayed_consequence_world":
        return "delayed_state_dependency"
    if state.world == "partially_contradictory_world":
        return "contradictory_observations"
    if curvature(coords) > 1.7:
        return "nonlinear_curvature"
    if float(np.mean(errors)) > 1.0:
        return "large_affine_residual"
    return "no_major_failure"


def curvature(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    angles = []
    for a, b, c in zip(coords, coords[1:], coords[2:]):
        v1 = b - a
        v2 = c - b
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 0:
            angles.append(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))))
    return float(np.mean(angles)) if angles else 0.0


def build_failure_atlas(results: list[OODCaseResult]) -> dict[str, object]:
    failures = [result for result in results if result.failure_mode != "no_major_failure" or result.frozen_affine_r2 < 0.65]
    return {
        "summary": {
            "cases": len(results),
            "failures": len(failures),
            "failure_rate": len(failures) / max(1, len(results)),
            "failure_modes": count_by(results, "failure_mode"),
        },
        "by_world": group_stats(results, "world"),
        "by_architecture": group_stats(results, "architecture"),
        "by_update_rule": group_stats(results, "update_rule"),
        "worst_cases": [asdict(item) for item in sorted(results, key=lambda row: row.frozen_affine_r2)[:30]],
    }


def applicability_boundary(results: list[OODCaseResult], threshold: float = 0.65) -> dict[str, object]:
    axes = {
        "world": group_stats(results, "world"),
        "architecture": group_stats(results, "architecture"),
        "update_rule": group_stats(results, "update_rule"),
    }
    works = {}
    breaks = {}
    for axis, rows in axes.items():
        works[axis] = [name for name, data in rows.items() if data["mean_r2"] >= threshold and data["failure_rate"] <= 0.35]
        breaks[axis] = [name for name, data in rows.items() if data["mean_r2"] < threshold or data["failure_rate"] > 0.35]
    return {
        "threshold": threshold,
        "summary": {
            "works": works,
            "breaks": breaks,
        },
        "axis_statistics": axes,
    }


def competing_model_comparison(
    reference_states: list[FlowState],
    ood_states: list[FlowState],
    frozen: FrozenKoopmanCoordinates,
    affine_law: FrozenAffineLaw,
) -> dict[str, object]:
    ref_coords = transform(reference_states, frozen)
    ood_coords = transform(ood_states, frozen)
    ref_pairs = transition_pairs(rows_from_states(reference_states))
    ood_pairs = transition_pairs(rows_from_states(ood_states))
    models: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "frozen_2d_affine": affine_law.predict,
        "frozen_2d_nonlinear_quadratic": fit_feature_model(ref_coords, ref_pairs, lambda x: np.hstack([x, quadratic_features(x), np.ones((len(x), 1))])),
        "frozen_2d_recurrent_delta": fit_recurrent_delta_model(ref_coords, reference_states),
    }
    ref_values = observable_values(reference_states)
    ood_values = observable_values(ood_states)
    higher = fit_higher_dim_koopman(reference_states, ref_values, dim=4)
    higher_coords = transform_with_higher_dim(ood_values, higher)
    higher_ref_coords = transform_with_higher_dim(ref_values, higher)
    models_on_coords = evaluate_models_on_coords(models, ood_coords, ood_pairs)
    models_on_coords["discovery_trained_4d_affine"] = eval_affine_from_train(higher_ref_coords, ref_pairs, higher_coords, ood_pairs)
    models_on_coords["graph_latent_proxy"] = graph_latent_proxy_score(ood_coords, ood_pairs, ood_states)
    best_name = max(models_on_coords, key=lambda name: models_on_coords[name]["r2"])
    return {
        "summary": {
            "best_model": best_name,
            "frozen_2d_affine_r2": models_on_coords["frozen_2d_affine"]["r2"],
            "best_r2": models_on_coords[best_name]["r2"],
            "frozen_rank": sorted(models_on_coords, key=lambda name: models_on_coords[name]["r2"], reverse=True).index("frozen_2d_affine") + 1,
        },
        "models": models_on_coords,
        "rule": "all competing models are trained on discovery reference data or are fixed evaluators; no model is fit on OOD targets",
    }


def fit_feature_model(coords: np.ndarray, pairs: list[tuple[int, int]], featurize: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    coef, *_ = np.linalg.lstsq(featurize(x), y, rcond=None)
    return lambda new_x: featurize(new_x) @ coef


def fit_recurrent_delta_model(coords: np.ndarray, states: list[FlowState]) -> Callable[[np.ndarray], np.ndarray]:
    rows = []
    targets = []
    by_traj = group_indices_by_trajectory(states)
    for indices in by_traj.values():
        for prev_i, i, j in zip(indices, indices[1:], indices[2:]):
            delta = coords[i] - coords[prev_i]
            rows.append(np.concatenate([coords[i], delta, [1.0]]))
            targets.append(coords[j])
    coef, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)

    def predict(new_x: np.ndarray) -> np.ndarray:
        deltas = np.vstack([np.zeros((1, new_x.shape[1])), np.diff(new_x, axis=0)])
        features = np.hstack([new_x, deltas, np.ones((len(new_x), 1))])
        return features @ coef

    return predict


def evaluate_models_on_coords(
    models: dict[str, Callable[[np.ndarray], np.ndarray]],
    coords: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    rows = {}
    for name, model in models.items():
        pred = model(x)
        rows[name] = {
            "r2": r2_score(y, pred),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        }
    return rows


def observable_values(states: list[FlowState]) -> np.ndarray:
    return np.array([[state.observables[name] for name in OBSERVABLES] for state in states], dtype=float)


def fit_higher_dim_koopman(states: list[FlowState], values: np.ndarray, dim: int) -> dict[str, object]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (values - mean) / std
    features = quadratic_features(scaled[:, : min(10, scaled.shape[1])])
    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return {
        "observable_mean": mean,
        "observable_std": std,
        "feature_mean": feature_mean,
        "components": vt[:dim],
        "states": len(states),
    }


def transform_with_higher_dim(values: np.ndarray, model: dict[str, object]) -> np.ndarray:
    scaled = (values - model["observable_mean"]) / model["observable_std"]
    features = quadratic_features(scaled[:, : min(10, scaled.shape[1])])
    return (features - model["feature_mean"]) @ model["components"].T


def eval_affine_from_train(
    train_coords: np.ndarray,
    train_pairs: list[tuple[int, int]],
    test_coords: np.ndarray,
    test_pairs: list[tuple[int, int]],
) -> dict[str, float]:
    law = fit_affine_law(train_coords, train_pairs)
    x = np.array([test_coords[i] for i, _ in test_pairs], dtype=float)
    y = np.array([test_coords[j] for _, j in test_pairs], dtype=float)
    pred = law.predict(x)
    return {"r2": r2_score(y, pred), "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}


def graph_latent_proxy_score(coords: np.ndarray, pairs: list[tuple[int, int]], states: list[FlowState]) -> dict[str, float]:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    pred = np.array(x, copy=True)
    by_key: dict[tuple[str, str], list[int]] = {}
    for pair_index, (i, _) in enumerate(pairs):
        state = states[i]
        by_key.setdefault((state.world, state.update_rule), []).append(pair_index)
    for indices in by_key.values():
        centroid = np.mean(x[indices], axis=0)
        pred[indices] = 0.82 * x[indices] + 0.18 * centroid
    return {
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "note": "fixed graph-family message-passing evaluator; uses OOD graph grouping but no OOD target fitting",
    }


def robustness_report(
    results: list[OODCaseResult],
    failure_atlas: dict[str, object],
    boundary: dict[str, object],
    competing: dict[str, object],
) -> dict[str, object]:
    r2_values = [result.frozen_affine_r2 for result in results]
    return {
        "frozen_theory": "2D latent coordinates with affine dynamics",
        "ood_case_count": len(results),
        "r2_distribution": distribution_summary(r2_values),
        "failure_rate": failure_atlas["summary"]["failure_rate"],
        "works_on_worlds": boundary["summary"]["works"]["world"],
        "breaks_on_worlds": boundary["summary"]["breaks"]["world"],
        "competing_model_best": competing["summary"]["best_model"],
        "frozen_model_rank": competing["summary"]["frozen_rank"],
        "interpretation": robustness_interpretation(results, boundary, competing),
    }


def robustness_interpretation(
    results: list[OODCaseResult],
    boundary: dict[str, object],
    competing: dict[str, object],
) -> str:
    mean_r2 = distribution_summary([result.frozen_affine_r2 for result in results])["mean"]
    break_count = sum(len(items) for items in boundary["summary"]["breaks"].values())
    if mean_r2 >= 0.75 and break_count <= 3 and competing["summary"]["frozen_rank"] <= 2:
        return "The frozen law remains broadly robust under OOD shift."
    if mean_r2 >= 0.45 and boundary["summary"]["works"]["world"]:
        return "The frozen law identifies a restricted universality region, not a global law."
    return "The frozen law is likely an artifact of the original benchmark family under this OOD suite."


def confidence_intervals(results: list[OODCaseResult], seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    values = [result.frozen_affine_r2 for result in results]
    failure_flags = [1.0 if result.frozen_affine_r2 < 0.65 or result.failure_mode != "no_major_failure" else 0.0 for result in results]
    samples = []
    failure_samples = []
    for _ in range(500):
        indices = [rng.randrange(len(results)) for _ in results]
        samples.append(sum(values[index] for index in indices) / len(indices))
        failure_samples.append(sum(failure_flags[index] for index in indices) / len(indices))
    return {
        "mean_r2_95ci": percentile_interval(samples),
        "failure_rate_95ci": percentile_interval(failure_samples),
    }


def scientific_assessment(
    results: list[OODCaseResult],
    boundary: dict[str, object],
    competing: dict[str, object],
    confidence: dict[str, object],
) -> dict[str, object]:
    r2 = distribution_summary([result.frozen_affine_r2 for result in results])
    world_breaks = boundary["summary"]["breaks"]["world"]
    frozen_rank = competing["summary"]["frozen_rank"]
    if r2["mean"] >= 0.75 and r2["p05"] >= 0.50 and len(world_breaks) <= 2 and frozen_rank <= 2:
        verdict = "A. General epistemic law"
    elif r2["mean"] >= 0.40 and boundary["summary"]["works"]["world"]:
        verdict = "B. Law valid only inside one universality region"
    else:
        verdict = "C. Artifact of the original benchmark family"
    return {
        "verdict": verdict,
        "evidence_for_A": {
            "mean_r2": r2["mean"],
            "p05_r2": r2["p05"],
            "frozen_model_rank": frozen_rank,
        },
        "evidence_for_B": {
            "works": boundary["summary"]["works"],
            "breaks": boundary["summary"]["breaks"],
        },
        "evidence_for_C": {
            "failure_rate_ci": confidence["failure_rate_95ci"],
            "best_competing_model": competing["summary"]["best_model"],
            "best_competing_r2": competing["summary"]["best_r2"],
            "frozen_2d_affine_r2": competing["summary"]["frozen_2d_affine_r2"],
        },
        "note": "Success criterion is scope determination, not performance maximization.",
    }


def suite_manifest(states: list[FlowState]) -> dict[str, object]:
    trajectories = sorted({state.trajectory_id for state in states})
    return {
        "trajectory_count": len(trajectories),
        "state_count": len(states),
        "worlds": NOVEL_WORLDS,
        "architectures": NOVEL_ARCHITECTURES,
        "learning_rules": NOVEL_LEARNING_RULES,
        "never_reuses_discovery_worlds": True,
    }


def frozen_rule_manifest(path: Path, law: FrozenAffineLaw) -> dict[str, object]:
    return {
        "coordinate_system_path": str(path),
        "latent_dimension": FROZEN_DIMENSION,
        "coordinate_system": "koopman_quadratic_pca",
        "dynamics": "affine",
        "affine_coef": law.coef,
        "rule_0": {
            "no_retraining": True,
            "no_coordinate_optimization": True,
            "no_parameter_tuning_on_ood": True,
        },
    }


def group_indices_by_trajectory(states: list[FlowState]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        grouped.setdefault(state.trajectory_id, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda item: states[item].step)
    return grouped


def group_stats(results: list[OODCaseResult], field: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[OODCaseResult]] = {}
    for result in results:
        grouped.setdefault(str(getattr(result, field)), []).append(result)
    rows = {}
    for key, items in grouped.items():
        failures = [item for item in items if item.failure_mode != "no_major_failure" or item.frozen_affine_r2 < 0.65]
        rows[key] = {
            "cases": len(items),
            "mean_r2": mean([item.frozen_affine_r2 for item in items]),
            "p05_r2": quantile([item.frozen_affine_r2 for item in items], 0.05),
            "mean_error": mean([item.mean_prediction_error for item in items]),
            "failure_rate": len(failures) / len(items),
        }
    return dict(sorted(rows.items(), key=lambda item: item[1]["mean_r2"]))


def count_by(results: list[OODCaseResult], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        key = str(getattr(result, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def distribution_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": float(np.std(np.array(values, dtype=float))) if values else 0.0,
        "p05": quantile(values, 0.05),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
    }


def percentile_interval(values: list[float]) -> dict[str, float]:
    return {
        "low": quantile(values, 0.025),
        "median": quantile(values, 0.5),
        "high": quantile(values, 0.975),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.array(values, dtype=float), q)) if values else 0.0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
