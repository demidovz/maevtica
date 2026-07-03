from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.epistemic_flow import OBSERVABLES, FlowState, generate_trajectories, observe_state
from epistemic_engine.epistemic_phase_transition import PARAMETERS, sample_parameters
from epistemic_engine.natural_coordinates import quadratic_features, r2_score, transition_pairs
from epistemic_engine.natural_coordinates_validation import (
    FrozenKoopmanCoordinates,
    fit_frozen_coordinates,
    observable_values,
    rows_from_states,
    transform,
)


OOD_WORLDS = (
    "adversarial",
    "changing_rules",
    "delayed_consequences",
    "deceptive_regularities",
    "hierarchical_environment",
    "sparse_observations",
    "continuous_environment",
    "partially_contradictory",
    "self_modifying",
    "no_reusable_structure",
)

OOD_ARCHITECTURES = (
    "reservoir_memory",
    "tensor_factorization",
    "proof_net",
    "neural_cellular",
    "causal_program_synthesizer",
    "energy_based",
    "swarm_protocol",
    "category_rewriter",
)

OOD_RULES = (
    "reinforcement_like",
    "evolutionary",
    "bayesian",
    "symbolic_theorem_proving",
    "constraint_satisfaction",
    "message_passing_ood",
    "differentiable_optimization_ood",
    "random_search",
    "hybrid_system",
)

ARCHITECTURE_TO_PARADIGM = {
    "reservoir_memory": "random_interaction",
    "tensor_factorization": "differentiable_system",
    "proof_net": "symbolic_system",
    "neural_cellular": "cellular_automata",
    "causal_program_synthesizer": "rewriting_system",
    "energy_based": "differentiable_system",
    "swarm_protocol": "graph_system",
    "category_rewriter": "hypergraph",
}


@dataclass(frozen=True)
class FrozenAffineLaw:
    coefficients: list[list[float]]


def ood_validation_report(out_dir: Path, initial_conditions: int = 6, steps: int = 10, seed: int = 41) -> dict[str, object]:
    reference_states = generate_trajectories(initial_conditions, steps, 29)
    frozen = fit_frozen_coordinates(reference_states)
    reference_coords = transform(reference_states, frozen)
    law = fit_affine_law(reference_coords, transition_pairs(rows_from_states(reference_states)))

    suites = []
    for world in OOD_WORLDS:
        states = generate_ood_trajectories(initial_conditions, steps, seed, worlds=(world,), architectures=OOD_ARCHITECTURES[:3], rules=OOD_RULES[:3])
        suites.append(evaluate_suite(f"world::{world}", states, frozen, law))
    for architecture in OOD_ARCHITECTURES:
        states = generate_ood_trajectories(initial_conditions, steps, seed + 100, worlds=OOD_WORLDS[:3], architectures=(architecture,), rules=OOD_RULES[:3])
        suites.append(evaluate_suite(f"architecture::{architecture}", states, frozen, law))
    for rule in OOD_RULES:
        states = generate_ood_trajectories(initial_conditions, steps, seed + 200, worlds=OOD_WORLDS[:3], architectures=OOD_ARCHITECTURES[:3], rules=(rule,))
        suites.append(evaluate_suite(f"rule::{rule}", states, frozen, law))

    all_states = generate_ood_trajectories(initial_conditions, steps, seed + 300, worlds=OOD_WORLDS, architectures=OOD_ARCHITECTURES, rules=OOD_RULES)
    aggregate = evaluate_suite("aggregate_ood", all_states, frozen, law)
    failures = failure_atlas(suites)
    boundary = applicability_boundary(suites)
    competitors = competing_models(reference_states, all_states, frozen)
    confidence = confidence_intervals(suites)
    verdict = scientific_verdict(aggregate, boundary, competitors)
    summary = {
        "frozen_coordinate_system": "koopman_quadratic_pca_dim2",
        "frozen_affine_law": True,
        "ood_suites": len(suites),
        "aggregate": aggregate,
        "prediction_accuracy_distribution": distribution_summary([suite["r2"] for suite in suites]),
        "failure_count": len(failures["failures"]),
        "applicability_boundary": boundary,
        "competing_models": competitors,
        "confidence_intervals": confidence,
        "scientific_verdict": verdict,
    }
    write_outputs(out_dir, frozen, law, suites, aggregate, failures, boundary, competitors, confidence, summary)
    return summary


def fit_affine_law(coords: np.ndarray, pairs: list[tuple[int, int]]) -> FrozenAffineLaw:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    features = np.hstack([x, np.ones((len(x), 1))])
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return FrozenAffineLaw(coef.tolist())


def predict_with_law(coords: np.ndarray, pairs: list[tuple[int, int]], law: FrozenAffineLaw) -> dict[str, object]:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    features = np.hstack([x, np.ones((len(x), 1))])
    pred = features @ np.array(law.coefficients)
    errors = np.linalg.norm(y - pred, axis=1)
    return {
        "r2": r2_score(y, pred),
        "mae": float(np.mean(errors)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "error_series": errors.tolist(),
    }


def generate_ood_trajectories(
    initial_conditions: int,
    steps: int,
    seed: int,
    *,
    worlds: tuple[str, ...],
    architectures: tuple[str, ...],
    rules: tuple[str, ...],
) -> list[FlowState]:
    rng = random.Random(seed)
    states: list[FlowState] = []
    for index in range(initial_conditions):
        base = sample_parameters(index, max(1, initial_conditions), rng)
        for architecture in architectures:
            paradigm = ARCHITECTURE_TO_PARADIGM[architecture]
            for world in worlds:
                for rule in rules:
                    params = initialize_ood_params(base, architecture, world, rng)
                    trajectory_id = f"ood_{index:03d}_{architecture}_{world}_{rule}"
                    for step in range(steps):
                        observables = observe_state(trajectory_id, step, paradigm, params, seed)
                        observables = distort_observables(observables, world, architecture, rule, step, rng)
                        states.append(FlowState(trajectory_id, step, rule, world, paradigm, dict(params), observables))
                        params = update_ood_params(params, world, architecture, rule, step, rng)
    return states


def initialize_ood_params(base: dict[str, float], architecture: str, world: str, rng: random.Random) -> dict[str, float]:
    params = dict(base)
    if architecture in {"reservoir_memory", "proof_net"}:
        params["memory_capacity"] = min(1.0, params["memory_capacity"] + 0.45)
    if architecture in {"tensor_factorization", "energy_based"}:
        params["computational_budget"] = min(1.0, params["computational_budget"] + 0.50)
        params["abstraction_budget"] = min(1.0, params["abstraction_budget"] + 0.25)
    if architecture in {"swarm_protocol", "category_rewriter"}:
        params["graph_connectivity"] = min(1.0, params["graph_connectivity"] + 0.55)
        params["communication_bandwidth"] = min(1.0, params["communication_bandwidth"] + 0.35)
    if world == "no_reusable_structure":
        params["environmental_regularity"] = 0.0
        params["stochasticity"] = 1.0
    return params


def update_ood_params(params: dict[str, float], world: str, architecture: str, rule: str, step: int, rng: random.Random) -> dict[str, float]:
    delta = {name: 0.0 for name in PARAMETERS}
    if rule == "reinforcement_like":
        delta["feedback_strength"] += 0.050
        delta["prediction_horizon"] += 0.020
    elif rule == "evolutionary":
        delta["stochasticity"] += rng.uniform(-0.060, 0.060)
        delta["abstraction_budget"] += 0.025
    elif rule == "bayesian":
        delta["stochasticity"] -= 0.035
        delta["memory_capacity"] += 0.030
    elif rule == "symbolic_theorem_proving":
        delta["constraint_density"] += 0.055
        delta["computational_budget"] += 0.030
    elif rule == "constraint_satisfaction":
        delta["constraint_density"] += 0.070
        delta["interaction_locality"] += 0.020
    elif rule == "message_passing_ood":
        delta["graph_connectivity"] += 0.045
        delta["communication_bandwidth"] += 0.045
    elif rule == "differentiable_optimization_ood":
        delta["computational_budget"] += 0.045
        delta["prediction_horizon"] += 0.035
    elif rule == "random_search":
        for name in PARAMETERS:
            delta[name] += rng.uniform(-0.055, 0.055)
    elif rule == "hybrid_system":
        delta["feedback_strength"] += 0.025
        delta["constraint_density"] += 0.025
        delta["graph_connectivity"] += 0.025

    if world == "adversarial":
        delta["environmental_regularity"] -= 0.055
        delta["stochasticity"] += 0.040
    elif world == "changing_rules":
        delta["feedback_strength"] += 0.08 * math.sin(step)
        delta["environmental_regularity"] += -0.04 if step % 2 else 0.04
    elif world == "delayed_consequences":
        delta["memory_capacity"] += 0.055
        delta["prediction_horizon"] += 0.055
    elif world == "deceptive_regularities":
        delta["prediction_horizon"] += 0.050
        if step > 4:
            delta["environmental_regularity"] -= 0.090
    elif world == "hierarchical_environment":
        delta["hierarchy_depth"] += 0.070
        delta["compositionality"] += 0.050
    elif world == "sparse_observations":
        delta["communication_bandwidth"] -= 0.060
        delta["memory_capacity"] += 0.030
    elif world == "continuous_environment":
        delta["interaction_locality"] -= 0.030
        delta["graph_connectivity"] += 0.040
    elif world == "partially_contradictory":
        delta["constraint_density"] += 0.030
        delta["stochasticity"] += 0.050
    elif world == "self_modifying":
        delta["computational_budget"] += 0.060
        delta["feedback_strength"] += rng.uniform(-0.080, 0.080)
    elif world == "no_reusable_structure":
        delta["stochasticity"] += 0.020
        delta["environmental_regularity"] -= 0.050

    next_params = {}
    for name in PARAMETERS:
        next_params[name] = max(0.0, min(1.0, params[name] + delta[name] + rng.uniform(-0.010, 0.010)))
    return next_params


def distort_observables(
    observables: dict[str, float],
    world: str,
    architecture: str,
    rule: str,
    step: int,
    rng: random.Random,
) -> dict[str, float]:
    result = dict(observables)
    if world in {"adversarial", "deceptive_regularities"} and step > 3:
        result["order_transfer_efficiency"] = max(0.0, result["order_transfer_efficiency"] - 0.25)
        result["reconstruction"] = max(0.0, result["reconstruction"] - 0.20)
    if world == "partially_contradictory":
        result["uncertainty_handling"] = min(1.0, result["uncertainty_handling"] + 0.35)
        result["order_uncertainty_persistence"] = min(1.0, result["order_uncertainty_persistence"] + 0.30)
    if world == "no_reusable_structure":
        for key in ("abstraction_growth", "order_compression_ratio", "order_reuse", "stability"):
            result[key] = max(0.0, result[key] * 0.25 + rng.random() * 0.05)
    if architecture == "self_modifying" or world == "self_modifying":
        result["translation_complexity"] += 0.5 * step
    if rule == "random_search":
        result["order_behavioral_entropy"] = min(1.0, result["order_behavioral_entropy"] + rng.random() * 0.2)
    return result


def evaluate_suite(name: str, states: list[FlowState], frozen: FrozenKoopmanCoordinates, law: FrozenAffineLaw) -> dict[str, object]:
    coords = transform(states, frozen)
    pairs = transition_pairs(rows_from_states(states))
    prediction = predict_with_law(coords, pairs, law)
    return {
        "suite": name,
        "states": len(states),
        "trajectories": len({state.trajectory_id for state in states}),
        "r2": prediction["r2"],
        "mae": prediction["mae"],
        "p95_error": prediction["p95_error"],
        "works": prediction["r2"] >= 0.65,
        "weak": 0.35 <= prediction["r2"] < 0.65,
        "fails": prediction["r2"] < 0.35,
    }


def failure_atlas(suites: list[dict[str, object]]) -> dict[str, object]:
    failures = [suite for suite in suites if suite["fails"]]
    patterns: dict[str, int] = {}
    for suite in failures:
        kind, value = str(suite["suite"]).split("::", 1)
        patterns[kind] = patterns.get(kind, 0) + 1
    return {"failures": failures, "failure_patterns": patterns}


def applicability_boundary(suites: list[dict[str, object]]) -> dict[str, object]:
    by_kind: dict[str, list[dict[str, object]]] = {}
    for suite in suites:
        kind, _ = str(suite["suite"]).split("::", 1)
        by_kind.setdefault(kind, []).append(suite)
    return {
        kind: {
            "works": [suite["suite"] for suite in values if suite["works"]],
            "weak": [suite["suite"] for suite in values if suite["weak"]],
            "fails": [suite["suite"] for suite in values if suite["fails"]],
            "mean_r2": float(np.mean([suite["r2"] for suite in values])),
        }
        for kind, values in sorted(by_kind.items())
    }


def competing_models(reference_states: list[FlowState], ood_states: list[FlowState], frozen: FrozenKoopmanCoordinates) -> dict[str, object]:
    reference_coords = transform(reference_states, frozen)
    ood_coords = transform(ood_states, frozen)
    train_pairs = transition_pairs(rows_from_states(reference_states))
    test_pairs = transition_pairs(rows_from_states(ood_states))
    return {
        "frozen_affine_dim2": evaluate_competing(reference_coords, train_pairs, ood_coords, test_pairs, "affine"),
        "frozen_polynomial_dim2": evaluate_competing(reference_coords, train_pairs, ood_coords, test_pairs, "polynomial"),
        "frozen_recurrent_proxy_dim2": evaluate_competing(reference_coords, train_pairs, ood_coords, test_pairs, "recurrent_proxy"),
        "higher_dim_koopman_dim4": higher_dim_competing(reference_states, ood_states, 4),
    }


def evaluate_competing(train: np.ndarray, train_pairs: list[tuple[int, int]], test: np.ndarray, test_pairs: list[tuple[int, int]], model: str) -> float:
    x_train = np.array([train[i] for i, _ in train_pairs], dtype=float)
    y_train = np.array([train[j] for _, j in train_pairs], dtype=float)
    x_test = np.array([test[i] for i, _ in test_pairs], dtype=float)
    y_test = np.array([test[j] for _, j in test_pairs], dtype=float)
    if model == "affine":
        f_train = np.hstack([x_train, np.ones((len(x_train), 1))])
        f_test = np.hstack([x_test, np.ones((len(x_test), 1))])
    elif model == "polynomial":
        f_train = np.hstack([x_train, quadratic_features(x_train), np.ones((len(x_train), 1))])
        f_test = np.hstack([x_test, quadratic_features(x_test), np.ones((len(x_test), 1))])
    elif model == "recurrent_proxy":
        f_train = np.hstack([x_train, np.sin(x_train), np.ones((len(x_train), 1))])
        f_test = np.hstack([x_test, np.sin(x_test), np.ones((len(x_test), 1))])
    else:
        raise ValueError(model)
    coef, *_ = np.linalg.lstsq(f_train, y_train, rcond=None)
    return r2_score(y_test, f_test @ coef)


def higher_dim_competing(reference_states: list[FlowState], ood_states: list[FlowState], dim: int) -> float:
    values = observable_values(reference_states)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (values - mean) / std
    features = quadratic_features(scaled[:, : min(10, scaled.shape[1])])
    feature_mean = features.mean(axis=0)
    _, _, vt = np.linalg.svd(features - feature_mean, full_matrices=False)
    frozen = FrozenKoopmanCoordinates(mean.tolist(), std.tolist(), feature_mean.tolist(), vt[:dim].tolist())
    train_coords = transform(reference_states, frozen)
    test_coords = transform(ood_states, frozen)
    return evaluate_competing(train_coords, transition_pairs(rows_from_states(reference_states)), test_coords, transition_pairs(rows_from_states(ood_states)), "affine")


def confidence_intervals(suites: list[dict[str, object]]) -> dict[str, object]:
    values = [float(suite["r2"]) for suite in suites]
    return {"r2": distribution_summary(values)}


def scientific_verdict(aggregate: dict[str, object], boundary: dict[str, object], competitors: dict[str, object]) -> str:
    if aggregate["r2"] >= 0.70 and not aggregate["fails"]:
        return "A. candidate general epistemic law under this OOD suite"
    if aggregate["r2"] >= 0.35 or any(section["works"] for section in boundary.values()):
        return "B. law valid only inside a restricted universality region"
    if max(float(value) for value in competitors.values()) > aggregate["r2"] + 0.25:
        return "C. likely artifact of the original benchmark family; competing dynamics transfer better"
    return "C. law fails under broad distribution shift"


def distribution_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def write_outputs(
    out_dir: Path,
    frozen: FrozenKoopmanCoordinates,
    law: FrozenAffineLaw,
    suites: list[dict[str, object]],
    aggregate: dict[str, object],
    failures: dict[str, object],
    boundary: dict[str, object],
    competitors: dict[str, object],
    confidence: dict[str, object],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_coordinate_system.json", frozen.__dict__)
    write_json(out_dir / "frozen_affine_law.json", law.__dict__)
    write_csv(out_dir / "ood_benchmark_suite.csv", suites)
    write_json(out_dir / "prediction_accuracy_distributions.json", confidence)
    write_json(out_dir / "failure_atlas.json", failures)
    write_json(out_dir / "applicability_boundary.json", boundary)
    write_json(out_dir / "competing_model_comparison.json", competitors)
    write_json(out_dir / "robustness_report.json", {"aggregate": aggregate, "boundary": boundary})
    write_json(out_dir / "ood_validation_summary.json", summary)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
