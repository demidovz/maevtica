from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epistemic_engine.abstractions.unsupervised import kmeans, pca, standardize


METADATA_COLUMNS = {"trajectory_id", "step", "update_rule", "world", "paradigm", "x", "y", "z"}


@dataclass(frozen=True)
class CoordinateResult:
    name: str
    dimension: int
    dynamics: str
    predictive_r2: float
    simplicity_score: float
    sparsity: float
    curvature: float
    smoothness: float
    variance_score: float
    mdl_cost: float


def natural_coordinate_report(trajectory_path: Path, out_dir: Path, max_dim: int = 8) -> dict[str, object]:
    rows, columns = load_rows(trajectory_path)
    observable_columns = [
        column
        for column in columns
        if column not in METADATA_COLUMNS and is_float(rows[0].get(column, ""))
    ]
    values = np.array([[float(row[column]) for column in observable_columns] for row in rows], dtype=float)
    scaled, mean, std = standardize(values)
    groups = {
        "world": [row["world"] for row in rows],
        "update_rule": [row["update_rule"] for row in rows],
        "paradigm": [row["paradigm"] for row in rows],
    }
    transitions = transition_pairs(rows)
    candidates = build_coordinate_candidates(scaled, transitions, max_dim)
    results = evaluate_candidates(candidates, transitions)
    best = max(results, key=lambda item: item.simplicity_score)
    best_coords = candidates[(best.name, best.dimension)]
    hidden = hidden_state_report(results)
    geometry = latent_geometry_report(best_coords, transitions)
    conserved = conserved_quantity_report(best_coords, transitions)
    stability = coordinate_stability_report(scaled, rows, observable_columns, best.name, best.dimension, groups, max_dim)
    interpretation = interpretation_report(best_coords, scaled, observable_columns)
    canonical = canonical_space_report(best_coords, rows)
    summary = {
        "input": str(trajectory_path),
        "states": len(rows),
        "observables": len(observable_columns),
        "candidate_count": len(results),
        "best_coordinate_system": best.__dict__,
        "previous_flow_r2": previous_flow_r2(out_dir.parent / "epistemic_flow" / "empirical_dynamical_law.json"),
        "hidden_state": hidden,
        "coordinate_stability": stability["summary"],
        "geometry": geometry["summary"],
        "conserved_quantities": conserved["summary"],
        "canonical_latent_space": canonical["summary"],
        "success_assessment": success_assessment(best, hidden, stability),
    }
    write_outputs(out_dir, rows, observable_columns, best_coords, results, hidden, geometry, conserved, stability, interpretation, canonical, summary)
    return summary


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader)
        header = deduplicate(raw_header)
        rows = [dict(zip(header, row)) for row in reader]
    return rows, header


def deduplicate(header: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in header:
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}__{count + 1}")
    return result


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def transition_pairs(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    index = {(row["trajectory_id"], int(row["step"])): i for i, row in enumerate(rows)}
    pairs = []
    for i, row in enumerate(rows):
        j = index.get((row["trajectory_id"], int(row["step"]) + 1))
        if j is not None:
            pairs.append((i, j))
    return pairs


def build_coordinate_candidates(values: np.ndarray, pairs: list[tuple[int, int]], max_dim: int) -> dict[tuple[str, int], np.ndarray]:
    candidates: dict[tuple[str, int], np.ndarray] = {}
    coords, _, _ = pca(values, components=max_dim)
    for dim in range(1, max_dim + 1):
        candidates[("pca", dim)] = coords[:, :dim]

    deltas = np.array([values[j] - values[i] for i, j in pairs], dtype=float)
    _, _, vt_delta = np.linalg.svd(deltas, full_matrices=False)
    slow_components = vt_delta[::-1]
    fast_components = vt_delta
    for dim in range(1, min(max_dim, slow_components.shape[0]) + 1):
        candidates[("slow_svd", dim)] = values @ slow_components[:dim].T
        candidates[("delta_svd", dim)] = values @ fast_components[:dim].T

    future = np.zeros_like(values)
    for i, j in pairs:
        future[i] = values[j]
    predictive = np.hstack([values, future - values])
    pred_coords, _, _ = pca(predictive, components=max_dim)
    for dim in range(1, max_dim + 1):
        candidates[("predictive_state_pca", dim)] = pred_coords[:, :dim]

    koopman_features = quadratic_features(values[:, : min(values.shape[1], 10)])
    koopman_coords, _, _ = pca(koopman_features, components=max_dim)
    for dim in range(1, max_dim + 1):
        candidates[("koopman_quadratic_pca", dim)] = koopman_coords[:, :dim]

    random = random_projection(values, max_dim, seed=113)
    for dim in range(1, max_dim + 1):
        candidates[("random_projection_control", dim)] = random[:, :dim]
    return candidates


def quadratic_features(values: np.ndarray) -> np.ndarray:
    columns = [values]
    products = []
    for i in range(values.shape[1]):
        for j in range(i, values.shape[1]):
            products.append((values[:, i] * values[:, j])[:, None])
    columns.append(np.hstack(products))
    return np.hstack(columns)


def random_projection(values: np.ndarray, components: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(values.shape[1], components))
    return values @ projection / math.sqrt(values.shape[1])


def evaluate_candidates(candidates: dict[tuple[str, int], np.ndarray], pairs: list[tuple[int, int]]) -> list[CoordinateResult]:
    results = []
    for (name, dim), coords in candidates.items():
        for dynamics in ("affine", "polynomial2", "piecewise_affine"):
            fit = fit_dynamics(coords, pairs, dynamics)
            curvature = curvature_score(coords, pairs)
            smoothness = 1.0 / (1.0 + fit["coef_norm"] + curvature)
            variance = variance_score(coords)
            mdl = mdl_cost(dim, dynamics, fit["sparsity"], fit["predictive_r2"], curvature, variance)
            simplicity = (
                fit["predictive_r2"]
                + 0.20 * fit["sparsity"]
                + 0.16 * smoothness
                + 0.20 * variance
                - 0.045 * dim
                - 0.018 * mdl
            )
            results.append(
                CoordinateResult(
                    name,
                    dim,
                    dynamics,
                    fit["predictive_r2"],
                    simplicity,
                    fit["sparsity"],
                    curvature,
                    smoothness,
                    variance,
                    mdl,
                )
            )
    return sorted(results, key=lambda item: -item.simplicity_score)


def fit_dynamics(coords: np.ndarray, pairs: list[tuple[int, int]], dynamics: str) -> dict[str, float]:
    x = np.array([coords[i] for i, _ in pairs], dtype=float)
    y = np.array([coords[j] for _, j in pairs], dtype=float)
    if dynamics == "affine":
        features = np.hstack([x, np.ones((len(x), 1))])
        pred, coef = least_squares_predict(features, y)
    elif dynamics == "polynomial2":
        features = np.hstack([x, quadratic_features(x), np.ones((len(x), 1))])
        pred, coef = least_squares_predict(features, y)
    elif dynamics == "piecewise_affine":
        labels = kmeans(x, min(5, max(2, len(x) // 120)), seed=211)
        pred = np.zeros_like(y)
        coefs = []
        for label in sorted(set(labels)):
            mask = np.array(labels) == label
            if mask.sum() < coords.shape[1] + 2:
                pred[mask] = x[mask]
                continue
            features = np.hstack([x[mask], np.ones((mask.sum(), 1))])
            local_pred, local_coef = least_squares_predict(features, y[mask])
            pred[mask] = local_pred
            coefs.append(local_coef.ravel())
        coef = np.concatenate(coefs) if coefs else np.zeros(1)
    else:
        raise ValueError(dynamics)
    return {
        "predictive_r2": r2_score(y, pred),
        "sparsity": float(np.mean(np.abs(coef) < 1e-3)),
        "coef_norm": float(np.linalg.norm(coef)),
    }


def least_squares_predict(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return features @ coef, coef


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def curvature_score(coords: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    by_left = {i: j for i, j in pairs}
    angles = []
    for i, j in pairs:
        k = by_left.get(j)
        if k is None:
            continue
        v1 = coords[j] - coords[i]
        v2 = coords[k] - coords[j]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 0:
            angles.append(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))))
    return float(np.mean(angles)) if angles else 0.0


def variance_score(coords: np.ndarray) -> float:
    spread = float(np.mean(np.std(coords, axis=0)))
    return max(0.0, min(1.0, spread / 0.25))


def mdl_cost(dim: int, dynamics: str, sparsity: float, r2: float, curvature: float, variance: float) -> float:
    model_factor = {"affine": 1.0, "piecewise_affine": 2.1, "polynomial2": 2.8}[dynamics]
    return model_factor * dim * (1.0 - 0.5 * sparsity) + 4.0 * max(0.0, 1.0 - r2) + curvature + 4.0 * (1.0 - variance)


def hidden_state_report(results: list[CoordinateResult]) -> dict[str, object]:
    best_r2 = max(item.predictive_r2 for item in results)
    viable = [
        item for item in results
        if item.predictive_r2 >= 0.95 * best_r2 and item.simplicity_score >= max(r.simplicity_score for r in results) - 0.15
        and item.variance_score >= 0.25
    ]
    min_dim = min(item.dimension for item in viable) if viable else min(item.dimension for item in results)
    by_dim = {}
    for dim in sorted({item.dimension for item in results}):
        best = max((item for item in results if item.dimension == dim), key=lambda item: item.simplicity_score)
        by_dim[str(dim)] = best.__dict__
    return {
        "best_predictive_r2": best_r2,
        "minimum_viable_dimension": min_dim,
        "previous_estimate": 5,
        "dimension_table": by_dim,
    }


def latent_geometry_report(coords: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, object]:
    distances = pairwise_sample_distances(coords, sample=500)
    curvature = curvature_score(coords, pairs)
    neighborhood = neighborhood_preservation(coords, pairs)
    metric = local_metric_tensor(coords, pairs)
    return {
        "summary": {
            "curvature": curvature,
            "neighborhood_preservation": neighborhood,
            "distance_mean": float(np.mean(distances)) if len(distances) else 0.0,
            "distance_std": float(np.std(distances)) if len(distances) else 0.0,
        },
        "local_metric_tensor": metric.tolist(),
    }


def pairwise_sample_distances(coords: np.ndarray, sample: int) -> np.ndarray:
    n = min(sample, len(coords))
    indices = np.linspace(0, len(coords) - 1, n, dtype=int)
    values = coords[indices]
    result = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            result.append(float(np.linalg.norm(values[i] - values[j])))
    return np.array(result, dtype=float)


def neighborhood_preservation(coords: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    jumps = [float(np.linalg.norm(coords[j] - coords[i])) for i, j in pairs]
    if not jumps:
        return 0.0
    threshold = float(np.quantile(jumps, 0.35))
    return sum(1 for jump in jumps if jump <= threshold) / len(jumps)


def local_metric_tensor(coords: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    deltas = np.array([coords[j] - coords[i] for i, j in pairs], dtype=float)
    if len(deltas) == 0:
        return np.eye(coords.shape[1])
    return np.cov(deltas.T)


def conserved_quantity_report(coords: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, object]:
    deltas = np.array([coords[j] - coords[i] for i, j in pairs], dtype=float)
    _, _, vt = np.linalg.svd(deltas, full_matrices=False)
    candidates = []
    for vector in vt[::-1][: min(5, len(vt))]:
        projected_delta = deltas @ vector
        projected_level = coords[[i for i, _ in pairs]] @ vector
        candidates.append(
            {
                "drift_mean_abs": float(np.mean(np.abs(projected_delta))),
                "drift_std": float(np.std(projected_delta)),
                "level_std": float(np.std(projected_level)),
                "weights": [float(x) for x in vector],
            }
        )
    return {
        "summary": {
            "best_drift_mean_abs": candidates[0]["drift_mean_abs"] if candidates else 0.0,
            "candidate_count": len(candidates),
        },
        "candidates": candidates,
    }


def coordinate_stability_report(
    values: np.ndarray,
    rows: list[dict[str, str]],
    observable_columns: list[str],
    best_name: str,
    best_dim: int,
    groups: dict[str, list[str]],
    max_dim: int,
) -> dict[str, object]:
    full_candidates = build_coordinate_candidates(values, transition_pairs(rows), max_dim)
    full = full_candidates[(best_name, best_dim)]
    report = {}
    for group_name, labels in groups.items():
        group_scores = []
        for label in sorted(set(labels)):
            indices = [i for i, item in enumerate(labels) if item == label]
            if len(indices) < best_dim + 4:
                continue
            sub_values = values[indices]
            sub_rows = [rows[i] for i in indices]
            sub_pairs = transition_pairs(sub_rows)
            if len(sub_pairs) < best_dim + 4:
                continue
            sub_candidates = build_coordinate_candidates(sub_values, sub_pairs, max_dim)
            sub = sub_candidates.get((best_name, best_dim))
            if sub is None:
                continue
            group_scores.append(subspace_similarity(full[indices], sub))
        report[group_name] = {
            "mean_subspace_similarity": sum(group_scores) / len(group_scores) if group_scores else 0.0,
            "groups_evaluated": len(group_scores),
        }
    return {"summary": report}


def subspace_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_u, _, _ = np.linalg.svd(left - left.mean(axis=0), full_matrices=False)
    right_u, _, _ = np.linalg.svd(right - right.mean(axis=0), full_matrices=False)
    dim = min(left_u.shape[1], right_u.shape[1], 6)
    if dim == 0:
        return 0.0
    return float(np.linalg.norm(left_u[:, :dim].T @ right_u[:, :dim], ord="fro") / math.sqrt(dim))


def interpretation_report(coords: np.ndarray, values: np.ndarray, observable_columns: list[str]) -> dict[str, object]:
    correlations = {}
    for dim in range(coords.shape[1]):
        scores = []
        for index, name in enumerate(observable_columns):
            corr = correlation(coords[:, dim], values[:, index])
            scores.append((name, corr))
        correlations[f"z{dim + 1}"] = [
            {"observable": name, "correlation": float(corr)}
            for name, corr in sorted(scores, key=lambda item: -abs(item[1]))[:10]
        ]
    return {
        "coordinate_interpretations": correlations,
        "note": "Names are post-hoc correlations with observations, not imposed semantic labels.",
    }


def canonical_space_report(coords: np.ndarray, rows: list[dict[str, str]]) -> dict[str, object]:
    distortions = {}
    for key in ("world", "update_rule", "paradigm"):
        labels = sorted(set(row[key] for row in rows))
        centroids = []
        for label in labels:
            members = coords[[i for i, row in enumerate(rows) if row[key] == label]]
            centroids.append(members.mean(axis=0))
        if len(centroids) < 2:
            distortions[key] = 0.0
            continue
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        distortions[key] = float(np.mean(distances))
    incompatible = [key for key, value in distortions.items() if value > 2.0]
    return {
        "summary": {
            "embedding_distortion_by_family": distortions,
            "incompatible_family_axes": incompatible,
            "common_space_supported": len(incompatible) == 0,
        }
    }


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, str]],
    observable_columns: list[str],
    best_coords: np.ndarray,
    results: list[CoordinateResult],
    hidden: dict[str, object],
    geometry: dict[str, object],
    conserved: dict[str, object],
    stability: dict[str, object],
    interpretation: dict[str, object],
    canonical: dict[str, object],
    summary: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_coordinates(out_dir / "natural_coordinate_system.csv", rows, best_coords)
    write_json(out_dir / "dynamical_simplicity_analysis.json", [item.__dict__ for item in results[:80]])
    write_json(out_dir / "hidden_state_report.json", hidden)
    write_json(out_dir / "latent_geometry_report.json", geometry)
    write_json(out_dir / "conserved_quantity_report.json", conserved)
    write_json(out_dir / "coordinate_stability_analysis.json", stability)
    write_json(out_dir / "interpretation_report.json", interpretation)
    write_json(out_dir / "canonical_latent_space_proposal.json", canonical)
    write_json(out_dir / "natural_coordinates_summary.json", summary)


def write_coordinates(path: Path, rows: list[dict[str, str]], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "step", "update_rule", "world", "paradigm", *[f"z{i + 1}" for i in range(coords.shape[1])]])
        for row, coord in zip(rows, coords):
            writer.writerow([row["trajectory_id"], row["step"], row["update_rule"], row["world"], row["paradigm"], *coord])


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def previous_flow_r2(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("predictive_r2")


def success_assessment(best: CoordinateResult, hidden: dict[str, object], stability: dict[str, object]) -> str:
    stable_axes = [
        value["mean_subspace_similarity"]
        for value in stability["summary"].values()
        if value["groups_evaluated"] > 0
    ]
    mean_stability = sum(stable_axes) / len(stable_axes) if stable_axes else 0.0
    if best.predictive_r2 >= 0.62 and best.dimension <= 5 and mean_stability >= 0.55:
        return "A candidate natural coordinate system substantially simplifies the observed dynamics."
    if best.predictive_r2 < 0.45 or mean_stability < 0.35:
        return "No stable natural coordinates were found; dynamics remain high-order or family-dependent."
    return "A partial coordinate improvement was found, but it is not yet strong enough to treat as natural state variables."


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
