from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NUMERIC_FEATURES = (
    "abstraction_count",
    "alive_count",
    "graph_depth",
    "dag_width",
    "mean_lifetime",
    "mean_exposure_lifetime",
    "mean_reuse",
    "valid_reuse",
    "raw_transfer_reuse",
    "valid_transfer_reuse",
    "transfer_correctness",
    "graph_compression_ratio",
    "semantic_subsumption_ratio",
    "branching_factor",
    "hierarchy_emergence_time",
    "births",
    "deaths",
)


@dataclass(frozen=True)
class ClusterResult:
    method: str
    labels: list[int]
    cluster_count: int
    noise_count: int
    stability_ari: float


def load_phase_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix(rows: list[dict[str, str]], features: tuple[str, ...] = NUMERIC_FEATURES) -> np.ndarray:
    values = np.array([[float(row[feature]) for feature in features] for row in rows], dtype=float)
    return values


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    return (values - mean) / std, mean, std


def pca(values: np.ndarray, components: int = 3) -> tuple[np.ndarray, list[float], np.ndarray]:
    centered = values - values.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:components].T
    variance = singular_values**2
    explained = variance / variance.sum() if variance.sum() else variance
    return coords, explained[:components].tolist(), vt[:components]


def kmeans(values: np.ndarray, k: int, seed: int = 0, iterations: int = 80) -> list[int]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(values)), k)
    centers = values[indices].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels.tolist()


def agglomerative(values: np.ndarray, k: int) -> list[int]:
    clusters: list[list[int]] = [[index] for index in range(len(values))]
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2) ** 0.5
    while len(clusters) > k:
        best_pair = (0, 1)
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = np.mean([distances[a, b] for a in clusters[i] for b in clusters[j]])
                if d < best_distance:
                    best_distance = d
                    best_pair = (i, j)
        i, j = best_pair
        clusters[i].extend(clusters[j])
        del clusters[j]
    labels = [-1] * len(values)
    for label, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = label
    return labels


def dbscan(values: np.ndarray, eps: float, min_samples: int) -> list[int]:
    labels = [-99] * len(values)
    cluster_id = 0
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2) ** 0.5

    for index in range(len(values)):
        if labels[index] != -99:
            continue
        neighbors = [i for i, d in enumerate(distances[index]) if d <= eps]
        if len(neighbors) < min_samples:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        queue = list(neighbors)
        while queue:
            current = queue.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -99:
                continue
            labels[current] = cluster_id
            current_neighbors = [i for i, d in enumerate(distances[current]) if d <= eps]
            if len(current_neighbors) >= min_samples:
                queue.extend(current_neighbors)
        cluster_id += 1
    return labels


def adjusted_rand_index(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    labels_a = sorted(set(a))
    labels_b = sorted(set(b))
    contingency = np.zeros((len(labels_a), len(labels_b)), dtype=int)
    index_a = {label: i for i, label in enumerate(labels_a)}
    index_b = {label: i for i, label in enumerate(labels_b)}
    for left, right in zip(a, b):
        contingency[index_a[left], index_b[right]] += 1

    def comb2(x: int) -> float:
        return x * (x - 1) / 2

    sum_comb = sum(comb2(int(x)) for x in contingency.flat)
    row_comb = sum(comb2(int(x)) for x in contingency.sum(axis=1))
    col_comb = sum(comb2(int(x)) for x in contingency.sum(axis=0))
    total = comb2(n)
    expected = row_comb * col_comb / total if total else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return (sum_comb - expected) / denom if denom else 0.0


def cluster_count(labels: list[int]) -> int:
    return len({label for label in labels if label >= 0})


def bootstrap_stability(values: np.ndarray, method: str, labels: list[int], seed: int = 0) -> float:
    rng = random.Random(seed)
    scores: list[float] = []
    n = len(values)
    if n < 8:
        return 0.0
    trials = 8
    sample_fraction = 0.45 if method.startswith("agglomerative") else 0.65
    for trial in range(trials):
        sample = sorted(rng.sample(range(n), max(6, int(n * sample_fraction))))
        sub_values = values[sample]
        if method.startswith("kmeans"):
            k = int(method.split("_")[-1])
            sub_labels = kmeans(sub_values, k, seed=seed + trial)
        elif method.startswith("agglomerative"):
            k = int(method.split("_")[-1])
            sub_labels = agglomerative(sub_values, k)
        else:
            sub_labels = dbscan(sub_values, eps=2.2, min_samples=4)
        original = [labels[index] for index in sample]
        scores.append(adjusted_rand_index(original, sub_labels))
    return sum(scores) / len(scores)


def cluster_all(values: np.ndarray) -> list[ClusterResult]:
    results: list[ClusterResult] = []
    for k in (2, 3, 4, 5, 6):
        method = f"kmeans_{k}"
        labels = kmeans(values, k, seed=17)
        results.append(
            ClusterResult(method, labels, cluster_count(labels), labels.count(-1), bootstrap_stability(values, method, labels))
        )
    for k in (2, 3, 4, 5, 6):
        method = f"agglomerative_{k}"
        labels = agglomerative(values, k)
        results.append(
            ClusterResult(method, labels, cluster_count(labels), labels.count(-1), bootstrap_stability(values, method, labels))
        )
    for eps in (1.6, 2.0, 2.4, 2.8):
        method = f"dbscan_{eps}"
        labels = dbscan(values, eps=eps, min_samples=4)
        results.append(
            ClusterResult(method, labels, cluster_count(labels), labels.count(-1), bootstrap_stability(values, method, labels))
        )
    return results


def describe_clusters(values: np.ndarray, labels: list[int], features: tuple[str, ...]) -> dict[str, list[str]]:
    descriptions: dict[str, list[str]] = {}
    global_mean = values.mean(axis=0)
    global_std = values.std(axis=0)
    global_std[global_std == 0.0] = 1.0
    for label in sorted(set(labels)):
        if label < 0:
            continue
        members = values[np.array(labels) == label]
        z = (members.mean(axis=0) - global_mean) / global_std
        ranked = sorted(zip(features, z), key=lambda item: -abs(item[1]))[:5]
        descriptions[str(label)] = [
            f"{feature} {'high' if score > 0 else 'low'} ({score:.2f}z)"
            for feature, score in ranked
        ]
    return descriptions


def feature_ablation_report(rows: list[dict[str, str]], baseline_labels: list[int], groups: dict[str, tuple[str, ...]]) -> dict[str, float]:
    report: dict[str, float] = {}
    for name, removed in groups.items():
        features = tuple(feature for feature in NUMERIC_FEATURES if feature not in removed)
        values, _, _ = standardize(matrix(rows, features))
        labels = kmeans(values, max(2, cluster_count(baseline_labels)), seed=17)
        report[name] = adjusted_rand_index(baseline_labels, labels)
    return report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_projection(path: Path, rows: list[dict[str, str]], coords: np.ndarray, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "world", "ablation", "seed", "pc1", "pc2", "cluster"])
        for row, coord, label in zip(rows, coords, labels):
            writer.writerow([row["objective"], row["world"], row["ablation"], row["seed"], coord[0], coord[1], label])
