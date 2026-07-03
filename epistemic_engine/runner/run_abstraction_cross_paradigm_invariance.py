from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.dynamics import dynamics_agent_classes
from epistemic_engine.abstractions.phase_diagram import (
    AblationSpec,
    PhaseRunMetrics,
    PhaseWorld,
    PhaseWorldSpec,
    reviewer_ablations,
    world_specs,
)
from epistemic_engine.abstractions.unsupervised import (
    NUMERIC_FEATURES,
    adjusted_rand_index,
    cluster_all,
    cluster_count,
    describe_clusters,
    feature_ablation_report,
    matrix,
    pca,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Cross-Paradigm Invariance across distinct local epistemic dynamics."
    )
    parser.add_argument("--train-objects", type=int, default=80)
    parser.add_argument("--transfer-objects", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--reviewer-suite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_cross_paradigm"),
    )
    return parser.parse_args()


def run_one(
    agent_class,
    world_spec: PhaseWorldSpec,
    ablation: AblationSpec,
    seed: int,
    train_objects: int,
    transfer_objects: int,
) -> PhaseRunMetrics:
    world = PhaseWorld(world_spec, seed)
    train = world.train(train_objects)
    transfer = world.transfer(transfer_objects)
    if ablation.shuffle_stream:
        random.Random(seed + 10_000).shuffle(train)
    agent = agent_class(ablation)
    for step, toy_object in enumerate(train, start=1):
        agent.observe(toy_object, step)
    metrics = agent.metrics(transfer)
    return replace(metrics, objective=agent.name, world=world_spec.name, ablation=ablation.name, seed=seed)


def selected_worlds(quick: bool) -> list[PhaseWorldSpec]:
    specs = world_specs()
    if not quick:
        return specs
    keep = {"random", "no_regularities", "hierarchical", "compositional", "partial", "anti_compositional"}
    return [spec for spec in specs if spec.name in keep]


def selected_ablations(reviewer_suite: bool, quick: bool) -> list[AblationSpec]:
    if reviewer_suite:
        specs = reviewer_ablations()
        if quick:
            keep = {"standard", "equal_abstraction_budget", "remove_composition", "remove_hierarchy"}
            specs = [spec for spec in specs if spec.name in keep]
        return specs
    return [AblationSpec("standard")]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_projection(path: Path, rows: list[dict[str, object]], coords, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paradigm", "world", "ablation", "seed", "pc1", "pc2", "cluster"])
        for row, coord, label in zip(rows, coords, labels):
            writer.writerow([row["objective"], row["world"], row["ablation"], row["seed"], coord[0], coord[1], label])


def cross_paradigm_report(rows: list[dict[str, object]], labels: list[int]) -> dict[str, object]:
    paradigms = sorted({str(row["objective"]) for row in rows})
    row_labels = [{key: str(value) for key, value in row.items()} for row in rows]

    paradigm_by_cluster: dict[str, dict[str, int]] = {}
    for row, label in zip(row_labels, labels):
        if label < 0:
            continue
        bucket = paradigm_by_cluster.setdefault(str(label), {paradigm: 0 for paradigm in paradigms})
        bucket[row["objective"]] += 1

    cluster_balance_scores: list[float] = []
    for counts in paradigm_by_cluster.values():
        total = sum(counts.values())
        present = sum(1 for value in counts.values() if value > 0)
        if total and len(paradigms) > 1:
            cluster_balance_scores.append(present / len(paradigms))
    paradigm_mixing = sum(cluster_balance_scores) / len(cluster_balance_scores) if cluster_balance_scores else 0.0

    by_paradigm: dict[str, dict[tuple[str, str, str], int]] = {paradigm: {} for paradigm in paradigms}
    for row, label in zip(row_labels, labels):
        key = (row["world"], row["ablation"], row["seed"])
        by_paradigm[row["objective"]][key] = label

    pairwise: dict[str, float] = {}
    for left, right in combinations(paradigms, 2):
        keys = sorted(set(by_paradigm[left]) & set(by_paradigm[right]))
        if not keys:
            pairwise[f"{left}|{right}"] = 0.0
            continue
        pairwise[f"{left}|{right}"] = adjusted_rand_index(
            [by_paradigm[left][key] for key in keys],
            [by_paradigm[right][key] for key in keys],
        )
    mean_pairwise_ari = sum(pairwise.values()) / len(pairwise) if pairwise else 0.0

    paradigm_centroids: dict[str, dict[str, float]] = {}
    for paradigm in paradigms:
        subset = [row for row in row_labels if row["objective"] == paradigm]
        paradigm_centroids[paradigm] = {
            feature: sum(float(row[feature]) for row in subset) / len(subset)
            for feature in NUMERIC_FEATURES
        }
    macro_variable_spread = {
        feature: max(values) - min(values)
        for feature in NUMERIC_FEATURES
        for values in [[paradigm_centroids[paradigm][feature] for paradigm in paradigms]]
    }

    return {
        "paradigms": paradigms,
        "cluster_paradigm_mix": paradigm_by_cluster,
        "paradigm_mixing": paradigm_mixing,
        "pairwise_cluster_ari": pairwise,
        "mean_pairwise_cluster_ari": mean_pairwise_ari,
        "largest_macro_variable_spread": dict(sorted(macro_variable_spread.items(), key=lambda item: -item[1])[:8]),
        "interpretation": interpret(paradigm_mixing, mean_pairwise_ari, macro_variable_spread),
    }


def interpret(paradigm_mixing: float, mean_pairwise_ari: float, spread: dict[str, float]) -> str:
    large_spread = [feature for feature, value in spread.items() if value > 15]
    if paradigm_mixing < 0.35:
        return "Clusters mostly separate by local paradigm; Cross-Paradigm Invariance is weakened."
    if mean_pairwise_ari < 0.2:
        return "Macro-clusters mix paradigms, but matched world/ablation/seed cells do not preserve cluster identity."
    if large_spread:
        return "Some macro-regimes survive, but major macro-variable shifts remain: " + ", ".join(large_spread[:6])
    return "This run supports a weak Cross-Paradigm Invariance candidate."


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    agent_classes = dynamics_agent_classes()
    worlds = selected_worlds(args.quick)
    ablations = selected_ablations(args.reviewer_suite, args.quick)
    for agent_class in agent_classes:
        for world in worlds:
            for ablation in ablations:
                for seed in range(args.seeds):
                    metrics = run_one(agent_class, world, ablation, seed, args.train_objects, args.transfer_objects)
                    rows.append(asdict(metrics))

    row_strings = [{key: str(value) for key, value in row.items()} for row in rows]
    values, _, _ = standardize(matrix(row_strings))
    coords, explained, components = pca(values, components=3)
    cluster_results = cluster_all(values)
    candidates = [result for result in cluster_results if result.cluster_count >= 2]
    best = max(candidates, key=lambda item: (item.stability_ari, item.cluster_count)) if candidates else cluster_results[0]
    descriptions = describe_clusters(values, best.labels, NUMERIC_FEATURES)
    ablation_groups = {
        "without_depth": ("graph_depth", "dag_width", "branching_factor", "hierarchy_emergence_time"),
        "without_reuse": ("mean_reuse", "valid_reuse", "raw_transfer_reuse", "valid_transfer_reuse"),
        "without_lifetime": ("mean_lifetime", "mean_exposure_lifetime"),
        "without_transfer": ("raw_transfer_reuse", "valid_transfer_reuse", "transfer_correctness"),
        "without_compression": ("graph_compression_ratio",),
    }
    feature_ablation = feature_ablation_report(row_strings, best.labels, ablation_groups)
    cross_report = cross_paradigm_report(rows, best.labels)

    payload = {
        "runs": len(rows),
        "selected_method": best.method,
        "selected_cluster_count": cluster_count(best.labels),
        "selected_stability_ari": best.stability_ari,
        "pca_explained_variance": explained,
        "pca_components": {
            f"pc{index + 1}": [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in sorted(zip(NUMERIC_FEATURES, component), key=lambda item: -abs(item[1]))[:8]
            ]
            for index, component in enumerate(components)
        },
        "feature_ablation_ari": feature_ablation,
        "cluster_descriptions": descriptions,
        "cross_paradigm": cross_report,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "cross_paradigm_runs.csv", rows)
    write_projection(args.out_dir / "cross_paradigm_pca_projection.csv", rows, coords, best.labels)
    with (args.out_dir / "cross_paradigm_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("CROSS-PARADIGM INVARIANCE")
    print(f"runs: {len(rows)}")
    print(f"selected_method: {best.method}")
    print(f"clusters: {cluster_count(best.labels)}")
    print(f"stability_ari: {best.stability_ari:.3f}")
    print(f"paradigm_mixing: {cross_report['paradigm_mixing']:.3f}")
    print(f"mean_pairwise_cluster_ari: {cross_report['mean_pairwise_cluster_ari']:.3f}")
    print("feature_ablation_ari:")
    for name, score in sorted(feature_ablation.items()):
        print(f"  {name}: {score:.3f}")
    print("largest_macro_variable_spread:")
    for feature, value in cross_report["largest_macro_variable_spread"].items():
        print(f"  {feature}: {value:.3f}")
    print(f"interpretation: {cross_report['interpretation']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
