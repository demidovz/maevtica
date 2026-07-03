from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.phase_diagram import (
    AblationSpec,
    ObjectiveGrowthAgent,
    PhaseRunMetrics,
    PhaseWorld,
    PhaseWorldSpec,
    objective_specs,
    reviewer_ablations,
    world_specs,
)
from epistemic_engine.abstractions.representations import HypergraphObjectiveGrowthAgent
from epistemic_engine.abstractions.unsupervised import (
    NUMERIC_FEATURES,
    adjusted_rand_index,
    cluster_all,
    cluster_count,
    describe_clusters,
    matrix,
    pca,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether macro abstraction-growth behavior survives a representation rewrite."
    )
    parser.add_argument("--train-objects", type=int, default=80)
    parser.add_argument("--transfer-objects", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--reviewer-suite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_representation_invariance"),
    )
    return parser.parse_args()


def run_one(
    implementation: str,
    objective,
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
    if implementation == "dag":
        agent = ObjectiveGrowthAgent(objective, ablation)
    elif implementation == "hypergraph":
        agent = HypergraphObjectiveGrowthAgent(objective, ablation)
    else:
        raise ValueError(f"unknown implementation: {implementation}")
    for step, toy_object in enumerate(train, start=1):
        agent.observe(toy_object, step)
    metrics = agent.metrics(transfer)
    return replace(metrics, objective=objective.name, world=world_spec.name, ablation=ablation.name, seed=seed)


def selected_objectives(quick: bool):
    specs = objective_specs()
    if not quick:
        return specs
    keep = {"exact_memory", "accuracy", "compression", "lifetime_reuse", "novelty", "prediction_lifetime"}
    return [spec for spec in specs if spec.name in keep]


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


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["objective"], row["world"], row["ablation"], row["seed"])


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
        writer.writerow(["implementation", "objective", "world", "ablation", "seed", "pc1", "pc2", "cluster"])
        for row, coord, label in zip(rows, coords, labels):
            writer.writerow(
                [
                    row["implementation"],
                    row["objective"],
                    row["world"],
                    row["ablation"],
                    row["seed"],
                    coord[0],
                    coord[1],
                    label,
                ]
            )


def preservation_report(rows: list[dict[str, object]], labels: list[int]) -> dict[str, object]:
    by_impl: dict[str, dict[tuple[str, str, str, str], dict[str, object]]] = {"dag": {}, "hypergraph": {}}
    by_impl_labels: dict[str, dict[tuple[str, str, str, str], int]] = {"dag": {}, "hypergraph": {}}
    for row, label in zip(rows, labels):
        key = row_key({k: str(v) for k, v in row.items()})
        impl = str(row["implementation"])
        by_impl[impl][key] = row
        by_impl_labels[impl][key] = label

    shared_keys = sorted(set(by_impl["dag"]) & set(by_impl["hypergraph"]))
    dag_labels = [by_impl_labels["dag"][key] for key in shared_keys]
    hyper_labels = [by_impl_labels["hypergraph"][key] for key in shared_keys]
    paired_cluster_ari = adjusted_rand_index(dag_labels, hyper_labels)

    feature_deltas: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        deltas: list[float] = []
        for key in shared_keys:
            left = float(by_impl["dag"][key][feature])
            right = float(by_impl["hypergraph"][key][feature])
            denominator = max(1.0, abs(left), abs(right))
            deltas.append(abs(left - right) / denominator)
        feature_deltas[feature] = sum(deltas) / len(deltas) if deltas else 0.0

    cluster_mix: dict[str, dict[str, int]] = {}
    for row, label in zip(rows, labels):
        if label < 0:
            continue
        entry = cluster_mix.setdefault(str(label), {"dag": 0, "hypergraph": 0})
        entry[str(row["implementation"])] += 1
    balance_scores = []
    for counts in cluster_mix.values():
        total = counts["dag"] + counts["hypergraph"]
        if total:
            balance_scores.append(2 * min(counts["dag"], counts["hypergraph"]) / total)
    implementation_balance = sum(balance_scores) / len(balance_scores) if balance_scores else 0.0

    return {
        "paired_cells": len(shared_keys),
        "paired_cluster_ari": paired_cluster_ari,
        "mean_relative_feature_delta": feature_deltas,
        "largest_feature_deltas": dict(sorted(feature_deltas.items(), key=lambda item: -item[1])[:8]),
        "cluster_implementation_mix": cluster_mix,
        "implementation_balance": implementation_balance,
        "interpretation": interpret(paired_cluster_ari, implementation_balance, feature_deltas),
    }


def interpret(paired_cluster_ari: float, implementation_balance: float, feature_deltas: dict[str, float]) -> str:
    large_deltas = [feature for feature, value in feature_deltas.items() if value > 0.35]
    if paired_cluster_ari < 0.35 or implementation_balance < 0.35:
        return "Macro-behavior is representation-sensitive; current universality hypothesis is weakened."
    if large_deltas:
        return "Cluster structure partially survives, but important macro-metrics shift: " + ", ".join(large_deltas[:6])
    return "Macro-behavior is a candidate representation-invariant signal in this run."


def main() -> None:
    args = parse_args()
    objectives = selected_objectives(args.quick)
    worlds = selected_worlds(args.quick)
    ablations = selected_ablations(args.reviewer_suite, args.quick)
    rows: list[dict[str, object]] = []
    for implementation in ("dag", "hypergraph"):
        for objective in objectives:
            for world in worlds:
                for ablation in ablations:
                    for seed in range(args.seeds):
                        metrics = run_one(
                            implementation,
                            objective,
                            world,
                            ablation,
                            seed,
                            args.train_objects,
                            args.transfer_objects,
                        )
                        row = asdict(metrics)
                        row["implementation"] = implementation
                        rows.append(row)

    values, _, _ = standardize(matrix([{key: str(value) for key, value in row.items()} for row in rows]))
    coords, explained, _ = pca(values, components=3)
    cluster_results = cluster_all(values)
    candidates = [result for result in cluster_results if result.cluster_count >= 2]
    best = max(candidates, key=lambda item: (item.stability_ari, item.cluster_count)) if candidates else cluster_results[0]
    descriptions = describe_clusters(values, best.labels, NUMERIC_FEATURES)
    preservation = preservation_report(rows, best.labels)

    payload = {
        "runs": len(rows),
        "implementations": ["dag", "hypergraph"],
        "selected_method": best.method,
        "selected_cluster_count": cluster_count(best.labels),
        "selected_stability_ari": best.stability_ari,
        "pca_explained_variance": explained,
        "cluster_descriptions": descriptions,
        "preservation": preservation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "representation_runs.csv", rows)
    write_projection(args.out_dir / "representation_pca_projection.csv", rows, coords, best.labels)
    with (args.out_dir / "representation_invariance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("REPRESENTATION INVARIANCE")
    print(f"runs: {len(rows)}")
    print(f"selected_method: {best.method}")
    print(f"clusters: {cluster_count(best.labels)}")
    print(f"stability_ari: {best.stability_ari:.3f}")
    print(f"paired_cluster_ari: {preservation['paired_cluster_ari']:.3f}")
    print(f"implementation_balance: {preservation['implementation_balance']:.3f}")
    print("largest_feature_deltas:")
    for feature, value in preservation["largest_feature_deltas"].items():
        print(f"  {feature}: {value:.3f}")
    print(f"interpretation: {preservation['interpretation']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
