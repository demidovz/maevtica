from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.phase_diagram import (
    AblationSpec,
    ObjectiveGrowthAgent,
    PhaseWorld,
    PhaseWorldSpec,
    objective_specs,
)
from epistemic_engine.abstractions.unsupervised import (
    NUMERIC_FEATURES,
    cluster_all,
    cluster_count,
    describe_clusters,
    feature_ablation_report,
    matrix,
    pca,
    standardize,
    write_json,
)


PARAMETERS = ("compositionality", "noise", "regularity", "partial_observability", "volatility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan continuous world parameters for regime transitions.")
    parser.add_argument("--objective", default="lifetime_reuse")
    parser.add_argument("--train-objects", type=int, default=90)
    parser.add_argument("--transfer-objects", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--levels", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_continuous_phase_scan"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = [float(item) for item in args.levels.split(",")]
    objective = next(spec for spec in objective_specs() if spec.name == args.objective)
    ablation = AblationSpec("standard")
    rows: list[dict[str, object]] = []

    for compositionality in levels:
        for noise in levels:
            for regularity in levels:
                for partial in levels:
                    if partial <= 0:
                        continue
                    for volatility in (0.0, 0.5, 1.0):
                        for seed in range(args.seeds):
                            spec = PhaseWorldSpec(
                                name="continuous",
                                noise=noise,
                                partial_observability=partial,
                                compositionality=compositionality,
                                regularity=regularity,
                                volatility=volatility,
                            )
                            world = PhaseWorld(spec, seed)
                            train = world.train(args.train_objects)
                            transfer = world.transfer(args.transfer_objects)
                            agent = ObjectiveGrowthAgent(objective, ablation)
                            for step, toy_object in enumerate(train, start=1):
                                agent.observe(toy_object, step)
                            metrics = agent.metrics(transfer)
                            row = asdict(metrics)
                            row.update(
                                {
                                    "objective": objective.name,
                                    "world": "continuous",
                                    "ablation": ablation.name,
                                    "compositionality": compositionality,
                                    "noise": noise,
                                    "regularity": regularity,
                                    "partial_observability": partial,
                                    "volatility": volatility,
                                    "seed": seed,
                                }
                            )
                            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.objective}_continuous_scan.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    values, _, _ = standardize(matrix([{key: str(value) for key, value in row.items()} for row in rows]))
    coords, explained, components = pca(values, components=3)
    cluster_results = cluster_all(values)
    candidates = [result for result in cluster_results if result.cluster_count >= 2]
    best = max(candidates, key=lambda item: (item.stability_ari, item.cluster_count)) if candidates else cluster_results[0]
    ablation_groups = {
        "without_depth": ("graph_depth", "dag_width", "branching_factor", "hierarchy_emergence_time"),
        "without_reuse": ("mean_reuse", "valid_reuse", "raw_transfer_reuse", "valid_transfer_reuse"),
        "without_lifetime": ("mean_lifetime", "mean_exposure_lifetime"),
        "without_transfer": ("raw_transfer_reuse", "valid_transfer_reuse", "transfer_correctness"),
        "without_compression": ("graph_compression_ratio",),
    }
    feature_ablation = feature_ablation_report(
        [{key: str(value) for key, value in row.items()} for row in rows],
        best.labels,
        ablation_groups,
    )
    descriptions = describe_clusters(values, best.labels, NUMERIC_FEATURES)
    transition_report = transition_candidates(rows, best.labels)
    discovery_payload = {
        "input": str(path),
        "points": len(rows),
        "parameters": list(PARAMETERS),
        "features": list(NUMERIC_FEATURES),
        "pca_explained_variance": explained,
        "pca_components": {
            f"pc{index + 1}": [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in sorted(zip(NUMERIC_FEATURES, component), key=lambda item: -abs(item[1]))[:8]
            ]
            for index, component in enumerate(components)
        },
        "methods": [
            {
                "method": result.method,
                "cluster_count": result.cluster_count,
                "noise_count": result.noise_count,
                "stability_ari": result.stability_ari,
            }
            for result in cluster_results
        ],
        "selected_method": best.method,
        "selected_cluster_count": cluster_count(best.labels),
        "selected_noise_count": best.labels.count(-1),
        "selected_stability_ari": best.stability_ari,
        "feature_ablation_ari": feature_ablation,
        "cluster_descriptions": descriptions,
        "transition_candidates": transition_report,
    }
    write_json(args.out_dir / f"{args.objective}_continuous_discovery_summary.json", discovery_payload)
    write_continuous_projection(args.out_dir / f"{args.objective}_continuous_pca_projection.csv", rows, coords, best.labels)

    print("CONTINUOUS PHASE SCAN")
    print(f"objective: {args.objective}")
    print(f"runs: {len(rows)}")
    print(f"selected_method: {best.method}")
    print(f"clusters: {cluster_count(best.labels)}")
    print(f"stability_ari: {best.stability_ari:.3f}")
    print("feature_ablation_ari:")
    for name, score in sorted(feature_ablation.items()):
        print(f"  {name}: {score:.3f}")
    print("cluster_descriptions:")
    for label, description in descriptions.items():
        print(f"  cluster {label}: {', '.join(description)}")
    print("transition_counts:")
    for parameter, count in sorted(transition_report["counts_by_parameter"].items()):
        print(f"  {parameter}: {count}")
    print(f"outputs: {args.out_dir}")


def transition_candidates(rows: list[dict[str, object]], labels: list[int]) -> dict[str, object]:
    points: dict[tuple[float, ...], int] = {
        tuple(float(row[parameter]) for parameter in PARAMETERS): label
        for row, label in zip(rows, labels)
    }
    values_by_parameter = {
        parameter: sorted({float(row[parameter]) for row in rows})
        for parameter in PARAMETERS
    }
    transitions: list[dict[str, object]] = []
    counts = {parameter: 0 for parameter in PARAMETERS}
    for point, label in points.items():
        for index, parameter in enumerate(PARAMETERS):
            values = values_by_parameter[parameter]
            current = point[index]
            position = values.index(current)
            if position + 1 >= len(values):
                continue
            neighbor = list(point)
            neighbor[index] = values[position + 1]
            neighbor_key = tuple(neighbor)
            if neighbor_key not in points:
                continue
            neighbor_label = points[neighbor_key]
            if neighbor_label == label:
                continue
            counts[parameter] += 1
            transitions.append(
                {
                    "parameter": parameter,
                    "from": current,
                    "to": values[position + 1],
                    "cluster_from": label,
                    "cluster_to": neighbor_label,
                    "context": {
                        other: point[other_index]
                        for other_index, other in enumerate(PARAMETERS)
                        if other != parameter
                    },
                }
            )
    transitions.sort(key=lambda item: (item["parameter"], item["from"], item["to"], str(item["context"])))
    return {"counts_by_parameter": counts, "examples": transitions[:30]}


def write_continuous_projection(path: Path, rows: list[dict[str, object]], coords, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "seed", *PARAMETERS, "pc1", "pc2", "cluster"])
        for row, coord, label in zip(rows, coords, labels):
            writer.writerow(
                [
                    row["objective"],
                    row["seed"],
                    *(row[parameter] for parameter in PARAMETERS),
                    coord[0],
                    coord[1],
                    label,
                ]
            )


if __name__ == "__main__":
    main()
