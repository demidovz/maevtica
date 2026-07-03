from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.unsupervised import (
    NUMERIC_FEATURES,
    cluster_all,
    cluster_count,
    describe_clusters,
    feature_ablation_report,
    load_phase_rows,
    matrix,
    pca,
    standardize,
    write_json,
    write_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover abstraction-growth regimes without using regime labels."
    )
    parser.add_argument(
        "--phase-runs",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_phase_diagram/phase_runs.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_regime_discovery"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_phase_rows(args.phase_runs)
    raw = matrix(rows)
    values, _, _ = standardize(raw)
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
    feature_ablation = feature_ablation_report(rows, best.labels, ablation_groups)
    descriptions = describe_clusters(values, best.labels, NUMERIC_FEATURES)

    payload = {
        "input": str(args.phase_runs),
        "points": len(rows),
        "features": list(NUMERIC_FEATURES),
        "pca_explained_variance": explained,
        "pca_components": {
            f"pc{index + 1}": [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in sorted(zip(NUMERIC_FEATURES, component), key=lambda item: -abs(item[1]))[:8]
            ]
            for index, component in enumerate(components)
        },
        "methods": [asdict(result) | {"labels": "omitted"} for result in cluster_results],
        "selected_method": best.method,
        "selected_cluster_count": cluster_count(best.labels),
        "selected_noise_count": best.labels.count(-1),
        "selected_stability_ari": best.stability_ari,
        "cluster_descriptions": descriptions,
        "feature_ablation_ari": feature_ablation,
        "interpretation": interpret(best.stability_ari, feature_ablation),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "regime_discovery_summary.json", payload)
    write_projection(args.out_dir / "pca_projection.csv", rows, coords, best.labels)

    print("UNSUPERVISED REGIME DISCOVERY")
    print(f"points: {len(rows)}")
    print(f"selected_method: {best.method}")
    print(f"clusters: {cluster_count(best.labels)}")
    print(f"stability_ari: {best.stability_ari:.3f}")
    print("feature_ablation_ari:")
    for name, score in sorted(feature_ablation.items()):
        print(f"  {name}: {score:.3f}")
    print("cluster_descriptions:")
    for label, description in descriptions.items():
        print(f"  cluster {label}: {', '.join(description)}")
    print(f"outputs: {args.out_dir}")


def interpret(stability_ari: float, feature_ablation: dict[str, float]) -> str:
    fragile = [name for name, score in feature_ablation.items() if score < 0.45]
    if stability_ari < 0.35:
        return "Clusters are weak or unstable; current regime labels should be treated as suspect."
    if fragile:
        return "Clusters exist but depend on specific feature groups: " + ", ".join(sorted(fragile))
    return "Clusters are moderately stable across bootstrap and feature ablations in this dataset."


if __name__ == "__main__":
    main()
