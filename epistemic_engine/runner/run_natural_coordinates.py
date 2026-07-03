from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.natural_coordinates import natural_coordinate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover natural latent coordinates by maximizing dynamical simplicity."
    )
    parser.add_argument(
        "--trajectory-catalogue",
        type=Path,
        default=Path("epistemic_engine/outputs/epistemic_flow/trajectory_catalogue.csv"),
    )
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/natural_coordinates"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = natural_coordinate_report(args.trajectory_catalogue, args.out_dir, args.max_dim)
    best = summary["best_coordinate_system"]
    print("NATURAL COORDINATES PROGRAM")
    print(f"states: {summary['states']}")
    print(f"observables: {summary['observables']}")
    print(f"candidate_count: {summary['candidate_count']}")
    print(f"best: {best['name']} dim={best['dimension']} dynamics={best['dynamics']}")
    print(f"best_r2: {best['predictive_r2']:.3f}")
    print(f"simplicity_score: {best['simplicity_score']:.3f}")
    print(f"minimum_viable_dimension: {summary['hidden_state']['minimum_viable_dimension']}")
    print(f"previous_flow_r2: {summary['previous_flow_r2']}")
    print("coordinate_stability:")
    for axis, data in sorted(summary["coordinate_stability"].items()):
        print(f"  {axis}: {data['mean_subspace_similarity']:.3f} over {data['groups_evaluated']} groups")
    print(f"success_assessment: {summary['success_assessment']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
