from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.natural_coordinates_validation import validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen Koopman quadratic natural coordinates without retuning."
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--initial-conditions", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/natural_coordinates_validation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validation_report(args.seeds, args.out_dir, args.initial_conditions, args.steps)
    print("NATURAL COORDINATES VALIDATION PROGRAM")
    print(f"seeds: {summary['seeds']}")
    print(f"dimension: {summary['dimension']}")
    print(f"r2_mean: {summary['reproducibility']['mean']:.3f}")
    print(f"r2_p05: {summary['reproducibility']['p05']:.3f}")
    print(f"stability_mean: {summary['coordinate_stability']['mean']:.3f}")
    print(f"world_holdout_r2: {summary['holdout_worlds']['test_r2']:.3f}")
    print(f"paradigm_holdout_r2: {summary['holdout_paradigms']['test_r2']:.3f}")
    print(f"false_discovery_r2: {summary['false_discovery']['null_world_r2']:.3f}")
    print(f"verdict: {summary['verdict']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
