from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.ood_validation import ood_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OOD validation for the frozen 2D Koopman affine natural-coordinate law."
    )
    parser.add_argument("--initial-conditions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/ood_validation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ood_validation_report(args.out_dir, args.initial_conditions, args.steps, args.seed)
    aggregate = summary["aggregate"]
    dist = summary["prediction_accuracy_distribution"]
    print("OUT-OF-DISTRIBUTION VALIDATION PROGRAM")
    print(f"ood_suites: {summary['ood_suites']}")
    print(f"aggregate_r2: {aggregate['r2']:.3f}")
    print(f"r2_mean: {dist['mean']:.3f}")
    print(f"r2_p05: {dist['p05']:.3f}")
    print(f"r2_p95: {dist['p95']:.3f}")
    print(f"failure_count: {summary['failure_count']}")
    print("competing_models:")
    for name, value in sorted(summary["competing_models"].items()):
        print(f"  {name}: {value:.3f}")
    print(f"verdict: {summary['scientific_verdict']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
