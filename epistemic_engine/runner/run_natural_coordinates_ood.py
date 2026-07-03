from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.natural_coordinates_ood import ood_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict OOD validation for frozen 2D natural coordinates."
    )
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--frozen-path",
        type=Path,
        default=Path("epistemic_engine/outputs/natural_coordinates_validation/frozen_coordinate_system.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/natural_coordinates_ood"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ood_validation_report(
        out_dir=args.out_dir,
        frozen_path=args.frozen_path,
        cases_per_family=args.cases_per_family,
        steps=args.steps,
        seed=args.seed,
    )
    accuracy = summary["prediction_accuracy"]
    boundary = summary["applicability_boundary"]
    competing = summary["competing_models"]
    assessment = summary["scientific_assessment"]
    print("NATURAL COORDINATES OOD VALIDATION")
    print(f"rule_0: {summary['rule_0']}")
    print(f"cases: {summary['cases']}")
    print(f"transitions: {summary['transitions']}")
    print(f"r2_mean: {accuracy['mean']:.3f}")
    print(f"r2_p05: {accuracy['p05']:.3f}")
    print(f"failure_rate: {summary['failure_atlas']['failure_rate']:.3f}")
    print(f"works_worlds: {boundary['works']['world']}")
    print(f"breaks_worlds: {boundary['breaks']['world']}")
    print(f"best_competing_model: {competing['best_model']}")
    print(f"frozen_model_rank: {competing['frozen_rank']}")
    print(f"verdict: {assessment['verdict']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()

