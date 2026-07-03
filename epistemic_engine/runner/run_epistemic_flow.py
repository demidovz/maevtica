from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.epistemic_flow import flow_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct epistemic flow from architecture-as-initial-condition trajectories."
    )
    parser.add_argument("--initial-conditions", type=int, default=10)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/epistemic_flow"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_conditions = 4 if args.quick else args.initial_conditions
    steps = 8 if args.quick else args.steps
    summary = flow_report(initial_conditions, steps, args.seed, args.out_dir)
    print("EPISTEMIC FLOW PROGRAM")
    print(f"states: {summary['states']}")
    print(f"trajectories: {summary['trajectories']}")
    print(f"intrinsic_dimension_85pct: {summary['intrinsic_dimension_85pct']}")
    print(f"mean_speed: {summary['mean_speed']:.3f}")
    print(f"endpoint_attractor_count: {summary['endpoint_attractor_count']}")
    print(f"limit_cycle_candidates: {summary['limit_cycle_candidates']}")
    print(f"empirical_law_r2: {summary['empirical_law_r2']:.3f}")
    print(f"coordinate_invariance_mean: {summary['coordinate_invariance_mean']:.3f}")
    print("hypothesis_support:")
    for name, value in sorted(summary["hypothesis_support"].items()):
        print(f"  {name}: {value:.3f}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
