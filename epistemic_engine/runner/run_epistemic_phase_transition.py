from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.epistemic_phase_transition import phase_transition_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an empirical phase diagram of epistemic architecture space."
    )
    parser.add_argument("--points", type=int, default=900)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/epistemic_phase_transition"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = 240 if args.quick else args.points
    summary = phase_transition_report(points, args.seed, args.out_dir)
    print("EPISTEMIC PHASE TRANSITION PROGRAM")
    print(f"points: {summary['points']}")
    print(f"phase_count: {summary['phase_count']}")
    print(f"noise_fraction: {summary['noise_fraction']:.3f}")
    print(f"critical_edges: {summary['critical_edges']}")
    print(f"phase_change_rate: {summary['phase_change_rate']:.3f}")
    print(f"intrinsic_dimension_85pct: {summary['intrinsic_dimension_85pct']}")
    print(f"bootstrap_phase_ari: {summary['confidence_estimates']['bootstrap_phase_ari']:.3f}")
    print("top_order_parameters:")
    for name in summary["top_order_parameters"]:
        print(f"  {name}")
    print("hypothesis_support:")
    for name, value in sorted(summary["hypothesis_support"].items()):
        print(f"  {name}: {value:.3f}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
