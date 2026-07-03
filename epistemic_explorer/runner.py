from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_explorer.cycle import ResearchCycleRunner
from epistemic_explorer.objectives import KnowledgeGrowthObjective, ReuseObjective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Epistemic Explorer MVP research loop.")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path("epistemic_explorer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ResearchCycleRunner(
        root=args.root,
        objectives=[KnowledgeGrowthObjective(), ReuseObjective()],
        seed=args.seed,
    )
    logs = runner.run(args.cycles)
    summary = runner.summary(logs)
    print("EPISTEMIC EXPLORER MVP")
    print(f"root: {args.root}")
    print(f"cycles: {summary['cycles']}")
    print(f"nodes: {summary['nodes']}")
    print(f"edges: {summary['edges']}")
    print(f"open_questions: {summary['open_questions']}")
    print(f"hypotheses_survived: {summary['hypotheses_survived']}")
    print(f"hypotheses_falsified: {summary['hypotheses_falsified']}")
    print(f"max_depth: {summary['max_depth']}")
    print("objective_totals:")
    for name, total in summary["objective_totals"].items():
        print(f"  {name}: {total:.2f}")
    print(f"graph: {args.root / 'knowledge' / 'graph.dot'}")


if __name__ == "__main__":
    main()

