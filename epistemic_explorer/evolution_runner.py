from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_explorer.evolution import EvolutionaryEcosystem
from epistemic_explorer.evolution_models import EcosystemSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolutionary research-strategy ecosystem experiments.")
    parser.add_argument("--populations", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("epistemic_explorer/outputs/evolution"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[EcosystemSummary] = []
    for population in args.populations:
        run_dir = args.out_dir / f"pop_{population}"
        ecosystem = EvolutionaryEcosystem(
            population_size=population,
            epochs=args.epochs,
            seed=args.seed + population,
            out_dir=run_dir,
        )
        summary = ecosystem.run()
        summaries.append(summary)
        print_summary(summary, run_dir)
    write_comparison(args.out_dir / "comparison.csv", summaries)
    print()
    print(f"comparison: {args.out_dir / 'comparison.csv'}")


def print_summary(summary: EcosystemSummary, run_dir: Path) -> None:
    print(f"POPULATION {summary.population_target}")
    print(f"  outputs: {run_dir}")
    print(f"  total_epistemic_value: {summary.total_epistemic_value:.2f}")
    print(f"  mean_selection_pressure: {summary.mean_selection_pressure:.3f}")
    print(f"  mean_experiment_yield: {summary.mean_experiment_yield:.3f}")
    print(f"  mean_bridge_score: {summary.mean_bridge_score:.3f}")
    print(f"  mean_research_diversity: {summary.mean_research_diversity:.3f}")
    print(f"  mean_dead_end_rate: {summary.mean_dead_end_rate:.3f}")
    print(f"  final_species_counts: {summary.final_species_counts}")


def write_comparison(path: Path, summaries: list[EcosystemSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = summary.to_dict()
            row["final_species_counts"] = str(row["final_species_counts"])
            writer.writerow(row)


if __name__ == "__main__":
    main()

