from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.architecture_falsification import (
    evolutionary_counterexample_search,
    random_catalogue,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Architecture Falsification Program against C1-C5 necessity."
    )
    parser.add_argument("--architectures", type=int, default=2000)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/architecture_falsification"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architectures = 350 if args.quick else args.architectures
    population = 60 if args.quick else args.population
    generations = 4 if args.quick else args.generations

    catalogue = random_catalogue(architectures, args.seed)
    evolved = evolutionary_counterexample_search(population, generations, args.seed + 1)
    summary = write_outputs(args.out_dir, catalogue, evolved)
    universality = summary["universality_statistics"]

    print("ARCHITECTURE FALSIFICATION PROGRAM")
    print(f"architectures: {summary['architectures']}")
    print(f"random_architectures: {summary['random_architectures']}")
    print(f"evolved_architectures: {summary['evolved_architectures']}")
    print(f"e_star_complete: {summary['e_star_complete']}")
    print(f"counterexamples: {summary['counterexamples']}")
    print(f"representation_clusters: {summary['representation_emergence']['cluster_count']}")
    print(f"universality_successes: {universality['successful_count']}")
    print(f"mean_correspondence: {universality.get('mean_correspondence', 0.0):.3f}")
    print(f"empirical_confidence_lower_bound: {universality.get('empirical_confidence_lower_bound', 0.0):.3f}")
    print("negative_results:")
    for item in summary["negative_results"]:
        print(f"  {item}")
    if summary["genuine_counterexamples"]:
        print("best_counterexample:")
        best = summary["genuine_counterexamples"][0]
        print(f"  id: {best['architecture_id']}")
        print(f"  paradigm: {best['paradigm']}")
        print(f"  e_star_score: {best['e_star_score']:.3f}")
        print(f"  correspondence_score: {best['correspondence_score']:.3f}")
        print(f"  primitives: {' '.join(best['primitives'])}")
    else:
        best = summary["best_counterexample_candidates"][0] if summary["best_counterexample_candidates"] else None
        if best:
            print("best_near_counterexample:")
            print(f"  id: {best['architecture_id']}")
            print(f"  paradigm: {best['paradigm']}")
            print(f"  e_star_score: {best['e_star_score']:.3f}")
            print(f"  correspondence_score: {best['correspondence_score']:.3f}")
            print(f"  primitives: {' '.join(best['primitives'])}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
