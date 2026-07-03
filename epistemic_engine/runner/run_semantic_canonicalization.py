from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.semantic_canonicalization import semantic_canonicalization_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize anonymous architecture behavior and test whether counterexamples collapse semantically."
    )
    parser.add_argument("--architectures", type=int, default=2000)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--analysis-limit", type=int, default=650)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/semantic_canonicalization"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architectures = 350 if args.quick else args.architectures
    population = 60 if args.quick else args.population
    generations = 4 if args.quick else args.generations
    summary = semantic_canonicalization_report(
        architectures,
        population,
        generations,
        args.seed,
        args.out_dir,
        args.analysis_limit,
    )
    hypotheses = summary["hypothesis_support"]
    iso = summary["semantic_isomorphism"]
    print("SEMANTIC CANONICALIZATION PROGRAM")
    print(f"architectures: {summary['architectures']}")
    print(f"analyzed_architectures: {summary['analyzed_architectures']}")
    print(f"generated_e_star_complete: {summary['generated_e_star_complete']}")
    print(f"analyzed_e_star_complete: {summary['e_star_complete']}")
    print(f"role_count: {summary['role_count']}")
    print(f"canonical_class_count: {summary['canonical_class_count']}")
    print(f"successful_canonical_class_count: {summary['successful_canonical_class_count']}")
    print(f"mean_lab_distance: {iso['mean_lab_distance']:.3f}")
    print("weakest_correspondence_counts:")
    for name, count in sorted(iso["weakest_correspondence_counts"].items()):
        print(f"  {name}: {count}")
    print("hypothesis_support:")
    for name, value in sorted(hypotheses.items()):
        print(f"  {name}: {value:.3f}")
    print(f"genuine_counterexamples: {len(summary['genuine_counterexamples'])}")
    if summary["genuine_counterexamples"]:
        best = summary["genuine_counterexamples"][0]
        print("strongest_genuine_counterexample:")
        print(f"  id: {best['architecture_id']}")
        print(f"  paradigm: {best['paradigm']}")
        print(f"  e_star_score: {best['e_star_score']:.3f}")
        print(f"  lab_distance: {best['lab_distance']:.3f}")
        print(f"  canonical_class: {best['canonical_class']}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
