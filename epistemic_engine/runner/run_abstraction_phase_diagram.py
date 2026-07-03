from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.phase_diagram import (
    AblationSpec,
    ObjectiveGrowthAgent,
    PhaseRunMetrics,
    PhaseWorld,
    PhaseWorldSpec,
    objective_specs,
    reviewer_ablations,
    world_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small objective x world phase diagram for abstraction growth dynamics."
    )
    parser.add_argument("--train-objects", type=int, default=120)
    parser.add_argument("--transfer-objects", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--reviewer-suite", action="store_true")
    parser.add_argument("--noise-sweep", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small representative subset for fast smoke testing.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_phase_diagram"),
    )
    return parser.parse_args()


def run_one(
    objective,
    world_spec: PhaseWorldSpec,
    ablation: AblationSpec,
    seed: int,
    train_objects: int,
    transfer_objects: int,
) -> PhaseRunMetrics:
    world = PhaseWorld(world_spec, seed)
    train = world.train(train_objects)
    transfer = world.transfer(transfer_objects)
    if ablation.shuffle_stream:
        random.Random(seed + 10_000).shuffle(train)
    agent = ObjectiveGrowthAgent(objective, ablation)
    for step, toy_object in enumerate(train, start=1):
        agent.observe(toy_object, step)
    metrics = agent.metrics(transfer)
    return replace(metrics, objective=objective.name, world=world_spec.name, ablation=ablation.name, seed=seed)


def aggregate_regimes(rows: list[PhaseRunMetrics]) -> dict[tuple[str, str, str], str]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        key = (row.objective, row.world, row.ablation)
        grouped.setdefault(key, []).append(row.regime)
    result: dict[tuple[str, str, str], str] = {}
    for key, regimes in grouped.items():
        counts: dict[str, int] = {}
        for regime in regimes:
            counts[regime] = counts.get(regime, 0) + 1
        result[key] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return result


def write_rows(path: Path, rows: list[PhaseRunMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_phase_map(path: Path, rows: list[PhaseRunMetrics], ablation_name: str) -> None:
    aggregate = aggregate_regimes(rows)
    objectives = sorted({row.objective for row in rows})
    worlds = sorted({row.world for row in rows})
    with path.open("w", encoding="utf-8") as handle:
        handle.write("objective," + ",".join(worlds) + "\n")
        for objective in objectives:
            cells = [aggregate.get((objective, world, ablation_name), "") for world in worlds]
            handle.write(objective + "," + ",".join(cells) + "\n")


def write_summary(path: Path, rows: list[PhaseRunMetrics]) -> None:
    aggregate = aggregate_regimes(rows)
    regime_counts: dict[str, int] = {}
    for regime in aggregate.values():
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    payload = {
        "runs": len(rows),
        "cells": len(aggregate),
        "regime_counts": dict(sorted(regime_counts.items())),
        "top_regimes_by_cell": {
            "|".join(key): value
            for key, value in sorted(aggregate.items())
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def selected_objectives(quick: bool):
    specs = objective_specs()
    if not quick:
        return specs
    keep = {"exact_memory", "accuracy", "compression", "lifetime_reuse", "novelty", "prediction_lifetime"}
    return [spec for spec in specs if spec.name in keep]


def selected_worlds(noise_sweep: bool, quick: bool) -> list[PhaseWorldSpec]:
    specs = world_specs()
    if quick:
        keep = {"random", "no_regularities", "hierarchical", "compositional", "partial", "anti_compositional"}
        specs = [spec for spec in specs if spec.name in keep]
    if not noise_sweep:
        return specs
    extra = [PhaseWorldSpec(f"noise_{level:.2f}", noise=level) for level in (0.0, 0.05, 0.1, 0.2, 0.35)]
    return specs + extra


def selected_ablations(reviewer_suite: bool, quick: bool) -> list[AblationSpec]:
    if reviewer_suite:
        specs = reviewer_ablations()
        if quick:
            keep = {"standard", "equal_abstraction_budget", "remove_composition", "remove_hierarchy"}
            specs = [spec for spec in specs if spec.name in keep]
        return specs
    return [AblationSpec("standard")]


def print_readout(rows: list[PhaseRunMetrics]) -> None:
    aggregate = aggregate_regimes(rows)
    regime_counts: dict[str, int] = {}
    for regime in aggregate.values():
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    print("ABSTRACTION PHASE DIAGRAM")
    print(f"runs: {len(rows)}")
    print(f"cells: {len(aggregate)}")
    print("regimes:")
    for regime, count in sorted(regime_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {regime}: {count}")
    print("sample_cells:")
    for key, regime in list(sorted(aggregate.items()))[:12]:
        objective, world, ablation = key
        print(f"  {objective} x {world} x {ablation} -> {regime}")


def main() -> None:
    args = parse_args()
    rows: list[PhaseRunMetrics] = []
    objectives = selected_objectives(args.quick)
    worlds = selected_worlds(args.noise_sweep, args.quick)
    ablations = selected_ablations(args.reviewer_suite, args.quick)
    seeds = list(range(args.seeds))

    for objective in objectives:
        for world in worlds:
            for ablation in ablations:
                for seed in seeds:
                    rows.append(
                        run_one(
                            objective,
                            world,
                            ablation,
                            seed,
                            args.train_objects,
                            args.transfer_objects,
                        )
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "phase_runs.csv", rows)
    write_phase_map(args.out_dir / "phase_map_standard.csv", rows, "standard")
    write_summary(args.out_dir / "phase_summary.json", rows)
    print_readout(rows)
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
