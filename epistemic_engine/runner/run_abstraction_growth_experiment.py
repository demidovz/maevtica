from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epistemic_engine.abstractions.agents import ErrorMinimizingBaseline, LifetimeReuseAgent
from epistemic_engine.abstractions.models import Abstraction, ExperimentSummary, StepMetrics
from epistemic_engine.abstractions.world import HiddenRegularityWorld, ToyWorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal abstraction-growth experiment: compare a "
            "lifetime/reuse abstraction builder against an error-minimizing baseline."
        )
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--train-objects", type=int, default=180)
    parser.add_argument("--transfer-objects", type=int, default=60)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("epistemic_engine/outputs/abstraction_growth"),
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def run_agent(agent, train_objects, transfer_objects, log_every: int) -> tuple[ExperimentSummary, list[StepMetrics]]:
    metrics: list[StepMetrics] = []
    for step, toy_object in enumerate(train_objects, start=1):
        agent.observe(toy_object, step)
        if step == 1 or step % log_every == 0 or step == len(train_objects):
            metrics.append(agent.metrics(step))
    return agent.summary(transfer_objects), metrics


def write_metrics(path: Path, metrics: list[StepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow(asdict(row))


def write_summary(path: Path, summaries: list[ExperimentSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(summary) for summary in summaries], handle, indent=2, ensure_ascii=False)


def write_graphviz(path: Path, abstractions: dict[str, Abstraction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("digraph knowledge_growth {\n")
        handle.write("  rankdir=TB;\n")
        handle.write("  node [shape=box, fontsize=10];\n")
        for abstraction in abstractions.values():
            conditions = " & ".join(f"{key}={value}" for key, value in sorted(abstraction.conditions))
            effects = ",".join(sorted(abstraction.expected_effects))
            status = "alive" if abstraction.alive else f"dead@{abstraction.destroyed_at}"
            label = (
                f"{abstraction.abstraction_id}\\n"
                f"{conditions}\\n"
                f"effects: {effects}\\n"
                f"life={abstraction.lifetime} reuse={abstraction.use_count} "
                f"depth={abstraction.depth}\\n"
                f"{status}"
            )
            color = "black" if abstraction.alive else "gray"
            handle.write(f'  "{abstraction.abstraction_id}" [label="{label}", color="{color}"];\n')
        for abstraction in abstractions.values():
            for parent in abstraction.parents:
                if parent in abstractions:
                    handle.write(f'  "{parent}" -> "{abstraction.abstraction_id}";\n')
        handle.write("}\n")


def print_summary(summary: ExperimentSummary) -> None:
    print(summary.agent_name)
    print(f"  observations: {summary.observations}")
    print(f"  abstractions: {summary.abstractions}")
    print(f"  alive_abstractions: {summary.alive_abstractions}")
    print(f"  max_depth: {summary.max_depth}")
    print(f"  mean_lifetime: {summary.mean_lifetime:.2f}")
    print(f"  mean_reuse: {summary.mean_reuse:.2f}")
    print(f"  total_life_reuse: {summary.total_life_reuse}")
    print(f"  transfer_reuse: {summary.transfer_reuse}")
    print(f"  destroyed: {summary.destroyed}")
    print(f"  merges: {summary.merges}")
    print(f"  splits: {summary.splits}")
    print(f"  compositions: {summary.compositions}")


def main() -> None:
    args = parse_args()
    world = HiddenRegularityWorld(
        ToyWorldConfig(
            seed=args.seed,
            train_objects=args.train_objects,
            transfer_objects=args.transfer_objects,
        )
    )
    train_objects = world.generate_train()
    transfer_objects = world.generate_transfer()

    lifetime_agent = LifetimeReuseAgent()
    baseline_agent = ErrorMinimizingBaseline()

    lifetime_summary, lifetime_metrics = run_agent(
        lifetime_agent,
        train_objects,
        transfer_objects,
        args.log_every,
    )
    baseline_summary, baseline_metrics = run_agent(
        baseline_agent,
        train_objects,
        transfer_objects,
        args.log_every,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(args.out_dir / "lifetime_reuse_metrics.csv", lifetime_metrics)
    write_metrics(args.out_dir / "error_baseline_metrics.csv", baseline_metrics)
    write_summary(args.out_dir / "summary.json", [lifetime_summary, baseline_summary])
    write_graphviz(args.out_dir / "lifetime_reuse_graph.dot", lifetime_agent.abstractions)
    write_graphviz(args.out_dir / "error_baseline_graph.dot", baseline_agent.abstractions)

    print("ABSTRACTION GROWTH EXPERIMENT")
    print(f"outputs: {args.out_dir}")
    print()
    print_summary(lifetime_summary)
    print()
    print_summary(baseline_summary)
    print()
    print("research_readout:")
    if lifetime_summary.max_depth > baseline_summary.max_depth:
        print("  hierarchy_depth: lifetime/reuse agent produced deeper structure.")
    else:
        print("  hierarchy_depth: no depth advantage for lifetime/reuse agent.")
    if lifetime_summary.transfer_reuse > baseline_summary.transfer_reuse:
        print("  transfer: lifetime/reuse abstractions reused more on new objects.")
    else:
        print("  transfer: baseline reused as much or more on new objects.")
    if lifetime_summary.total_life_reuse > baseline_summary.total_life_reuse:
        print("  life_reuse: lifetime/reuse objective produced longer-lived reusable abstractions.")
    else:
        print("  life_reuse: lifetime/reuse objective did not beat baseline on life*reuse.")


if __name__ == "__main__":
    main()
