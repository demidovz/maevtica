from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefArchiveClassSelectionConfig:
    feature_count: int = 8
    action_episodes: int = 64
    diagnostic_budget: int = 4
    seed: int = 7


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    exact_mode: str


@dataclass(frozen=True)
class Hypothesis:
    name: str
    family: str
    kind: str
    low_kind: str | None = None
    high_kind: str | None = None

    def predict(self, world: World) -> int:
        if self.family == "hybrid":
            if self.low_kind is None or self.high_kind is None:
                raise ValueError("Hybrid hypothesis requires both low_kind and high_kind")
            kind = self.low_kind if world[2] == 0 else self.high_kind
            return base_predict(kind, world)
        return base_predict(self.kind, world)


@dataclass(frozen=True)
class PolicyResult:
    scenario_name: str
    policy_name: str
    selected_family: str
    selected_mode: str
    exact_mode: int
    table_accuracy: float
    first_ten_accuracy: float
    overall_accuracy: float


def parse_args() -> BeliefArchiveClassSelectionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the system chooses hybrid composition when archived parts "
            "are sufficient and switches to a novel principle when composition fails."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--action-episodes", type=int, default=64)
    parser.add_argument("--diagnostic-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = BeliefArchiveClassSelectionConfig(
        feature_count=args.feature_count,
        action_episodes=args.action_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefArchiveClassSelectionConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_archive_class_selection_test currently expects feature_count == 8")
    if config.action_episodes <= 0:
        raise ValueError("action_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")


def base_predict(kind: str, world: World) -> int:
    feature_0 = world[0]
    feature_2 = world[2]
    feature_5 = world[5]

    if kind == "xor":
        return feature_0 ^ feature_5
    if kind == "and":
        return feature_0 & feature_5
    if kind == "xor3":
        return feature_0 ^ feature_2 ^ feature_5
    if kind == "maj3":
        return 1 if (feature_0 + feature_2 + feature_5) >= 2 else 0
    if kind == "nand3":
        return 1 - (feature_0 & feature_2 & feature_5)
    raise ValueError(f"Unknown kind: {kind}")


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="composable",
            description="if feature2=0 -> and(0, 5), else -> xor(0, 5)",
            exact_mode="gate_f2(and->xor)",
        ),
        Scenario(
            name="novel",
            description="majority(0, 2, 5)",
            exact_mode="maj3_0_2_5",
        ),
    ]


def build_archive_candidates() -> list[Hypothesis]:
    return [
        Hypothesis(name="archive_xor", family="archive", kind="xor"),
        Hypothesis(name="archive_and", family="archive", kind="and"),
        Hypothesis(name="archive_xor3", family="archive", kind="xor3"),
    ]


def build_hybrid_candidates() -> list[Hypothesis]:
    kinds = ["xor", "and", "xor3"]
    hybrids: list[Hypothesis] = []
    for low_kind in kinds:
        for high_kind in kinds:
            hybrids.append(
                Hypothesis(
                    name=f"gate_f2({low_kind}->{high_kind})",
                    family="hybrid",
                    kind="gate_f2",
                    low_kind=low_kind,
                    high_kind=high_kind,
                )
            )
    return hybrids


def build_novel_candidates() -> list[Hypothesis]:
    return [
        Hypothesis(name="maj3_0_2_5", family="novel", kind="maj3"),
        Hypothesis(name="nand3_0_2_5", family="novel", kind="nand3"),
    ]


def build_diagnostic_worlds(feature_count: int) -> list[World]:
    sequence = [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    worlds: list[World] = []
    for feature_0, feature_2, feature_5 in sequence:
        values = [0] * feature_count
        values[0] = feature_0
        values[2] = feature_2
        values[5] = feature_5
        worlds.append(tuple(values))
    return worlds


def build_truth_table(feature_count: int) -> list[World]:
    worlds: list[World] = []
    for feature_0 in (0, 1):
        for feature_2 in (0, 1):
            for feature_5 in (0, 1):
                values = [0] * feature_count
                values[0] = feature_0
                values[2] = feature_2
                values[5] = feature_5
                worlds.append(tuple(values))
    return worlds


def sample_action_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


def target_value(scenario: Scenario, world: World) -> int:
    if scenario.name == "composable":
        if world[2] == 0:
            return world[0] & world[5]
        return world[0] ^ world[5]
    if scenario.name == "novel":
        return 1 if (world[0] + world[2] + world[5]) >= 2 else 0
    raise ValueError(f"Unknown scenario: {scenario.name}")


def diagnostic_accuracy(
    hypothesis: Hypothesis,
    scenario: Scenario,
    diagnostic_worlds: Sequence[World],
) -> float:
    accuracies = [
        1 if hypothesis.predict(world) == target_value(scenario, world) else 0
        for world in diagnostic_worlds
    ]
    return statistics.mean(accuracies)


def family_rank(family: str) -> int:
    if family == "archive":
        return 0
    if family == "hybrid":
        return 1
    if family == "novel":
        return 2
    raise ValueError(f"Unknown family: {family}")


def select_best_candidate(
    candidates: Sequence[Hypothesis],
    scenario: Scenario,
    diagnostic_worlds: Sequence[World],
) -> Hypothesis:
    scored = [
        (diagnostic_accuracy(candidate, scenario, diagnostic_worlds), -family_rank(candidate.family), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return scored[0][2]


def run_policy(
    scenario: Scenario,
    policy_name: str,
    diagnostic_worlds: Sequence[World],
    truth_table: Sequence[World],
    action_worlds: Sequence[World],
) -> PolicyResult:
    archives = build_archive_candidates()
    hybrids = build_hybrid_candidates()
    novels = build_novel_candidates()

    if policy_name == "single_router":
        selected = select_best_candidate(archives, scenario, diagnostic_worlds)
    elif policy_name == "hybrid_only":
        selected = select_best_candidate(hybrids, scenario, diagnostic_worlds)
    elif policy_name == "adaptive_class_selector":
        selected = select_best_candidate(archives + hybrids + novels, scenario, diagnostic_worlds)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    table_accuracies = [
        1 if selected.predict(world) == target_value(scenario, world) else 0
        for world in truth_table
    ]
    action_accuracies = [
        1 if selected.predict(world) == target_value(scenario, world) else 0
        for world in action_worlds
    ]

    return PolicyResult(
        scenario_name=scenario.name,
        policy_name=policy_name,
        selected_family=selected.family,
        selected_mode=selected.name,
        exact_mode=1 if selected.name == scenario.exact_mode else 0,
        table_accuracy=statistics.mean(table_accuracies),
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
    )


def print_report(
    config: BeliefArchiveClassSelectionConfig,
    scenarios: Sequence[Scenario],
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: class selection between archive reuse, hybrid composition, and novelty")
    print("Archive library: xor(0, 5), and(0, 5), xor3(0, 2, 5)")
    print("Novel library: maj3(0, 2, 5), nand3(0, 2, 5)")
    print("Diagnostic sequence on features (0, 2, 5): 001, 011, 101, 111")
    print(f"Seed: {config.seed}")
    print()

    for scenario in scenarios:
        print(f"Scenario: {scenario.name}")
        print(f"True target: {scenario.description}")
        header = (
            f"{'Policy':<24}"
            f"{'Family':<10}"
            f"{'Selected':<22}"
            f"{'Exact':>8}"
            f"{'TableAcc':>10}"
            f"{'First10':>10}"
            f"{'Overall':>10}"
        )
        print(header)
        print("-" * len(header))
        for result in results:
            if result.scenario_name != scenario.name:
                continue
            print(
                f"{result.policy_name:<24}"
                f"{result.selected_family:<10}"
                f"{result.selected_mode:<22}"
                f"{result.exact_mode:>8}"
                f"{result.table_accuracy:>10.3f}"
                f"{result.first_ten_accuracy:>10.3f}"
                f"{result.overall_accuracy:>10.3f}"
            )
        print()


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)
    scenarios = build_scenarios()
    diagnostic_worlds = build_diagnostic_worlds(config.feature_count)[: config.diagnostic_budget]
    truth_table = build_truth_table(config.feature_count)
    action_worlds = sample_action_worlds(rng, config.feature_count, config.action_episodes)

    results: list[PolicyResult] = []
    for scenario in scenarios:
        for policy_name in ("single_router", "hybrid_only", "adaptive_class_selector"):
            results.append(
                run_policy(
                    scenario=scenario,
                    policy_name=policy_name,
                    diagnostic_worlds=diagnostic_worlds,
                    truth_table=truth_table,
                    action_worlds=action_worlds,
                )
            )

    print_report(config, scenarios, results)


if __name__ == "__main__":
    main()
