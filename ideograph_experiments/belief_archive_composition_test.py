from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefArchiveCompositionConfig:
    feature_count: int = 8
    action_episodes: int = 64
    diagnostic_budget: int = 4
    seed: int = 7
    consistency: float = 0.92
    prior_xor: float = 0.25
    prior_and: float = 0.20
    prior_xor3: float = 0.55


@dataclass(frozen=True)
class ArchiveHypothesis:
    name: str
    kind: str

    def predict(self, world: World) -> int:
        if self.kind == "xor":
            return world[0] ^ world[5]
        if self.kind == "and":
            return world[0] & world[5]
        if self.kind == "xor3":
            return world[0] ^ world[2] ^ world[5]
        raise ValueError(f"Unknown archive kind: {self.kind}")


@dataclass(frozen=True)
class HybridHypothesis:
    name: str
    low_context_archive: str
    high_context_archive: str


@dataclass(frozen=True)
class PolicyResult:
    policy_name: str
    probe_plan: str
    selected_mode: str
    exact_mode: int
    table_accuracy: float
    first_ten_accuracy: float
    overall_accuracy: float


def parse_args() -> BeliefArchiveCompositionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the system can compose a new regime from parts of "
            "multiple archived regimes when no single archive fits fully."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--action-episodes", type=int, default=64)
    parser.add_argument("--diagnostic-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--prior-xor", type=float, default=0.25)
    parser.add_argument("--prior-and", type=float, default=0.20)
    parser.add_argument("--prior-xor3", type=float, default=0.55)
    args = parser.parse_args()

    config = BeliefArchiveCompositionConfig(
        feature_count=args.feature_count,
        action_episodes=args.action_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
        consistency=args.consistency,
        prior_xor=args.prior_xor,
        prior_and=args.prior_and,
        prior_xor3=args.prior_xor3,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefArchiveCompositionConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_archive_composition_test currently expects feature_count == 8")
    if config.action_episodes <= 0:
        raise ValueError("action_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    prior_total = config.prior_xor + config.prior_and + config.prior_xor3
    if not math.isclose(prior_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("archive priors must sum to 1.0")
    if min(config.prior_xor, config.prior_and, config.prior_xor3) <= 0.0:
        raise ValueError("all archive priors must be > 0")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def build_archives() -> list[ArchiveHypothesis]:
    return [
        ArchiveHypothesis(name="archive_xor", kind="xor"),
        ArchiveHypothesis(name="archive_and", kind="and"),
        ArchiveHypothesis(name="archive_xor3", kind="xor3"),
    ]


def initial_scores(config: BeliefArchiveCompositionConfig) -> list[float]:
    return [
        math.log(config.prior_xor),
        math.log(config.prior_and),
        math.log(config.prior_xor3),
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


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


def target_value(world: World) -> int:
    if world[2] == 0:
        return world[0] & world[5]
    return world[0] ^ world[5]


def update_scores(
    scores: list[float],
    hypotheses: Sequence[ArchiveHypothesis],
    world: World,
    true_action: int,
    consistency: float,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        likelihood = consistency if hypothesis.predict(world) == true_action else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[ArchiveHypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def predict_with_archive_mix(
    scores: Sequence[float],
    hypotheses: Sequence[ArchiveHypothesis],
    world: World,
) -> int:
    belief = softmax(scores)
    probability_action_one = sum(
        weight * hypothesis.predict(world)
        for weight, hypothesis in zip(belief, hypotheses)
    )
    return 1 if probability_action_one >= 0.5 else 0


def choose_best_archive_for_context(
    hypotheses: Sequence[ArchiveHypothesis],
    observations: Sequence[tuple[World, int]],
    context_value: int,
) -> ArchiveHypothesis:
    candidates = [item for item in observations if item[0][2] == context_value]
    if not candidates:
        raise ValueError("No observations for requested context")

    best_archive = hypotheses[0]
    best_accuracy = -1.0
    for hypothesis in hypotheses:
        accuracy = statistics.mean(
            1 if hypothesis.predict(world) == true_action else 0
            for world, true_action in candidates
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_archive = hypothesis
    return best_archive


def predict_with_hybrid(
    hybrid: HybridHypothesis,
    archive_map: dict[str, ArchiveHypothesis],
    world: World,
) -> int:
    if world[2] == 0:
        return archive_map[hybrid.low_context_archive].predict(world)
    return archive_map[hybrid.high_context_archive].predict(world)


def run_policy(
    config: BeliefArchiveCompositionConfig,
    policy_name: str,
    diagnostic_worlds: Sequence[World],
    truth_table: Sequence[World],
    action_worlds: Sequence[World],
) -> PolicyResult:
    archives = build_archives()
    archive_map = {archive.name: archive for archive in archives}
    scores = initial_scores(config)
    observations: list[tuple[World, int]] = []
    probe_plan: list[str] = []

    for world in diagnostic_worlds[: config.diagnostic_budget]:
        true_action = target_value(world)
        observations.append((world, true_action))
        probe_plan.append(probe_label(world))
        update_scores(scores, archives, world, true_action, config.consistency)

    if policy_name == "single_router":
        selected_mode, _ = top_hypothesis(scores, archives)
        predictor = lambda world: archive_map[selected_mode].predict(world)
    elif policy_name == "diagnostic_mix":
        selected_mode, _ = top_hypothesis(scores, archives)
        predictor = lambda world: predict_with_archive_mix(scores, archives, world)
    elif policy_name == "hybrid_composer":
        low_archive = choose_best_archive_for_context(archives, observations, context_value=0)
        high_archive = choose_best_archive_for_context(archives, observations, context_value=1)
        hybrid = HybridHypothesis(
            name=f"gate_f2({low_archive.kind}->{high_archive.kind})",
            low_context_archive=low_archive.name,
            high_context_archive=high_archive.name,
        )
        selected_mode = hybrid.name
        predictor = lambda world: predict_with_hybrid(hybrid, archive_map, world)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    table_accuracies = [
        1 if predictor(world) == target_value(world) else 0
        for world in truth_table
    ]
    action_accuracies = [
        1 if predictor(world) == target_value(world) else 0
        for world in action_worlds
    ]

    exact_mode = 1 if selected_mode == "gate_f2(and->xor)" else 0
    return PolicyResult(
        policy_name=policy_name,
        probe_plan=",".join(probe_plan),
        selected_mode=selected_mode,
        exact_mode=exact_mode,
        table_accuracy=statistics.mean(table_accuracies),
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
    )


def print_report(
    config: BeliefArchiveCompositionConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: composing a new regime from multiple archived regimes")
    print("Archive A: xor(0, 5)")
    print("Archive B: and(0, 5)")
    print("Archive C: xor3(0, 2, 5)")
    print("True target: if feature2=0 -> and(0, 5), else -> xor(0, 5)")
    print("Diagnostic sequence on features (0, 2, 5): 001, 011, 101, 111")
    print(
        f"Archive priors: xor={config.prior_xor:.2f}, "
        f"and={config.prior_and:.2f}, xor3={config.prior_xor3:.2f}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<18}"
        f"{'Plan':<16}"
        f"{'Selected':<22}"
        f"{'Exact':>8}"
        f"{'TableAcc':>10}"
        f"{'First10':>10}"
        f"{'Overall':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.policy_name:<18}"
            f"{result.probe_plan:<16}"
            f"{result.selected_mode:<22}"
            f"{result.exact_mode:>8}"
            f"{result.table_accuracy:>10.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)
    diagnostic_worlds = build_diagnostic_worlds(config.feature_count)
    truth_table = build_truth_table(config.feature_count)
    action_worlds = sample_action_worlds(rng, config.feature_count, config.action_episodes)

    results = [
        run_policy(
            config,
            policy_name="single_router",
            diagnostic_worlds=diagnostic_worlds,
            truth_table=truth_table,
            action_worlds=action_worlds,
        ),
        run_policy(
            config,
            policy_name="diagnostic_mix",
            diagnostic_worlds=diagnostic_worlds,
            truth_table=truth_table,
            action_worlds=action_worlds,
        ),
        run_policy(
            config,
            policy_name="hybrid_composer",
            diagnostic_worlds=diagnostic_worlds,
            truth_table=truth_table,
            action_worlds=action_worlds,
        ),
    ]

    print_report(config, results)


if __name__ == "__main__":
    main()
