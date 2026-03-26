from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefClassSwitchConfig:
    feature_count: int = 8
    warmup_repeats: int = 2
    reveal_repeats: int = 4
    hold_repeats: int = 4
    seed: int = 7


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
            active_kind = self.low_kind if world[2] == 0 else self.high_kind
            return base_predict(active_kind, world)
        return base_predict(self.kind, world)


@dataclass(frozen=True)
class PolicyResult:
    policy_name: str
    initial_mode: str
    final_mode: str
    first_novel_step: int | None
    warmup_accuracy: float
    reveal_accuracy: float
    hold_accuracy: float
    overall_accuracy: float


def parse_args() -> BeliefClassSwitchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the system can switch from a plausible hybrid class "
            "to a genuinely novel class when later evidence breaks the hybrid."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--reveal-repeats", type=int, default=4)
    parser.add_argument("--hold-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = BeliefClassSwitchConfig(
        feature_count=args.feature_count,
        warmup_repeats=args.warmup_repeats,
        reveal_repeats=args.reveal_repeats,
        hold_repeats=args.hold_repeats,
        seed=args.seed,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefClassSwitchConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_class_switch_test currently expects feature_count == 8")
    if min(config.warmup_repeats, config.reveal_repeats, config.hold_repeats) <= 0:
        raise ValueError("all repeat counts must be > 0")


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


def build_world(feature_count: int, feature_0: int, feature_2: int, feature_5: int) -> World:
    values = [0] * feature_count
    values[0] = feature_0
    values[2] = feature_2
    values[5] = feature_5
    return tuple(values)


def build_diagnostic_worlds(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, 0, 0, 1),
        build_world(feature_count, 0, 1, 1),
        build_world(feature_count, 1, 0, 1),
    ]


def build_warmup_worlds(feature_count: int, repeats: int) -> list[World]:
    template = build_diagnostic_worlds(feature_count)
    return template * repeats


def build_reveal_worlds(feature_count: int, repeats: int) -> list[World]:
    return [build_world(feature_count, 1, 1, 1)] * repeats


def build_hold_worlds(feature_count: int, repeats: int, seed: int) -> list[World]:
    worlds = [
        build_world(feature_count, feature_0, feature_2, feature_5)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
    ] * repeats
    rng = random.Random(seed)
    rng.shuffle(worlds)
    return worlds


def true_target(world: World) -> int:
    return 1 if (world[0] + world[2] + world[5]) >= 2 else 0


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
    observations: Sequence[tuple[World, int]],
) -> Hypothesis:
    scored = []
    for candidate in candidates:
        accuracy = statistics.mean(
            1 if candidate.predict(world) == true_action else 0
            for world, true_action in observations
        )
        scored.append((accuracy, -family_rank(candidate.family), candidate.name, candidate))
    scored.sort(reverse=True)
    return scored[0][3]


def run_policy(
    policy_name: str,
    diagnostic_worlds: Sequence[World],
    warmup_worlds: Sequence[World],
    reveal_worlds: Sequence[World],
    hold_worlds: Sequence[World],
) -> PolicyResult:
    archives = build_archive_candidates()
    hybrids = build_hybrid_candidates()
    novels = build_novel_candidates()

    if policy_name == "static_commit":
        candidate_pool = archives + hybrids + novels
        reselection_pool: Sequence[Hypothesis] | None = None
    elif policy_name == "hybrid_revision":
        candidate_pool = archives + hybrids + novels
        reselection_pool = archives + hybrids
    elif policy_name == "adaptive_class_switch":
        candidate_pool = archives + hybrids + novels
        reselection_pool = archives + hybrids + novels
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    observations = [(world, true_target(world)) for world in diagnostic_worlds]
    selected = select_best_candidate(candidate_pool, observations)
    initial_mode = selected.name
    first_novel_step: int | None = None

    all_worlds = list(warmup_worlds) + list(reveal_worlds) + list(hold_worlds)
    accuracies: list[int] = []

    for step_index, world in enumerate(all_worlds):
        prediction = selected.predict(world)
        actual = true_target(world)
        accuracies.append(1 if prediction == actual else 0)
        observations.append((world, actual))

        if reselection_pool is not None:
            updated = select_best_candidate(reselection_pool, observations)
            if first_novel_step is None and updated.family == "novel":
                first_novel_step = step_index
            selected = updated

    warmup_end = len(warmup_worlds)
    reveal_end = warmup_end + len(reveal_worlds)

    return PolicyResult(
        policy_name=policy_name,
        initial_mode=initial_mode,
        final_mode=selected.name,
        first_novel_step=first_novel_step,
        warmup_accuracy=statistics.mean(accuracies[:warmup_end]),
        reveal_accuracy=statistics.mean(accuracies[warmup_end:reveal_end]),
        hold_accuracy=statistics.mean(accuracies[reveal_end:]),
        overall_accuracy=statistics.mean(accuracies),
    )


def print_report(
    config: BeliefClassSwitchConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: online class switching from hybrid reuse to novel principle")
    print("True world rule: majority(0, 2, 5)")
    print("Initial diagnostic worlds: 001, 011, 101")
    print("These initial worlds are perfectly consistent with gate_f2(and->xor)")
    print("Reveal world: 111 repeated to break the hybrid interpretation")
    print(
        f"Warmup repeats: {config.warmup_repeats}, "
        f"Reveal repeats: {config.reveal_repeats}, "
        f"Hold repeats: {config.hold_repeats}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<24}"
        f"{'Initial':<22}"
        f"{'Final':<22}"
        f"{'NovelAt':>10}"
        f"{'Warmup':>10}"
        f"{'Reveal':>10}"
        f"{'Hold':>10}"
        f"{'Overall':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        switch_value = "-" if result.first_novel_step is None else str(result.first_novel_step)
        print(
            f"{result.policy_name:<24}"
            f"{result.initial_mode:<22}"
            f"{result.final_mode:<22}"
            f"{switch_value:>10}"
            f"{result.warmup_accuracy:>10.3f}"
            f"{result.reveal_accuracy:>10.3f}"
            f"{result.hold_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
        )


def main() -> None:
    config = parse_args()
    diagnostic_worlds = build_diagnostic_worlds(config.feature_count)
    warmup_worlds = build_warmup_worlds(config.feature_count, config.warmup_repeats)
    reveal_worlds = build_reveal_worlds(config.feature_count, config.reveal_repeats)
    hold_worlds = build_hold_worlds(config.feature_count, config.hold_repeats, config.seed)

    results = [
        run_policy(
            policy_name="static_commit",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
        ),
        run_policy(
            policy_name="hybrid_revision",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
        ),
        run_policy(
            policy_name="adaptive_class_switch",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
        ),
    ]

    print_report(config, results)


if __name__ == "__main__":
    main()
