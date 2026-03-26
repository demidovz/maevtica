from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefClassSwitchDiagnosticConfig:
    feature_count: int = 8
    warmup_repeats: int = 2
    reveal_repeats: int = 4
    hold_repeats: int = 4
    seed: int = 7
    consistency: float = 0.8
    prior_archive: float = 0.15
    prior_hybrid: float = 0.80
    prior_novel: float = 0.05
    diagnostic_questions: int = 1


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
    diagnostic_step: int | None
    diagnostic_probe: str
    questions_asked: int
    reveal_accuracy: float
    hold_accuracy: float
    overall_accuracy: float


def parse_args() -> BeliefClassSwitchDiagnosticConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a diagnostic question shortens the lag between early "
            "class-misfit and the switch to a novel class."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--reveal-repeats", type=int, default=4)
    parser.add_argument("--hold-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.8)
    parser.add_argument("--prior-archive", type=float, default=0.15)
    parser.add_argument("--prior-hybrid", type=float, default=0.80)
    parser.add_argument("--prior-novel", type=float, default=0.05)
    parser.add_argument("--diagnostic-questions", type=int, default=1)
    args = parser.parse_args()

    config = BeliefClassSwitchDiagnosticConfig(
        feature_count=args.feature_count,
        warmup_repeats=args.warmup_repeats,
        reveal_repeats=args.reveal_repeats,
        hold_repeats=args.hold_repeats,
        seed=args.seed,
        consistency=args.consistency,
        prior_archive=args.prior_archive,
        prior_hybrid=args.prior_hybrid,
        prior_novel=args.prior_novel,
        diagnostic_questions=args.diagnostic_questions,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefClassSwitchDiagnosticConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_class_switch_diagnostic_test currently expects feature_count == 8")
    if min(config.warmup_repeats, config.reveal_repeats, config.hold_repeats) <= 0:
        raise ValueError("all repeat counts must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.diagnostic_questions <= 0:
        raise ValueError("diagnostic_questions must be > 0")
    prior_total = config.prior_archive + config.prior_hybrid + config.prior_novel
    if not math.isclose(prior_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("family priors must sum to 1.0")
    if min(config.prior_archive, config.prior_hybrid, config.prior_novel) <= 0.0:
        raise ValueError("all family priors must be > 0")


def base_predict(kind: str, world: World) -> int:
    feature_0 = world[0]
    feature_2 = world[2]
    feature_5 = world[5]

    if kind == "xor_05":
        return feature_0 ^ feature_5
    if kind == "and_05":
        return feature_0 & feature_5
    if kind == "xor3_025":
        return feature_0 ^ feature_2 ^ feature_5
    if kind == "or_02":
        return feature_0 | feature_2
    if kind == "or_25":
        return feature_2 | feature_5
    if kind == "maj3_025":
        return 1 if (feature_0 + feature_2 + feature_5) >= 2 else 0
    if kind == "nand3_025":
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
    return build_diagnostic_worlds(feature_count) * repeats


def build_reveal_worlds(feature_count: int, repeats: int) -> list[World]:
    return [build_world(feature_count, 1, 1, 1)] * repeats


def build_truth_table(feature_count: int) -> list[World]:
    return [
        build_world(feature_count, feature_0, feature_2, feature_5)
        for feature_0 in (0, 1)
        for feature_2 in (0, 1)
        for feature_5 in (0, 1)
    ]


def build_hold_worlds(feature_count: int, repeats: int, seed: int) -> list[World]:
    worlds = build_truth_table(feature_count) * repeats
    rng = random.Random(seed)
    rng.shuffle(worlds)
    return worlds


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


def true_target(world: World) -> int:
    return world[0] | world[2]


def build_archive_candidates() -> list[Hypothesis]:
    return [
        Hypothesis(name="archive_xor", family="archive", kind="xor_05"),
        Hypothesis(name="archive_and", family="archive", kind="and_05"),
        Hypothesis(name="archive_xor3", family="archive", kind="xor3_025"),
    ]


def build_hybrid_candidates() -> list[Hypothesis]:
    kinds = ["xor_05", "and_05", "xor3_025"]
    candidates: list[Hypothesis] = []
    for low_kind in kinds:
        for high_kind in kinds:
            candidates.append(
                Hypothesis(
                    name=f"gate_f2({low_kind}->{high_kind})",
                    family="hybrid",
                    kind="gate_f2",
                    low_kind=low_kind,
                    high_kind=high_kind,
                )
            )
    return candidates


def build_novel_candidates() -> list[Hypothesis]:
    return [
        Hypothesis(name="or_0_2", family="novel", kind="or_02"),
        Hypothesis(name="or_2_5", family="novel", kind="or_25"),
        Hypothesis(name="maj3_0_2_5", family="novel", kind="maj3_025"),
        Hypothesis(name="nand3_0_2_5", family="novel", kind="nand3_025"),
    ]


def initial_scores(
    config: BeliefClassSwitchDiagnosticConfig,
    archives: Sequence[Hypothesis],
    hybrids: Sequence[Hypothesis],
    novels: Sequence[Hypothesis],
) -> list[float]:
    archive_prior = config.prior_archive / len(archives)
    hybrid_prior = config.prior_hybrid / len(hybrids)
    novel_prior = config.prior_novel / len(novels)
    return (
        [math.log(archive_prior)] * len(archives)
        + [math.log(hybrid_prior)] * len(hybrids)
        + [math.log(novel_prior)] * len(novels)
    )


def update_scores(
    scores: list[float],
    hypotheses: Sequence[Hypothesis],
    world: World,
    actual: int,
    consistency: float,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        likelihood = consistency if hypothesis.predict(world) == actual else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def family_rank(family: str) -> int:
    if family == "archive":
        return 0
    if family == "hybrid":
        return 1
    if family == "novel":
        return 2
    raise ValueError(f"Unknown family: {family}")


def top_index(scores: Sequence[float], hypotheses: Sequence[Hypothesis]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (scores[index], -family_rank(hypotheses[index].family), hypotheses[index].name),
    )


def choose_diagnostic_probe(
    hypotheses: Sequence[Hypothesis],
    scores: Sequence[float],
    observed_labels: set[str],
    truth_table: Sequence[World],
) -> World:
    top_overall = hypotheses[top_index(scores, hypotheses)]
    novel_indices = [index for index, item in enumerate(hypotheses) if item.family == "novel"]
    best_novel_index = max(novel_indices, key=lambda index: (scores[index], hypotheses[index].name))
    best_novel = hypotheses[best_novel_index]

    candidate_worlds = [world for world in truth_table if probe_label(world) not in observed_labels]
    if not candidate_worlds:
        raise ValueError("No diagnostic worlds left to query")

    best_world = candidate_worlds[0]
    best_score = -1.0
    for world in candidate_worlds:
        novel_support = statistics.mean(
            1 if hypotheses[index].predict(world) == 1 else 0
            for index in novel_indices
        )
        separation = abs(top_overall.predict(world) - best_novel.predict(world))
        score = separation + abs(novel_support - top_overall.predict(world))
        if score > best_score:
            best_score = score
            best_world = world
    return best_world


def run_policy(
    config: BeliefClassSwitchDiagnosticConfig,
    policy_name: str,
    diagnostic_worlds: Sequence[World],
    warmup_worlds: Sequence[World],
    reveal_worlds: Sequence[World],
    hold_worlds: Sequence[World],
    truth_table: Sequence[World],
) -> PolicyResult:
    archives = build_archive_candidates()
    hybrids = build_hybrid_candidates()
    novels = build_novel_candidates()
    hypotheses = archives + hybrids + novels
    scores = initial_scores(config, archives, hybrids, novels)

    observed_labels = set()
    for world in diagnostic_worlds:
        update_scores(scores, hypotheses, world, true_target(world), config.consistency)
        observed_labels.add(probe_label(world))

    selected = hypotheses[top_index(scores, hypotheses)]
    initial_mode = selected.name
    first_novel_step: int | None = None
    diagnostic_step: int | None = None
    diagnostic_probe = "-"
    questions_asked = 0

    all_worlds = list(warmup_worlds) + list(reveal_worlds) + list(hold_worlds)
    reveal_start = len(warmup_worlds)
    reveal_end = reveal_start + len(reveal_worlds)
    accuracies: list[int] = []

    for step_index, world in enumerate(all_worlds):
        prediction = selected.predict(world)
        actual = true_target(world)
        correct = 1 if prediction == actual else 0
        accuracies.append(correct)
        update_scores(scores, hypotheses, world, actual, config.consistency)
        observed_labels.add(probe_label(world))

        if (
            policy_name == "diagnostic_switch"
            and questions_asked < config.diagnostic_questions
            and step_index >= reveal_start
            and step_index < reveal_end
            and correct == 0
        ):
            probe = choose_diagnostic_probe(hypotheses, scores, observed_labels, truth_table)
            diagnostic_step = step_index
            diagnostic_probe = probe_label(probe)
            questions_asked += 1
            update_scores(scores, hypotheses, probe, true_target(probe), config.consistency)
            observed_labels.add(probe_label(probe))

        if policy_name == "static_commit":
            updated = selected
        else:
            updated = hypotheses[top_index(scores, hypotheses)]

        if first_novel_step is None and updated.family == "novel":
            first_novel_step = step_index
        selected = updated

    return PolicyResult(
        policy_name=policy_name,
        initial_mode=initial_mode,
        final_mode=selected.name,
        first_novel_step=first_novel_step,
        diagnostic_step=diagnostic_step,
        diagnostic_probe=diagnostic_probe,
        questions_asked=questions_asked,
        reveal_accuracy=statistics.mean(accuracies[reveal_start:reveal_end]),
        hold_accuracy=statistics.mean(accuracies[reveal_end:]),
        overall_accuracy=statistics.mean(accuracies),
    )


def print_report(
    config: BeliefClassSwitchDiagnosticConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: diagnostic question during online class switching")
    print("True world rule: or(0, 2)")
    print("Initial diagnostic worlds: 001, 011, 101")
    print("These worlds support both gate_f2(and->xor) and the novel rule or(0, 2)")
    print("Reveal world: 111 repeated")
    print(
        f"Family priors: archive={config.prior_archive:.2f}, "
        f"hybrid={config.prior_hybrid:.2f}, novel={config.prior_novel:.2f}"
    )
    print(f"Consistency: {config.consistency:.2f}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<20}"
        f"{'Initial':<24}"
        f"{'Final':<18}"
        f"{'NovelAt':>9}"
        f"{'DiagAt':>8}"
        f"{'Probe':>8}"
        f"{'Q':>4}"
        f"{'Reveal':>10}"
        f"{'Hold':>10}"
        f"{'Overall':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        novel_at = "-" if result.first_novel_step is None else str(result.first_novel_step)
        diag_at = "-" if result.diagnostic_step is None else str(result.diagnostic_step)
        print(
            f"{result.policy_name:<20}"
            f"{result.initial_mode:<24}"
            f"{result.final_mode:<18}"
            f"{novel_at:>9}"
            f"{diag_at:>8}"
            f"{result.diagnostic_probe:>8}"
            f"{result.questions_asked:>4}"
            f"{result.reveal_accuracy:>10.3f}"
            f"{result.hold_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
        )


def main() -> None:
    config = parse_args()
    diagnostic_worlds = build_diagnostic_worlds(config.feature_count)
    warmup_worlds = build_warmup_worlds(config.feature_count, config.warmup_repeats)
    reveal_worlds = build_reveal_worlds(config.feature_count, config.reveal_repeats)
    truth_table = build_truth_table(config.feature_count)
    hold_worlds = build_hold_worlds(config.feature_count, config.hold_repeats, config.seed)

    results = [
        run_policy(
            config=config,
            policy_name="static_commit",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
            truth_table=truth_table,
        ),
        run_policy(
            config=config,
            policy_name="passive_switch",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
            truth_table=truth_table,
        ),
        run_policy(
            config=config,
            policy_name="diagnostic_switch",
            diagnostic_worlds=diagnostic_worlds,
            warmup_worlds=warmup_worlds,
            reveal_worlds=reveal_worlds,
            hold_worlds=hold_worlds,
            truth_table=truth_table,
        ),
    ]

    print_report(config, results)


if __name__ == "__main__":
    main()
