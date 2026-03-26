from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefConflictConfig:
    feature_count: int = 8
    pretrain_episodes: int = 200
    conflict_episodes: int = 240
    seed: int = 7
    consistency: float = 0.92
    spawn_after: int = 20
    spawn_window: int = 12
    spawn_threshold: float = 0.70


@dataclass(frozen=True)
class Hypothesis:
    name: str
    kind: str
    feature_a: int
    feature_b: int | None = None

    def predict(self, world: World) -> int:
        if self.kind == "feature":
            return world[self.feature_a]
        if self.kind == "or":
            return world[self.feature_a] | world[self.feature_b]
        if self.kind == "and":
            return world[self.feature_a] & world[self.feature_b]
        if self.kind == "xor":
            return world[self.feature_a] ^ world[self.feature_b]
        if self.kind == "xnor":
            return 1 - (world[self.feature_a] ^ world[self.feature_b])
        raise ValueError(f"Unknown hypothesis kind: {self.kind}")


@dataclass(frozen=True)
class PretrainResult:
    context_name: str
    top_hypothesis: str
    top_mass: float


@dataclass(frozen=True)
class ConflictResult:
    name: str
    overall_accuracy: float
    early_conflict_accuracy: float
    late_conflict_accuracy: float
    final_top_hypothesis: str
    final_top_mass: float
    synthetic_spawn_episode: int | None


def parse_args() -> BeliefConflictConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether systematic conflict between two stable beliefs can produce "
            "a new synthetic belief instead of only choosing one old belief."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--pretrain-episodes", type=int, default=200)
    parser.add_argument("--conflict-episodes", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--spawn-after", type=int, default=20)
    parser.add_argument("--spawn-window", type=int, default=12)
    parser.add_argument("--spawn-threshold", type=float, default=0.70)
    args = parser.parse_args()

    config = BeliefConflictConfig(
        feature_count=args.feature_count,
        pretrain_episodes=args.pretrain_episodes,
        conflict_episodes=args.conflict_episodes,
        seed=args.seed,
        consistency=args.consistency,
        spawn_after=args.spawn_after,
        spawn_window=args.spawn_window,
        spawn_threshold=args.spawn_threshold,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefConflictConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_conflict_test currently expects feature_count == 8")
    if config.pretrain_episodes <= 0 or config.conflict_episodes <= 0:
        raise ValueError("pretrain_episodes and conflict_episodes must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.spawn_after < 0:
        raise ValueError("spawn_after must be >= 0")
    if config.spawn_window <= 0:
        raise ValueError("spawn_window must be > 0")
    if not 0.0 < config.spawn_threshold < 1.0:
        raise ValueError("spawn_threshold must be in (0, 1)")


def build_feature_hypotheses(feature_count: int) -> list[Hypothesis]:
    return [
        Hypothesis(name=f"feat_{feature_index}", kind="feature", feature_a=feature_index)
        for feature_index in range(feature_count)
    ]


def build_conflict_composites() -> list[Hypothesis]:
    return [
        Hypothesis(name="or_0_4", kind="or", feature_a=0, feature_b=4),
        Hypothesis(name="and_0_4", kind="and", feature_a=0, feature_b=4),
        Hypothesis(name="xor_0_4", kind="xor", feature_a=0, feature_b=4),
        Hypothesis(name="xnor_0_4", kind="xnor", feature_a=0, feature_b=4),
    ]


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def sample_world(rng: random.Random, feature_count: int) -> World:
    return tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))


def update_scores(
    scores: list[float],
    hypotheses: Sequence[Hypothesis],
    world: World,
    true_action: int,
    consistency: float,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        likelihood = consistency if hypothesis.predict(world) == true_action else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[Hypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def pretrain_context(
    rng: random.Random,
    config: BeliefConflictConfig,
    context_name: str,
    target_feature: int,
) -> tuple[PretrainResult, Hypothesis]:
    hypotheses = build_feature_hypotheses(config.feature_count)
    scores = [0.0] * len(hypotheses)

    for _ in range(config.pretrain_episodes):
        world = sample_world(rng, config.feature_count)
        true_action = world[target_feature]
        update_scores(scores, hypotheses, world, true_action, config.consistency)

    top_name, top_mass = top_hypothesis(scores, hypotheses)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return (
        PretrainResult(
            context_name=context_name,
            top_hypothesis=top_name,
            top_mass=top_mass,
        ),
        hypotheses[top_index],
    )


def run_winner_take_all(
    rng: random.Random,
    config: BeliefConflictConfig,
    source_hypotheses: Sequence[Hypothesis],
) -> ConflictResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []

    for _ in range(config.conflict_episodes):
        world = sample_world(rng, config.feature_count)
        true_action = world[0] ^ world[4]
        top_index = max(range(len(scores)), key=scores.__getitem__)
        predicted_action = hypotheses[top_index].predict(world)
        accuracies.append(1 if predicted_action == true_action else 0)
        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, final_top_mass = top_hypothesis(scores, hypotheses)
    return ConflictResult(
        name="winner_take_all",
        overall_accuracy=statistics.mean(accuracies),
        early_conflict_accuracy=statistics.mean(accuracies[:40]),
        late_conflict_accuracy=statistics.mean(accuracies[-40:]),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
        synthetic_spawn_episode=None,
    )


def run_compromise_old(
    rng: random.Random,
    config: BeliefConflictConfig,
    source_hypotheses: Sequence[Hypothesis],
) -> ConflictResult:
    accuracies: list[int] = []
    for _ in range(config.conflict_episodes):
        world = sample_world(rng, config.feature_count)
        true_action = world[0] ^ world[4]
        average_vote = sum(hypothesis.predict(world) for hypothesis in source_hypotheses) / len(source_hypotheses)
        predicted_action = 1 if average_vote >= 0.5 else 0
        accuracies.append(1 if predicted_action == true_action else 0)

    return ConflictResult(
        name="compromise_old",
        overall_accuracy=statistics.mean(accuracies),
        early_conflict_accuracy=statistics.mean(accuracies[:40]),
        late_conflict_accuracy=statistics.mean(accuracies[-40:]),
        final_top_hypothesis="blend(feat_0,feat_4)",
        final_top_mass=1.0,
        synthetic_spawn_episode=None,
    )


def run_synthesis_enabled(
    rng: random.Random,
    config: BeliefConflictConfig,
    source_hypotheses: Sequence[Hypothesis],
) -> ConflictResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []
    recent_accuracies: list[int] = []
    synthetic_spawn_episode: int | None = None

    for episode_index in range(config.conflict_episodes):
        world = sample_world(rng, config.feature_count)
        true_action = world[0] ^ world[4]
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        is_correct = 1 if predicted_action == true_action else 0
        accuracies.append(is_correct)
        recent_accuracies.append(is_correct)
        if len(recent_accuracies) > config.spawn_window:
            recent_accuracies.pop(0)

        if (
            synthetic_spawn_episode is None
            and episode_index >= config.spawn_after
            and statistics.mean(recent_accuracies) < config.spawn_threshold
        ):
            hypotheses.extend(build_conflict_composites())
            scores.extend([0.0] * 4)
            synthetic_spawn_episode = episode_index

        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, final_top_mass = top_hypothesis(scores, hypotheses)
    return ConflictResult(
        name="synthesis_enabled",
        overall_accuracy=statistics.mean(accuracies),
        early_conflict_accuracy=statistics.mean(accuracies[:40]),
        late_conflict_accuracy=statistics.mean(accuracies[-40:]),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
        synthetic_spawn_episode=synthetic_spawn_episode,
    )


def print_report(
    config: BeliefConflictConfig,
    pretrain_results: Sequence[PretrainResult],
    conflict_results: Sequence[ConflictResult],
) -> None:
    print("Experiment: belief conflict / synthetic belief emergence")
    print("Source contexts:")
    print("A -> action = feat_0")
    print("B -> action = feat_4")
    print("Conflict context:")
    print("C -> action = xor(feat_0, feat_4)")
    print(f"Pretrain episodes per source context: {config.pretrain_episodes}")
    print(f"Conflict episodes: {config.conflict_episodes}")
    print(f"Spawn trigger: after {config.spawn_after} episodes, rolling window {config.spawn_window}, threshold {config.spawn_threshold:.2f}")
    print(f"Seed: {config.seed}")
    print()

    print("Pretrain results:")
    pretrain_header = f"{'Context':<10}{'Top belief':<16}{'Top mass':>10}"
    print(pretrain_header)
    print("-" * len(pretrain_header))
    for result in pretrain_results:
        print(
            f"{result.context_name:<10}"
            f"{result.top_hypothesis:<16}"
            f"{result.top_mass:>10.3f}"
        )

    print()
    print("Conflict results:")
    conflict_header = (
        f"{'Policy':<18}"
        f"{'Overall':>10}"
        f"{'Early':>10}"
        f"{'Late':>10}"
        f"{'Top belief':>24}"
        f"{'Top mass':>10}"
        f"{'Spawn':>8}"
    )
    print(conflict_header)
    print("-" * len(conflict_header))
    for result in conflict_results:
        spawn_text = "-" if result.synthetic_spawn_episode is None else str(result.synthetic_spawn_episode)
        print(
            f"{result.name:<18}"
            f"{result.overall_accuracy:>10.3f}"
            f"{result.early_conflict_accuracy:>10.3f}"
            f"{result.late_conflict_accuracy:>10.3f}"
            f"{result.final_top_hypothesis:>24}"
            f"{result.final_top_mass:>10.3f}"
            f"{spawn_text:>8}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)

    pretrain_a, belief_a = pretrain_context(rng, config, "source_A", 0)
    pretrain_b, belief_b = pretrain_context(rng, config, "source_B", 4)
    source_hypotheses = [belief_a, belief_b]

    conflict_results = [
        run_winner_take_all(rng, config, source_hypotheses),
        run_compromise_old(rng, config, source_hypotheses),
        run_synthesis_enabled(rng, config, source_hypotheses),
    ]
    print_report(config, [pretrain_a, pretrain_b], conflict_results)


if __name__ == "__main__":
    main()
