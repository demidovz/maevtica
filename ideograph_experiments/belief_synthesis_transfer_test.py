from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefSynthesisTransferConfig:
    feature_count: int = 8
    pretrain_episodes: int = 150
    base_conflict_episodes: int = 120
    transfer_conflict_episodes: int = 40
    seed: int = 7
    consistency: float = 0.92
    spawn_after: int = 12
    spawn_window: int = 8
    spawn_threshold: float = 0.72
    transfer_prior_strength: float = 4.0


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
    source_name: str
    top_hypothesis: str
    top_mass: float


@dataclass(frozen=True)
class BaseConflictResult:
    overall_accuracy: float
    early_accuracy: float
    late_accuracy: float
    final_top_hypothesis: str
    final_top_mass: float
    synthetic_spawn_episode: int | None
    operator_prior: dict[str, float]


@dataclass(frozen=True)
class TransferResult:
    name: str
    overall_accuracy: float
    first_ten_accuracy: float
    last_ten_accuracy: float
    final_top_hypothesis: str
    final_top_mass: float
    synthetic_spawn_episode: int | None


def parse_args() -> BeliefSynthesisTransferConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a synthetic belief learned in one conflict transfers "
            "to a new partially similar conflict without full retraining."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--pretrain-episodes", type=int, default=150)
    parser.add_argument("--base-conflict-episodes", type=int, default=120)
    parser.add_argument("--transfer-conflict-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--spawn-after", type=int, default=12)
    parser.add_argument("--spawn-window", type=int, default=8)
    parser.add_argument("--spawn-threshold", type=float, default=0.72)
    parser.add_argument("--transfer-prior-strength", type=float, default=4.0)
    args = parser.parse_args()

    config = BeliefSynthesisTransferConfig(
        feature_count=args.feature_count,
        pretrain_episodes=args.pretrain_episodes,
        base_conflict_episodes=args.base_conflict_episodes,
        transfer_conflict_episodes=args.transfer_conflict_episodes,
        seed=args.seed,
        consistency=args.consistency,
        spawn_after=args.spawn_after,
        spawn_window=args.spawn_window,
        spawn_threshold=args.spawn_threshold,
        transfer_prior_strength=args.transfer_prior_strength,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefSynthesisTransferConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_synthesis_transfer_test currently expects feature_count == 8")
    if config.pretrain_episodes <= 0:
        raise ValueError("pretrain_episodes must be > 0")
    if config.base_conflict_episodes <= 0 or config.transfer_conflict_episodes <= 0:
        raise ValueError("conflict episode counts must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.spawn_after < 0:
        raise ValueError("spawn_after must be >= 0")
    if config.spawn_window <= 0:
        raise ValueError("spawn_window must be > 0")
    if not 0.0 < config.spawn_threshold < 1.0:
        raise ValueError("spawn_threshold must be in (0, 1)")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def build_feature_hypotheses(feature_count: int) -> list[Hypothesis]:
    return [
        Hypothesis(name=f"feat_{feature_index}", kind="feature", feature_a=feature_index)
        for feature_index in range(feature_count)
    ]


def build_pair_composites(feature_a: int, feature_b: int) -> list[Hypothesis]:
    return [
        Hypothesis(name=f"or_{feature_a}_{feature_b}", kind="or", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"and_{feature_a}_{feature_b}", kind="and", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"xor_{feature_a}_{feature_b}", kind="xor", feature_a=feature_a, feature_b=feature_b),
        Hypothesis(name=f"xnor_{feature_a}_{feature_b}", kind="xnor", feature_a=feature_a, feature_b=feature_b),
    ]


def sample_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


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


def pretrain_source(
    config: BeliefSynthesisTransferConfig,
    worlds: Sequence[World],
    target_feature: int,
    source_name: str,
) -> tuple[PretrainResult, Hypothesis]:
    hypotheses = build_feature_hypotheses(config.feature_count)
    scores = [0.0] * len(hypotheses)

    for world in worlds:
        true_action = world[target_feature]
        update_scores(scores, hypotheses, world, true_action, config.consistency)

    top_name, top_mass = top_hypothesis(scores, hypotheses)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return (
        PretrainResult(
            source_name=source_name,
            top_hypothesis=top_name,
            top_mass=top_mass,
        ),
        hypotheses[top_index],
    )


def run_base_conflict(
    config: BeliefSynthesisTransferConfig,
    source_hypotheses: Sequence[Hypothesis],
    worlds: Sequence[World],
    target_pair: tuple[int, int],
) -> BaseConflictResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []
    recent_accuracies: list[int] = []
    synthetic_spawn_episode: int | None = None

    for episode_index, world in enumerate(worlds):
        true_action = world[target_pair[0]] ^ world[target_pair[1]]
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
            hypotheses.extend(build_pair_composites(*target_pair))
            scores.extend([0.0] * 4)
            synthetic_spawn_episode = episode_index

        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, final_top_mass = top_hypothesis(scores, hypotheses)
    operator_prior = {kind: 0.0 for kind in ("or", "and", "xor", "xnor")}
    belief = softmax(scores)
    for weight, hypothesis in zip(belief, hypotheses):
        if hypothesis.kind != "feature":
            operator_prior[hypothesis.kind] += weight

    return BaseConflictResult(
        overall_accuracy=statistics.mean(accuracies),
        early_accuracy=statistics.mean(accuracies[:20]),
        late_accuracy=statistics.mean(accuracies[-20:]),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
        synthetic_spawn_episode=synthetic_spawn_episode,
        operator_prior=operator_prior,
    )


def run_compromise_old(
    source_hypotheses: Sequence[Hypothesis],
    worlds: Sequence[World],
    target_pair: tuple[int, int],
) -> TransferResult:
    accuracies: list[int] = []

    for world in worlds:
        true_action = world[target_pair[0]] ^ world[target_pair[1]]
        average_vote = sum(
            hypothesis.predict(world)
            for hypothesis in source_hypotheses
        ) / len(source_hypotheses)
        predicted_action = 1 if average_vote >= 0.5 else 0
        accuracies.append(1 if predicted_action == true_action else 0)

    return TransferResult(
        name="compromise_old",
        overall_accuracy=statistics.mean(accuracies),
        first_ten_accuracy=statistics.mean(accuracies[:10]),
        last_ten_accuracy=statistics.mean(accuracies[-10:]),
        final_top_hypothesis="blend(feat_0,feat_5)",
        final_top_mass=1.0,
        synthetic_spawn_episode=None,
    )


def run_synthesis_conflict(
    config: BeliefSynthesisTransferConfig,
    source_hypotheses: Sequence[Hypothesis],
    worlds: Sequence[World],
    target_pair: tuple[int, int],
    transfer_operator_prior: dict[str, float] | None,
    name: str,
) -> TransferResult:
    hypotheses = list(source_hypotheses)
    scores = [0.0] * len(hypotheses)
    accuracies: list[int] = []
    recent_accuracies: list[int] = []
    synthetic_spawn_episode: int | None = None

    if transfer_operator_prior is not None:
        candidates = build_pair_composites(*target_pair)
        hypotheses.extend(candidates)
        uniform_mass = 1.0 / len(candidates)
        scores.extend(
            [
                config.transfer_prior_strength
                * (transfer_operator_prior.get(candidate.kind, 0.0) - uniform_mass)
                for candidate in candidates
            ]
        )
        synthetic_spawn_episode = 0

    for episode_index, world in enumerate(worlds):
        true_action = world[target_pair[0]] ^ world[target_pair[1]]
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
            transfer_operator_prior is None
            and synthetic_spawn_episode is None
            and episode_index >= config.spawn_after
            and statistics.mean(recent_accuracies) < config.spawn_threshold
        ):
            hypotheses.extend(build_pair_composites(*target_pair))
            scores.extend([0.0] * 4)
            synthetic_spawn_episode = episode_index

        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_name, final_top_mass = top_hypothesis(scores, hypotheses)
    return TransferResult(
        name=name,
        overall_accuracy=statistics.mean(accuracies),
        first_ten_accuracy=statistics.mean(accuracies[:10]),
        last_ten_accuracy=statistics.mean(accuracies[-10:]),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
        synthetic_spawn_episode=synthetic_spawn_episode,
    )


def print_report(
    config: BeliefSynthesisTransferConfig,
    pretrain_results: Sequence[PretrainResult],
    base_conflict_result: BaseConflictResult,
    transfer_results: Sequence[TransferResult],
) -> None:
    print("Experiment: synthetic belief transfer")
    print("Source beliefs:")
    print("A -> feat_0, B -> feat_4, C -> feat_5")
    print("Base conflict:")
    print("xor(feat_0, feat_4)")
    print("Transfer conflict:")
    print("xor(feat_0, feat_5)")
    print(f"Pretrain episodes per source: {config.pretrain_episodes}")
    print(f"Base conflict episodes: {config.base_conflict_episodes}")
    print(f"Transfer conflict episodes: {config.transfer_conflict_episodes}")
    print(f"Seed: {config.seed}")
    print()

    print("Pretrain results:")
    pretrain_header = f"{'Source':<10}{'Top belief':<16}{'Top mass':>10}"
    print(pretrain_header)
    print("-" * len(pretrain_header))
    for result in pretrain_results:
        print(
            f"{result.source_name:<10}"
            f"{result.top_hypothesis:<16}"
            f"{result.top_mass:>10.3f}"
        )

    print()
    print("Base conflict synthesis:")
    print(
        f"overall={base_conflict_result.overall_accuracy:.3f}, "
        f"early={base_conflict_result.early_accuracy:.3f}, "
        f"late={base_conflict_result.late_accuracy:.3f}, "
        f"top={base_conflict_result.final_top_hypothesis}, "
        f"spawn={base_conflict_result.synthetic_spawn_episode}"
    )
    print(
        "operator prior: "
        + ", ".join(
            f"{kind}={mass:.3f}"
            for kind, mass in base_conflict_result.operator_prior.items()
        )
    )

    print()
    print("Transfer conflict results:")
    transfer_header = (
        f"{'Policy':<22}"
        f"{'Overall':>10}"
        f"{'First10':>10}"
        f"{'Last10':>10}"
        f"{'Top belief':>24}"
        f"{'Spawn':>8}"
    )
    print(transfer_header)
    print("-" * len(transfer_header))
    for result in transfer_results:
        spawn_text = "-" if result.synthetic_spawn_episode is None else str(result.synthetic_spawn_episode)
        print(
            f"{result.name:<22}"
            f"{result.overall_accuracy:>10.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.last_ten_accuracy:>10.3f}"
            f"{result.final_top_hypothesis:>24}"
            f"{spawn_text:>8}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)

    pretrain_worlds_a = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    pretrain_worlds_b = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    pretrain_worlds_c = sample_worlds(rng, config.feature_count, config.pretrain_episodes)
    base_conflict_worlds = sample_worlds(rng, config.feature_count, config.base_conflict_episodes)
    transfer_conflict_worlds = sample_worlds(rng, config.feature_count, config.transfer_conflict_episodes)

    pretrain_a, belief_a = pretrain_source(config, pretrain_worlds_a, 0, "source_A")
    pretrain_b, belief_b = pretrain_source(config, pretrain_worlds_b, 4, "source_B")
    pretrain_c, belief_c = pretrain_source(config, pretrain_worlds_c, 5, "source_C")

    base_conflict_result = run_base_conflict(
        config,
        [belief_a, belief_b],
        base_conflict_worlds,
        target_pair=(0, 4),
    )
    transfer_results = [
        run_compromise_old([belief_a, belief_c], transfer_conflict_worlds, target_pair=(0, 5)),
        run_synthesis_conflict(
            config,
            [belief_a, belief_c],
            transfer_conflict_worlds,
            target_pair=(0, 5),
            transfer_operator_prior=None,
            name="scratch_synthesis",
        ),
        run_synthesis_conflict(
            config,
            [belief_a, belief_c],
            transfer_conflict_worlds,
            target_pair=(0, 5),
            transfer_operator_prior=base_conflict_result.operator_prior,
            name="transfer_synthesis",
        ),
    ]

    print_report(config, [pretrain_a, pretrain_b, pretrain_c], base_conflict_result, transfer_results)


if __name__ == "__main__":
    main()
