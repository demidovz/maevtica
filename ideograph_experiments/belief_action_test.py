from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence

from question_uncertainty_test import (
    ExperimentConfig,
    World,
    build_prior,
    build_worlds,
    entropy,
    update_posterior,
    validate_config,
)


Observation = tuple[int, int]


@dataclass(frozen=True)
class BeliefActionConfig:
    feature_count: int = 8
    initial_observations: int = 2
    question_budget: int = 4
    sensor_reliability: float = 0.82
    episodes: int = 800
    seed: int = 7
    question_cost: float = 0.15
    block_size: int = 40
    mutation_rate: float = 0.05
    reentry_window: int = 5

    def to_base_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            feature_count=self.feature_count,
            initial_observations=self.initial_observations,
            question_budget=self.question_budget,
            sensor_reliability=self.sensor_reliability,
            episodes=self.episodes,
            seed=self.seed,
        )


@dataclass(frozen=True)
class Episode:
    context_id: int
    block_index: int
    block_position: int
    true_world_id: int
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class PolicyResult:
    name: str
    action_accuracy: float
    average_questions_asked: float
    average_action_confidence: float
    average_final_entropy: float
    average_intrinsic_value: float
    early_block_accuracy: float
    late_block_accuracy: float
    early_block_questions: float
    late_block_questions: float
    reentry_accuracy: float
    reentry_questions: float
    final_rule_accuracy: float
    final_rule_entropy: float
    final_rule_top1_mass: float


def parse_args() -> BeliefActionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether stable beliefs appear only when questions close into "
            "prediction/action with explicit error cost."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--initial-observations", type=int, default=2)
    parser.add_argument("--question-budget", type=int, default=4)
    parser.add_argument("--sensor-reliability", type=float, default=0.82)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question-cost", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--reentry-window", type=int, default=5)
    args = parser.parse_args()

    config = BeliefActionConfig(
        feature_count=args.feature_count,
        initial_observations=args.initial_observations,
        question_budget=args.question_budget,
        sensor_reliability=args.sensor_reliability,
        episodes=args.episodes,
        seed=args.seed,
        question_cost=args.question_cost,
        block_size=args.block_size,
        mutation_rate=args.mutation_rate,
        reentry_window=args.reentry_window,
    )
    validate_config(config.to_base_config())
    if config.feature_count != 8:
        raise ValueError("belief_action_test currently expects feature_count == 8")
    if config.block_size <= 1:
        raise ValueError("block_size must be > 1")
    if not 0.0 <= config.mutation_rate < 1.0:
        raise ValueError("mutation_rate must be in [0, 1)")
    if config.question_cost < 0:
        raise ValueError("question_cost must be non-negative")
    if config.reentry_window <= 0 or config.reentry_window > config.block_size:
        raise ValueError("reentry_window must be in [1, block_size]")
    return config


def build_context_prototypes() -> dict[int, tuple[int, ...]]:
    return {
        0: (3, 11, 35, 99),
        1: (144, 176, 208, 240),
        2: (18, 22, 82, 86),
        3: (129, 153, 165, 189),
    }


def build_context_rules() -> dict[int, int]:
    return {
        0: 0,
        1: 4,
        2: 2,
        3: 6,
    }


def mutate_world_id(
    rng: random.Random,
    worlds: Sequence[World],
    prototype_world_id: int,
    mutation_rate: float,
) -> int:
    bits = list(worlds[prototype_world_id])
    for feature_index in range(len(bits)):
        if rng.random() < mutation_rate:
            bits[feature_index] = 1 - bits[feature_index]

    world_id = 0
    for bit_index, bit_value in enumerate(bits):
        world_id |= bit_value << bit_index
    return world_id


def generate_episodes(
    config: BeliefActionConfig,
    worlds: Sequence[World],
) -> list[Episode]:
    rng = random.Random(config.seed)
    context_prototypes = build_context_prototypes()
    prototype_weights = (0.45, 0.25, 0.20, 0.10)
    context_ids = tuple(sorted(context_prototypes))

    episodes: list[Episode] = []
    for episode_index in range(config.episodes):
        block_index = episode_index // config.block_size
        block_position = episode_index % config.block_size
        context_id = context_ids[block_index % len(context_ids)]
        prototype_world_id = rng.choices(
            context_prototypes[context_id],
            weights=prototype_weights,
            k=1,
        )[0]
        true_world_id = mutate_world_id(
            rng=rng,
            worlds=worlds,
            prototype_world_id=prototype_world_id,
            mutation_rate=config.mutation_rate,
        )

        observed_features = rng.sample(range(config.feature_count), config.initial_observations)
        observations: list[Observation] = []
        for feature_index in observed_features:
            true_value = worlds[true_world_id][feature_index]
            if rng.random() < config.sensor_reliability:
                observed_value = true_value
            else:
                observed_value = 1 - true_value
            observations.append((feature_index, observed_value))

        episodes.append(
            Episode(
                context_id=context_id,
                block_index=block_index,
                block_position=block_position,
                true_world_id=true_world_id,
                observations=tuple(observations),
            )
        )

    return episodes


def normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def binary_entropy(probability_one: float) -> float:
    if probability_one <= 0.0 or probability_one >= 1.0:
        return 0.0
    probability_zero = 1.0 - probability_one
    return -(
        probability_one * math.log2(probability_one)
        + probability_zero * math.log2(probability_zero)
    )


def world_feature_probability(
    posterior: Sequence[float],
    worlds: Sequence[World],
    feature_index: int,
) -> float:
    return sum(
        probability
        for probability, world in zip(posterior, worlds)
        if world[feature_index] == 1
    )


def action_probability(
    posterior: Sequence[float],
    worlds: Sequence[World],
    rule_belief: Sequence[float],
) -> float:
    return sum(
        belief * world_feature_probability(posterior, worlds, feature_index)
        for feature_index, belief in enumerate(rule_belief)
    )


def select_world_question(
    posterior: Sequence[float],
    worlds: Sequence[World],
    asked_features: set[int],
) -> tuple[int, float]:
    base_entropy = entropy(posterior)
    best_feature = -1
    best_gain = float("-inf")

    for feature_index in range(len(worlds[0])):
        if feature_index in asked_features:
            continue

        probability_one = world_feature_probability(posterior, worlds, feature_index)
        probability_zero = 1.0 - probability_one

        expected_entropy = 0.0
        for observed_value, answer_probability in ((0, probability_zero), (1, probability_one)):
            if answer_probability <= 0:
                continue
            conditional = [
                probability if world[feature_index] == observed_value else 0.0
                for probability, world in zip(posterior, worlds)
            ]
            conditional = normalize(conditional)
            expected_entropy += answer_probability * entropy(conditional)

        information_gain = base_entropy - expected_entropy
        if information_gain > best_gain or (
            math.isclose(information_gain, best_gain) and feature_index < best_feature
        ):
            best_feature = feature_index
            best_gain = information_gain

    return best_feature, best_gain


def select_action_question(
    posterior: Sequence[float],
    worlds: Sequence[World],
    asked_features: set[int],
    rule_belief: Sequence[float],
) -> tuple[int, float]:
    base_action_entropy = binary_entropy(action_probability(posterior, worlds, rule_belief))
    best_feature = -1
    best_gain = float("-inf")

    for feature_index in range(len(worlds[0])):
        if feature_index in asked_features:
            continue

        probability_one = world_feature_probability(posterior, worlds, feature_index)
        probability_zero = 1.0 - probability_one

        expected_action_entropy = 0.0
        for observed_value, answer_probability in ((0, probability_zero), (1, probability_one)):
            if answer_probability <= 0:
                continue
            conditional = [
                probability if world[feature_index] == observed_value else 0.0
                for probability, world in zip(posterior, worlds)
            ]
            conditional = normalize(conditional)
            expected_action_entropy += answer_probability * binary_entropy(
                action_probability(conditional, worlds, rule_belief)
            )

        information_gain = base_action_entropy - expected_action_entropy
        if information_gain > best_gain or (
            math.isclose(information_gain, best_gain) and feature_index < best_feature
        ):
            best_feature = feature_index
            best_gain = information_gain

    return best_feature, best_gain


def update_rule_belief(
    rule_belief: Sequence[float],
    posterior: Sequence[float],
    worlds: Sequence[World],
    true_action: int,
) -> list[float]:
    likelihoods = []
    for feature_index in range(len(worlds[0])):
        probability_one = world_feature_probability(posterior, worlds, feature_index)
        likelihood = probability_one if true_action == 1 else (1.0 - probability_one)
        likelihoods.append(max(likelihood, 1e-9))
    updated = [belief * likelihood for belief, likelihood in zip(rule_belief, likelihoods)]
    return normalize(updated)


def run_policy(
    policy_name: str,
    config: BeliefActionConfig,
    worlds: Sequence[World],
    world_prior: Sequence[float],
    episodes: Sequence[Episode],
    context_rules: dict[int, int],
) -> PolicyResult:
    context_ids = tuple(sorted(context_rules))
    rule_beliefs = {
        context_id: [1.0 / config.feature_count] * config.feature_count
        for context_id in context_ids
    }

    action_accuracies: list[int] = []
    questions_asked: list[int] = []
    action_confidences: list[float] = []
    final_entropies: list[float] = []
    intrinsic_values: list[float] = []
    early_block_accuracies: list[int] = []
    late_block_accuracies: list[int] = []
    early_block_questions: list[int] = []
    late_block_questions: list[int] = []
    reentry_accuracies: list[int] = []
    reentry_questions: list[int] = []

    for episode in episodes:
        posterior = list(world_prior)
        asked_features: set[int] = set()
        rule_belief = rule_beliefs[episode.context_id]

        for feature_index, observed_value in episode.observations:
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_features.add(feature_index)

        episode_questions = 0
        for _ in range(config.question_budget):
            if len(asked_features) >= config.feature_count:
                break

            if policy_name == "action_with_belief":
                feature_index, expected_gain = select_action_question(
                    posterior, worlds, asked_features, rule_belief
                )
            else:
                feature_index, expected_gain = select_world_question(
                    posterior, worlds, asked_features
                )

            if expected_gain <= config.question_cost:
                break

            asked_features.add(feature_index)
            answer = worlds[episode.true_world_id][feature_index]
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=answer,
                reliability=1.0,
            )
            episode_questions += 1

        probability_action_one = action_probability(posterior, worlds, rule_belief)
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        action_confidence = max(probability_action_one, 1.0 - probability_action_one)
        true_action = worlds[episode.true_world_id][context_rules[episode.context_id]]
        reward = 1 if predicted_action == true_action else 0

        action_accuracies.append(reward)
        questions_asked.append(episode_questions)
        action_confidences.append(action_confidence)
        final_entropies.append(entropy(posterior))
        intrinsic_values.append(reward - config.question_cost * episode_questions)

        if episode.block_position < config.block_size // 2:
            early_block_accuracies.append(reward)
            early_block_questions.append(episode_questions)
        else:
            late_block_accuracies.append(reward)
            late_block_questions.append(episode_questions)

        if (
            episode.block_index >= len(context_ids)
            and episode.block_position < config.reentry_window
        ):
            reentry_accuracies.append(reward)
            reentry_questions.append(episode_questions)

        if policy_name != "world_no_belief":
            rule_beliefs[episode.context_id] = update_rule_belief(
                rule_belief=rule_belief,
                posterior=posterior,
                worlds=worlds,
                true_action=true_action,
            )

    final_rule_accuracy = statistics.mean(
        1 if max(range(config.feature_count), key=rule_beliefs[context_id].__getitem__) == context_rules[context_id] else 0
        for context_id in context_ids
    )
    final_rule_entropies = [entropy(rule_beliefs[context_id]) for context_id in context_ids]
    final_rule_top1_mass = statistics.mean(
        max(rule_beliefs[context_id]) for context_id in context_ids
    )

    return PolicyResult(
        name=policy_name,
        action_accuracy=statistics.mean(action_accuracies),
        average_questions_asked=statistics.mean(questions_asked),
        average_action_confidence=statistics.mean(action_confidences),
        average_final_entropy=statistics.mean(final_entropies),
        average_intrinsic_value=statistics.mean(intrinsic_values),
        early_block_accuracy=statistics.mean(early_block_accuracies),
        late_block_accuracy=statistics.mean(late_block_accuracies),
        early_block_questions=statistics.mean(early_block_questions),
        late_block_questions=statistics.mean(late_block_questions),
        reentry_accuracy=statistics.mean(reentry_accuracies),
        reentry_questions=statistics.mean(reentry_questions),
        final_rule_accuracy=final_rule_accuracy,
        final_rule_entropy=statistics.mean(final_rule_entropies),
        final_rule_top1_mass=final_rule_top1_mass,
    )


def print_report(
    config: BeliefActionConfig,
    results: Sequence[PolicyResult],
    context_rules: dict[int, int],
) -> None:
    print("Experiment: beliefs via prediction/action with error cost")
    print("Hidden context rules (context -> decisive feature):")
    print(", ".join(f"{context_id}->{feature_index}" for context_id, feature_index in sorted(context_rules.items())))
    print(f"Question cost: {config.question_cost:.3f}")
    print(f"Question budget: {config.question_budget}")
    print(f"Block size: {config.block_size}")
    print(f"Reentry window: {config.reentry_window}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<20}"
        f"{'ActAcc':>9}"
        f"{'Q asked':>10}"
        f"{'Conf':>9}"
        f"{'Net':>9}"
        f"{'RuleAcc':>10}"
        f"{'RuleTop1':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<20}"
            f"{result.action_accuracy:>9.3f}"
            f"{result.average_questions_asked:>10.3f}"
            f"{result.average_action_confidence:>9.3f}"
            f"{result.average_intrinsic_value:>9.3f}"
            f"{result.final_rule_accuracy:>10.3f}"
            f"{result.final_rule_top1_mass:>11.3f}"
        )

    print()
    print("Transfer after context return:")
    transfer_header = (
        f"{'Policy':<20}"
        f"{'EarlyAcc':>10}"
        f"{'LateAcc':>10}"
        f"{'ReentryAcc':>12}"
        f"{'EarlyQ':>10}"
        f"{'LateQ':>10}"
        f"{'ReentryQ':>12}"
    )
    print(transfer_header)
    print("-" * len(transfer_header))
    for result in results:
        print(
            f"{result.name:<20}"
            f"{result.early_block_accuracy:>10.3f}"
            f"{result.late_block_accuracy:>10.3f}"
            f"{result.reentry_accuracy:>12.3f}"
            f"{result.early_block_questions:>10.3f}"
            f"{result.late_block_questions:>10.3f}"
            f"{result.reentry_questions:>12.3f}"
        )


def main() -> None:
    config = parse_args()
    worlds = build_worlds(config.feature_count)
    world_prior = build_prior(worlds)
    context_rules = build_context_rules()
    episodes = generate_episodes(config, worlds)

    results = [
        run_policy("world_no_belief", config, worlds, world_prior, episodes, context_rules),
        run_policy("world_with_belief", config, worlds, world_prior, episodes, context_rules),
        run_policy("action_with_belief", config, worlds, world_prior, episodes, context_rules),
    ]
    print_report(config, results, context_rules)


if __name__ == "__main__":
    main()
