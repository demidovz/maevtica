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
    argmax,
    build_prior,
    build_worlds,
    entropy,
    generate_episodes,
    update_posterior,
    validate_config,
)


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    feature_count: int = 8
    initial_observations: int = 2
    question_budget: int = 4
    sensor_reliability: float = 0.82
    episodes: int = 2000
    seed: int = 7
    question_cost: float = 0.85

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
class PolicyResult:
    name: str
    accuracy: float
    average_questions_asked: float
    average_final_entropy: float
    average_total_information_gain: float
    information_gain_per_question: float
    average_true_world_probability: float
    average_intrinsic_value: float
    average_expected_gain_of_asked_questions: float
    entropy_trajectory: tuple[float, ...]


def parse_args() -> IntrinsicRewardConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a cost-aware question policy asks fewer but more informative questions."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--initial-observations", type=int, default=2)
    parser.add_argument("--question-budget", type=int, default=4)
    parser.add_argument("--sensor-reliability", type=float, default=0.82)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question-cost", type=float, default=0.85)
    args = parser.parse_args()

    config = IntrinsicRewardConfig(
        feature_count=args.feature_count,
        initial_observations=args.initial_observations,
        question_budget=args.question_budget,
        sensor_reliability=args.sensor_reliability,
        episodes=args.episodes,
        seed=args.seed,
        question_cost=args.question_cost,
    )
    validate_config(config.to_base_config())
    if config.question_cost < 0:
        raise ValueError("question_cost must be non-negative")
    return config


def select_random_question(
    rng: random.Random,
    feature_count: int,
    asked_features: set[int],
) -> int:
    available_features = [
        feature_index
        for feature_index in range(feature_count)
        if feature_index not in asked_features
    ]
    return rng.choice(available_features)


def best_entropy_question_with_gain(
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

        probability_one = sum(
            probability
            for probability, world in zip(posterior, worlds)
            if world[feature_index] == 1
        )
        probability_zero = 1.0 - probability_one

        expected_entropy = 0.0
        for observed_value, answer_probability in ((0, probability_zero), (1, probability_one)):
            if answer_probability <= 0:
                continue
            conditional = [
                probability if world[feature_index] == observed_value else 0.0
                for probability, world in zip(posterior, worlds)
            ]
            conditional_mass = sum(conditional)
            conditional = [value / conditional_mass for value in conditional]
            expected_entropy += answer_probability * entropy(conditional)

        information_gain = base_entropy - expected_entropy
        if information_gain > best_gain or (
            math.isclose(information_gain, best_gain) and feature_index < best_feature
        ):
            best_feature = feature_index
            best_gain = information_gain

    return best_feature, best_gain


def run_policy(
    policy_name: str,
    policy_seed: int,
    config: IntrinsicRewardConfig,
    worlds: Sequence[World],
    prior: Sequence[float],
) -> PolicyResult:
    rng = random.Random(policy_seed)
    base_config = config.to_base_config()
    episodes = generate_episodes(base_config, worlds, prior)

    correct_guesses: list[int] = []
    questions_asked: list[int] = []
    final_entropies: list[float] = []
    total_information_gains: list[float] = []
    true_world_probabilities: list[float] = []
    intrinsic_values: list[float] = []
    expected_gains_on_asked_questions: list[float] = []
    entropy_buckets: list[list[float]] = [[] for _ in range(config.question_budget + 1)]

    for episode in episodes:
        posterior = list(prior)
        asked_features: set[int] = set()
        total_information_gain = 0.0
        episode_questions_asked = 0

        for feature_index, observed_value in episode.observations:
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_features.add(feature_index)

        entropy_buckets[0].append(entropy(posterior))

        for step_index in range(config.question_budget):
            if len(asked_features) >= config.feature_count:
                break

            if policy_name == "no_questions":
                break

            if policy_name == "random_budgeted":
                feature_index = select_random_question(rng, config.feature_count, asked_features)
                expected_gain = float("nan")
            elif policy_name == "entropy_budgeted":
                feature_index, expected_gain = best_entropy_question_with_gain(
                    posterior, worlds, asked_features
                )
            elif policy_name == "entropy_cost_aware":
                feature_index, expected_gain = best_entropy_question_with_gain(
                    posterior, worlds, asked_features
                )
                if expected_gain <= config.question_cost:
                    break
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            before_entropy = entropy(posterior)
            asked_features.add(feature_index)
            answer = worlds[episode.true_world_id][feature_index]
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=answer,
                reliability=1.0,
            )
            after_entropy = entropy(posterior)
            total_information_gain += before_entropy - after_entropy
            episode_questions_asked += 1
            entropy_buckets[step_index + 1].append(after_entropy)

            if not math.isnan(expected_gain):
                expected_gains_on_asked_questions.append(expected_gain)

        if episode_questions_asked < config.question_budget:
            frozen_entropy = entropy(posterior)
            for step_index in range(episode_questions_asked + 1, config.question_budget + 1):
                entropy_buckets[step_index].append(frozen_entropy)

        guessed_world_id = argmax(posterior)
        correct_guesses.append(1 if guessed_world_id == episode.true_world_id else 0)
        questions_asked.append(episode_questions_asked)
        final_entropies.append(entropy(posterior))
        total_information_gains.append(total_information_gain)
        true_world_probabilities.append(posterior[episode.true_world_id])
        intrinsic_values.append(total_information_gain - config.question_cost * episode_questions_asked)

    total_questions = sum(questions_asked)
    total_information_gain = sum(total_information_gains)
    information_gain_per_question = (
        total_information_gain / total_questions if total_questions else 0.0
    )

    return PolicyResult(
        name=policy_name,
        accuracy=statistics.mean(correct_guesses),
        average_questions_asked=statistics.mean(questions_asked),
        average_final_entropy=statistics.mean(final_entropies),
        average_total_information_gain=statistics.mean(total_information_gains),
        information_gain_per_question=information_gain_per_question,
        average_true_world_probability=statistics.mean(true_world_probabilities),
        average_intrinsic_value=statistics.mean(intrinsic_values),
        average_expected_gain_of_asked_questions=(
            statistics.mean(expected_gains_on_asked_questions)
            if expected_gains_on_asked_questions
            else 0.0
        ),
        entropy_trajectory=tuple(statistics.mean(bucket) for bucket in entropy_buckets),
    )


def print_report(config: IntrinsicRewardConfig, results: Sequence[PolicyResult]) -> None:
    print("Experiment: intrinsic reward with costly questions")
    print(f"Question cost: {config.question_cost:.3f}")
    print(f"Question budget: {config.question_budget}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<20}"
        f"{'Q asked':>10}"
        f"{'Accuracy':>10}"
        f"{'Final H':>10}"
        f"{'Total IG':>10}"
        f"{'IG/Q':>10}"
        f"{'Intrinsic':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<20}"
            f"{result.average_questions_asked:>10.3f}"
            f"{result.accuracy:>10.3f}"
            f"{result.average_final_entropy:>10.3f}"
            f"{result.average_total_information_gain:>10.3f}"
            f"{result.information_gain_per_question:>10.3f}"
            f"{result.average_intrinsic_value:>11.3f}"
        )

    print()
    print("Expected gain of asked questions:")
    for result in results:
        print(
            f"{result.name:<20}"
            f"{result.average_expected_gain_of_asked_questions:>10.3f}"
        )

    print()
    print("Entropy trajectory (after initial observations and each question slot):")
    step_header = "Step".ljust(20) + "".join(f"{step:>10}" for step in range(config.question_budget + 1))
    print(step_header)
    print("-" * len(step_header))
    for result in results:
        trajectory = "".join(f"{value:>10.3f}" for value in result.entropy_trajectory)
        print(f"{result.name:<20}{trajectory}")


def main() -> None:
    config = parse_args()
    worlds = build_worlds(config.feature_count)
    prior = build_prior(worlds)

    results = [
        run_policy("no_questions", config.seed + 11, config, worlds, prior),
        run_policy("random_budgeted", config.seed + 23, config, worlds, prior),
        run_policy("entropy_budgeted", config.seed + 37, config, worlds, prior),
        run_policy("entropy_cost_aware", config.seed + 41, config, worlds, prior),
    ]
    print_report(config, results)


if __name__ == "__main__":
    main()
