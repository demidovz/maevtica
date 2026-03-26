from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]
Observation = tuple[int, int]


@dataclass(frozen=True)
class ExperimentConfig:
    feature_count: int = 8
    initial_observations: int = 2
    question_budget: int = 4
    sensor_reliability: float = 0.82
    episodes: int = 2000
    seed: int = 7


@dataclass(frozen=True)
class Episode:
    true_world_id: int
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class PolicyResult:
    name: str
    accuracy: float
    average_final_entropy: float
    average_total_information_gain: float
    average_information_gain_per_question: float
    average_true_world_probability: float
    entropy_trajectory: tuple[float, ...]


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Minimal test: are questions useful because they reduce uncertainty?"
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--initial-observations", type=int, default=2)
    parser.add_argument("--question-budget", type=int, default=4)
    parser.add_argument("--sensor-reliability", type=float, default=0.82)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    config = ExperimentConfig(
        feature_count=args.feature_count,
        initial_observations=args.initial_observations,
        question_budget=args.question_budget,
        sensor_reliability=args.sensor_reliability,
        episodes=args.episodes,
        seed=args.seed,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.feature_count < 2:
        raise ValueError("feature_count must be >= 2")
    if not 0.5 < config.sensor_reliability <= 1.0:
        raise ValueError("sensor_reliability must be in (0.5, 1.0]")
    if config.initial_observations < 0 or config.question_budget < 0:
        raise ValueError("initial_observations and question_budget must be non-negative")
    if config.initial_observations >= config.feature_count:
        raise ValueError("initial_observations must be smaller than feature_count")
    if config.episodes <= 0:
        raise ValueError("episodes must be > 0")


def build_worlds(feature_count: int) -> list[World]:
    worlds: list[World] = []
    for world_id in range(1 << feature_count):
        bits = tuple((world_id >> bit_index) & 1 for bit_index in range(feature_count))
        worlds.append(bits)
    return worlds


def world_weight(world: World) -> float:
    score = 0.0

    if len(world) > 0:
        score += 1.2 if world[0] else -0.4
    if len(world) > 1:
        score += 0.8 if world[1] else 0.0
    if len(world) > 2:
        score += 0.9 if not world[2] else -0.2
    if len(world) > 4:
        score += 1.1 if world[3] == world[4] else -0.7
    if len(world) > 6:
        score += 1.0 if (world[5] and not world[6]) else -0.1
    if len(world) > 7:
        score += 0.7 if (world[0] and world[7]) else -0.3

    for feature_index in range(8, len(world)):
        if feature_index % 2 == 0:
            score += 0.2 if world[feature_index] else -0.05
        else:
            score += 0.15 if world[feature_index] else 0.0

    return math.exp(score)


def normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("Cannot normalize a distribution with non-positive mass")
    return [value / total for value in values]


def entropy(distribution: Sequence[float]) -> float:
    return -sum(probability * math.log2(probability) for probability in distribution if probability > 0)


def build_prior(worlds: Sequence[World]) -> list[float]:
    raw_weights = [world_weight(world) for world in worlds]
    return normalize(raw_weights)


def update_posterior(
    posterior: Sequence[float],
    worlds: Sequence[World],
    feature_index: int,
    observed_value: int,
    reliability: float,
) -> list[float]:
    updated = []
    for probability, world in zip(posterior, worlds):
        likelihood = reliability if world[feature_index] == observed_value else (1.0 - reliability)
        updated.append(probability * likelihood)
    return normalize(updated)


def argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def select_entropy_question(
    posterior: Sequence[float],
    worlds: Sequence[World],
    asked_features: set[int],
) -> int:
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
            conditional = normalize(conditional)
            expected_entropy += answer_probability * entropy(conditional)

        information_gain = base_entropy - expected_entropy
        if information_gain > best_gain or (
            math.isclose(information_gain, best_gain) and feature_index < best_feature
        ):
            best_gain = information_gain
            best_feature = feature_index

    return best_feature


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


def generate_episodes(
    config: ExperimentConfig,
    worlds: Sequence[World],
    prior: Sequence[float],
) -> list[Episode]:
    rng = random.Random(config.seed)
    true_world_ids = rng.choices(range(len(worlds)), weights=prior, k=config.episodes)
    episodes: list[Episode] = []

    for true_world_id in true_world_ids:
        true_world = worlds[true_world_id]
        observed_features = rng.sample(range(config.feature_count), config.initial_observations)
        observations: list[Observation] = []

        for feature_index in observed_features:
            if rng.random() < config.sensor_reliability:
                observed_value = true_world[feature_index]
            else:
                observed_value = 1 - true_world[feature_index]
            observations.append((feature_index, observed_value))

        episodes.append(Episode(true_world_id=true_world_id, observations=tuple(observations)))

    return episodes


def run_policy(
    policy_name: str,
    policy_seed: int,
    config: ExperimentConfig,
    worlds: Sequence[World],
    prior: Sequence[float],
    episodes: Sequence[Episode],
) -> PolicyResult:
    rng = random.Random(policy_seed)

    correct_guesses: list[int] = []
    final_entropies: list[float] = []
    total_information_gains: list[float] = []
    true_world_probabilities: list[float] = []
    entropy_buckets: list[list[float]] = [[] for _ in range(config.question_budget + 1)]

    for episode in episodes:
        posterior = list(prior)
        asked_features = set()
        total_information_gain = 0.0

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

        steps = 0 if policy_name == "no_questions" else min(
            config.question_budget,
            config.feature_count - len(asked_features),
        )

        for step_index in range(steps):
            before_entropy = entropy(posterior)
            if policy_name == "random_questions":
                feature_index = select_random_question(rng, config.feature_count, asked_features)
            elif policy_name == "entropy_questions":
                feature_index = select_entropy_question(posterior, worlds, asked_features)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

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
            entropy_buckets[step_index + 1].append(after_entropy)

        if steps < config.question_budget:
            frozen_entropy = entropy(posterior)
            for step_index in range(steps + 1, config.question_budget + 1):
                entropy_buckets[step_index].append(frozen_entropy)

        guessed_world_id = argmax(posterior)
        correct_guesses.append(1 if guessed_world_id == episode.true_world_id else 0)
        final_entropies.append(entropy(posterior))
        total_information_gains.append(total_information_gain)
        true_world_probabilities.append(posterior[episode.true_world_id])

    asked_questions = config.question_budget if policy_name != "no_questions" else 0
    average_information_gain_per_question = (
        statistics.mean(total_information_gains) / asked_questions if asked_questions else 0.0
    )

    return PolicyResult(
        name=policy_name,
        accuracy=statistics.mean(correct_guesses),
        average_final_entropy=statistics.mean(final_entropies),
        average_total_information_gain=statistics.mean(total_information_gains),
        average_information_gain_per_question=average_information_gain_per_question,
        average_true_world_probability=statistics.mean(true_world_probabilities),
        entropy_trajectory=tuple(statistics.mean(bucket) for bucket in entropy_buckets),
    )


def print_report(config: ExperimentConfig, worlds: Sequence[World], results: Sequence[PolicyResult]) -> None:
    print("Experiment: active questions as uncertainty reduction")
    print(f"Worlds: {len(worlds)}")
    print(f"Features per world: {config.feature_count}")
    print(
        "Initial noisy observations: "
        f"{config.initial_observations} (reliability={config.sensor_reliability:.2f})"
    )
    print(f"Question budget: {config.question_budget}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<20}"
        f"{'Accuracy':>10}"
        f"{'Final H':>12}"
        f"{'Total IG':>12}"
        f"{'IG/Q':>10}"
        f"{'P(true)':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<20}"
            f"{result.accuracy:>10.3f}"
            f"{result.average_final_entropy:>12.3f}"
            f"{result.average_total_information_gain:>12.3f}"
            f"{result.average_information_gain_per_question:>10.3f}"
            f"{result.average_true_world_probability:>12.3f}"
        )

    print()
    print("Entropy trajectory (after initial observations and each question):")
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
    episodes = generate_episodes(config, worlds, prior)

    results = [
        run_policy("no_questions", config.seed + 11, config, worlds, prior, episodes),
        run_policy("random_questions", config.seed + 23, config, worlds, prior, episodes),
        run_policy("entropy_questions", config.seed + 37, config, worlds, prior, episodes),
    ]
    print_report(config, worlds, results)


if __name__ == "__main__":
    main()
