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
    update_posterior,
    validate_config,
)


Observation = tuple[int, int]


@dataclass(frozen=True)
class MemoryTransferConfig:
    feature_count: int = 8
    initial_observations: int = 2
    question_budget: int = 4
    sensor_reliability: float = 0.82
    episodes: int = 800
    seed: int = 7
    question_cost: float = 0.85
    block_size: int = 40
    mutation_rate: float = 0.05
    memory_decay: float = 0.92
    memory_prior_strength: float = 40.0

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
    block_position: int
    true_world_id: int
    observations: tuple[Observation, ...]


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
    early_block_accuracy: float
    late_block_accuracy: float
    early_block_questions: float
    late_block_questions: float


def parse_args() -> MemoryTransferConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether episodic memory reduces question load and improves transfer "
            "inside repeated world contexts."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--initial-observations", type=int, default=2)
    parser.add_argument("--question-budget", type=int, default=4)
    parser.add_argument("--sensor-reliability", type=float, default=0.82)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question-cost", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--memory-decay", type=float, default=0.92)
    parser.add_argument("--memory-prior-strength", type=float, default=40.0)
    args = parser.parse_args()

    config = MemoryTransferConfig(
        feature_count=args.feature_count,
        initial_observations=args.initial_observations,
        question_budget=args.question_budget,
        sensor_reliability=args.sensor_reliability,
        episodes=args.episodes,
        seed=args.seed,
        question_cost=args.question_cost,
        block_size=args.block_size,
        mutation_rate=args.mutation_rate,
        memory_decay=args.memory_decay,
        memory_prior_strength=args.memory_prior_strength,
    )

    validate_config(config.to_base_config())
    if config.feature_count != 8:
        raise ValueError("memory_transfer_test currently expects feature_count == 8")
    if config.block_size <= 1:
        raise ValueError("block_size must be > 1")
    if not 0.0 <= config.mutation_rate < 1.0:
        raise ValueError("mutation_rate must be in [0, 1)")
    if not 0.0 < config.memory_decay <= 1.0:
        raise ValueError("memory_decay must be in (0, 1]")
    if config.memory_prior_strength <= 0:
        raise ValueError("memory_prior_strength must be > 0")
    if config.question_cost < 0:
        raise ValueError("question_cost must be non-negative")
    return config


def build_context_prototypes() -> dict[int, tuple[int, ...]]:
    return {
        0: (3, 11, 35, 99),
        1: (144, 176, 208, 240),
        2: (18, 22, 82, 86),
        3: (129, 153, 165, 189),
    }


def mutate_world_id(
    rng: random.Random,
    worlds: Sequence[World],
    prototype_world_id: int,
    mutation_rate: float,
) -> int:
    mutated_bits = list(worlds[prototype_world_id])
    for feature_index in range(len(mutated_bits)):
        if rng.random() < mutation_rate:
            mutated_bits[feature_index] = 1 - mutated_bits[feature_index]

    world_id = 0
    for bit_index, bit_value in enumerate(mutated_bits):
        world_id |= bit_value << bit_index
    return world_id


def generate_context_episodes(
    config: MemoryTransferConfig,
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
                block_position=block_position,
                true_world_id=true_world_id,
                observations=tuple(observations),
            )
        )

    return episodes


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


def normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def run_policy(
    policy_name: str,
    config: MemoryTransferConfig,
    worlds: Sequence[World],
    static_prior: Sequence[float],
    episodes: Sequence[Episode],
) -> PolicyResult:
    memory_counts = [config.memory_prior_strength * probability for probability in static_prior]

    accuracies: list[int] = []
    questions_asked: list[int] = []
    final_entropies: list[float] = []
    total_information_gains: list[float] = []
    true_world_probabilities: list[float] = []
    intrinsic_values: list[float] = []
    early_block_accuracies: list[int] = []
    late_block_accuracies: list[int] = []
    early_block_questions: list[int] = []
    late_block_questions: list[int] = []

    for episode in episodes:
        if policy_name.startswith("memory_"):
            prior = normalize(memory_counts)
        elif policy_name.startswith("static_"):
            prior = list(static_prior)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

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

        for _ in range(config.question_budget):
            if len(asked_features) >= config.feature_count:
                break

            feature_index, expected_gain = best_entropy_question_with_gain(
                posterior=posterior,
                worlds=worlds,
                asked_features=asked_features,
            )

            if policy_name.endswith("cost_aware") and expected_gain <= config.question_cost:
                break

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

        guessed_world_id = argmax(posterior)
        is_correct = 1 if guessed_world_id == episode.true_world_id else 0
        accuracies.append(is_correct)
        questions_asked.append(episode_questions_asked)
        final_entropies.append(entropy(posterior))
        total_information_gains.append(total_information_gain)
        true_world_probabilities.append(posterior[episode.true_world_id])
        intrinsic_values.append(
            total_information_gain - config.question_cost * episode_questions_asked
        )

        if episode.block_position < config.block_size // 2:
            early_block_accuracies.append(is_correct)
            early_block_questions.append(episode_questions_asked)
        else:
            late_block_accuracies.append(is_correct)
            late_block_questions.append(episode_questions_asked)

        if policy_name.startswith("memory_"):
            memory_counts = [config.memory_decay * count for count in memory_counts]
            memory_counts[episode.true_world_id] += 1.0

    total_questions = sum(questions_asked)
    information_gain_per_question = (
        sum(total_information_gains) / total_questions if total_questions else 0.0
    )

    return PolicyResult(
        name=policy_name,
        accuracy=statistics.mean(accuracies),
        average_questions_asked=statistics.mean(questions_asked),
        average_final_entropy=statistics.mean(final_entropies),
        average_total_information_gain=statistics.mean(total_information_gains),
        information_gain_per_question=information_gain_per_question,
        average_true_world_probability=statistics.mean(true_world_probabilities),
        average_intrinsic_value=statistics.mean(intrinsic_values),
        early_block_accuracy=statistics.mean(early_block_accuracies),
        late_block_accuracy=statistics.mean(late_block_accuracies),
        early_block_questions=statistics.mean(early_block_questions),
        late_block_questions=statistics.mean(late_block_questions),
    )


def print_report(config: MemoryTransferConfig, results: Sequence[PolicyResult]) -> None:
    print("Experiment: memory transfer across repeated contexts")
    print("Environment: episodes come in blocks with recurring world prototypes")
    print("Feedback: true world is revealed after each episode and can update memory")
    print(f"Block size: {config.block_size}")
    print(f"Mutation rate: {config.mutation_rate:.3f}")
    print(f"Question cost: {config.question_cost:.3f}")
    print(f"Memory decay: {config.memory_decay:.3f}")
    print(f"Memory prior strength: {config.memory_prior_strength:.1f}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<22}"
        f"{'Q asked':>10}"
        f"{'Accuracy':>10}"
        f"{'Final H':>10}"
        f"{'IG/Q':>10}"
        f"{'Intrinsic':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<22}"
            f"{result.average_questions_asked:>10.3f}"
            f"{result.accuracy:>10.3f}"
            f"{result.average_final_entropy:>10.3f}"
            f"{result.information_gain_per_question:>10.3f}"
            f"{result.average_intrinsic_value:>11.3f}"
        )

    print()
    print("Transfer inside blocks:")
    transfer_header = (
        f"{'Policy':<22}"
        f"{'Early Q':>10}"
        f"{'Late Q':>10}"
        f"{'Early Acc':>12}"
        f"{'Late Acc':>11}"
    )
    print(transfer_header)
    print("-" * len(transfer_header))
    for result in results:
        print(
            f"{result.name:<22}"
            f"{result.early_block_questions:>10.3f}"
            f"{result.late_block_questions:>10.3f}"
            f"{result.early_block_accuracy:>12.3f}"
            f"{result.late_block_accuracy:>11.3f}"
        )


def main() -> None:
    config = parse_args()
    worlds = build_worlds(config.feature_count)
    static_prior = build_prior(worlds)
    episodes = generate_context_episodes(config, worlds)

    results = [
        run_policy("static_budgeted", config, worlds, static_prior, episodes),
        run_policy("memory_budgeted", config, worlds, static_prior, episodes),
        run_policy("static_cost_aware", config, worlds, static_prior, episodes),
        run_policy("memory_cost_aware", config, worlds, static_prior, episodes),
    ]
    print_report(config, results)


if __name__ == "__main__":
    main()
