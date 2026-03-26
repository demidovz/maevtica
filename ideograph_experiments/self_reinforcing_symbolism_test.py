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
class SelfReinforcingConfig:
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
    true_world_id: int
    grounded_observations: tuple[Observation, ...]
    symbolic_observations: tuple[Observation, ...]
    symbolic_permutation: tuple[int, ...]


@dataclass(frozen=True)
class PolicyResult:
    name: str
    accuracy: float
    average_questions_asked: float
    average_confidence: float
    average_final_entropy: float
    overconfidence_gap: float
    average_intrinsic_value: float
    average_true_world_probability: float
    information_gain_per_question: float
    final_memory_entropy: float
    final_memory_top1_mass: float
    unique_final_guesses: int


def parse_args() -> SelfReinforcingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether closed symbolic self-feedback creates pseudo-semantics: "
            "high confidence, fewer questions, and low truth alignment."
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

    config = SelfReinforcingConfig(
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
        raise ValueError("self_reinforcing_symbolism_test currently expects feature_count == 8")
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
    bits = list(worlds[prototype_world_id])
    for feature_index in range(len(bits)):
        if rng.random() < mutation_rate:
            bits[feature_index] = 1 - bits[feature_index]

    world_id = 0
    for bit_index, bit_value in enumerate(bits):
        world_id |= bit_value << bit_index
    return world_id


def generate_episodes(
    config: SelfReinforcingConfig,
    worlds: Sequence[World],
) -> list[Episode]:
    rng = random.Random(config.seed)
    context_prototypes = build_context_prototypes()
    prototype_weights = (0.45, 0.25, 0.20, 0.10)
    context_ids = tuple(sorted(context_prototypes))

    episodes: list[Episode] = []
    for episode_index in range(config.episodes):
        block_index = episode_index // config.block_size
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
        symbolic_permutation = tuple(rng.sample(range(config.feature_count), config.feature_count))

        observed_channels = rng.sample(range(config.feature_count), config.initial_observations)
        grounded_observations: list[Observation] = []
        symbolic_observations: list[Observation] = []

        for channel in observed_channels:
            grounded_value = worlds[true_world_id][channel]
            symbolic_value = worlds[true_world_id][symbolic_permutation[channel]]

            if rng.random() >= config.sensor_reliability:
                grounded_value = 1 - grounded_value
            if rng.random() >= config.sensor_reliability:
                symbolic_value = 1 - symbolic_value

            grounded_observations.append((channel, grounded_value))
            symbolic_observations.append((channel, symbolic_value))

        episodes.append(
            Episode(
                context_id=context_id,
                true_world_id=true_world_id,
                grounded_observations=tuple(grounded_observations),
                symbolic_observations=tuple(symbolic_observations),
                symbolic_permutation=symbolic_permutation,
            )
        )

    return episodes


def best_entropy_question_with_gain(
    posterior: Sequence[float],
    worlds: Sequence[World],
    asked_channels: set[int],
) -> tuple[int, float]:
    base_entropy = entropy(posterior)
    best_channel = -1
    best_gain = float("-inf")

    for channel in range(len(worlds[0])):
        if channel in asked_channels:
            continue

        probability_one = sum(
            probability
            for probability, world in zip(posterior, worlds)
            if world[channel] == 1
        )
        probability_zero = 1.0 - probability_one

        expected_entropy = 0.0
        for observed_value, answer_probability in ((0, probability_zero), (1, probability_one)):
            if answer_probability <= 0:
                continue
            conditional = [
                probability if world[channel] == observed_value else 0.0
                for probability, world in zip(posterior, worlds)
            ]
            conditional_mass = sum(conditional)
            conditional = [value / conditional_mass for value in conditional]
            expected_entropy += answer_probability * entropy(conditional)

        information_gain = base_entropy - expected_entropy
        if information_gain > best_gain or (
            math.isclose(information_gain, best_gain) and channel < best_channel
        ):
            best_channel = channel
            best_gain = information_gain

    return best_channel, best_gain


def normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def run_policy(
    name: str,
    mode: str,
    config: SelfReinforcingConfig,
    worlds: Sequence[World],
    static_prior: Sequence[float],
    episodes: Sequence[Episode],
) -> PolicyResult:
    memory_counts = [config.memory_prior_strength * probability for probability in static_prior]

    accuracies: list[int] = []
    questions_asked: list[int] = []
    confidences: list[float] = []
    final_entropies: list[float] = []
    intrinsic_values: list[float] = []
    true_world_probabilities: list[float] = []
    total_information_gains: list[float] = []
    final_guesses: list[int] = []

    for episode in episodes:
        prior = normalize(memory_counts)
        posterior = list(prior)
        asked_channels: set[int] = set()
        total_information_gain = 0.0
        episode_questions = 0

        if mode == "grounded_feedback":
            observations = episode.grounded_observations
            symbolic = False
        elif mode in {"symbolic_feedback", "symbolic_self_bootstrap"}:
            observations = episode.symbolic_observations
            symbolic = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for channel, observed_value in observations:
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_channels.add(channel)

        for _ in range(config.question_budget):
            if len(asked_channels) >= config.feature_count:
                break

            channel, expected_gain = best_entropy_question_with_gain(
                posterior=posterior,
                worlds=worlds,
                asked_channels=asked_channels,
            )
            if expected_gain <= config.question_cost:
                break

            actual_feature = (
                channel
                if not symbolic
                else episode.symbolic_permutation[channel]
            )
            answer = worlds[episode.true_world_id][actual_feature]
            before_entropy = entropy(posterior)
            posterior = update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=answer,
                reliability=1.0,
            )
            asked_channels.add(channel)
            total_information_gain += before_entropy - entropy(posterior)
            episode_questions += 1

        guess = argmax(posterior)
        confidence = max(posterior)
        is_correct = 1 if guess == episode.true_world_id else 0

        accuracies.append(is_correct)
        questions_asked.append(episode_questions)
        confidences.append(confidence)
        final_entropies.append(entropy(posterior))
        intrinsic_values.append(
            total_information_gain - config.question_cost * episode_questions
        )
        true_world_probabilities.append(posterior[episode.true_world_id])
        total_information_gains.append(total_information_gain)
        final_guesses.append(guess)

        memory_counts = [config.memory_decay * value for value in memory_counts]
        if mode in {"grounded_feedback", "symbolic_feedback"}:
            memory_counts[episode.true_world_id] += 1.0
        else:
            memory_counts[guess] += 1.0

    final_memory = normalize(memory_counts)
    total_questions = sum(questions_asked)
    information_gain_per_question = (
        sum(total_information_gains) / total_questions if total_questions else 0.0
    )
    accuracy = statistics.mean(accuracies)
    average_confidence = statistics.mean(confidences)

    return PolicyResult(
        name=name,
        accuracy=accuracy,
        average_questions_asked=statistics.mean(questions_asked),
        average_confidence=average_confidence,
        average_final_entropy=statistics.mean(final_entropies),
        overconfidence_gap=average_confidence - accuracy,
        average_intrinsic_value=statistics.mean(intrinsic_values),
        average_true_world_probability=statistics.mean(true_world_probabilities),
        information_gain_per_question=information_gain_per_question,
        final_memory_entropy=entropy(final_memory),
        final_memory_top1_mass=max(final_memory),
        unique_final_guesses=len(set(final_guesses)),
    )


def print_report(
    config: SelfReinforcingConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: self-reinforcing symbolism without external feedback")
    print("Grounded feedback: questions and memory are tied to the true world")
    print("Symbolic feedback: channel labels persist but their referents permute each episode")
    print("Self-bootstrap: symbolic agent updates memory from its own final guess instead of truth")
    print(f"Question cost: {config.question_cost:.3f}")
    print(f"Block size: {config.block_size}")
    print(f"Mutation rate: {config.mutation_rate:.3f}")
    print(f"Memory decay: {config.memory_decay:.3f}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<24}"
        f"{'Q asked':>10}"
        f"{'Acc':>8}"
        f"{'Conf':>8}"
        f"{'Over':>8}"
        f"{'Final H':>10}"
        f"{'IG/Q':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<24}"
            f"{result.average_questions_asked:>10.3f}"
            f"{result.accuracy:>8.3f}"
            f"{result.average_confidence:>8.3f}"
            f"{result.overconfidence_gap:>8.3f}"
            f"{result.average_final_entropy:>10.3f}"
            f"{result.information_gain_per_question:>8.3f}"
        )

    print()
    print("Memory state after all episodes:")
    memory_header = (
        f"{'Policy':<24}"
        f"{'Memory H':>10}"
        f"{'Top1':>8}"
        f"{'Guesses':>10}"
        f"{'Intrinsic':>11}"
    )
    print(memory_header)
    print("-" * len(memory_header))
    for result in results:
        print(
            f"{result.name:<24}"
            f"{result.final_memory_entropy:>10.3f}"
            f"{result.final_memory_top1_mass:>8.3f}"
            f"{result.unique_final_guesses:>10}"
            f"{result.average_intrinsic_value:>11.3f}"
        )

    result_map = {result.name: result for result in results}
    if (
        "grounded_feedback" in result_map
        and "symbolic_feedback" in result_map
        and "symbolic_self_bootstrap" in result_map
    ):
        grounded = result_map["grounded_feedback"]
        symbolic = result_map["symbolic_feedback"]
        bootstrap = result_map["symbolic_self_bootstrap"]
        print()
        print("Pseudo-semantics gap:")
        print(
            "bootstrap vs grounded: "
            f"accuracy {bootstrap.accuracy - grounded.accuracy:.3f}, "
            f"confidence {bootstrap.average_confidence - grounded.average_confidence:.3f}, "
            f"overconfidence {bootstrap.overconfidence_gap - grounded.overconfidence_gap:.3f}"
        )
        print(
            "bootstrap vs symbolic_feedback: "
            f"questions {bootstrap.average_questions_asked - symbolic.average_questions_asked:.3f}, "
            f"memory entropy {bootstrap.final_memory_entropy - symbolic.final_memory_entropy:.3f}, "
            f"top1 mass {bootstrap.final_memory_top1_mass - symbolic.final_memory_top1_mass:.3f}"
        )


def main() -> None:
    config = parse_args()
    worlds = build_worlds(config.feature_count)
    static_prior = build_prior(worlds)
    episodes = generate_episodes(config, worlds)

    results = [
        run_policy(
            name="grounded_feedback",
            mode="grounded_feedback",
            config=config,
            worlds=worlds,
            static_prior=static_prior,
            episodes=episodes,
        ),
        run_policy(
            name="symbolic_feedback",
            mode="symbolic_feedback",
            config=config,
            worlds=worlds,
            static_prior=static_prior,
            episodes=episodes,
        ),
        run_policy(
            name="symbolic_self_bootstrap",
            mode="symbolic_self_bootstrap",
            config=config,
            worlds=worlds,
            static_prior=static_prior,
            episodes=episodes,
        ),
    ]
    print_report(config, results)


if __name__ == "__main__":
    main()
