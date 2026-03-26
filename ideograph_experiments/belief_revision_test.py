from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


ContextId = int
World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefRevisionConfig:
    feature_count: int = 8
    episodes: int = 800
    block_size: int = 40
    seed: int = 7
    change_block: int = 10
    feedback_flip_rate: float = 0.08
    consistency: float = 0.92
    reentry_window: int = 5


@dataclass(frozen=True)
class Episode:
    block_index: int
    context_id: ContextId
    block_position: int
    world: World
    active_rule: int
    contradictory_feedback: bool


@dataclass(frozen=True)
class PolicySpec:
    name: str
    persistence: float
    learning_rate: float


@dataclass(frozen=True)
class PolicyResult:
    name: str
    action_accuracy: float
    noise_hold_rate: float
    mean_switch_lag: float
    post_change_rule_match_rate: float
    late_post_rule_match_rate: float
    reentry_rule_match_rate: float
    final_rule_accuracy: float
    final_rule_entropy: float
    average_action_confidence: float


def parse_args() -> BeliefRevisionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test belief hysteresis: a useful belief should resist isolated contradictions "
            "but revise under sustained rule change."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--change-block", type=int, default=10)
    parser.add_argument("--feedback-flip-rate", type=float, default=0.08)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--reentry-window", type=int, default=5)
    args = parser.parse_args()

    config = BeliefRevisionConfig(
        feature_count=args.feature_count,
        episodes=args.episodes,
        block_size=args.block_size,
        seed=args.seed,
        change_block=args.change_block,
        feedback_flip_rate=args.feedback_flip_rate,
        consistency=args.consistency,
        reentry_window=args.reentry_window,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefRevisionConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_revision_test currently expects feature_count == 8")
    if config.episodes <= 0:
        raise ValueError("episodes must be > 0")
    if config.block_size <= 1:
        raise ValueError("block_size must be > 1")
    if not 0.0 <= config.feedback_flip_rate < 1.0:
        raise ValueError("feedback_flip_rate must be in [0, 1)")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.change_block <= 0:
        raise ValueError("change_block must be > 0")
    if config.change_block >= config.episodes // config.block_size:
        raise ValueError("change_block must be smaller than the total number of blocks")
    if config.reentry_window <= 0 or config.reentry_window > config.block_size:
        raise ValueError("reentry_window must be in [1, block_size]")


def build_old_rules() -> dict[ContextId, int]:
    return {0: 0, 1: 4, 2: 2, 3: 6}


def build_new_rules() -> dict[ContextId, int]:
    return {0: 5, 1: 1, 2: 7, 3: 3}


def build_context_biases() -> dict[ContextId, dict[int, float]]:
    return {
        0: {0: 0.70, 5: 0.30},
        1: {4: 0.70, 1: 0.30},
        2: {2: 0.70, 7: 0.30},
        3: {6: 0.70, 3: 0.30},
    }


def build_context_ids() -> tuple[ContextId, ...]:
    return (0, 1, 2, 3)


def sample_world(
    rng: random.Random,
    context_id: ContextId,
    feature_count: int,
    context_biases: dict[ContextId, dict[int, float]],
) -> World:
    bits = []
    for feature_index in range(feature_count):
        probability_one = context_biases[context_id].get(feature_index, 0.5)
        bits.append(1 if rng.random() < probability_one else 0)
    return tuple(bits)


def generate_episodes(config: BeliefRevisionConfig) -> tuple[list[Episode], dict[ContextId, int], dict[ContextId, int]]:
    rng = random.Random(config.seed)
    context_ids = build_context_ids()
    old_rules = build_old_rules()
    new_rules = build_new_rules()
    context_biases = build_context_biases()

    episodes: list[Episode] = []
    for episode_index in range(config.episodes):
        block_index = episode_index // config.block_size
        block_position = episode_index % config.block_size
        context_id = context_ids[block_index % len(context_ids)]
        active_rules = old_rules if block_index < config.change_block else new_rules
        active_rule = active_rules[context_id]
        world = sample_world(rng, context_id, config.feature_count, context_biases)
        contradictory_feedback = rng.random() < config.feedback_flip_rate
        episodes.append(
            Episode(
                block_index=block_index,
                context_id=context_id,
                block_position=block_position,
                world=world,
                active_rule=active_rule,
                contradictory_feedback=contradictory_feedback,
            )
        )

    return episodes, old_rules, new_rules


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def entropy(distribution: Sequence[float]) -> float:
    return -sum(probability * math.log2(probability) for probability in distribution if probability > 0)


def predict_action(world: World, belief: Sequence[float]) -> tuple[int, float]:
    probability_action_one = sum(weight * world[feature_index] for feature_index, weight in enumerate(belief))
    predicted_action = 1 if probability_action_one >= 0.5 else 0
    confidence = max(probability_action_one, 1.0 - probability_action_one)
    return predicted_action, confidence


def update_scores(
    scores: list[float],
    world: World,
    feedback_action: int,
    persistence: float,
    learning_rate: float,
    consistency: float,
) -> list[float]:
    updated = []
    for feature_index, score in enumerate(scores):
        likelihood = consistency if world[feature_index] == feedback_action else (1.0 - consistency)
        updated.append(persistence * score + learning_rate * math.log(likelihood))
    return updated


def run_policy(
    spec: PolicySpec,
    config: BeliefRevisionConfig,
    episodes: Sequence[Episode],
    old_rules: dict[ContextId, int],
    new_rules: dict[ContextId, int],
) -> PolicyResult:
    context_ids = tuple(sorted(old_rules))
    scores = {
        context_id: [0.0] * config.feature_count
        for context_id in context_ids
    }

    action_accuracies: list[int] = []
    action_confidences: list[float] = []
    noise_hold_values: list[int] = []
    post_change_rule_matches: list[int] = []
    late_post_rule_matches: list[int] = []
    reentry_rule_matches: list[int] = []
    post_change_seen: dict[ContextId, int] = {context_id: 0 for context_id in context_ids}
    switch_lags: dict[ContextId, int | None] = {context_id: None for context_id in context_ids}

    for episode in episodes:
        belief = softmax(scores[episode.context_id])
        predicted_action, confidence = predict_action(episode.world, belief)
        true_action = episode.world[episode.active_rule]
        action_accuracies.append(1 if predicted_action == true_action else 0)
        action_confidences.append(confidence)

        top_rule_before_update = max(range(config.feature_count), key=belief.__getitem__)
        if (
            episode.block_index < config.change_block
            and episode.contradictory_feedback
        ):
            noise_hold_values.append(1 if top_rule_before_update == old_rules[episode.context_id] else 0)

        feedback_action = 1 - true_action if episode.contradictory_feedback else true_action
        scores[episode.context_id] = update_scores(
            scores=scores[episode.context_id],
            world=episode.world,
            feedback_action=feedback_action,
            persistence=spec.persistence,
            learning_rate=spec.learning_rate,
            consistency=config.consistency,
        )

        top_rule_after_update = max(
            range(config.feature_count),
            key=scores[episode.context_id].__getitem__,
        )

        if episode.block_index >= config.change_block:
            post_change_rule_matches.append(
                1 if top_rule_after_update == new_rules[episode.context_id] else 0
            )
            if episode.block_index >= config.change_block + 2:
                late_post_rule_matches.append(
                    1 if top_rule_after_update == new_rules[episode.context_id] else 0
                )
            if (
                episode.block_index >= config.change_block + 4
                and episode.block_position < config.reentry_window
            ):
                reentry_rule_matches.append(
                    1 if top_rule_after_update == new_rules[episode.context_id] else 0
                )

            if (
                switch_lags[episode.context_id] is None
                and top_rule_after_update == new_rules[episode.context_id]
            ):
                switch_lags[episode.context_id] = post_change_seen[episode.context_id]
            post_change_seen[episode.context_id] += 1

    for context_id in context_ids:
        if switch_lags[context_id] is None:
            switch_lags[context_id] = post_change_seen[context_id]

    final_rule_beliefs = {
        context_id: softmax(scores[context_id])
        for context_id in context_ids
    }
    final_rule_accuracy = statistics.mean(
        1
        if max(range(config.feature_count), key=final_rule_beliefs[context_id].__getitem__) == new_rules[context_id]
        else 0
        for context_id in context_ids
    )
    final_rule_entropy = statistics.mean(
        entropy(final_rule_beliefs[context_id])
        for context_id in context_ids
    )

    return PolicyResult(
        name=spec.name,
        action_accuracy=statistics.mean(action_accuracies),
        noise_hold_rate=statistics.mean(noise_hold_values),
        mean_switch_lag=statistics.mean(switch_lags.values()),
        post_change_rule_match_rate=statistics.mean(post_change_rule_matches),
        late_post_rule_match_rate=statistics.mean(late_post_rule_matches),
        reentry_rule_match_rate=statistics.mean(reentry_rule_matches),
        final_rule_accuracy=final_rule_accuracy,
        final_rule_entropy=final_rule_entropy,
        average_action_confidence=statistics.mean(action_confidences),
    )


def print_report(
    config: BeliefRevisionConfig,
    results: Sequence[PolicyResult],
    old_rules: dict[ContextId, int],
    new_rules: dict[ContextId, int],
) -> None:
    print("Experiment: belief hysteresis / belief revision")
    print("Old rules (context -> decisive feature):")
    print(", ".join(f"{context_id}->{feature_index}" for context_id, feature_index in sorted(old_rules.items())))
    print("New rules after shift:")
    print(", ".join(f"{context_id}->{feature_index}" for context_id, feature_index in sorted(new_rules.items())))
    print(f"Change block: {config.change_block}")
    print(f"Feedback flip rate: {config.feedback_flip_rate:.3f}")
    print(f"Consistency: {config.consistency:.3f}")
    print(f"Episodes: {config.episodes}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<14}"
        f"{'ActAcc':>9}"
        f"{'NoiseHold':>11}"
        f"{'Lag':>8}"
        f"{'PostMatch':>11}"
        f"{'LatePost':>10}"
        f"{'RuleAcc':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<14}"
            f"{result.action_accuracy:>9.3f}"
            f"{result.noise_hold_rate:>11.3f}"
            f"{result.mean_switch_lag:>8.3f}"
            f"{result.post_change_rule_match_rate:>11.3f}"
            f"{result.late_post_rule_match_rate:>10.3f}"
            f"{result.final_rule_accuracy:>10.3f}"
        )

    print()
    print("Reentry and confidence:")
    sub_header = (
        f"{'Policy':<14}"
        f"{'Reentry':>10}"
        f"{'Conf':>10}"
        f"{'Belief H':>10}"
    )
    print(sub_header)
    print("-" * len(sub_header))
    for result in results:
        print(
            f"{result.name:<14}"
            f"{result.reentry_rule_match_rate:>10.3f}"
            f"{result.average_action_confidence:>10.3f}"
            f"{result.final_rule_entropy:>10.3f}"
        )


def main() -> None:
    config = parse_args()
    episodes, old_rules, new_rules = generate_episodes(config)
    specs = [
        PolicySpec(name="fragile", persistence=0.30, learning_rate=1.00),
        PolicySpec(name="adaptive", persistence=0.90, learning_rate=0.25),
        PolicySpec(name="rigid", persistence=0.99, learning_rate=0.05),
    ]
    results = [
        run_policy(spec, config, episodes, old_rules, new_rules)
        for spec in specs
    ]
    print_report(config, results, old_rules, new_rules)


if __name__ == "__main__":
    main()
