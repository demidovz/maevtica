from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefPruningReactivationConfig:
    feature_count: int = 8
    pretrain_episodes: int = 120
    diagnostic_budget: int = 4
    ood_retention_episodes: int = 32
    return_episodes: int = 32
    seed: int = 7
    consistency: float = 0.92
    forgetting_decay: float = 0.93
    transfer_keep_mass: float = 0.80
    synthesis_keep_mass: float = 0.20
    synthesis_prior: float = 0.25


@dataclass(frozen=True)
class Hypothesis:
    name: str
    kind: str

    def predict(self, world: World) -> int:
        feature_0 = world[0]
        feature_2 = world[2]
        feature_5 = world[5]

        if self.kind == "xor":
            return feature_0 ^ feature_5
        if self.kind == "xnor":
            return 1 - (feature_0 ^ feature_5)
        if self.kind == "and":
            return feature_0 & feature_5
        if self.kind == "or":
            return feature_0 | feature_5
        if self.kind == "xor3":
            return feature_0 ^ feature_2 ^ feature_5
        if self.kind == "nand3":
            return 1 - (feature_0 & feature_2 & feature_5)
        if self.kind == "gate_or_and":
            return (feature_0 & feature_5) if feature_2 == 1 else (feature_0 | feature_5)
        if self.kind == "maj3":
            return 1 if (feature_0 + feature_2 + feature_5) >= 2 else 0
        raise ValueError(f"Unknown hypothesis kind: {self.kind}")


@dataclass(frozen=True)
class PolicyResult:
    policy_name: str
    synthesis_spawn_step: int | None
    reactivation_episode: int | None
    active_hypotheses: int
    ood_retention_accuracy: float
    return_first_ten_accuracy: float
    return_overall_accuracy: float
    final_top_hypothesis: str
    final_top_mass: float


def parse_args() -> BeliefPruningReactivationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether pruning obsolete transfer beliefs can coexist with fast "
            "reactivation when the world returns to an old regime."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--pretrain-episodes", type=int, default=120)
    parser.add_argument("--diagnostic-budget", type=int, default=4)
    parser.add_argument("--ood-retention-episodes", type=int, default=32)
    parser.add_argument("--return-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--forgetting-decay", type=float, default=0.93)
    parser.add_argument("--transfer-keep-mass", type=float, default=0.80)
    parser.add_argument("--synthesis-keep-mass", type=float, default=0.20)
    parser.add_argument("--synthesis-prior", type=float, default=0.25)
    args = parser.parse_args()

    config = BeliefPruningReactivationConfig(
        feature_count=args.feature_count,
        pretrain_episodes=args.pretrain_episodes,
        diagnostic_budget=args.diagnostic_budget,
        ood_retention_episodes=args.ood_retention_episodes,
        return_episodes=args.return_episodes,
        seed=args.seed,
        consistency=args.consistency,
        forgetting_decay=args.forgetting_decay,
        transfer_keep_mass=args.transfer_keep_mass,
        synthesis_keep_mass=args.synthesis_keep_mass,
        synthesis_prior=args.synthesis_prior,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefPruningReactivationConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_pruning_reactivation_test currently expects feature_count == 8")
    if config.pretrain_episodes <= 0:
        raise ValueError("pretrain_episodes must be > 0")
    if config.diagnostic_budget <= 0 or config.ood_retention_episodes <= 0 or config.return_episodes <= 0:
        raise ValueError("episode counts must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if not 0.0 < config.forgetting_decay <= 1.0:
        raise ValueError("forgetting_decay must be in (0, 1]")
    if config.transfer_keep_mass <= 0.0 or config.synthesis_keep_mass <= 0.0:
        raise ValueError("keep masses must be > 0")
    if not math.isclose(config.transfer_keep_mass + config.synthesis_keep_mass, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("transfer_keep_mass + synthesis_keep_mass must equal 1.0")
    if config.synthesis_prior <= 0.0:
        raise ValueError("synthesis_prior must be > 0")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def build_transfer_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(name="xor_0_5", kind="xor"),
        Hypothesis(name="xnor_0_5", kind="xnor"),
        Hypothesis(name="and_0_5", kind="and"),
        Hypothesis(name="or_0_5", kind="or"),
    ]


def build_synthesis_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(name="xor3_0_2_5", kind="xor3"),
        Hypothesis(name="nand3_0_2_5", kind="nand3"),
        Hypothesis(name="gate_or_and_0_2_5", kind="gate_or_and"),
        Hypothesis(name="maj3_0_2_5", kind="maj3"),
    ]


def sample_world(rng: random.Random, feature_count: int) -> World:
    return tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))


def sample_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [sample_world(rng, feature_count) for _ in range(count)]


def build_ood_diagnostic_sequence(feature_count: int) -> list[World]:
    sequence_025 = [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    worlds: list[World] = []
    for feature_0, feature_2, feature_5 in sequence_025:
        values = [0] * feature_count
        values[0] = feature_0
        values[2] = feature_2
        values[5] = feature_5
        worlds.append(tuple(values))
    return worlds


def build_return_worlds(
    rng: random.Random,
    feature_count: int,
    count: int,
) -> list[World]:
    diagnostic_prefix = [
        (0, 1, 1),
        (1, 1, 1),
    ]
    worlds: list[World] = []
    for feature_0, feature_2, feature_5 in diagnostic_prefix:
        values = [0] * feature_count
        values[0] = feature_0
        values[2] = feature_2
        values[5] = feature_5
        worlds.append(tuple(values))

    while len(worlds) < count:
        worlds.append(sample_world(rng, feature_count))
    return worlds


def true_ood_target(world: World) -> int:
    return world[0] ^ world[2] ^ world[5]


def true_return_target(world: World) -> int:
    return world[0] ^ world[5]


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


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


def replay_observations(
    config: BeliefPruningReactivationConfig,
    hypotheses: Sequence[Hypothesis],
    priors: Sequence[float],
    observations: Sequence[tuple[World, int]],
) -> list[float]:
    scores = [math.log(prior) for prior in priors]
    for world, true_action in observations:
        update_scores(scores, hypotheses, world, true_action, config.consistency)
    return scores


def pair_inconsistency_detected(observations: Sequence[tuple[World, int]]) -> bool:
    seen_labels: dict[tuple[int, int], int] = {}
    for world, true_action in observations:
        pair_key = (world[0], world[5])
        previous = seen_labels.get(pair_key)
        if previous is not None and previous != true_action:
            return True
        seen_labels[pair_key] = true_action
    return False


def predict_probability_one(
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    world: World,
) -> float:
    belief = softmax(scores)
    return sum(
        weight * hypothesis.predict(world)
        for weight, hypothesis in zip(belief, hypotheses)
    )


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[Hypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def apply_forgetting(scores: list[float], priors: Sequence[float], decay: float) -> None:
    for index, prior in enumerate(priors):
        prior_score = math.log(prior)
        scores[index] = prior_score + decay * (scores[index] - prior_score)


def pretrain_transfer_library(
    config: BeliefPruningReactivationConfig,
    rng: random.Random,
) -> tuple[list[Hypothesis], list[float], list[float]]:
    hypotheses = build_transfer_hypotheses()
    scores = [0.0] * len(hypotheses)

    for _ in range(config.pretrain_episodes):
        world = sample_world(rng, config.feature_count)
        update_scores(scores, hypotheses, world, true_return_target(world), config.consistency)

    posterior = softmax(scores)
    return hypotheses, scores, posterior


def spawn_keep_all_state(
    config: BeliefPruningReactivationConfig,
    transfer_hypotheses: Sequence[Hypothesis],
    transfer_posterior: Sequence[float],
    ood_observations: Sequence[tuple[World, int]],
) -> tuple[list[Hypothesis], list[float], list[float]]:
    synthesis_hypotheses = build_synthesis_hypotheses()
    transfer_priors = [
        config.transfer_keep_mass * probability
        for probability in transfer_posterior
    ]
    synthesis_priors = [
        config.synthesis_keep_mass * config.synthesis_prior
        for _ in synthesis_hypotheses
    ]
    active_hypotheses = list(transfer_hypotheses) + synthesis_hypotheses
    active_priors = transfer_priors + synthesis_priors
    scores = replay_observations(config, active_hypotheses, active_priors, ood_observations)
    return active_hypotheses, scores, active_priors


def spawn_synthesis_only_state(
    config: BeliefPruningReactivationConfig,
    ood_observations: Sequence[tuple[World, int]],
) -> tuple[list[Hypothesis], list[float], list[float]]:
    synthesis_hypotheses = build_synthesis_hypotheses()
    synthesis_priors = [config.synthesis_prior] * len(synthesis_hypotheses)
    scores = replay_observations(config, synthesis_hypotheses, synthesis_priors, ood_observations)
    return synthesis_hypotheses, scores, synthesis_priors


def run_policy(
    config: BeliefPruningReactivationConfig,
    policy_name: str,
    transfer_hypotheses: Sequence[Hypothesis],
    transfer_scores: Sequence[float],
    transfer_posterior: Sequence[float],
    ood_diagnostic_worlds: Sequence[World],
    ood_retention_worlds: Sequence[World],
    return_worlds: Sequence[World],
) -> PolicyResult:
    active_hypotheses = list(transfer_hypotheses)
    active_scores = list(transfer_scores)
    active_priors = list(transfer_posterior)
    archive_hypotheses = list(transfer_hypotheses)
    archive_scores = list(transfer_scores)
    archive_priors = list(transfer_posterior)
    observations: list[tuple[World, int]] = []
    synthesis_spawn_step: int | None = None
    probe_plan: list[str] = []

    for step_index, world in enumerate(ood_diagnostic_worlds[: config.diagnostic_budget]):
        true_action = true_ood_target(world)
        observations.append((world, true_action))
        probe_plan.append(probe_label(world))
        update_scores(active_scores, active_hypotheses, world, true_action, config.consistency)

        if synthesis_spawn_step is None and pair_inconsistency_detected(observations):
            synthesis_spawn_step = step_index + 1
            if policy_name == "keep_all":
                active_hypotheses, active_scores, active_priors = spawn_keep_all_state(
                    config,
                    transfer_hypotheses,
                    transfer_posterior,
                    observations,
                )
            else:
                active_hypotheses, active_scores, active_priors = spawn_synthesis_only_state(
                    config,
                    observations,
                )

    ood_retention_accuracies: list[int] = []
    retention_scores = list(active_scores)
    for world in ood_retention_worlds:
        apply_forgetting(retention_scores, active_priors, config.forgetting_decay)
        probability_action_one = predict_probability_one(retention_scores, active_hypotheses, world)
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        ood_retention_accuracies.append(1 if predicted_action == true_ood_target(world) else 0)

    active_scores = list(retention_scores)
    reactivation_episode: int | None = None
    return_accuracies: list[int] = []

    for episode_index, world in enumerate(return_worlds):
        apply_forgetting(active_scores, active_priors, config.forgetting_decay)
        active_probability_one = predict_probability_one(active_scores, active_hypotheses, world)
        active_prediction = 1 if active_probability_one >= 0.5 else 0
        archive_prediction = None
        if policy_name == "prune_archive":
            archive_probability_one = predict_probability_one(archive_scores, archive_hypotheses, world)
            archive_prediction = 1 if archive_probability_one >= 0.5 else 0

        true_action = true_return_target(world)
        return_accuracies.append(1 if active_prediction == true_action else 0)
        update_scores(active_scores, active_hypotheses, world, true_action, config.consistency)

        if policy_name == "prune_archive":
            update_scores(archive_scores, archive_hypotheses, world, true_action, config.consistency)
            if (
                reactivation_episode is None
                and archive_prediction is not None
                and active_prediction != true_action
                and archive_prediction == true_action
            ):
                active_hypotheses = list(archive_hypotheses)
                active_scores = list(archive_scores)
                active_priors = list(archive_priors)
                reactivation_episode = episode_index

    final_top_name, final_top_mass = top_hypothesis(active_scores, active_hypotheses)
    return PolicyResult(
        policy_name=policy_name,
        synthesis_spawn_step=synthesis_spawn_step,
        reactivation_episode=reactivation_episode,
        active_hypotheses=len(active_hypotheses),
        ood_retention_accuracy=statistics.mean(ood_retention_accuracies),
        return_first_ten_accuracy=statistics.mean(return_accuracies[:10]),
        return_overall_accuracy=statistics.mean(return_accuracies),
        final_top_hypothesis=final_top_name,
        final_top_mass=final_top_mass,
    )


def print_report(
    config: BeliefPruningReactivationConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: selective pruning with belief reactivation")
    print("Old in-distribution rule: xor(0, 5)")
    print("OOD rule: xor3(0, 2, 5)")
    print("After OOD synthesis, the world returns to xor(0, 5)")
    print(f"Forgetting decay: {config.forgetting_decay:.2f}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<18}"
        f"{'Spawn':>8}"
        f"{'React':>8}"
        f"{'ActH':>6}"
        f"{'OODHold':>10}"
        f"{'Ret10':>8}"
        f"{'RetAll':>8}"
        f"{'TopMass':>9}"
        f"{'Top belief':>18}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        spawn_text = "-" if result.synthesis_spawn_step is None else str(result.synthesis_spawn_step)
        react_text = "-" if result.reactivation_episode is None else str(result.reactivation_episode)
        print(
            f"{result.policy_name:<18}"
            f"{spawn_text:>8}"
            f"{react_text:>8}"
            f"{result.active_hypotheses:>6}"
            f"{result.ood_retention_accuracy:>10.3f}"
            f"{result.return_first_ten_accuracy:>8.3f}"
            f"{result.return_overall_accuracy:>8.3f}"
            f"{result.final_top_mass:>9.3f}"
            f"{result.final_top_hypothesis:>18}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)

    transfer_hypotheses, transfer_scores, transfer_posterior = pretrain_transfer_library(config, rng)
    ood_diagnostic_worlds = build_ood_diagnostic_sequence(config.feature_count)
    ood_retention_worlds = sample_worlds(rng, config.feature_count, config.ood_retention_episodes)
    return_worlds = build_return_worlds(rng, config.feature_count, config.return_episodes)

    results = [
        run_policy(
            config,
            policy_name="keep_all",
            transfer_hypotheses=transfer_hypotheses,
            transfer_scores=transfer_scores,
            transfer_posterior=transfer_posterior,
            ood_diagnostic_worlds=ood_diagnostic_worlds,
            ood_retention_worlds=ood_retention_worlds,
            return_worlds=return_worlds,
        ),
        run_policy(
            config,
            policy_name="prune_drop",
            transfer_hypotheses=transfer_hypotheses,
            transfer_scores=transfer_scores,
            transfer_posterior=transfer_posterior,
            ood_diagnostic_worlds=ood_diagnostic_worlds,
            ood_retention_worlds=ood_retention_worlds,
            return_worlds=return_worlds,
        ),
        run_policy(
            config,
            policy_name="prune_archive",
            transfer_hypotheses=transfer_hypotheses,
            transfer_scores=transfer_scores,
            transfer_posterior=transfer_posterior,
            ood_diagnostic_worlds=ood_diagnostic_worlds,
            ood_retention_worlds=ood_retention_worlds,
            return_worlds=return_worlds,
        ),
    ]

    print_report(config, results)


if __name__ == "__main__":
    main()
