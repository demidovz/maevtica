from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefOODSynthesisConfig:
    feature_count: int = 8
    action_episodes: int = 64
    diagnostic_budget: int = 4
    seed: int = 7
    consistency: float = 0.92
    question_cost: float = 0.09
    synthesis_trigger_accuracy: float = 0.65
    min_misfit_observations: int = 2
    transfer_prior_xor: float = 0.40
    transfer_prior_xnor: float = 0.25
    transfer_prior_and: float = 0.20
    transfer_prior_or: float = 0.15
    synthesis_prior: float = 0.05


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
    questions_asked: int
    probe_plan: str
    stop_reason: str
    synthesis_spawn_step: int | None
    table_accuracy: float
    net_value: float
    first_ten_accuracy: float
    overall_accuracy: float
    final_top_hypothesis: str


def parse_args() -> BeliefOODSynthesisConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the system switches from transfer to synthesis when the "
            "correct rule lies outside the current transfer library."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--action-episodes", type=int, default=64)
    parser.add_argument("--diagnostic-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--question-cost", type=float, default=0.09)
    parser.add_argument("--synthesis-trigger-accuracy", type=float, default=0.65)
    parser.add_argument("--min-misfit-observations", type=int, default=2)
    parser.add_argument("--transfer-prior-xor", type=float, default=0.40)
    parser.add_argument("--transfer-prior-xnor", type=float, default=0.25)
    parser.add_argument("--transfer-prior-and", type=float, default=0.20)
    parser.add_argument("--transfer-prior-or", type=float, default=0.15)
    parser.add_argument("--synthesis-prior", type=float, default=0.05)
    args = parser.parse_args()

    config = BeliefOODSynthesisConfig(
        feature_count=args.feature_count,
        action_episodes=args.action_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
        consistency=args.consistency,
        question_cost=args.question_cost,
        synthesis_trigger_accuracy=args.synthesis_trigger_accuracy,
        min_misfit_observations=args.min_misfit_observations,
        transfer_prior_xor=args.transfer_prior_xor,
        transfer_prior_xnor=args.transfer_prior_xnor,
        transfer_prior_and=args.transfer_prior_and,
        transfer_prior_or=args.transfer_prior_or,
        synthesis_prior=args.synthesis_prior,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefOODSynthesisConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_ood_synthesis_test currently expects feature_count == 8")
    if config.action_episodes <= 0:
        raise ValueError("action_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.question_cost <= 0.0:
        raise ValueError("question_cost must be > 0")
    if not 0.0 < config.synthesis_trigger_accuracy < 1.0:
        raise ValueError("synthesis_trigger_accuracy must be in (0, 1)")
    if config.min_misfit_observations <= 0:
        raise ValueError("min_misfit_observations must be > 0")
    transfer_total = (
        config.transfer_prior_xor
        + config.transfer_prior_xnor
        + config.transfer_prior_and
        + config.transfer_prior_or
    )
    if not math.isclose(transfer_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("transfer priors must sum to 1.0")
    if min(
        config.transfer_prior_xor,
        config.transfer_prior_xnor,
        config.transfer_prior_and,
        config.transfer_prior_or,
        config.synthesis_prior,
    ) <= 0.0:
        raise ValueError("all priors must be > 0")


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


def initial_transfer_scores(config: BeliefOODSynthesisConfig) -> list[float]:
    return [
        math.log(config.transfer_prior_xor),
        math.log(config.transfer_prior_xnor),
        math.log(config.transfer_prior_and),
        math.log(config.transfer_prior_or),
    ]


def build_full_truth_table(feature_count: int) -> list[World]:
    worlds: list[World] = []
    for feature_0 in (0, 1):
        for feature_2 in (0, 1):
            for feature_5 in (0, 1):
                values = [0] * feature_count
                values[0] = feature_0
                values[2] = feature_2
                values[5] = feature_5
                worlds.append(tuple(values))
    return worlds


def build_diagnostic_sequence(feature_count: int) -> list[World]:
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


def sample_action_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


def true_target(world: World) -> int:
    return world[0] ^ world[2] ^ world[5]


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
    config: BeliefOODSynthesisConfig,
    hypotheses: Sequence[Hypothesis],
    observations: Sequence[tuple[World, int]],
    priors: Sequence[float],
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


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[Hypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def predicted_table_accuracy(
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    eval_worlds: Sequence[World],
) -> float:
    belief = softmax(scores)
    accuracies: list[float] = []
    for world in eval_worlds:
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        accuracies.append(max(probability_action_one, 1.0 - probability_action_one))
    return statistics.mean(accuracies)


def true_table_accuracy(
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    eval_worlds: Sequence[World],
) -> float:
    belief = softmax(scores)
    accuracies: list[int] = []
    for world in eval_worlds:
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        accuracies.append(1 if predicted_action == true_target(world) else 0)
    return statistics.mean(accuracies)


def expected_next_gain(
    config: BeliefOODSynthesisConfig,
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    next_world: World,
    eval_worlds: Sequence[World],
) -> float:
    current_accuracy = predicted_table_accuracy(scores, hypotheses, eval_worlds)
    belief = softmax(scores)
    expected_future_accuracy = 0.0

    for outcome in (0, 1):
        outcome_probability = sum(
            weight
            for weight, hypothesis in zip(belief, hypotheses)
            if hypothesis.predict(next_world) == outcome
        )
        if outcome_probability == 0.0:
            continue

        projected_scores = list(scores)
        update_scores(projected_scores, hypotheses, next_world, outcome, config.consistency)
        expected_future_accuracy += outcome_probability * predicted_table_accuracy(
            projected_scores,
            hypotheses,
            eval_worlds,
        )

    return expected_future_accuracy - current_accuracy


def run_policy(
    config: BeliefOODSynthesisConfig,
    policy_name: str,
    diagnostic_worlds: Sequence[World],
    eval_worlds: Sequence[World],
    action_worlds: Sequence[World],
) -> PolicyResult:
    transfer_hypotheses = build_transfer_hypotheses()
    synthesis_hypotheses = build_synthesis_hypotheses()

    active_hypotheses = list(transfer_hypotheses)
    active_priors = [
        config.transfer_prior_xor,
        config.transfer_prior_xnor,
        config.transfer_prior_and,
        config.transfer_prior_or,
    ]
    scores = [math.log(prior) for prior in active_priors]
    observations: list[tuple[World, int]] = []
    probe_plan: list[str] = []
    synthesis_spawn_step: int | None = None
    stop_reason = "budget_exhausted"
    synthesis_active = False

    for step_index, world in enumerate(diagnostic_worlds[: config.diagnostic_budget]):
        true_action = true_target(world)
        observations.append((world, true_action))
        probe_plan.append(probe_label(world))
        update_scores(scores, active_hypotheses, world, true_action, config.consistency)

        if (
            policy_name == "synthesis_on_misfit"
            and not synthesis_active
            and len(observations) >= config.min_misfit_observations
        ):
            transfer_accuracy = predicted_table_accuracy(scores, active_hypotheses, eval_worlds)
            has_next_transfer_probe = step_index + 1 < min(config.diagnostic_budget, len(diagnostic_worlds))
            next_transfer_gain = (
                expected_next_gain(
                    config,
                    scores,
                    active_hypotheses,
                    diagnostic_worlds[step_index + 1],
                    eval_worlds,
                )
                if has_next_transfer_probe
                else 0.0
            )
            if (
                pair_inconsistency_detected(observations)
                or (transfer_accuracy < config.synthesis_trigger_accuracy and next_transfer_gain <= config.question_cost)
            ):
                synthesis_active = True
                synthesis_spawn_step = step_index + 1
                active_hypotheses = list(transfer_hypotheses) + list(synthesis_hypotheses)
                active_priors = [
                    config.transfer_prior_xor,
                    config.transfer_prior_xnor,
                    config.transfer_prior_and,
                    config.transfer_prior_or,
                    config.synthesis_prior,
                    config.synthesis_prior,
                    config.synthesis_prior,
                    config.synthesis_prior,
                ]
                scores = replay_observations(config, active_hypotheses, observations, active_priors)

        if policy_name == "transfer_fixed_budget":
            continue

        has_next_probe = step_index + 1 < min(config.diagnostic_budget, len(diagnostic_worlds))
        if not has_next_probe:
            stop_reason = "budget_exhausted"
            break

        if (
            policy_name == "synthesis_on_misfit"
            and not synthesis_active
            and len(observations) < config.min_misfit_observations
        ):
            continue

        if (
            policy_name == "synthesis_on_misfit"
            and synthesis_active
            and synthesis_spawn_step is not None
            and len(observations) == synthesis_spawn_step
        ):
            continue

        next_gain = expected_next_gain(
            config,
            scores,
            active_hypotheses,
            diagnostic_worlds[step_index + 1],
            eval_worlds,
        )
        if next_gain <= config.question_cost:
            stop_reason = "gain_below_cost"
            break

    table_accuracy = true_table_accuracy(scores, active_hypotheses, eval_worlds)
    net_value = table_accuracy - config.question_cost * len(probe_plan)

    action_accuracies: list[int] = []
    for world in action_worlds:
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, active_hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        action_accuracies.append(1 if predicted_action == true_target(world) else 0)

    final_top_hypothesis, _ = top_hypothesis(scores, active_hypotheses)
    return PolicyResult(
        policy_name=policy_name,
        questions_asked=len(probe_plan),
        probe_plan=",".join(probe_plan),
        stop_reason=stop_reason,
        synthesis_spawn_step=synthesis_spawn_step,
        table_accuracy=table_accuracy,
        net_value=net_value,
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
        final_top_hypothesis=final_top_hypothesis,
    )


def print_report(
    config: BeliefOODSynthesisConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: out-of-distribution conflict and belief synthesis")
    print("Transfer library: xor/xnor/and/or on features (0, 5)")
    print("OOD synthesis bank: xor3/nand3/gate_or_and/maj3 on features (0, 2, 5)")
    print("True target: xor3(0, 2, 5)")
    print("Diagnostic sequence on features (0, 2, 5): 001, 011, 101, 111")
    print(f"Question cost: {config.question_cost:.2f}")
    print(f"Synthesis trigger accuracy: {config.synthesis_trigger_accuracy:.2f}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<22}"
        f"{'Q':>4}"
        f"{'Plan':<16}"
        f"{'Stop':<18}"
        f"{'Spawn':>8}"
        f"{'TableAcc':>10}"
        f"{'NetV':>8}"
        f"{'First10':>10}"
        f"{'Overall':>10}"
        f"{'Top belief':>18}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        spawn_text = "-" if result.synthesis_spawn_step is None else str(result.synthesis_spawn_step)
        print(
            f"{result.policy_name:<22}"
            f"{result.questions_asked:>4}"
            f"{result.probe_plan:<16}"
            f"{result.stop_reason:<18}"
            f"{spawn_text:>8}"
            f"{result.table_accuracy:>10.3f}"
            f"{result.net_value:>8.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
            f"{result.final_top_hypothesis:>18}"
        )


def main() -> None:
    config = parse_args()
    rng = random.Random(config.seed)
    diagnostic_worlds = build_diagnostic_sequence(config.feature_count)
    eval_worlds = build_full_truth_table(config.feature_count)
    action_worlds = sample_action_worlds(rng, config.feature_count, config.action_episodes)

    results = [
        run_policy(
            config,
            policy_name="transfer_fixed_budget",
            diagnostic_worlds=diagnostic_worlds,
            eval_worlds=eval_worlds,
            action_worlds=action_worlds,
        ),
        run_policy(
            config,
            policy_name="transfer_value_stop",
            diagnostic_worlds=diagnostic_worlds,
            eval_worlds=eval_worlds,
            action_worlds=action_worlds,
        ),
        run_policy(
            config,
            policy_name="synthesis_on_misfit",
            diagnostic_worlds=diagnostic_worlds,
            eval_worlds=eval_worlds,
            action_worlds=action_worlds,
        ),
    ]

    print_report(config, results)


if __name__ == "__main__":
    main()
