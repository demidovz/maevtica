from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefMultiTransferRiskConfig:
    feature_count: int = 8
    target_episodes: int = 40
    diagnostic_budget: int = 3
    seed: int = 7
    consistency: float = 0.92
    question_cost: float = 0.09
    top_mass_threshold: float = 0.55
    default_error_cost: float = 1.0
    rare_error_cost: float = 2.0
    rare_target: str = "or"
    prior_xor: float = 0.40
    prior_xnor: float = 0.25
    prior_and: float = 0.20
    prior_or: float = 0.15


@dataclass(frozen=True)
class Hypothesis:
    name: str
    kind: str
    feature_a: int
    feature_b: int

    def predict(self, world: World) -> int:
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
class PolicyResult:
    target_name: str
    policy_name: str
    questions_asked: int
    probe_plan: str
    stop_reason: str
    top_after_diagnostics: str
    top_mass_after_diagnostics: float
    predicted_utility: float
    true_utility: float
    first_ten_accuracy: float
    overall_accuracy: float
    final_top_hypothesis: str


def parse_args() -> BeliefMultiTransferRiskConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether stopping should change when some transfer mistakes are rarer "
            "but more expensive than others."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--target-episodes", type=int, default=40)
    parser.add_argument("--diagnostic-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--question-cost", type=float, default=0.09)
    parser.add_argument("--top-mass-threshold", type=float, default=0.55)
    parser.add_argument("--default-error-cost", type=float, default=1.0)
    parser.add_argument("--rare-error-cost", type=float, default=2.0)
    parser.add_argument("--rare-target", type=str, default="or")
    parser.add_argument("--prior-xor", type=float, default=0.40)
    parser.add_argument("--prior-xnor", type=float, default=0.25)
    parser.add_argument("--prior-and", type=float, default=0.20)
    parser.add_argument("--prior-or", type=float, default=0.15)
    args = parser.parse_args()

    config = BeliefMultiTransferRiskConfig(
        feature_count=args.feature_count,
        target_episodes=args.target_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
        consistency=args.consistency,
        question_cost=args.question_cost,
        top_mass_threshold=args.top_mass_threshold,
        default_error_cost=args.default_error_cost,
        rare_error_cost=args.rare_error_cost,
        rare_target=args.rare_target,
        prior_xor=args.prior_xor,
        prior_xnor=args.prior_xnor,
        prior_and=args.prior_and,
        prior_or=args.prior_or,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefMultiTransferRiskConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_multi_transfer_risk_test currently expects feature_count == 8")
    if config.target_episodes <= 0:
        raise ValueError("target_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if config.question_cost <= 0.0:
        raise ValueError("question_cost must be > 0")
    if not 0.0 < config.top_mass_threshold < 1.0:
        raise ValueError("top_mass_threshold must be in (0, 1)")
    if config.default_error_cost <= 0.0 or config.rare_error_cost <= 0.0:
        raise ValueError("error costs must be > 0")
    if config.rare_target not in {"xor", "xnor", "and", "or"}:
        raise ValueError("rare_target must be one of xor/xnor/and/or")
    prior_total = config.prior_xor + config.prior_xnor + config.prior_and + config.prior_or
    if not math.isclose(prior_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("transfer priors must sum to 1.0")
    if min(config.prior_xor, config.prior_xnor, config.prior_and, config.prior_or) <= 0.0:
        raise ValueError("all transfer priors must be > 0")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def build_transfer_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(name="xor_0_5", kind="xor", feature_a=0, feature_b=5),
        Hypothesis(name="xnor_0_5", kind="xnor", feature_a=0, feature_b=5),
        Hypothesis(name="and_0_5", kind="and", feature_a=0, feature_b=5),
        Hypothesis(name="or_0_5", kind="or", feature_a=0, feature_b=5),
    ]


def initial_scores(config: BeliefMultiTransferRiskConfig) -> list[float]:
    return [
        math.log(config.prior_xor),
        math.log(config.prior_xnor),
        math.log(config.prior_and),
        math.log(config.prior_or),
    ]


def sample_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


def build_probe_bank(feature_count: int) -> list[World]:
    bank: list[World] = []
    for left, right in ((0, 1), (1, 1), (0, 0), (1, 0)):
        values = [0] * feature_count
        values[0] = left
        values[5] = right
        bank.append(tuple(values))
    return bank


def probe_label(world: World) -> str:
    return f"{world[0]}{world[5]}"


def target_value(operator: str, world: World) -> int:
    if operator == "xor":
        return world[0] ^ world[5]
    if operator == "xnor":
        return 1 - (world[0] ^ world[5])
    if operator == "and":
        return world[0] & world[5]
    if operator == "or":
        return world[0] | world[5]
    raise ValueError(f"Unknown operator: {operator}")


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


def error_cost(config: BeliefMultiTransferRiskConfig, target_name: str) -> float:
    return config.rare_error_cost if target_name == config.rare_target else config.default_error_cost


def predicted_utility(
    config: BeliefMultiTransferRiskConfig,
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    target_name: str,
    eval_worlds: Sequence[World],
) -> float:
    belief = softmax(scores)
    mistake_cost = error_cost(config, target_name)
    utilities: list[float] = []
    for world in eval_worlds:
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        success_probability = max(probability_action_one, 1.0 - probability_action_one)
        expected_error = 1.0 - success_probability
        utilities.append(1.0 - mistake_cost * expected_error)
    return statistics.mean(utilities)


def true_utility(
    config: BeliefMultiTransferRiskConfig,
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    target_name: str,
    eval_worlds: Sequence[World],
) -> float:
    belief = softmax(scores)
    mistake_cost = error_cost(config, target_name)
    values: list[float] = []
    for world in eval_worlds:
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        is_correct = predicted_action == target_value(target_name, world)
        values.append(1.0 if is_correct else 1.0 - mistake_cost)
    return statistics.mean(values)


def best_probe_future_utility(
    config: BeliefMultiTransferRiskConfig,
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    target_name: str,
    probe_bank: Sequence[World],
    asked_indices: set[int],
    consistency: float,
    eval_worlds: Sequence[World],
) -> tuple[int, float]:
    belief = softmax(scores)
    best_index = -1
    best_value = -math.inf

    for probe_index, probe in enumerate(probe_bank):
        if probe_index in asked_indices:
            continue

        expected_future_utility = 0.0
        for outcome in (0, 1):
            outcome_probability = sum(
                weight
                for weight, hypothesis in zip(belief, hypotheses)
                if hypothesis.predict(probe) == outcome
            )
            if outcome_probability == 0.0:
                continue

            projected_scores = list(scores)
            update_scores(projected_scores, hypotheses, probe, outcome, consistency)
            expected_future_utility += outcome_probability * predicted_utility(
                config,
                projected_scores,
                hypotheses,
                target_name,
                eval_worlds,
            )

        if expected_future_utility > best_value:
            best_value = expected_future_utility
            best_index = probe_index

    if best_index < 0:
        raise ValueError("No diagnostic probe available")
    return best_index, best_value


def run_policy(
    config: BeliefMultiTransferRiskConfig,
    target_name: str,
    action_worlds: Sequence[World],
    policy_name: str,
) -> PolicyResult:
    hypotheses = build_transfer_hypotheses()
    scores = initial_scores(config)
    probe_bank = build_probe_bank(config.feature_count)
    asked_indices: set[int] = set()
    probe_plan: list[str] = []
    stop_reason = "budget_exhausted"
    eval_worlds = probe_bank

    for _ in range(config.diagnostic_budget):
        probe_index, _ = best_probe_future_utility(
            config,
            scores,
            hypotheses,
            target_name,
            probe_bank,
            asked_indices,
            config.consistency,
            eval_worlds,
        )
        probe = probe_bank[probe_index]
        asked_indices.add(probe_index)
        probe_plan.append(probe_label(probe))
        true_action = target_value(target_name, probe)
        update_scores(scores, hypotheses, probe, true_action, config.consistency)

        if policy_name == "fixed_budget":
            continue

        if policy_name == "topmass_stop":
            _, top_mass = top_hypothesis(scores, hypotheses)
            if top_mass >= config.top_mass_threshold:
                stop_reason = "top_mass"
                break
            continue

        if policy_name == "value_stop":
            current_value = predicted_utility(config, scores, hypotheses, target_name, eval_worlds)
            remaining_probes = len(probe_bank) - len(asked_indices)
            if remaining_probes == 0:
                stop_reason = "probe_bank_exhausted"
                break
            _, next_future_value = best_probe_future_utility(
                config,
                scores,
                hypotheses,
                target_name,
                probe_bank,
                asked_indices,
                config.consistency,
                eval_worlds,
            )
            expected_gain = next_future_value - current_value
            if expected_gain <= config.question_cost:
                stop_reason = "utility_below_cost"
                break
            continue

        raise ValueError(f"Unknown policy: {policy_name}")

    top_after_diagnostics, top_mass_after_diagnostics = top_hypothesis(scores, hypotheses)
    predicted_value = predicted_utility(config, scores, hypotheses, target_name, eval_worlds)
    realized_value = true_utility(config, scores, hypotheses, target_name, eval_worlds)
    net_true_value = realized_value - config.question_cost * len(probe_plan)

    action_accuracies: list[int] = []
    for world in action_worlds:
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        true_action = target_value(target_name, world)
        action_accuracies.append(1 if predicted_action == true_action else 0)
        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_hypothesis, _ = top_hypothesis(scores, hypotheses)
    return PolicyResult(
        target_name=target_name,
        policy_name=policy_name,
        questions_asked=len(probe_plan),
        probe_plan=",".join(probe_plan),
        stop_reason=stop_reason,
        top_after_diagnostics=top_after_diagnostics,
        top_mass_after_diagnostics=top_mass_after_diagnostics,
        predicted_utility=predicted_value,
        true_utility=net_true_value,
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
        final_top_hypothesis=final_top_hypothesis,
    )


def print_report(
    config: BeliefMultiTransferRiskConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: risk-sensitive stopping under competing transfer schemas")
    print("Transfer library priors:")
    print(
        f"xor={config.prior_xor:.2f}, "
        f"xnor={config.prior_xnor:.2f}, "
        f"and={config.prior_and:.2f}, "
        f"or={config.prior_or:.2f}"
    )
    print(f"Question cost: {config.question_cost:.2f}")
    print(
        f"Error cost: default={config.default_error_cost:.2f}, "
        f"{config.rare_target}={config.rare_error_cost:.2f}"
    )
    print(f"Top-mass threshold: {config.top_mass_threshold:.2f}")
    print(f"Diagnostic budget: {config.diagnostic_budget}")
    print(f"Seed: {config.seed}")
    print()

    print("Per-target results:")
    header = (
        f"{'Target':<8}"
        f"{'Policy':<16}"
        f"{'Q':>4}"
        f"{'Plan':<10}"
        f"{'Stop':<18}"
        f"{'TopAfter':>12}"
        f"{'Mass':>8}"
        f"{'PredU':>8}"
        f"{'NetU':>8}"
        f"{'First10':>10}"
        f"{'Overall':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.target_name:<8}"
            f"{result.policy_name:<16}"
            f"{result.questions_asked:>4}"
            f"{result.probe_plan:<10}"
            f"{result.stop_reason:<18}"
            f"{result.top_after_diagnostics:>12}"
            f"{result.top_mass_after_diagnostics:>8.3f}"
            f"{result.predicted_utility:>8.3f}"
            f"{result.true_utility:>8.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
        )

    grouped: dict[str, list[PolicyResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_name, []).append(result)

    print()
    print("Policy summary:")
    header = (
        f"{'Policy':<16}"
        f"{'MeanQ':>8}"
        f"{'MeanNetU':>11}"
        f"{'MeanFirst10':>13}"
        f"{'MeanOverall':>13}"
        f"{'RarePlan':>14}"
    )
    print(header)
    print("-" * len(header))
    for policy_name, policy_results in grouped.items():
        rare_plan = next(result.probe_plan for result in policy_results if result.target_name == config.rare_target)
        print(
            f"{policy_name:<16}"
            f"{statistics.mean(result.questions_asked for result in policy_results):>8.3f}"
            f"{statistics.mean(result.true_utility for result in policy_results):>11.3f}"
            f"{statistics.mean(result.first_ten_accuracy for result in policy_results):>13.3f}"
            f"{statistics.mean(result.overall_accuracy for result in policy_results):>13.3f}"
            f"{rare_plan:>14}"
        )


def main() -> None:
    config = parse_args()
    base_rng = random.Random(config.seed)
    target_worlds = {
        target_name: sample_worlds(base_rng, config.feature_count, config.target_episodes)
        for target_name in ("xor", "xnor", "and", "or")
    }

    results: list[PolicyResult] = []
    for target_name in ("xor", "xnor", "and", "or"):
        for policy_name in ("fixed_budget", "topmass_stop", "value_stop"):
            results.append(
                run_policy(
                    config,
                    target_name=target_name,
                    action_worlds=target_worlds[target_name],
                    policy_name=policy_name,
                )
            )

    print_report(config, results)


if __name__ == "__main__":
    main()
