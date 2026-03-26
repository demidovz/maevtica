from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefMultiTransferQuestionConfig:
    feature_count: int = 8
    target_episodes: int = 40
    diagnostic_budget: int = 2
    seed: int = 7
    consistency: float = 0.92
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
    first_probe: str
    probe_plan: str
    exact_after_diagnostics: int
    top_after_diagnostics: str
    top_mass_after_diagnostics: float
    first_action_accuracy: float
    first_ten_accuracy: float
    overall_accuracy: float
    final_top_hypothesis: str


def parse_args() -> BeliefMultiTransferQuestionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test which diagnostic question should be asked first when multiple "
            "transferable synthetic beliefs compete under a tight question budget."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--target-episodes", type=int, default=40)
    parser.add_argument("--diagnostic-budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--prior-xor", type=float, default=0.40)
    parser.add_argument("--prior-xnor", type=float, default=0.25)
    parser.add_argument("--prior-and", type=float, default=0.20)
    parser.add_argument("--prior-or", type=float, default=0.15)
    args = parser.parse_args()

    config = BeliefMultiTransferQuestionConfig(
        feature_count=args.feature_count,
        target_episodes=args.target_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
        consistency=args.consistency,
        prior_xor=args.prior_xor,
        prior_xnor=args.prior_xnor,
        prior_and=args.prior_and,
        prior_or=args.prior_or,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefMultiTransferQuestionConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_multi_transfer_question_test currently expects feature_count == 8")
    if config.target_episodes <= 0:
        raise ValueError("target_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
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


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(probability * math.log2(probability) for probability in probabilities if probability > 0.0)


def build_transfer_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(name="xor_0_5", kind="xor", feature_a=0, feature_b=5),
        Hypothesis(name="xnor_0_5", kind="xnor", feature_a=0, feature_b=5),
        Hypothesis(name="and_0_5", kind="and", feature_a=0, feature_b=5),
        Hypothesis(name="or_0_5", kind="or", feature_a=0, feature_b=5),
    ]


def initial_scores(config: BeliefMultiTransferQuestionConfig) -> list[float]:
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


def select_active_probe(
    scores: Sequence[float],
    hypotheses: Sequence[Hypothesis],
    probe_bank: Sequence[World],
    asked_indices: set[int],
    consistency: float,
) -> int:
    current_belief = softmax(scores)
    best_index = -1
    best_expected_entropy = math.inf

    for probe_index, probe in enumerate(probe_bank):
        if probe_index in asked_indices:
            continue

        expected_entropy = 0.0
        for outcome in (0, 1):
            outcome_probability = sum(
                weight
                for weight, hypothesis in zip(current_belief, hypotheses)
                if hypothesis.predict(probe) == outcome
            )
            if outcome_probability == 0.0:
                continue

            projected_scores = list(scores)
            update_scores(projected_scores, hypotheses, probe, outcome, consistency)
            projected_belief = softmax(projected_scores)
            expected_entropy += outcome_probability * entropy(projected_belief)

        if expected_entropy < best_expected_entropy:
            best_expected_entropy = expected_entropy
            best_index = probe_index

    if best_index < 0:
        raise ValueError("No active diagnostic probe available")
    return best_index


def choose_fixed_probe(
    step_index: int,
    probe_bank: Sequence[World],
    asked_indices: set[int],
) -> int:
    fixed_order = ["11", "00", "01", "10"]
    labels_to_indices = {probe_label(probe): index for index, probe in enumerate(probe_bank)}
    for label in fixed_order[step_index:]:
        probe_index = labels_to_indices[label]
        if probe_index not in asked_indices:
            return probe_index
    raise ValueError("No fixed diagnostic probe available")


def choose_random_probe(
    rng: random.Random,
    probe_bank: Sequence[World],
    asked_indices: set[int],
) -> int:
    available = [index for index in range(len(probe_bank)) if index not in asked_indices]
    return rng.choice(available)


def run_policy(
    config: BeliefMultiTransferQuestionConfig,
    target_name: str,
    action_worlds: Sequence[World],
    policy_name: str,
    rng: random.Random,
) -> PolicyResult:
    hypotheses = build_transfer_hypotheses()
    scores = initial_scores(config)
    probe_bank = build_probe_bank(config.feature_count)
    asked_indices: set[int] = set()
    probe_plan: list[str] = []

    diagnostic_questions = config.diagnostic_budget if policy_name != "no_diagnostics" else 0
    for step_index in range(diagnostic_questions):
        if policy_name == "fixed_schema_checks":
            probe_index = choose_fixed_probe(step_index, probe_bank, asked_indices)
        elif policy_name == "random_diagnostics":
            probe_index = choose_random_probe(rng, probe_bank, asked_indices)
        elif policy_name == "active_diagnostics":
            probe_index = select_active_probe(
                scores,
                hypotheses,
                probe_bank,
                asked_indices,
                config.consistency,
            )
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        asked_indices.add(probe_index)
        probe = probe_bank[probe_index]
        probe_plan.append(probe_label(probe))
        true_action = target_value(target_name, probe)
        update_scores(scores, hypotheses, probe, true_action, config.consistency)

    top_after_diagnostics, top_mass_after_diagnostics = top_hypothesis(scores, hypotheses)
    exact_after_diagnostics = 1 if top_after_diagnostics == f"{target_name}_0_5" else 0

    action_accuracies: list[int] = []
    for world in action_worlds:
        belief = softmax(scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        true_action = target_value(target_name, world)
        is_correct = 1 if predicted_action == true_action else 0
        action_accuracies.append(is_correct)
        update_scores(scores, hypotheses, world, true_action, config.consistency)

    final_top_hypothesis, _ = top_hypothesis(scores, hypotheses)
    first_probe = "-" if not probe_plan else probe_plan[0]
    probe_plan_text = "-" if not probe_plan else ",".join(probe_plan)

    return PolicyResult(
        target_name=target_name,
        policy_name=policy_name,
        first_probe=first_probe,
        probe_plan=probe_plan_text,
        exact_after_diagnostics=exact_after_diagnostics,
        top_after_diagnostics=top_after_diagnostics,
        top_mass_after_diagnostics=top_mass_after_diagnostics,
        first_action_accuracy=float(action_accuracies[0]),
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
        final_top_hypothesis=final_top_hypothesis,
    )


def print_report(
    config: BeliefMultiTransferQuestionConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: first diagnostic question under competing transfer schemas")
    print("Transfer library priors:")
    print(
        f"xor={config.prior_xor:.2f}, "
        f"xnor={config.prior_xnor:.2f}, "
        f"and={config.prior_and:.2f}, "
        f"or={config.prior_or:.2f}"
    )
    print("Probe bank on features (0, 5): 01, 11, 00, 10")
    print(f"Diagnostic budget: {config.diagnostic_budget}")
    print(f"Seed: {config.seed}")
    print()

    print("Per-target results:")
    header = (
        f"{'Target':<8}"
        f"{'Policy':<22}"
        f"{'Q1':<6}"
        f"{'Plan':<10}"
        f"{'Exact':>8}"
        f"{'TopAfter':>14}"
        f"{'Mass':>8}"
        f"{'First1':>8}"
        f"{'First10':>10}"
        f"{'Overall':>10}"
        f"{'FinalTop':>14}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.target_name:<8}"
            f"{result.policy_name:<22}"
            f"{result.first_probe:<6}"
            f"{result.probe_plan:<10}"
            f"{result.exact_after_diagnostics:>8}"
            f"{result.top_after_diagnostics:>14}"
            f"{result.top_mass_after_diagnostics:>8.3f}"
            f"{result.first_action_accuracy:>8.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
            f"{result.final_top_hypothesis:>14}"
        )

    grouped: dict[str, list[PolicyResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_name, []).append(result)

    print()
    print("Policy summary:")
    summary_header = (
        f"{'Policy':<22}"
        f"{'MeanExact':>12}"
        f"{'MeanFirst1':>12}"
        f"{'MeanFirst10':>13}"
        f"{'MeanOverall':>13}"
        f"{'Q1 choices':>18}"
    )
    print(summary_header)
    print("-" * len(summary_header))
    for policy_name, policy_results in grouped.items():
        q1_choices = ",".join(result.first_probe for result in policy_results)
        print(
            f"{policy_name:<22}"
            f"{statistics.mean(result.exact_after_diagnostics for result in policy_results):>12.3f}"
            f"{statistics.mean(result.first_action_accuracy for result in policy_results):>12.3f}"
            f"{statistics.mean(result.first_ten_accuracy for result in policy_results):>13.3f}"
            f"{statistics.mean(result.overall_accuracy for result in policy_results):>13.3f}"
            f"{q1_choices:>18}"
        )


def main() -> None:
    config = parse_args()
    base_rng = random.Random(config.seed)
    target_worlds = {
        target_name: sample_worlds(base_rng, config.feature_count, config.target_episodes)
        for target_name in ("xor", "xnor", "and", "or")
    }

    results: list[PolicyResult] = []
    policies = [
        "no_diagnostics",
        "fixed_schema_checks",
        "random_diagnostics",
        "active_diagnostics",
    ]
    for target_index, target_name in enumerate(("xor", "xnor", "and", "or")):
        for policy_index, policy_name in enumerate(policies):
            policy_rng = random.Random(config.seed + 100 * target_index + policy_index)
            results.append(
                run_policy(
                    config,
                    target_name=target_name,
                    action_worlds=target_worlds[target_name],
                    policy_name=policy_name,
                    rng=policy_rng,
                )
            )

    print_report(config, results)


if __name__ == "__main__":
    main()
