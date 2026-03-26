from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


World = tuple[int, ...]


@dataclass(frozen=True)
class BeliefArchiveSelectionConfig:
    feature_count: int = 8
    action_episodes: int = 40
    diagnostic_budget: int = 2
    seed: int = 7
    consistency: float = 0.92
    forgetting_decay: float = 0.93
    prior_xor: float = 0.25
    prior_and: float = 0.20
    prior_xor3: float = 0.55


@dataclass(frozen=True)
class ArchiveHypothesis:
    name: str
    kind: str

    def predict(self, world: World) -> int:
        if self.kind == "xor":
            return world[0] ^ world[5]
        if self.kind == "and":
            return world[0] & world[5]
        if self.kind == "xor3":
            return world[0] ^ world[2] ^ world[5]
        raise ValueError(f"Unknown archive kind: {self.kind}")


@dataclass(frozen=True)
class PolicyResult:
    target_name: str
    policy_name: str
    first_probe: str
    probe_plan: str
    selected_archive: str
    exact_archive: int
    top_mass_after_diagnostics: float
    first_ten_accuracy: float
    overall_accuracy: float


def parse_args() -> BeliefArchiveSelectionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the system can rapidly select the correct archived regime "
            "when multiple past world-models are available."
        )
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--action-episodes", type=int, default=40)
    parser.add_argument("--diagnostic-budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--consistency", type=float, default=0.92)
    parser.add_argument("--forgetting-decay", type=float, default=0.93)
    parser.add_argument("--prior-xor", type=float, default=0.25)
    parser.add_argument("--prior-and", type=float, default=0.20)
    parser.add_argument("--prior-xor3", type=float, default=0.55)
    args = parser.parse_args()

    config = BeliefArchiveSelectionConfig(
        feature_count=args.feature_count,
        action_episodes=args.action_episodes,
        diagnostic_budget=args.diagnostic_budget,
        seed=args.seed,
        consistency=args.consistency,
        forgetting_decay=args.forgetting_decay,
        prior_xor=args.prior_xor,
        prior_and=args.prior_and,
        prior_xor3=args.prior_xor3,
    )
    validate_config(config)
    return config


def validate_config(config: BeliefArchiveSelectionConfig) -> None:
    if config.feature_count != 8:
        raise ValueError("belief_archive_selection_test currently expects feature_count == 8")
    if config.action_episodes <= 0:
        raise ValueError("action_episodes must be > 0")
    if config.diagnostic_budget <= 0:
        raise ValueError("diagnostic_budget must be > 0")
    if not 0.5 < config.consistency < 1.0:
        raise ValueError("consistency must be in (0.5, 1.0)")
    if not 0.0 < config.forgetting_decay <= 1.0:
        raise ValueError("forgetting_decay must be in (0, 1]")
    prior_total = config.prior_xor + config.prior_and + config.prior_xor3
    if not math.isclose(prior_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("archive priors must sum to 1.0")
    if min(config.prior_xor, config.prior_and, config.prior_xor3) <= 0.0:
        raise ValueError("all archive priors must be > 0")


def softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exponentials = [math.exp(score - max_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(probability * math.log2(probability) for probability in probabilities if probability > 0.0)


def build_archives() -> list[ArchiveHypothesis]:
    return [
        ArchiveHypothesis(name="archive_xor", kind="xor"),
        ArchiveHypothesis(name="archive_and", kind="and"),
        ArchiveHypothesis(name="archive_xor3", kind="xor3"),
    ]


def initial_scores(config: BeliefArchiveSelectionConfig) -> list[float]:
    return [
        math.log(config.prior_xor),
        math.log(config.prior_and),
        math.log(config.prior_xor3),
    ]


def sample_worlds(rng: random.Random, feature_count: int, count: int) -> list[World]:
    return [
        tuple(1 if rng.random() < 0.5 else 0 for _ in range(feature_count))
        for _ in range(count)
    ]


def build_probe_bank(feature_count: int) -> list[World]:
    bank: list[World] = []
    for feature_0 in (0, 1):
        for feature_2 in (0, 1):
            for feature_5 in (0, 1):
                values = [0] * feature_count
                values[0] = feature_0
                values[2] = feature_2
                values[5] = feature_5
                bank.append(tuple(values))
    return bank


def probe_label(world: World) -> str:
    return f"{world[0]}{world[2]}{world[5]}"


def target_value(target_name: str, world: World) -> int:
    if target_name == "xor":
        return world[0] ^ world[5]
    if target_name == "and":
        return world[0] & world[5]
    if target_name == "xor3":
        return world[0] ^ world[2] ^ world[5]
    raise ValueError(f"Unknown target: {target_name}")


def update_scores(
    scores: list[float],
    hypotheses: Sequence[ArchiveHypothesis],
    world: World,
    true_action: int,
    consistency: float,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        likelihood = consistency if hypothesis.predict(world) == true_action else (1.0 - consistency)
        scores[index] += math.log(likelihood)


def top_hypothesis(scores: Sequence[float], hypotheses: Sequence[ArchiveHypothesis]) -> tuple[str, float]:
    belief = softmax(scores)
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return hypotheses[top_index].name, belief[top_index]


def select_active_probe(
    scores: Sequence[float],
    hypotheses: Sequence[ArchiveHypothesis],
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
        raise ValueError("No active archive probe available")
    return best_index


def apply_forgetting(scores: list[float], priors: Sequence[float], decay: float) -> None:
    for index, prior in enumerate(priors):
        prior_score = math.log(prior)
        scores[index] = prior_score + decay * (scores[index] - prior_score)


def run_policy(
    config: BeliefArchiveSelectionConfig,
    target_name: str,
    action_worlds: Sequence[World],
    policy_name: str,
) -> PolicyResult:
    archives = build_archives()
    scores = initial_scores(config)
    priors = [config.prior_xor, config.prior_and, config.prior_xor3]
    probe_bank = build_probe_bank(config.feature_count)
    asked_indices: set[int] = set()
    probe_plan: list[str] = []

    if policy_name in {"diagnostic_mix", "diagnostic_router"}:
        for _ in range(config.diagnostic_budget):
            probe_index = select_active_probe(
                scores,
                archives,
                probe_bank,
                asked_indices,
                config.consistency,
            )
            asked_indices.add(probe_index)
            probe = probe_bank[probe_index]
            probe_plan.append(probe_label(probe))
            update_scores(scores, archives, probe, target_value(target_name, probe), config.consistency)

    if policy_name == "latest_archive":
        chosen_hypotheses = [archives[2]]
        chosen_scores = [0.0]
        chosen_priors = [1.0]
        selected_archive = archives[2].name
        top_mass = 1.0
    elif policy_name == "flat_mix":
        chosen_hypotheses = list(archives)
        chosen_scores = initial_scores(config)
        chosen_priors = priors
        selected_archive, top_mass = top_hypothesis(chosen_scores, chosen_hypotheses)
    elif policy_name == "diagnostic_mix":
        chosen_hypotheses = list(archives)
        chosen_scores = list(scores)
        chosen_priors = priors
        selected_archive, top_mass = top_hypothesis(chosen_scores, chosen_hypotheses)
    elif policy_name == "diagnostic_router":
        top_index = max(range(len(scores)), key=scores.__getitem__)
        chosen_hypotheses = [archives[top_index]]
        chosen_scores = [0.0]
        chosen_priors = [1.0]
        selected_archive = archives[top_index].name
        top_mass = 1.0
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    action_scores = list(chosen_scores)
    action_accuracies: list[int] = []
    for world in action_worlds:
        apply_forgetting(action_scores, chosen_priors, config.forgetting_decay)
        belief = softmax(action_scores)
        probability_action_one = sum(
            weight * hypothesis.predict(world)
            for weight, hypothesis in zip(belief, chosen_hypotheses)
        )
        predicted_action = 1 if probability_action_one >= 0.5 else 0
        action_accuracies.append(1 if predicted_action == target_value(target_name, world) else 0)

    first_probe = "-" if not probe_plan else probe_plan[0]
    exact_archive = 1 if selected_archive == f"archive_{target_name}" else 0
    return PolicyResult(
        target_name=target_name,
        policy_name=policy_name,
        first_probe=first_probe,
        probe_plan="-" if not probe_plan else ",".join(probe_plan),
        selected_archive=selected_archive,
        exact_archive=exact_archive,
        top_mass_after_diagnostics=top_mass,
        first_ten_accuracy=statistics.mean(action_accuracies[:10]),
        overall_accuracy=statistics.mean(action_accuracies),
    )


def print_report(
    config: BeliefArchiveSelectionConfig,
    results: Sequence[PolicyResult],
) -> None:
    print("Experiment: selecting the correct archive among multiple past regimes")
    print("Archive A: xor(0, 5)")
    print("Archive B: and(0, 5)")
    print("Archive C: xor3(0, 2, 5)")
    print(
        f"Archive priors: xor={config.prior_xor:.2f}, "
        f"and={config.prior_and:.2f}, xor3={config.prior_xor3:.2f}"
    )
    print(f"Diagnostic budget: {config.diagnostic_budget}")
    print(f"Forgetting decay: {config.forgetting_decay:.2f}")
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Target':<8}"
        f"{'Policy':<20}"
        f"{'Q1':<6}"
        f"{'Plan':<10}"
        f"{'Selected':<16}"
        f"{'Exact':>8}"
        f"{'Mass':>8}"
        f"{'First10':>10}"
        f"{'Overall':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.target_name:<8}"
            f"{result.policy_name:<20}"
            f"{result.first_probe:<6}"
            f"{result.probe_plan:<10}"
            f"{result.selected_archive:<16}"
            f"{result.exact_archive:>8}"
            f"{result.top_mass_after_diagnostics:>8.3f}"
            f"{result.first_ten_accuracy:>10.3f}"
            f"{result.overall_accuracy:>10.3f}"
        )

    grouped: dict[str, list[PolicyResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_name, []).append(result)

    print()
    print("Policy summary:")
    summary_header = (
        f"{'Policy':<20}"
        f"{'MeanExact':>12}"
        f"{'MeanFirst10':>13}"
        f"{'MeanOverall':>13}"
        f"{'Selections':>26}"
    )
    print(summary_header)
    print("-" * len(summary_header))
    for policy_name, policy_results in grouped.items():
        selections = ",".join(result.selected_archive.replace("archive_", "") for result in policy_results)
        print(
            f"{policy_name:<20}"
            f"{statistics.mean(result.exact_archive for result in policy_results):>12.3f}"
            f"{statistics.mean(result.first_ten_accuracy for result in policy_results):>13.3f}"
            f"{statistics.mean(result.overall_accuracy for result in policy_results):>13.3f}"
            f"{selections:>26}"
        )


def main() -> None:
    config = parse_args()
    base_rng = random.Random(config.seed)
    target_worlds = {
        target_name: sample_worlds(base_rng, config.feature_count, config.action_episodes)
        for target_name in ("xor", "and", "xor3")
    }

    results: list[PolicyResult] = []
    for target_name in ("xor", "and", "xor3"):
        for policy_name in ("latest_archive", "flat_mix", "diagnostic_mix", "diagnostic_router"):
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
