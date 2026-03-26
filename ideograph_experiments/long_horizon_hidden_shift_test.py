from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque


MODE_TABLES = {
    "color": {"00": 0, "01": 0, "10": 1, "11": 1},
    "shape": {"00": 0, "01": 1, "10": 0, "11": 1},
    "xor": {"00": 0, "01": 1, "10": 1, "11": 0},
}
MODE_CONTEXT_PROBS = {
    "color": {"00": 0.55, "01": 0.15, "10": 0.25, "11": 0.05},
    "shape": {"00": 0.15, "01": 0.55, "10": 0.05, "11": 0.25},
    "xor": {"00": 0.15, "01": 0.05, "10": 0.50, "11": 0.30},
}
MODE_NAMES = tuple(MODE_TABLES.keys())
CONTEXTS = tuple(MODE_TABLES["color"].keys())


@dataclass(frozen=True)
class HiddenShiftConfig:
    seed: int = 7
    block_length: int = 16
    probe_cost: float = 0.18
    high_risk_cost: float = 4.0
    low_risk_cost: float = 1.0
    random_high_risk_prob: float = 0.10
    window_size: int = 2
    consistency: float = 0.93
    uncertainty_threshold: float = 1.10


@dataclass(frozen=True)
class Episode:
    episode_index: int
    block_index: int
    block_offset: int
    mode_name: str
    context: str
    true_label: int
    risk_cost: float
    return_block: bool


@dataclass(frozen=True)
class PolicyMetrics:
    policy_name: str
    mean_utility: float
    accuracy: float
    catastrophic_error_rate: float
    probe_rate: float
    return_first4_accuracy: float
    avg_shift_lag: str
    fast_return_rate: str


def parse_args() -> HiddenShiftConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether an integrated agent can detect hidden regime shifts without "
            "an explicit drift hint, using only uncertainty, prediction errors, and risk."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--probe-cost", type=float, default=0.18)
    parser.add_argument("--high-risk-cost", type=float, default=4.0)
    parser.add_argument("--low-risk-cost", type=float, default=1.0)
    parser.add_argument("--random-high-risk-prob", type=float, default=0.10)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--consistency", type=float, default=0.93)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.10)
    args = parser.parse_args()
    return HiddenShiftConfig(
        seed=args.seed,
        block_length=args.block_length,
        probe_cost=args.probe_cost,
        high_risk_cost=args.high_risk_cost,
        low_risk_cost=args.low_risk_cost,
        random_high_risk_prob=args.random_high_risk_prob,
        window_size=args.window_size,
        consistency=args.consistency,
        uncertainty_threshold=args.uncertainty_threshold,
    )


def weighted_choice(rng: random.Random, probabilities: dict[str, float]) -> str:
    target = rng.random()
    cumulative = 0.0
    for key, probability in probabilities.items():
        cumulative += probability
        if target <= cumulative:
            return key
    return next(reversed(probabilities))


def build_episodes(
    rng: random.Random,
    config: HiddenShiftConfig,
) -> list[Episode]:
    schedule = ("color", "shape", "color", "xor", "shape", "xor")
    seen_modes: set[str] = set()
    episodes: list[Episode] = []

    for block_index, mode_name in enumerate(schedule):
        return_block = mode_name in seen_modes
        for block_offset in range(config.block_length):
            context = weighted_choice(rng, MODE_CONTEXT_PROBS[mode_name])
            risk_cost = (
                config.high_risk_cost
                if block_offset == 0 or rng.random() < config.random_high_risk_prob
                else config.low_risk_cost
            )
            episodes.append(
                Episode(
                    episode_index=len(episodes),
                    block_index=block_index,
                    block_offset=block_offset,
                    mode_name=mode_name,
                    context=context,
                    true_label=MODE_TABLES[mode_name][context],
                    risk_cost=risk_cost,
                    return_block=return_block,
                )
            )
        seen_modes.add(mode_name)
    return episodes


def softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exponentials = {
        mode_name: math.exp(score - max_score)
        for mode_name, score in scores.items()
    }
    total = sum(exponentials.values())
    return {
        mode_name: value / total
        for mode_name, value in exponentials.items()
    }


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )


def posterior_from_window(
    history: Deque[tuple[str, int]],
    current_context: str,
    config: HiddenShiftConfig,
) -> dict[str, float]:
    log_prior = -math.log(len(MODE_NAMES))
    scores = {mode_name: log_prior for mode_name in MODE_NAMES}
    for mode_name in MODE_NAMES:
        scores[mode_name] += math.log(MODE_CONTEXT_PROBS[mode_name][current_context])
    for context, true_label in history:
        for mode_name in MODE_NAMES:
            scores[mode_name] += math.log(MODE_CONTEXT_PROBS[mode_name][context])
            predicted = MODE_TABLES[mode_name][context]
            likelihood = config.consistency if predicted == true_label else (1.0 - config.consistency)
            scores[mode_name] += math.log(likelihood)
    return softmax(scores)


def top_mode(probabilities: dict[str, float]) -> tuple[str, float]:
    mode_name = max(probabilities, key=probabilities.get)
    return mode_name, probabilities[mode_name]


def expected_error_cost(
    probabilities: dict[str, float],
    context: str,
    risk_cost: float,
) -> float:
    chosen_mode, _ = top_mode(probabilities)
    chosen_label = MODE_TABLES[chosen_mode][context]
    wrong_probability = sum(
        probability
        for mode_name, probability in probabilities.items()
        if MODE_TABLES[mode_name][context] != chosen_label
    )
    return wrong_probability * risk_cost


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def compute_metrics(
    policy_name: str,
    utilities: list[float],
    correct_flags: list[int],
    catastrophic_flags: list[int],
    high_risk_total: int,
    probe_flags: list[int],
    return_first4_correct: list[int],
    shift_lags: list[int],
) -> PolicyMetrics:
    catastrophic_error_rate = 0.0 if high_risk_total == 0 else sum(catastrophic_flags) / high_risk_total
    fast_return_rate = None if not shift_lags else statistics.mean(1 if lag <= 1 else 0 for lag in shift_lags)
    avg_shift_lag = None if not shift_lags else statistics.mean(shift_lags)
    return PolicyMetrics(
        policy_name=policy_name,
        mean_utility=statistics.mean(utilities),
        accuracy=statistics.mean(correct_flags),
        catastrophic_error_rate=catastrophic_error_rate,
        probe_rate=statistics.mean(probe_flags),
        return_first4_accuracy=statistics.mean(return_first4_correct),
        avg_shift_lag=format_optional(avg_shift_lag),
        fast_return_rate=format_optional(fast_return_rate),
    )


def run_error_reactive(
    episodes: list[Episode],
    config: HiddenShiftConfig,
) -> PolicyMetrics:
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    active_mode: str | None = None
    force_probe = True
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    shift_lags: list[int] = []
    high_risk_total = 0
    found_return_blocks: set[int] = set()

    for episode in episodes:
        if force_probe or active_mode is None:
            probed = True
            prediction = episode.true_label
        else:
            probed = False
            prediction = MODE_TABLES[active_mode][episode.context]

        correct = int(prediction == episode.true_label)
        utility = (1.0 - config.probe_cost) if probed else (1.0 if correct else -episode.risk_cost)

        history.append((episode.context, episode.true_label))
        posterior = posterior_from_window(history, episode.context, config)
        if probed:
            active_mode, _ = top_mode(posterior)
            force_probe = False
        elif not correct:
            force_probe = True

        if episode.return_block and episode.block_index not in found_return_blocks:
            if active_mode == episode.mode_name:
                shift_lags.append(episode.block_offset)
                found_return_blocks.add(episode.block_index)

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(probed))
        if episode.risk_cost == config.high_risk_cost:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="error_reactive_archive",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        shift_lags=shift_lags,
    )


def run_uncertainty_probe(
    episodes: list[Episode],
    config: HiddenShiftConfig,
) -> PolicyMetrics:
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    shift_lags: list[int] = []
    high_risk_total = 0
    found_return_blocks: set[int] = set()

    for episode in episodes:
        posterior = posterior_from_window(history, episode.context, config)
        candidate_mode, _ = top_mode(posterior)
        probed = entropy(posterior) > config.uncertainty_threshold
        if probed:
            prediction = episode.true_label
        else:
            prediction = MODE_TABLES[candidate_mode][episode.context]

        correct = int(prediction == episode.true_label)
        utility = (1.0 - config.probe_cost) if probed else (1.0 if correct else -episode.risk_cost)

        history.append((episode.context, episode.true_label))
        posterior_after = posterior_from_window(history, episode.context, config)
        selected_mode, _ = top_mode(posterior_after)
        if episode.return_block and episode.block_index not in found_return_blocks and selected_mode == episode.mode_name:
            shift_lags.append(episode.block_offset)
            found_return_blocks.add(episode.block_index)

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(probed))
        if episode.risk_cost == config.high_risk_cost:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="uncertainty_probe_archive",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        shift_lags=shift_lags,
    )


def run_risk_aware_hidden_shift(
    episodes: list[Episode],
    config: HiddenShiftConfig,
) -> PolicyMetrics:
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    shift_lags: list[int] = []
    high_risk_total = 0
    found_return_blocks: set[int] = set()

    for episode in episodes:
        posterior = posterior_from_window(history, episode.context, config)
        candidate_mode, _ = top_mode(posterior)
        risk_of_not_probing = expected_error_cost(posterior, episode.context, episode.risk_cost)
        current_entropy = entropy(posterior)
        high_risk_uncertainty = (
            episode.risk_cost > config.low_risk_cost
            and current_entropy > (config.uncertainty_threshold * 0.78)
        )
        probed = risk_of_not_probing > config.probe_cost or high_risk_uncertainty
        if probed:
            prediction = episode.true_label
        else:
            prediction = MODE_TABLES[candidate_mode][episode.context]

        correct = int(prediction == episode.true_label)
        utility = (1.0 - config.probe_cost) if probed else (1.0 if correct else -episode.risk_cost)

        history.append((episode.context, episode.true_label))
        posterior_after = posterior_from_window(history, episode.context, config)
        selected_mode, _ = top_mode(posterior_after)
        if episode.return_block and episode.block_index not in found_return_blocks and selected_mode == episode.mode_name:
            shift_lags.append(episode.block_offset)
            found_return_blocks.add(episode.block_index)

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(probed))
        if episode.risk_cost == config.high_risk_cost:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    return compute_metrics(
        policy_name="risk_aware_hidden_shift",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        shift_lags=shift_lags,
    )


def run_always_probe(
    episodes: list[Episode],
    config: HiddenShiftConfig,
) -> PolicyMetrics:
    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    high_risk_total = 0

    for episode in episodes:
        utilities.append(1.0 - config.probe_cost)
        correct_flags.append(1)
        probe_flags.append(1)
        if episode.risk_cost == config.high_risk_cost:
            high_risk_total += 1
            catastrophic_flags.append(0)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(1)

    return compute_metrics(
        policy_name="always_probe_causal",
        utilities=utilities,
        correct_flags=correct_flags,
        catastrophic_flags=catastrophic_flags,
        high_risk_total=high_risk_total,
        probe_flags=probe_flags,
        return_first4_correct=return_first4_correct,
        shift_lags=[],
    )


def print_report(
    config: HiddenShiftConfig,
    episodes: list[Episode],
    metrics: list[PolicyMetrics],
) -> None:
    total_episodes = len(episodes)
    return_blocks = len({episode.block_index for episode in episodes if episode.return_block})
    high_risk_total = sum(1 for episode in episodes if episode.risk_cost == config.high_risk_cost)

    print("Experiment: hidden regime-shift detection without drift hints")
    print("The agent sees only context and risk before acting; true labels arrive after the action")
    print("World modes recur, but the regime switch itself is never announced")
    print(
        f"Episodes: {total_episodes} | Block length: {config.block_length} | "
        f"Return blocks: {return_blocks} | High-risk episodes: {high_risk_total}"
    )
    print(
        f"Probe cost: {config.probe_cost:.2f} | "
        f"Low-risk error: {config.low_risk_cost:.1f} | High-risk error: {config.high_risk_cost:.1f} | "
        f"Window size: {config.window_size}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<26}"
        f"{'MeanU':>10}"
        f"{'Acc':>10}"
        f"{'CatErr':>10}"
        f"{'Probe':>10}"
        f"{'Return4':>10}"
        f"{'Lag':>10}"
        f"{'FastRet':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics:
        print(
            f"{row.policy_name:<26}"
            f"{row.mean_utility:>10.3f}"
            f"{row.accuracy:>10.3f}"
            f"{row.catastrophic_error_rate:>10.3f}"
            f"{row.probe_rate:>10.3f}"
            f"{row.return_first4_accuracy:>10.3f}"
            f"{row.avg_shift_lag:>10}"
            f"{row.fast_return_rate:>10}"
        )


def main() -> None:
    config = parse_args()
    episodes = build_episodes(random.Random(config.seed), config)
    metrics = [
        run_error_reactive(episodes, config),
        run_uncertainty_probe(episodes, config),
        run_risk_aware_hidden_shift(episodes, config),
        run_always_probe(episodes, config),
    ]
    print_report(config, episodes, metrics)


if __name__ == "__main__":
    main()
