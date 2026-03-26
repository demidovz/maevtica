from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import learned_hidden_shift_batch_suite as flat_suite
import long_horizon_hidden_shift_test as reference_suite


ACTIONS = flat_suite.ACTIONS
SCENARIO_NAMES = (
    "baseline",
    "similar",
    "dissimilar",
    "drifted",
    "high_risk",
)


@dataclass(frozen=True)
class RecurrentHiddenShiftConfig:
    seed: int = 7
    seed_count: int = 10
    block_length: int = 16
    probe_cost: float = 0.18
    high_risk_cost: float = 5.0
    low_risk_cost: float = 1.0
    random_high_risk_prob: float = 0.18
    window_size: int = 2
    consistency: float = 0.93
    epsilon_start: float = 0.30
    epsilon_end: float = 0.03
    train_cycles: int = 18
    evidence_decay: float = 0.6037456976449606
    surprise_decay: float = 0.788973494774893
    context_weight: float = 0.7346422332836298
    outcome_weight: float = 0.9040552205915373
    ent_hi: float = 1.0738587717729853
    ent_mid: float = 0.6297964259154952
    conf_lo: float = 0.5712230161811868
    conf_hi: float = 0.7840829696930371
    gap_small: float = 0.06032455454516584
    gap_mid: float = 0.15425212147830095
    surprise_hi: float = 0.500729531175961
    surprise_mid: float = 0.16655341358101067
    ratio_xh: float = 2.2860520742121477
    ratio_hi: float = 0.8512636320106663
    ratio_mid: float = 0.5726935970274007


class RecurrentBeliefState:
    def __init__(
        self,
        config: RecurrentHiddenShiftConfig,
        context_probs: dict[str, dict[str, float]],
        probe_cost: float,
        low_risk_cost: float,
    ) -> None:
        self.config = config
        self.context_probs = context_probs
        self.probe_cost = probe_cost
        self.low_risk_cost = low_risk_cost
        self.log_scores = {mode_name: 0.0 for mode_name in flat_suite.MODE_NAMES}
        self.surprise_ema = 0.0
        self.error_streak = 0
        self.steps_since_probe = 0

    def posterior(self, context: str) -> dict[str, float]:
        scores = {
            mode_name: (
                self.log_scores[mode_name]
                + (
                    self.config.context_weight
                    * math.log(self.context_probs[mode_name][context])
                )
            )
            for mode_name in flat_suite.MODE_NAMES
        }
        return softmax(scores)

    def predict_label(self, context: str) -> int:
        posterior = self.posterior(context)
        mode_name = max(posterior, key=posterior.get)
        return flat_suite.MODE_TABLES[mode_name][context]

    def state_key(self, episode: Any) -> tuple[str, ...]:
        posterior = self.posterior(episode.context)
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        top_mode, top_mass = ranked[0]
        second_mass = ranked[1][1]

        entropy_value = entropy(posterior)
        ent_bucket = bucket_ternary(
            entropy_value,
            self.config.ent_mid,
            self.config.ent_hi,
        )
        conf_bucket = confidence_bucket(
            top_mass,
            self.config.conf_lo,
            self.config.conf_hi,
        )
        gap_bucket = gap_bucket_for_value(
            top_mass - second_mass,
            self.config.gap_small,
            self.config.gap_mid,
        )
        surprise_bucket = bucket_ternary(
            self.surprise_ema,
            self.config.surprise_mid,
            self.config.surprise_hi,
        )
        streak_bucket = "H" if self.error_streak >= 2 else "L"

        predicted_label = flat_suite.MODE_TABLES[top_mode][episode.context]
        wrong_probability = sum(
            probability
            for mode_name, probability in posterior.items()
            if flat_suite.MODE_TABLES[mode_name][episode.context] != predicted_label
        )
        cost_ratio = (wrong_probability * episode.risk_cost) / max(self.probe_cost, 1e-9)
        cost_bucket = ratio_bucket(
            cost_ratio,
            self.config.ratio_mid,
            self.config.ratio_hi,
            self.config.ratio_xh,
        )

        risk_bucket = "H" if episode.risk_cost > self.low_risk_cost else "L"
        probe_recency = "far" if self.steps_since_probe >= 4 else "near"
        return (
            risk_bucket,
            ent_bucket,
            conf_bucket,
            gap_bucket,
            surprise_bucket,
            streak_bucket,
            cost_bucket,
            probe_recency,
            top_mode,
        )

    def update(self, episode: Any, *, probed: bool) -> None:
        predicted_label = self.predict_label(episode.context)
        surprise = float(predicted_label != episode.true_label)
        scaled_surprise = surprise * max(
            1.0,
            episode.risk_cost / max(1.0, self.low_risk_cost),
        )
        self.surprise_ema = (
            self.config.surprise_decay * self.surprise_ema
            + ((1.0 - self.config.surprise_decay) * min(scaled_surprise / 3.0, 1.0))
        )
        self.error_streak = self.error_streak + 1 if surprise else 0
        self.steps_since_probe = 0 if probed else self.steps_since_probe + 1

        for mode_name in flat_suite.MODE_NAMES:
            likelihood = (
                self.config.consistency
                if flat_suite.MODE_TABLES[mode_name][episode.context] == episode.true_label
                else (1.0 - self.config.consistency)
            )
            gain = (
                (self.config.context_weight * math.log(self.context_probs[mode_name][episode.context]))
                + (self.config.outcome_weight * math.log(likelihood))
            )
            self.log_scores[mode_name] = (
                self.config.evidence_decay * self.log_scores[mode_name]
            ) + gain


def parse_args() -> RecurrentHiddenShiftConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Test a recurrent hidden-belief-state detector against the flat learned "
            "contextual hidden-shift gate and the hand-crafted risk-aware long-horizon baseline."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--probe-cost", type=float, default=0.18)
    parser.add_argument("--high-risk-cost", type=float, default=5.0)
    parser.add_argument("--low-risk-cost", type=float, default=1.0)
    parser.add_argument("--random-high-risk-prob", type=float, default=0.18)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--consistency", type=float, default=0.93)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--train-cycles", type=int, default=18)
    parser.add_argument("--evidence-decay", type=float, default=0.6037456976449606)
    parser.add_argument("--surprise-decay", type=float, default=0.788973494774893)
    parser.add_argument("--context-weight", type=float, default=0.7346422332836298)
    parser.add_argument("--outcome-weight", type=float, default=0.9040552205915373)
    parser.add_argument("--ent-hi", type=float, default=1.0738587717729853)
    parser.add_argument("--ent-mid", type=float, default=0.6297964259154952)
    parser.add_argument("--conf-lo", type=float, default=0.5712230161811868)
    parser.add_argument("--conf-hi", type=float, default=0.7840829696930371)
    parser.add_argument("--gap-small", type=float, default=0.06032455454516584)
    parser.add_argument("--gap-mid", type=float, default=0.15425212147830095)
    parser.add_argument("--surprise-hi", type=float, default=0.500729531175961)
    parser.add_argument("--surprise-mid", type=float, default=0.16655341358101067)
    parser.add_argument("--ratio-xh", type=float, default=2.2860520742121477)
    parser.add_argument("--ratio-hi", type=float, default=0.8512636320106663)
    parser.add_argument("--ratio-mid", type=float, default=0.5726935970274007)
    args = parser.parse_args()
    return RecurrentHiddenShiftConfig(
        seed=args.seed,
        seed_count=args.seed_count,
        block_length=args.block_length,
        probe_cost=args.probe_cost,
        high_risk_cost=args.high_risk_cost,
        low_risk_cost=args.low_risk_cost,
        random_high_risk_prob=args.random_high_risk_prob,
        window_size=args.window_size,
        consistency=args.consistency,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        train_cycles=args.train_cycles,
        evidence_decay=args.evidence_decay,
        surprise_decay=args.surprise_decay,
        context_weight=args.context_weight,
        outcome_weight=args.outcome_weight,
        ent_hi=args.ent_hi,
        ent_mid=args.ent_mid,
        conf_lo=args.conf_lo,
        conf_hi=args.conf_hi,
        gap_small=args.gap_small,
        gap_mid=args.gap_mid,
        surprise_hi=args.surprise_hi,
        surprise_mid=args.surprise_mid,
        ratio_xh=args.ratio_xh,
        ratio_hi=args.ratio_hi,
        ratio_mid=args.ratio_mid,
    )


def softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exponentials = {
        key: math.exp(score - max_score)
        for key, score in scores.items()
    }
    total = sum(exponentials.values())
    return {
        key: value / total
        for key, value in exponentials.items()
    }


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )


def bucket_ternary(value: float, low_threshold: float, high_threshold: float) -> str:
    if value > high_threshold:
        return "H"
    if value > low_threshold:
        return "M"
    return "L"


def confidence_bucket(value: float, low_threshold: float, high_threshold: float) -> str:
    if value < low_threshold:
        return "L"
    if value < high_threshold:
        return "M"
    return "H"


def gap_bucket_for_value(value: float, small_threshold: float, mid_threshold: float) -> str:
    if value < small_threshold:
        return "S"
    if value < mid_threshold:
        return "M"
    return "L"


def ratio_bucket(value: float, mid_threshold: float, high_threshold: float, extreme_threshold: float) -> str:
    if value > extreme_threshold:
        return "XH"
    if value > high_threshold:
        return "H"
    if value > mid_threshold:
        return "M"
    return "L"


def scenario_by_name(
    scenario_name: str,
    config: RecurrentHiddenShiftConfig,
) -> flat_suite.Scenario:
    if scenario_name == "baseline":
        return flat_suite.baseline_scenario(config)
    if scenario_name == "similar":
        return flat_suite.similar_scenario(config)
    if scenario_name == "dissimilar":
        return flat_suite.dissimilar_scenario(config)
    if scenario_name == "drifted":
        return flat_suite.drifted_scenario(config)
    if scenario_name == "high_risk":
        return flat_suite.high_risk_scenario(config)
    raise ValueError(f"Unknown scenario: {scenario_name}")


def epsilon_for_step(step: int, total_steps: int, config: RecurrentHiddenShiftConfig) -> float:
    if total_steps <= 1:
        return config.epsilon_end
    progress = step / (total_steps - 1)
    return config.epsilon_start + ((config.epsilon_end - config.epsilon_start) * progress)


def default_action_values() -> dict[str, float]:
    return {action: 0.0 for action in ACTIONS}


def choose_action(
    values: dict[tuple[str, ...], dict[str, float]],
    counts: dict[tuple[str, ...], dict[str, int]],
    state: tuple[str, ...],
    epsilon: float,
    rng: random.Random,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    for action in ACTIONS:
        if counts[state][action] == 0:
            return action
    return greedy_action(values, state)


def greedy_action(
    values: dict[tuple[str, ...], dict[str, float]],
    state: tuple[str, ...],
) -> str:
    state_values = values.get(state)
    if state_values is None:
        state_values = default_action_values()
    return max(ACTIONS, key=lambda action: (state_values[action], -ACTIONS.index(action)))


def reward_for_action(
    action: str,
    belief: RecurrentBeliefState,
    episode: Any,
) -> tuple[float, int]:
    if action == "probe":
        return 1.0 - belief.probe_cost, 1
    correct = int(belief.predict_label(episode.context) == episode.true_label)
    return (1.0 if correct else -episode.risk_cost), correct


def train_recurrent_policy(
    config: RecurrentHiddenShiftConfig,
    scenario: flat_suite.Scenario,
    seed: int,
) -> dict[tuple[str, ...], dict[str, float]]:
    values: dict[tuple[str, ...], dict[str, float]] = defaultdict(default_action_values)
    counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: {action: 0 for action in ACTIONS}
    )
    rng = random.Random(seed)
    episodes = flat_suite.build_episodes(rng, scenario, config.block_length) * config.train_cycles
    belief = RecurrentBeliefState(
        config,
        scenario.context_probs,
        scenario.probe_cost,
        scenario.low_risk_cost,
    )
    total_steps = len(episodes)

    for step, episode in enumerate(episodes):
        state = belief.state_key(episode)
        action = choose_action(
            values,
            counts,
            state,
            epsilon_for_step(step, total_steps, config),
            rng,
        )
        reward, _correct = reward_for_action(action, belief, episode)
        counts[state][action] += 1
        count = counts[state][action]
        values[state][action] += (reward - values[state][action]) / count
        belief.update(episode, probed=(action == "probe"))

    return values


def evaluate_recurrent_policy(
    config: RecurrentHiddenShiftConfig,
    values: dict[tuple[str, ...], dict[str, float]],
    scenario: flat_suite.Scenario,
    seed: int,
) -> dict[str, float]:
    episodes = flat_suite.build_episodes(random.Random(seed), scenario, config.block_length)
    belief = RecurrentBeliefState(
        config,
        scenario.context_probs,
        scenario.probe_cost,
        scenario.low_risk_cost,
    )
    rewards: list[float] = []
    correct_flags: list[int] = []
    probe_flags: list[int] = []
    high_risk_rewards: list[float] = []

    for episode in episodes:
        state = belief.state_key(episode)
        action = greedy_action(values, state)
        reward, correct = reward_for_action(action, belief, episode)
        belief.update(episode, probed=(action == "probe"))
        rewards.append(reward)
        correct_flags.append(correct)
        probe_flags.append(int(action == "probe"))
        if episode.risk_cost > scenario.low_risk_cost:
            high_risk_rewards.append(reward)

    return {
        "Overall": round(statistics.mean(rewards), 3),
        "Acc": round(statistics.mean(correct_flags), 3),
        "Probe": round(statistics.mean(probe_flags), 3),
        "HighRisk": round(statistics.mean(high_risk_rewards), 3),
    }


def run_mixed_eval(config: RecurrentHiddenShiftConfig) -> tuple[list[dict[str, object]], dict[str, float]]:
    flat_totals = {scenario_name: [] for scenario_name in SCENARIO_NAMES}
    recurrent_totals = {scenario_name: [] for scenario_name in SCENARIO_NAMES}

    for train_seed in range(config.seed, config.seed + config.seed_count):
        seed_config = replace(config, seed=train_seed)
        baseline = flat_suite.baseline_scenario(seed_config)
        flat_values, flat_global_values = flat_suite.train_learners(
            baseline,
            seed_config,
            train_seed,
        )
        recurrent_values = train_recurrent_policy(seed_config, baseline, train_seed)

        for offset, scenario_name in enumerate(SCENARIO_NAMES):
            scenario = scenario_by_name(scenario_name, seed_config)
            eval_seed = train_seed + 100 + offset
            flat_metrics = flat_suite.evaluate_policy(
                "learned_contextual",
                scenario,
                seed_config,
                eval_seed,
                flat_values,
                flat_global_values,
            )
            recurrent_metrics = evaluate_recurrent_policy(
                seed_config,
                recurrent_values,
                scenario,
                eval_seed,
            )
            flat_totals[scenario_name].append(flat_metrics["Overall"])
            recurrent_totals[scenario_name].append(recurrent_metrics["Overall"])

    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_NAMES:
        flat_mean = statistics.mean(flat_totals[scenario_name])
        recurrent_mean = statistics.mean(recurrent_totals[scenario_name])
        rows.append(
            {
                "scenario": scenario_name,
                "flat_contextual": round(flat_mean, 3),
                "recurrent_belief": round(recurrent_mean, 3),
                "delta": round(recurrent_mean - flat_mean, 3),
            }
        )

    flat_mean_reward = statistics.mean(statistics.mean(values) for values in flat_totals.values())
    recurrent_mean_reward = statistics.mean(
        statistics.mean(values) for values in recurrent_totals.values()
    )
    flat_floor = min(statistics.mean(values) for values in flat_totals.values())
    recurrent_floor = min(statistics.mean(values) for values in recurrent_totals.values())
    summary = {
        "flat_mean_reward": round(flat_mean_reward, 3),
        "recurrent_mean_reward": round(recurrent_mean_reward, 3),
        "flat_floor": round(flat_floor, 3),
        "recurrent_floor": round(recurrent_floor, 3),
    }
    return rows, summary


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def evaluate_recurrent_on_long_horizon(
    config: RecurrentHiddenShiftConfig,
    values: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, object]:
    reference_config = reference_suite.HiddenShiftConfig(seed=config.seed)
    episodes = reference_suite.build_episodes(random.Random(reference_config.seed), reference_config)
    belief = RecurrentBeliefState(
        config,
        reference_suite.MODE_CONTEXT_PROBS,
        reference_config.probe_cost,
        reference_config.low_risk_cost,
    )

    utilities: list[float] = []
    correct_flags: list[int] = []
    catastrophic_flags: list[int] = []
    probe_flags: list[int] = []
    return_first4_correct: list[int] = []
    reactivation_lags: list[int] = []
    high_risk_total = 0
    found_return_blocks: set[int] = set()

    for episode in episodes:
        state = belief.state_key(episode)
        action = greedy_action(values, state)
        utility, correct = reward_for_action(action, belief, episode)
        belief.update(episode, probed=(action == "probe"))

        posterior = belief.posterior(episode.context)
        selected_mode = max(posterior, key=posterior.get)
        if (
            episode.return_block
            and episode.block_index not in found_return_blocks
            and selected_mode == episode.mode_name
        ):
            reactivation_lags.append(episode.block_offset)
            found_return_blocks.add(episode.block_index)

        utilities.append(utility)
        correct_flags.append(correct)
        probe_flags.append(int(action == "probe"))
        if episode.risk_cost == reference_config.high_risk_cost:
            high_risk_total += 1
            catastrophic_flags.append(1 - correct)
        if episode.return_block and episode.block_offset < 4:
            return_first4_correct.append(correct)

    catastrophic_error_rate = (
        0.0 if high_risk_total == 0 else sum(catastrophic_flags) / high_risk_total
    )
    avg_reactivation_lag = (
        None if not reactivation_lags else statistics.mean(reactivation_lags)
    )
    return {
        "policy": "recurrent_belief_learned",
        "MeanU": round(statistics.mean(utilities), 3),
        "Acc": round(statistics.mean(correct_flags), 3),
        "CatErr": round(catastrophic_error_rate, 3),
        "Probe": round(statistics.mean(probe_flags), 3),
        "Return4": round(statistics.mean(return_first4_correct), 3),
        "Lag": format_optional(avg_reactivation_lag),
    }


def run_long_horizon_reference(config: RecurrentHiddenShiftConfig) -> list[dict[str, object]]:
    baseline = flat_suite.baseline_scenario(config)
    recurrent_values = train_recurrent_policy(config, baseline, config.seed)
    recurrent_row = evaluate_recurrent_on_long_horizon(config, recurrent_values)

    reference_config = reference_suite.HiddenShiftConfig(seed=config.seed)
    episodes = reference_suite.build_episodes(random.Random(reference_config.seed), reference_config)
    risk_aware = reference_suite.run_risk_aware_hidden_shift(episodes, reference_config)
    return [
        recurrent_row,
        {
            "policy": risk_aware.policy_name,
            "MeanU": round(risk_aware.mean_utility, 3),
            "Acc": round(risk_aware.accuracy, 3),
            "CatErr": round(risk_aware.catastrophic_error_rate, 3),
            "Probe": round(risk_aware.probe_rate, 3),
            "Return4": round(risk_aware.return_first4_accuracy, 3),
            "Lag": risk_aware.avg_shift_lag,
        },
    ]


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(title)
    if not rows:
        print()
        return
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = "".join(f"{column:<{widths[column] + 2}}" for column in columns)
    print(header.rstrip())
    print("-" * len(header.rstrip()))
    for row in rows:
        print(
            "".join(f"{str(row[column]):<{widths[column] + 2}}" for column in columns).rstrip()
        )
    print()


def main() -> None:
    config = parse_args()
    print("Hypothesis")
    print(
        "A recurrent hidden belief state with decayed mode evidence, surprise memory, "
        "and cost-aware buckets should beat the flat learned contextual gate on mixed hidden-shift evaluation."
    )
    print()
    print(
        f"Train/eval seeds: {config.seed_count} starting at {config.seed} | "
        f"Block length: {config.block_length} | Train cycles: {config.train_cycles}"
    )
    print(
        f"Probe cost: {config.probe_cost:.2f} | High-risk cost: {config.high_risk_cost:.1f} | "
        f"Low-risk cost: {config.low_risk_cost:.1f}"
    )
    print()

    mixed_rows, summary = run_mixed_eval(config)
    print_table(
        "Mixed hidden-shift evaluation (mean reward averaged across seeds)",
        mixed_rows,
    )
    print("Aggregate")
    print(
        f"flat_contextual mean={summary['flat_mean_reward']:.3f} "
        f"floor={summary['flat_floor']:.3f}"
    )
    print(
        f"recurrent_belief mean={summary['recurrent_mean_reward']:.3f} "
        f"floor={summary['recurrent_floor']:.3f}"
    )
    print()

    print_table(
        "Long-horizon reference benchmark",
        run_long_horizon_reference(config),
    )


if __name__ == "__main__":
    main()
