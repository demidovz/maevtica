from __future__ import annotations

import math
import random
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import learned_hidden_shift_batch_suite as flat_suite
import long_horizon_hidden_shift_test as long_suite


ACTIONS = flat_suite.ACTIONS
SCENARIO_ORDER = ("baseline", "similar", "dissimilar", "drifted", "high_risk")


@dataclass(frozen=True)
class IterationConfig:
    seed_start: int = 7
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


@dataclass(frozen=True)
class Variant:
    variant_id: str
    hypothesis: str
    trainer: str


@dataclass(frozen=True)
class VariantMetrics:
    variant_id: str
    mixed_mean: float
    mixed_floor: float
    long_mean: float


class RecurrentBeliefState:
    def __init__(
        self,
        config: IterationConfig,
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

    def state_key(self, episode) -> tuple[str, ...]:
        posterior = self.posterior(episode.context)
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        top_mode, top_mass = ranked[0]
        second_mass = ranked[1][1]

        entropy_value = entropy(posterior)
        ent_bucket = ternary_bucket(
            entropy_value,
            self.config.ent_mid,
            self.config.ent_hi,
        )
        conf_bucket = confidence_bucket(
            top_mass,
            self.config.conf_lo,
            self.config.conf_hi,
        )
        gap_bucket = spread_bucket(
            top_mass - second_mass,
            self.config.gap_small,
            self.config.gap_mid,
        )
        surprise_bucket = ternary_bucket(
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
        cost_bucket = cost_ratio_bucket(
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

    def update(self, episode, *, probed: bool) -> None:
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


def ternary_bucket(value: float, low_threshold: float, high_threshold: float) -> str:
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


def spread_bucket(value: float, low_threshold: float, high_threshold: float) -> str:
    if value < low_threshold:
        return "S"
    if value < high_threshold:
        return "M"
    return "L"


def cost_ratio_bucket(
    value: float,
    mid_threshold: float,
    high_threshold: float,
    extreme_threshold: float,
) -> str:
    if value > extreme_threshold:
        return "XH"
    if value > high_threshold:
        return "H"
    if value > mid_threshold:
        return "M"
    return "L"


def make_lightweight_config(config: IterationConfig, seed: int):
    return type(
        "LightweightConfig",
        (),
        {
            "seed": seed,
            "block_length": config.block_length,
            "probe_cost": config.probe_cost,
            "high_risk_cost": config.high_risk_cost,
            "low_risk_cost": config.low_risk_cost,
            "random_high_risk_prob": config.random_high_risk_prob,
            "window_size": config.window_size,
            "consistency": config.consistency,
            "epsilon_start": config.epsilon_start,
            "epsilon_end": config.epsilon_end,
            "train_cycles": config.train_cycles,
        },
    )


def scenario_by_name(name: str, config: IterationConfig, seed: int):
    light = make_lightweight_config(config, seed)
    if name == "baseline":
        return flat_suite.baseline_scenario(light)
    if name == "similar":
        return flat_suite.similar_scenario(light)
    if name == "dissimilar":
        return flat_suite.dissimilar_scenario(light)
    if name == "drifted":
        return flat_suite.drifted_scenario(light)
    if name == "high_risk":
        return flat_suite.high_risk_scenario(light)
    raise ValueError(f"Unknown scenario: {name}")


def epsilon_for_step(step: int, total_steps: int, config: IterationConfig) -> float:
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
    episode,
) -> tuple[float, int]:
    if action == "probe":
        return 1.0 - belief.probe_cost, 1
    correct = int(belief.predict_label(episode.context) == episode.true_label)
    return (1.0 if correct else -episode.risk_cost), correct


def train_flat_variant(
    config: IterationConfig,
    seed: int,
) -> tuple[dict[tuple[str, ...], dict[str, float]], dict[str, float]]:
    baseline = scenario_by_name("baseline", config, seed)
    light = make_lightweight_config(config, seed)
    return flat_suite.train_learners(baseline, light, seed)


def train_recurrent_variant(
    config: IterationConfig,
    seed: int,
    trainer: str,
) -> dict[tuple[str, ...], dict[str, float]]:
    values: dict[tuple[str, ...], dict[str, float]] = defaultdict(default_action_values)
    counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: {action: 0 for action in ACTIONS}
    )
    rng = random.Random(seed)
    blocks: list[tuple[dict[str, dict[str, float]], float, float, list]] = []

    if trainer == "baseline":
        baseline = scenario_by_name("baseline", config, seed)
        for cycle in range(config.train_cycles):
            episodes = flat_suite.build_episodes(
                random.Random(seed + cycle),
                baseline,
                config.block_length,
            )
            blocks.append(
                (
                    baseline.context_probs,
                    baseline.probe_cost,
                    baseline.low_risk_cost,
                    episodes,
                )
            )
    elif trainer in {"mixed", "mixed_plus_long"}:
        curriculum = ("baseline", "similar", "drifted", "high_risk", "dissimilar")
        for cycle in range(config.train_cycles):
            scenario_name = curriculum[cycle % len(curriculum)]
            scenario = scenario_by_name(scenario_name, config, seed)
            episodes = flat_suite.build_episodes(
                random.Random(seed + 1000 + cycle),
                scenario,
                config.block_length,
            )
            blocks.append(
                (
                    scenario.context_probs,
                    scenario.probe_cost,
                    scenario.low_risk_cost,
                    episodes,
                )
            )
        if trainer == "mixed_plus_long":
            long_config = long_suite.HiddenShiftConfig(
                seed=seed,
                block_length=config.block_length,
                probe_cost=config.probe_cost,
                high_risk_cost=4.0,
                low_risk_cost=config.low_risk_cost,
                random_high_risk_prob=0.10,
                window_size=config.window_size,
                consistency=config.consistency,
                uncertainty_threshold=1.10,
            )
            for cycle in range(6):
                episodes = long_suite.build_episodes(
                    random.Random(seed + 3000 + cycle),
                    long_config,
                )
                blocks.append(
                    (
                        long_suite.MODE_CONTEXT_PROBS,
                        long_config.probe_cost,
                        long_config.low_risk_cost,
                        episodes,
                    )
                )
    else:
        raise ValueError(f"Unknown trainer: {trainer}")

    total_steps = sum(len(episodes) for _, _, _, episodes in blocks)
    step = 0
    for context_probs, probe_cost, low_risk_cost, episodes in blocks:
        belief = RecurrentBeliefState(
            config,
            context_probs,
            probe_cost,
            low_risk_cost,
        )
        for episode in episodes:
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
            values[state][action] += (
                reward - values[state][action]
            ) / counts[state][action]
            belief.update(episode, probed=(action == "probe"))
            step += 1

    return values


def evaluate_flat_mixed(
    config: IterationConfig,
    values: dict[tuple[str, ...], dict[str, float]],
    global_values: dict[str, float],
    seed: int,
) -> dict[str, float]:
    light = make_lightweight_config(config, seed)
    results: dict[str, float] = {}
    for offset, scenario_name in enumerate(SCENARIO_ORDER):
        scenario = scenario_by_name(scenario_name, config, seed)
        metrics = flat_suite.evaluate_policy(
            "learned_contextual",
            scenario,
            light,
            seed + 100 + offset,
            values,
            global_values,
        )
        results[scenario_name] = metrics["Overall"]
    return results


def evaluate_recurrent_mixed(
    config: IterationConfig,
    values: dict[tuple[str, ...], dict[str, float]],
    seed: int,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for offset, scenario_name in enumerate(SCENARIO_ORDER):
        scenario = scenario_by_name(scenario_name, config, seed)
        belief = RecurrentBeliefState(
            config,
            scenario.context_probs,
            scenario.probe_cost,
            scenario.low_risk_cost,
        )
        episodes = flat_suite.build_episodes(
            random.Random(seed + 100 + offset),
            scenario,
            config.block_length,
        )
        rewards: list[float] = []
        for episode in episodes:
            state = belief.state_key(episode)
            action = greedy_action(values, state)
            reward, _correct = reward_for_action(action, belief, episode)
            rewards.append(reward)
            belief.update(episode, probed=(action == "probe"))
        results[scenario_name] = statistics.mean(rewards)
    return results


def evaluate_flat_long_horizon(
    config: IterationConfig,
    values: dict[tuple[str, ...], dict[str, float]],
    seed: int,
) -> float:
    reference = long_suite.HiddenShiftConfig(seed=seed)
    scenario = type(
        "FlatLongScenario",
        (),
        {
            "context_probs": long_suite.MODE_CONTEXT_PROBS,
            "probe_cost": reference.probe_cost,
            "high_risk_cost": reference.high_risk_cost,
            "low_risk_cost": reference.low_risk_cost,
            "random_high_risk_prob": reference.random_high_risk_prob,
            "schedule": tuple(),
        },
    )
    light = make_lightweight_config(config, seed)
    history = deque(maxlen=light.window_size)
    rewards: list[float] = []
    episodes = long_suite.build_episodes(random.Random(seed), reference)
    for episode in episodes:
        state = flat_suite.state_key(history, episode, light, scenario)
        action = greedy_action(values, state)
        reward, _correct = flat_suite.simulate_reward(action, history, episode, light, scenario)
        rewards.append(reward)
        history.append((episode.context, episode.true_label))
    return statistics.mean(rewards)


def evaluate_recurrent_long_horizon(
    config: IterationConfig,
    values: dict[tuple[str, ...], dict[str, float]],
    seed: int,
) -> float:
    reference = long_suite.HiddenShiftConfig(seed=seed)
    belief = RecurrentBeliefState(
        config,
        long_suite.MODE_CONTEXT_PROBS,
        reference.probe_cost,
        reference.low_risk_cost,
    )
    rewards: list[float] = []
    episodes = long_suite.build_episodes(random.Random(seed), reference)
    for episode in episodes:
        state = belief.state_key(episode)
        action = greedy_action(values, state)
        reward, _correct = reward_for_action(action, belief, episode)
        rewards.append(reward)
        belief.update(episode, probed=(action == "probe"))
    return statistics.mean(rewards)


def evaluate_variant(config: IterationConfig, variant: Variant) -> VariantMetrics:
    mixed_results = {scenario_name: [] for scenario_name in SCENARIO_ORDER}
    long_results: list[float] = []

    for seed in range(config.seed_start, config.seed_start + config.seed_count):
        if variant.trainer == "flat":
            values, global_values = train_flat_variant(config, seed)
            mixed_scores = evaluate_flat_mixed(config, values, global_values, seed)
            long_score = evaluate_flat_long_horizon(config, values, seed)
        else:
            values = train_recurrent_variant(config, seed, variant.trainer)
            mixed_scores = evaluate_recurrent_mixed(config, values, seed)
            long_score = evaluate_recurrent_long_horizon(config, values, seed)

        for scenario_name, score in mixed_scores.items():
            mixed_results[scenario_name].append(score)
        long_results.append(long_score)

    scenario_means = {
        scenario_name: statistics.mean(scores)
        for scenario_name, scores in mixed_results.items()
    }
    return VariantMetrics(
        variant_id=variant.variant_id,
        mixed_mean=statistics.mean(scenario_means.values()),
        mixed_floor=min(scenario_means.values()),
        long_mean=statistics.mean(long_results),
    )


def evaluate_risk_aware_reference(config: IterationConfig) -> float:
    rewards: list[float] = []
    for seed in range(config.seed_start, config.seed_start + config.seed_count):
        reference = long_suite.HiddenShiftConfig(seed=seed)
        episodes = long_suite.build_episodes(random.Random(seed), reference)
        rewards.append(long_suite.run_risk_aware_hidden_shift(episodes, reference).mean_utility)
    return statistics.mean(rewards)


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


def variants() -> list[Variant]:
    return [
        Variant(
            variant_id="iter1_flat_contextual",
            hypothesis="Плоского contextual gate по короткому снимку state достаточно.",
            trainer="flat",
        ),
        Variant(
            variant_id="iter2_recurrent_belief",
            hypothesis="Если хранить decayed mode evidence, surprise memory и историю ошибок, detector станет устойчивее.",
            trainer="baseline",
        ),
        Variant(
            variant_id="iter3_recurrent_mixed_curriculum",
            hypothesis="Если recurrent state дополнительно учить на смешанном curriculum, перенос станет ещё лучше.",
            trainer="mixed",
        ),
        Variant(
            variant_id="iter4_recurrent_long_finetune",
            hypothesis="Если после mixed curriculum отдельно доучить detector на return-block long horizon, gap до hand-crafted baseline сузится.",
            trainer="mixed_plus_long",
        ),
    ]


def main() -> None:
    config = IterationConfig()
    rows: list[dict[str, object]] = []
    for variant in variants():
        metrics = evaluate_variant(config, variant)
        rows.append(
            {
                "variant": variant.variant_id,
                "mixed_mean": round(metrics.mixed_mean, 3),
                "mixed_floor": round(metrics.mixed_floor, 3),
                "long_mean": round(metrics.long_mean, 3),
            }
        )

    print(
        f"Seeds: {config.seed_start}..{config.seed_start + config.seed_count - 1} | "
        f"block_length={config.block_length} | train_cycles={config.train_cycles}"
    )
    print_table("Four hypothesis iterations", rows)
    print(
        "Long-horizon hand-crafted reference: "
        f"{evaluate_risk_aware_reference(config):.3f}"
    )


if __name__ == "__main__":
    main()
