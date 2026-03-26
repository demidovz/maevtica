from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Deque


MODE_TABLES = {
    "color": {"00": 0, "01": 0, "10": 1, "11": 1},
    "shape": {"00": 0, "01": 1, "10": 0, "11": 1},
    "xor": {"00": 0, "01": 1, "10": 1, "11": 0},
}
MODE_CONTEXT_PROBS = {
    "color": {"00": 0.38, "01": 0.22, "10": 0.24, "11": 0.16},
    "shape": {"00": 0.22, "01": 0.38, "10": 0.16, "11": 0.24},
    "xor": {"00": 0.20, "01": 0.18, "10": 0.34, "11": 0.28},
}
SIMILAR_CONTEXT_PROBS = {
    "color": {"00": 0.35, "01": 0.24, "10": 0.23, "11": 0.18},
    "shape": {"00": 0.24, "01": 0.35, "10": 0.18, "11": 0.23},
    "xor": {"00": 0.19, "01": 0.21, "10": 0.31, "11": 0.29},
}
MODE_NAMES = tuple(MODE_TABLES.keys())
ACTIONS = ("act", "probe")


@dataclass(frozen=True)
class HiddenShiftLearningConfig:
    seed: int = 7
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


@dataclass(frozen=True)
class Scenario:
    name: str
    context_probs: dict[str, dict[str, float]]
    probe_cost: float
    high_risk_cost: float
    low_risk_cost: float
    random_high_risk_prob: float
    schedule: tuple[str, ...]


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


def parse_args() -> HiddenShiftLearningConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run four learned hidden-shift detector experiments: contextual probe "
            "learning, selective transfer, revision after cost drift, and mixed evaluation."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
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
    args = parser.parse_args()
    return HiddenShiftLearningConfig(
        seed=args.seed,
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
    )


def baseline_scenario(config: HiddenShiftLearningConfig) -> Scenario:
    return Scenario(
        name="baseline",
        context_probs=MODE_CONTEXT_PROBS,
        probe_cost=config.probe_cost,
        high_risk_cost=config.high_risk_cost,
        low_risk_cost=config.low_risk_cost,
        random_high_risk_prob=config.random_high_risk_prob,
        schedule=("color", "shape", "color", "xor", "shape", "xor"),
    )


def similar_scenario(config: HiddenShiftLearningConfig) -> Scenario:
    return Scenario(
        name="similar",
        context_probs=SIMILAR_CONTEXT_PROBS,
        probe_cost=config.probe_cost,
        high_risk_cost=config.high_risk_cost,
        low_risk_cost=config.low_risk_cost,
        random_high_risk_prob=config.random_high_risk_prob,
        schedule=("shape", "color", "shape", "xor", "color", "xor"),
    )


def dissimilar_scenario(config: HiddenShiftLearningConfig) -> Scenario:
    return Scenario(
        name="dissimilar",
        context_probs=MODE_CONTEXT_PROBS,
        probe_cost=0.42,
        high_risk_cost=1.4,
        low_risk_cost=config.low_risk_cost,
        random_high_risk_prob=0.05,
        schedule=("xor", "color", "xor", "shape", "color", "shape"),
    )


def drifted_scenario(config: HiddenShiftLearningConfig) -> Scenario:
    return Scenario(
        name="drifted",
        context_probs=MODE_CONTEXT_PROBS,
        probe_cost=0.42,
        high_risk_cost=1.4,
        low_risk_cost=config.low_risk_cost,
        random_high_risk_prob=0.05,
        schedule=("color", "shape", "color", "xor", "shape", "xor"),
    )


def high_risk_scenario(config: HiddenShiftLearningConfig) -> Scenario:
    return Scenario(
        name="high_risk",
        context_probs=SIMILAR_CONTEXT_PROBS,
        probe_cost=config.probe_cost,
        high_risk_cost=6.0,
        low_risk_cost=config.low_risk_cost,
        random_high_risk_prob=0.30,
        schedule=("xor", "shape", "xor", "color", "shape", "color"),
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
    scenario: Scenario,
    block_length: int,
) -> list[Episode]:
    seen_modes: set[str] = set()
    episodes: list[Episode] = []
    for block_index, mode_name in enumerate(scenario.schedule):
        return_block = mode_name in seen_modes
        for block_offset in range(block_length):
            context = weighted_choice(rng, scenario.context_probs[mode_name])
            risk_cost = (
                scenario.high_risk_cost
                if block_offset == 0 or rng.random() < scenario.random_high_risk_prob
                else scenario.low_risk_cost
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
    exps = {mode_name: math.exp(score - max_score) for mode_name, score in scores.items()}
    total = sum(exps.values())
    return {mode_name: value / total for mode_name, value in exps.items()}


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )


def posterior_from_history(
    history: Deque[tuple[str, int]],
    context: str,
    config: HiddenShiftLearningConfig,
    scenario: Scenario,
) -> dict[str, float]:
    log_prior = -math.log(len(MODE_NAMES))
    scores = {mode_name: log_prior for mode_name in MODE_NAMES}
    for mode_name in MODE_NAMES:
        scores[mode_name] += math.log(scenario.context_probs[mode_name][context])
    for hist_context, true_label in history:
        for mode_name in MODE_NAMES:
            scores[mode_name] += math.log(scenario.context_probs[mode_name][hist_context])
            predicted = MODE_TABLES[mode_name][hist_context]
            likelihood = config.consistency if predicted == true_label else (1.0 - config.consistency)
            scores[mode_name] += math.log(likelihood)
    return softmax(scores)


def top_mode(probabilities: dict[str, float]) -> tuple[str, float]:
    mode_name = max(probabilities, key=probabilities.get)
    return mode_name, probabilities[mode_name]


def state_key(
    history: Deque[tuple[str, int]],
    episode: Episode,
    config: HiddenShiftLearningConfig,
    scenario: Scenario,
) -> tuple[str, str, str, str, str]:
    posterior = posterior_from_history(history, episode.context, config, scenario)
    candidate_mode, top_mass = top_mode(posterior)
    ent_bucket = "H" if entropy(posterior) > 1.05 else "L"
    risk_bucket = "H" if episode.risk_cost > scenario.low_risk_cost else "L"
    conf_bucket = "L" if top_mass < 0.68 else "H"
    return ent_bucket, risk_bucket, conf_bucket, candidate_mode, episode.context


def epsilon_for_step(step: int, total_steps: int, config: HiddenShiftLearningConfig) -> float:
    if total_steps <= 1:
        return config.epsilon_end
    progress = step / (total_steps - 1)
    return config.epsilon_start + ((config.epsilon_end - config.epsilon_start) * progress)


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
    return max(ACTIONS, key=lambda action: (values[state][action], -ACTIONS.index(action)))


def update_average(
    values: dict[tuple[str, ...], dict[str, float]],
    counts: dict[tuple[str, ...], dict[str, int]],
    state: tuple[str, ...],
    action: str,
    reward: float,
) -> None:
    counts[state][action] += 1
    count = counts[state][action]
    values[state][action] += (reward - values[state][action]) / count


def choose_with_policy(
    policy_name: str,
    values: dict[tuple[str, ...], dict[str, float]],
    global_values: dict[str, float],
    history: Deque[tuple[str, int]],
    episode: Episode,
    config: HiddenShiftLearningConfig,
    scenario: Scenario,
) -> str:
    posterior = posterior_from_history(history, episode.context, config, scenario)
    ent = entropy(posterior)
    if policy_name == "fixed_no_probe":
        return "act"
    if policy_name == "fixed_uncertainty":
        return "probe" if ent > 1.05 else "act"
    if policy_name == "learned_global":
        return max(ACTIONS, key=lambda action: (global_values[action], -ACTIONS.index(action)))
    if policy_name == "learned_contextual":
        state = state_key(history, episode, config, scenario)
        state_values = values[state]
        return max(ACTIONS, key=lambda action: (state_values[action], -ACTIONS.index(action)))
    raise ValueError(f"Unknown policy: {policy_name}")


def simulate_reward(
    action: str,
    history: Deque[tuple[str, int]],
    episode: Episode,
    config: HiddenShiftLearningConfig,
    scenario: Scenario,
) -> tuple[float, int]:
    if action == "probe":
        return 1.0 - scenario.probe_cost, 1
    posterior = posterior_from_history(history, episode.context, config, scenario)
    mode_name, _ = top_mode(posterior)
    prediction = MODE_TABLES[mode_name][episode.context]
    correct = int(prediction == episode.true_label)
    reward = 1.0 if correct else -episode.risk_cost
    return reward, correct


def train_learners(
    scenario: Scenario,
    config: HiddenShiftLearningConfig,
    seed: int,
) -> tuple[dict[tuple[str, ...], dict[str, float]], dict[str, float]]:
    contextual_values: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {action: 0.0 for action in ACTIONS}
    )
    contextual_counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: {action: 0 for action in ACTIONS}
    )
    global_values = {action: 0.0 for action in ACTIONS}
    global_counts = {action: 0 for action in ACTIONS}

    rng = random.Random(seed)
    episodes = build_episodes(rng, scenario, config.block_length) * config.train_cycles
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    total_steps = len(episodes)

    for step, episode in enumerate(episodes):
        state = state_key(history, episode, config, scenario)
        epsilon = epsilon_for_step(step, total_steps, config)

        contextual_action = choose_action(contextual_values, contextual_counts, state, epsilon, rng)
        reward_contextual, _ = simulate_reward(contextual_action, history, episode, config, scenario)
        update_average(contextual_values, contextual_counts, state, contextual_action, reward_contextual)

        global_state = ("global",)
        global_state_values = {global_state: global_values}
        global_state_counts = {global_state: global_counts}
        global_action = choose_action(global_state_values, global_state_counts, global_state, epsilon, rng)
        reward_global, _ = simulate_reward(global_action, history, episode, config, scenario)
        global_counts[global_action] += 1
        global_values[global_action] += (reward_global - global_values[global_action]) / global_counts[global_action]

        history.append((episode.context, episode.true_label))

    return contextual_values, global_values


def evaluate_policy(
    policy_name: str,
    scenario: Scenario,
    config: HiddenShiftLearningConfig,
    seed: int,
    contextual_values: dict[tuple[str, ...], dict[str, float]] | None = None,
    global_values: dict[str, float] | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    episodes = build_episodes(rng, scenario, config.block_length)
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    rewards: list[float] = []
    correct_flags: list[int] = []
    probe_flags: list[int] = []
    high_risk_rewards: list[float] = []

    contextual_values = contextual_values or defaultdict(lambda: {action: 0.0 for action in ACTIONS})
    global_values = global_values or {action: 0.0 for action in ACTIONS}

    for episode in episodes:
        action = choose_with_policy(
            policy_name=policy_name,
            values=contextual_values,
            global_values=global_values,
            history=history,
            episode=episode,
            config=config,
            scenario=scenario,
        )
        reward, correct = simulate_reward(action, history, episode, config, scenario)
        history.append((episode.context, episode.true_label))

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


def run_transfer_sequence(
    contextual_values: dict[tuple[str, ...], dict[str, float]],
    scenario: Scenario,
    config: HiddenShiftLearningConfig,
    seed: int,
    gated: bool,
) -> dict[str, object]:
    rng = random.Random(seed)
    episodes = build_episodes(rng, scenario, config.block_length)
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)
    local_values = defaultdict(lambda: {action: 0.0 for action in ACTIONS})
    for state, action_values in contextual_values.items():
        local_values[state] = dict(action_values)
    local_counts = defaultdict(lambda: {action: 1 for action in ACTIONS})
    rewards: list[float] = []
    gate_open = False

    for step, episode in enumerate(episodes):
        state = state_key(history, episode, config, scenario)
        if gated and step >= 2 and statistics.mean(rewards[-2:]) < 0.72:
            gate_open = True
        if gate_open:
            epsilon = 0.25 if step < 6 else 0.05
            action = choose_action(local_values, local_counts, state, epsilon, rng)
        else:
            action = max(ACTIONS, key=lambda act: (local_values[state][act], -ACTIONS.index(act)))
        reward, _ = simulate_reward(action, history, episode, config, scenario)
        if gate_open:
            update_average(local_values, local_counts, state, action, reward)
        history.append((episode.context, episode.true_label))
        rewards.append(reward)

    return {
        "Overall": round(statistics.mean(rewards), 3),
        "Last6": round(statistics.mean(rewards[-6:]), 3),
        "Gate": "open" if gate_open else "closed",
    }


def revise_after_drift(
    contextual_values: dict[tuple[str, ...], dict[str, float]],
    pre_scenario: Scenario,
    post_scenario: Scenario,
    config: HiddenShiftLearningConfig,
    seed: int,
    adaptive: bool,
) -> dict[str, object]:
    rng = random.Random(seed)
    pre_episodes = build_episodes(rng, pre_scenario, config.block_length)
    post_episodes = build_episodes(rng, post_scenario, config.block_length)
    history: Deque[tuple[str, int]] = deque(maxlen=config.window_size)

    local_values = defaultdict(lambda: {action: 0.0 for action in ACTIONS})
    for state, action_values in contextual_values.items():
        local_values[state] = dict(action_values)
    local_counts = defaultdict(lambda: {action: 4 for action in ACTIONS})

    pre_rewards: list[float] = []
    post_rewards: list[float] = []
    probe_rates_post: list[int] = []
    switch_lag: int | None = None
    target_state = ("H", "L", "L", "xor", "10")

    for episode in pre_episodes:
        state = state_key(history, episode, config, pre_scenario)
        action = max(ACTIONS, key=lambda act: (local_values[state][act], -ACTIONS.index(act)))
        reward, _ = simulate_reward(action, history, episode, config, pre_scenario)
        pre_rewards.append(reward)
        history.append((episode.context, episode.true_label))

    history.clear()
    for step, episode in enumerate(post_episodes):
        state = state_key(history, episode, config, post_scenario)
        epsilon = 0.20 if adaptive and step < 8 else 0.0
        if adaptive:
            action = choose_action(local_values, local_counts, state, epsilon, rng)
        else:
            action = max(ACTIONS, key=lambda act: (local_values[state][act], -ACTIONS.index(act)))
        reward, _ = simulate_reward(action, history, episode, config, post_scenario)
        if adaptive:
            update_average(local_values, local_counts, state, action, reward)
        history.append((episode.context, episode.true_label))
        post_rewards.append(reward)
        probe_rates_post.append(int(action == "probe"))

        target_action = max(ACTIONS, key=lambda act: (local_values[target_state][act], -ACTIONS.index(act)))
        if switch_lag is None and target_action == "act":
            switch_lag = step

    return {
        "Pre": round(statistics.mean(pre_rewards), 3),
        "Post": round(statistics.mean(post_rewards), 3),
        "PostProbe": round(statistics.mean(probe_rates_post), 3),
        "Lag": "8+" if switch_lag is None else switch_lag,
    }


def stage1_contextual_probe_learning(config: HiddenShiftLearningConfig) -> list[dict[str, object]]:
    scenario = baseline_scenario(config)
    contextual_values, global_values = train_learners(scenario, config, config.seed)
    rows = []
    for policy_name in ("fixed_no_probe", "learned_global", "learned_contextual"):
        metrics = evaluate_policy(
            policy_name=policy_name,
            scenario=scenario,
            config=config,
            seed=config.seed + 101,
            contextual_values=contextual_values,
            global_values=global_values,
        )
        rows.append({"policy": policy_name, **metrics})
    return rows


def stage2_selective_transfer(config: HiddenShiftLearningConfig) -> list[dict[str, object]]:
    source = baseline_scenario(config)
    contextual_values, _global_values = train_learners(source, config, config.seed)
    rows = []
    for case_name, scenario in (
        ("similar", similar_scenario(config)),
        ("dissimilar", dissimilar_scenario(config)),
    ):
        blind = run_transfer_sequence(contextual_values, scenario, config, config.seed + 201, gated=False)
        gated = run_transfer_sequence(contextual_values, scenario, config, config.seed + 201, gated=True)
        rows.append({"case": case_name, "policy": "blind_transfer", **blind})
        rows.append({"case": case_name, "policy": "gated_transfer", **gated})
    return rows


def stage3_revision_after_drift(config: HiddenShiftLearningConfig) -> list[dict[str, object]]:
    pre = baseline_scenario(config)
    post = drifted_scenario(config)
    contextual_values, _global_values = train_learners(pre, config, config.seed)
    return [
        {"policy": "rigid_contextual", **revise_after_drift(contextual_values, pre, post, config, config.seed + 301, adaptive=False)},
        {"policy": "adaptive_contextual", **revise_after_drift(contextual_values, pre, post, config, config.seed + 301, adaptive=True)},
    ]


def stage4_mixed_eval(config: HiddenShiftLearningConfig) -> list[dict[str, object]]:
    train_scenario = baseline_scenario(config)
    contextual_values, global_values = train_learners(train_scenario, config, config.seed)
    scenarios = {
        "baseline": baseline_scenario(config),
        "similar_transfer": similar_scenario(config),
        "dissimilar_transfer": dissimilar_scenario(config),
        "probe_cost_drift": drifted_scenario(config),
        "high_risk_return": high_risk_scenario(config),
    }
    rows = []
    for policy_name in ("fixed_no_probe", "learned_global", "learned_contextual"):
        episode_rewards = []
        high_risk_rewards = []
        for offset, scenario in enumerate(scenarios.values()):
            metrics = evaluate_policy(
                policy_name=policy_name,
                scenario=scenario,
                config=config,
                seed=config.seed + 401 + offset,
                contextual_values=contextual_values,
                global_values=global_values,
            )
            episode_rewards.append(metrics["Overall"])
            if scenario.high_risk_cost > config.high_risk_cost or scenario.probe_cost > config.probe_cost:
                high_risk_rewards.append(metrics["Overall"])
        rows.append(
            {
                "policy": policy_name,
                "MeanReward": round(statistics.mean(episode_rewards), 3),
                "MinEpisode": round(min(episode_rewards), 3),
                "HighRiskMean": round(statistics.mean(high_risk_rewards), 3),
            }
        )
    return rows


def print_stage(title: str, rows: list[dict[str, object]]) -> None:
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
        print("".join(f"{str(row[column]):<{widths[column] + 2}}" for column in columns).rstrip())
    print()


def main() -> None:
    config = parse_args()
    print(f"Learned hidden-shift batch seed: {config.seed}")
    print(
        f"Probe cost={config.probe_cost:.2f}, high-risk cost={config.high_risk_cost:.1f}, "
        f"train cycles={config.train_cycles}"
    )
    print()
    print_stage("Stage 1: contextual learning of the hidden-shift probe gate", stage1_contextual_probe_learning(config))
    print_stage("Stage 2: selective transfer of the learned gate", stage2_selective_transfer(config))
    print_stage("Stage 3: revision after probe-cost drift", stage3_revision_after_drift(config))
    print_stage("Stage 4: mixed evaluation of the learned hidden-shift controller", stage4_mixed_eval(config))


if __name__ == "__main__":
    main()
