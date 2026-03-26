from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass


ACTIONS = ("ask_family", "ask_rule", "ask_mode", "act_0", "act_1")
QUESTION_ACTIONS = ("ask_family", "ask_rule", "ask_mode")
UNKNOWN = -1


@dataclass(frozen=True)
class LearnedInquiryPolicyConfig:
    phase1_episodes: int = 3500
    phase2_episodes: int = 3500
    eval_episodes: int = 2000
    seed: int = 7
    alpha: float = 0.22
    gamma: float = 0.95
    epsilon_start: float = 0.28
    epsilon_end: float = 0.03
    max_questions: int = 1
    ask_family_cost: float = 0.12
    ask_rule_cost: float = 0.12
    ask_mode_cost: float = 0.22
    normal_weight: float = 1.0
    mode_weight: float = 2.5
    switch_probe_interval: int = 25
    switch_stability_windows: int = 6


@dataclass(frozen=True)
class EpisodeSpec:
    signature: str
    family_value: int
    rule_value: int
    mode_value: int
    target: int
    weight: float


@dataclass(frozen=True)
class EvalMetrics:
    phase_name: str
    policy_name: str
    mean_utility: float
    risk_weighted_utility: float
    accuracy: float
    questions_per_episode: float


def parse_args() -> LearnedInquiryPolicyConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tabular inquiry policy to choose when to ask, which level to ask, "
            "and when to stop asking and act."
        )
    )
    parser.add_argument("--phase1-episodes", type=int, default=3500)
    parser.add_argument("--phase2-episodes", type=int, default=3500)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.22)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.28)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--max-questions", type=int, default=1)
    parser.add_argument("--ask-family-cost", type=float, default=0.12)
    parser.add_argument("--ask-rule-cost", type=float, default=0.12)
    parser.add_argument("--ask-mode-cost", type=float, default=0.22)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--mode-weight", type=float, default=2.5)
    parser.add_argument("--switch-probe-interval", type=int, default=25)
    parser.add_argument("--switch-stability-windows", type=int, default=6)
    args = parser.parse_args()

    config = LearnedInquiryPolicyConfig(
        phase1_episodes=args.phase1_episodes,
        phase2_episodes=args.phase2_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        max_questions=args.max_questions,
        ask_family_cost=args.ask_family_cost,
        ask_rule_cost=args.ask_rule_cost,
        ask_mode_cost=args.ask_mode_cost,
        normal_weight=args.normal_weight,
        mode_weight=args.mode_weight,
        switch_probe_interval=args.switch_probe_interval,
        switch_stability_windows=args.switch_stability_windows,
    )
    validate_config(config)
    return config


def validate_config(config: LearnedInquiryPolicyConfig) -> None:
    if min(config.phase1_episodes, config.phase2_episodes, config.eval_episodes) <= 0:
        raise ValueError("episode counts must be > 0")
    if not 0.0 < config.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= config.epsilon_end <= config.epsilon_start < 1.0:
        raise ValueError("epsilon range must satisfy 0 <= end <= start < 1")
    if config.max_questions <= 0:
        raise ValueError("max_questions must be > 0")
    if min(config.ask_family_cost, config.ask_rule_cost, config.ask_mode_cost) <= 0.0:
        raise ValueError("question costs must be > 0")
    if min(config.normal_weight, config.mode_weight) <= 0.0:
        raise ValueError("outcome weights must be > 0")
    if min(config.switch_probe_interval, config.switch_stability_windows) <= 0:
        raise ValueError("switch probe parameters must be > 0")


def signature_phase_kind(signature: str, phase: int) -> str:
    if signature.startswith("D"):
        return "direct"
    if signature == "R":
        return "rule"
    if signature == "M":
        return "mode"
    if signature == "F":
        return "family" if phase == 1 else "mode"
    raise ValueError(f"Unknown signature: {signature}")


def sample_signature(rng: random.Random, phase: int) -> str:
    if phase == 1:
        population = ("D0", "D1", "F", "R", "M")
        weights = (0.15, 0.15, 0.30, 0.20, 0.20)
    else:
        population = ("D0", "D1", "F", "R", "M")
        weights = (0.125, 0.125, 0.35, 0.15, 0.25)
    return rng.choices(population, weights=weights, k=1)[0]


def sample_episode(rng: random.Random, phase: int, config: LearnedInquiryPolicyConfig) -> EpisodeSpec:
    signature = sample_signature(rng, phase)
    kind = signature_phase_kind(signature, phase)

    if signature == "D0":
        target = 0
    elif signature == "D1":
        target = 1
    else:
        target = rng.randint(0, 1)

    family_value = rng.randint(0, 1)
    rule_value = rng.randint(0, 1)
    mode_value = rng.randint(0, 1)

    if kind == "family":
        family_value = target
        weight = config.normal_weight
    elif kind == "rule":
        rule_value = target
        weight = config.normal_weight
    elif kind == "mode":
        mode_value = target
        weight = config.mode_weight
    elif kind == "direct":
        weight = config.normal_weight
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return EpisodeSpec(
        signature=signature,
        family_value=family_value,
        rule_value=rule_value,
        mode_value=mode_value,
        target=target,
        weight=weight,
    )


def initial_state(spec: EpisodeSpec) -> tuple[str, int, int, int, int]:
    return (spec.signature, UNKNOWN, UNKNOWN, UNKNOWN, 0)


def state_after_question(
    state: tuple[str, int, int, int, int],
    action: str,
    spec: EpisodeSpec,
) -> tuple[str, int, int, int, int]:
    signature, family_obs, rule_obs, mode_obs, questions_asked = state
    if action == "ask_family":
        family_obs = spec.family_value
    elif action == "ask_rule":
        rule_obs = spec.rule_value
    elif action == "ask_mode":
        mode_obs = spec.mode_value
    else:
        raise ValueError(f"Unknown question action: {action}")
    return (signature, family_obs, rule_obs, mode_obs, questions_asked + 1)


def question_cost(action: str, config: LearnedInquiryPolicyConfig) -> float:
    if action == "ask_family":
        return config.ask_family_cost
    if action == "ask_rule":
        return config.ask_rule_cost
    if action == "ask_mode":
        return config.ask_mode_cost
    raise ValueError(f"Unknown action for question cost: {action}")


def available_actions(
    state: tuple[str, int, int, int, int],
    config: LearnedInquiryPolicyConfig,
) -> list[str]:
    _, family_obs, rule_obs, mode_obs, questions_asked = state
    actions = ["act_0", "act_1"]
    if questions_asked >= config.max_questions:
        return actions
    if family_obs == UNKNOWN:
        actions.append("ask_family")
    if rule_obs == UNKNOWN:
        actions.append("ask_rule")
    if mode_obs == UNKNOWN:
        actions.append("ask_mode")
    return actions


def greedy_action(
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]],
    state: tuple[str, int, int, int, int],
    config: LearnedInquiryPolicyConfig,
) -> str:
    actions = available_actions(state, config)
    return max(actions, key=lambda action: (q_table[state][action], -ACTIONS.index(action)))


def epsilon_greedy_action(
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]],
    state: tuple[str, int, int, int, int],
    epsilon: float,
    rng: random.Random,
    config: LearnedInquiryPolicyConfig,
) -> str:
    actions = available_actions(state, config)
    if rng.random() < epsilon:
        return rng.choice(actions)
    return greedy_action(q_table, state, config)


def transition(
    state: tuple[str, int, int, int, int],
    action: str,
    spec: EpisodeSpec,
    config: LearnedInquiryPolicyConfig,
) -> tuple[tuple[str, int, int, int, int] | None, float, float, int, bool]:
    if action in QUESTION_ACTIONS:
        next_state = state_after_question(state, action, spec)
        reward = -question_cost(action, config)
        return next_state, reward, 0.0, 0, False

    predicted = 0 if action == "act_0" else 1
    correct = 1 if predicted == spec.target else 0
    outcome_reward = spec.weight if correct else -spec.weight
    return None, outcome_reward, outcome_reward, correct, True


def oracle_first_action(signature: str, phase: int) -> str:
    kind = signature_phase_kind(signature, phase)
    if kind == "direct":
        return "act_0" if signature == "D0" else "act_1"
    if kind == "family":
        return "ask_family"
    if kind == "rule":
        return "ask_rule"
    if kind == "mode":
        return "ask_mode"
    raise ValueError(f"Unknown kind for oracle: {kind}")


def run_oracle_episode(
    spec: EpisodeSpec,
    phase: int,
    config: LearnedInquiryPolicyConfig,
) -> tuple[float, float, int, int]:
    action = oracle_first_action(spec.signature, phase)
    questions = 0
    utility = 0.0
    risk_weighted = 0.0

    if action in QUESTION_ACTIONS:
        utility -= question_cost(action, config)
        questions = 1
        if action == "ask_family":
            answer = spec.family_value
        elif action == "ask_rule":
            answer = spec.rule_value
        else:
            answer = spec.mode_value
        predicted = answer
    else:
        predicted = 0 if action == "act_0" else 1

    correct = 1 if predicted == spec.target else 0
    outcome = spec.weight if correct else -spec.weight
    utility += outcome
    risk_weighted += outcome
    return utility, risk_weighted, questions, correct


def evaluate_policy(
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]],
    phase: int,
    episodes: int,
    rng: random.Random,
    config: LearnedInquiryPolicyConfig,
) -> EvalMetrics:
    utilities: list[float] = []
    risk_weighted_utilities: list[float] = []
    questions_list: list[int] = []
    accuracies: list[int] = []

    for _ in range(episodes):
        spec = sample_episode(rng, phase, config)
        state = initial_state(spec)
        questions = 0
        utility = 0.0
        risk_utility = 0.0
        correct = 0

        for _step in range(config.max_questions + 1):
            action = greedy_action(q_table, state, config)
            next_state, reward, outcome_reward, correct, done = transition(state, action, spec, config)
            utility += reward
            risk_utility += outcome_reward
            if action in QUESTION_ACTIONS:
                questions += 1
            if done:
                break
            if next_state is None:
                raise ValueError("Non-terminal step returned no next state")
            state = next_state

        utilities.append(utility)
        risk_weighted_utilities.append(risk_utility)
        questions_list.append(questions)
        accuracies.append(correct)

    return EvalMetrics(
        phase_name=f"phase_{phase}",
        policy_name="learned_q_policy",
        mean_utility=statistics.mean(utilities),
        risk_weighted_utility=statistics.mean(risk_weighted_utilities),
        accuracy=statistics.mean(accuracies),
        questions_per_episode=statistics.mean(questions_list),
    )


def evaluate_oracle(
    phase: int,
    episodes: int,
    rng: random.Random,
    config: LearnedInquiryPolicyConfig,
) -> EvalMetrics:
    utilities: list[float] = []
    risk_weighted_utilities: list[float] = []
    questions_list: list[int] = []
    accuracies: list[int] = []

    for _ in range(episodes):
        spec = sample_episode(rng, phase, config)
        utility, risk_utility, questions, correct = run_oracle_episode(spec, phase, config)
        utilities.append(utility)
        risk_weighted_utilities.append(risk_utility)
        questions_list.append(questions)
        accuracies.append(correct)

    return EvalMetrics(
        phase_name=f"phase_{phase}",
        policy_name="oracle_handcrafted",
        mean_utility=statistics.mean(utilities),
        risk_weighted_utility=statistics.mean(risk_weighted_utilities),
        accuracy=statistics.mean(accuracies),
        questions_per_episode=statistics.mean(questions_list),
    )


def linear_epsilon(
    episode_index: int,
    total_episodes: int,
    config: LearnedInquiryPolicyConfig,
) -> float:
    fraction = episode_index / max(total_episodes - 1, 1)
    return config.epsilon_start + (config.epsilon_end - config.epsilon_start) * fraction


def first_action_for_signature(
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]],
    signature: str,
    config: LearnedInquiryPolicyConfig,
) -> str:
    state = (signature, UNKNOWN, UNKNOWN, UNKNOWN, 0)
    return greedy_action(q_table, state, config)


def measure_switch_lag(
    switch_probes: list[tuple[int, str]],
    config: LearnedInquiryPolicyConfig,
) -> int | None:
    if len(switch_probes) < config.switch_stability_windows:
        return None
    for start_index in range(len(switch_probes) - config.switch_stability_windows + 1):
        window = switch_probes[start_index : start_index + config.switch_stability_windows]
        if all(action == "ask_mode" for _episode, action in window):
            return window[0][0]
    return None


def clone_q_table(
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]],
) -> dict[tuple[str, int, int, int, int], dict[str, float]]:
    return {
        state: dict(action_values)
        for state, action_values in q_table.items()
    }


def train_q_policy(
    config: LearnedInquiryPolicyConfig,
) -> tuple[
    dict[tuple[str, int, int, int, int], dict[str, float]],
    dict[tuple[str, int, int, int, int], dict[str, float]],
    int | None,
]:
    q_table: dict[tuple[str, int, int, int, int], dict[str, float]] = defaultdict(
        lambda: {action: 0.0 for action in ACTIONS}
    )
    rng = random.Random(config.seed)
    total_episodes = config.phase1_episodes + config.phase2_episodes
    switch_probes: list[tuple[int, str]] = []
    phase1_snapshot: dict[tuple[str, int, int, int, int], dict[str, float]] | None = None

    for global_episode in range(total_episodes):
        phase = 1 if global_episode < config.phase1_episodes else 2
        epsilon = linear_epsilon(global_episode, total_episodes, config)
        spec = sample_episode(rng, phase, config)
        state = initial_state(spec)

        for _step in range(config.max_questions + 1):
            action = epsilon_greedy_action(q_table, state, epsilon, rng, config)
            next_state, reward, _outcome_reward, _correct, done = transition(state, action, spec, config)
            next_value = 0.0 if done or next_state is None else max(
                q_table[next_state][next_action]
                for next_action in available_actions(next_state, config)
            )
            q_table[state][action] += config.alpha * (
                reward + (config.gamma * next_value) - q_table[state][action]
            )
            if done:
                break
            if next_state is None:
                raise ValueError("Non-terminal Q step returned no next state")
            state = next_state

        if phase == 1 and global_episode == config.phase1_episodes - 1:
            phase1_snapshot = clone_q_table(q_table)

        if phase == 2 and (global_episode - config.phase1_episodes) % config.switch_probe_interval == 0:
            switch_probes.append(
                (
                    global_episode - config.phase1_episodes,
                    first_action_for_signature(q_table, "F", config),
                )
            )

    if phase1_snapshot is None:
        phase1_snapshot = clone_q_table(q_table)
    return phase1_snapshot, clone_q_table(q_table), measure_switch_lag(switch_probes, config)


def print_report(
    config: LearnedInquiryPolicyConfig,
    learned_phase1: EvalMetrics,
    learned_phase2: EvalMetrics,
    oracle_phase1: EvalMetrics,
    oracle_phase2: EvalMetrics,
    switch_lag: int | None,
    first_actions_phase1: dict[str, str],
    first_actions_phase2: dict[str, str],
) -> None:
    print("Experiment: learned inquiry policy vs oracle hand-crafted controller")
    print("Action space: ask_family, ask_rule, ask_mode, act_0, act_1")
    print("Phase 1: signature F requires ask_family")
    print("Phase 2: signature F switches and now requires ask_mode")
    print(
        f"Question costs: family={config.ask_family_cost:.2f}, "
        f"rule={config.ask_rule_cost:.2f}, mode={config.ask_mode_cost:.2f}"
    )
    print(
        f"Outcome weights: normal={config.normal_weight:.2f}, "
        f"mode={config.mode_weight:.2f}"
    )
    print(f"Seed: {config.seed}")
    print()

    header = (
        f"{'Policy':<20}"
        f"{'Phase':<10}"
        f"{'MeanU':>10}"
        f"{'RiskU':>10}"
        f"{'Acc':>10}"
        f"{'Q/Ep':>10}"
        f"{'Regret':>10}"
    )
    print(header)
    print("-" * len(header))

    paired = [
        (oracle_phase1, learned_phase1),
        (oracle_phase2, learned_phase2),
    ]
    for oracle_metrics, learned_metrics in paired:
        print(
            f"{oracle_metrics.policy_name:<20}"
            f"{oracle_metrics.phase_name:<10}"
            f"{oracle_metrics.mean_utility:>10.3f}"
            f"{oracle_metrics.risk_weighted_utility:>10.3f}"
            f"{oracle_metrics.accuracy:>10.3f}"
            f"{oracle_metrics.questions_per_episode:>10.3f}"
            f"{0.0:>10.3f}"
        )
        regret = oracle_metrics.mean_utility - learned_metrics.mean_utility
        print(
            f"{learned_metrics.policy_name:<20}"
            f"{learned_metrics.phase_name:<10}"
            f"{learned_metrics.mean_utility:>10.3f}"
            f"{learned_metrics.risk_weighted_utility:>10.3f}"
            f"{learned_metrics.accuracy:>10.3f}"
            f"{learned_metrics.questions_per_episode:>10.3f}"
            f"{regret:>10.3f}"
        )

    print()
    switch_text = "not reached" if switch_lag is None else str(switch_lag)
    print(f"Phase 2 switch lag for signature F -> ask_mode: {switch_text} episodes")
    print(
        "Greedy first actions after phase 1:"
        f" D0={first_actions_phase1['D0']},"
        f" D1={first_actions_phase1['D1']},"
        f" F={first_actions_phase1['F']},"
        f" R={first_actions_phase1['R']},"
        f" M={first_actions_phase1['M']}"
    )
    print(
        "Greedy first actions after phase 2:"
        f" D0={first_actions_phase2['D0']},"
        f" D1={first_actions_phase2['D1']},"
        f" F={first_actions_phase2['F']},"
        f" R={first_actions_phase2['R']},"
        f" M={first_actions_phase2['M']}"
    )


def main() -> None:
    config = parse_args()
    q_table_phase1, q_table_phase2, switch_lag = train_q_policy(config)

    learned_phase1 = evaluate_policy(
        q_table=q_table_phase1,
        phase=1,
        episodes=config.eval_episodes,
        rng=random.Random(config.seed + 101),
        config=config,
    )
    learned_phase2 = evaluate_policy(
        q_table=q_table_phase2,
        phase=2,
        episodes=config.eval_episodes,
        rng=random.Random(config.seed + 202),
        config=config,
    )
    oracle_phase1 = evaluate_oracle(
        phase=1,
        episodes=config.eval_episodes,
        rng=random.Random(config.seed + 101),
        config=config,
    )
    oracle_phase2 = evaluate_oracle(
        phase=2,
        episodes=config.eval_episodes,
        rng=random.Random(config.seed + 202),
        config=config,
    )
    first_actions_phase1 = {
        signature: first_action_for_signature(q_table_phase1, signature, config)
        for signature in ("D0", "D1", "F", "R", "M")
    }
    first_actions_phase2 = {
        signature: first_action_for_signature(q_table_phase2, signature, config)
        for signature in ("D0", "D1", "F", "R", "M")
    }

    print_report(
        config=config,
        learned_phase1=learned_phase1,
        learned_phase2=learned_phase2,
        oracle_phase1=oracle_phase1,
        oracle_phase2=oracle_phase2,
        switch_lag=switch_lag,
        first_actions_phase1=first_actions_phase1,
        first_actions_phase2=first_actions_phase2,
    )


if __name__ == "__main__":
    main()
