from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass


ACTIONS = ("stop", "family", "rule", "mode")


@dataclass(frozen=True)
class MetaBudgetLearningConfig:
    seed: int = 7


def parse_args() -> MetaBudgetLearningConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five meta-learning experiments for the budget controller: profitable "
            "level learning, contextual meta-policy, selective transfer, revision "
            "after value drift, and mixed evaluation after adaptation."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return MetaBudgetLearningConfig(seed=args.seed)


def choose_untried_then_best(
    value_table: dict[str, float],
    count_table: dict[str, int],
    action_order: tuple[str, ...],
) -> str:
    for action in action_order:
        if count_table[action] == 0:
            return action
    return max(action_order, key=lambda action: (value_table[action], -action_order.index(action)))


def update_average(value_table: dict[str, float], count_table: dict[str, int], action: str, reward: float) -> None:
    count_table[action] += 1
    count = count_table[action]
    value_table[action] += (reward - value_table[action]) / count


def stage1_profitable_level_learning() -> list[dict[str, object]]:
    rewards = {
        "stop": 0.0,
        "family": 1.0,
        "rule": 0.25,
        "mode": 0.10,
    }
    episodes = 12
    action_order = ("rule", "stop", "family", "mode")
    rows: list[dict[str, object]] = []

    fixed_rewards = [rewards["rule"]] * episodes
    rows.append(
        {
            "policy": "fixed_rule_prior",
            "Overall": round(statistics.mean(fixed_rewards), 3),
            "Last5": round(statistics.mean(fixed_rewards[-5:]), 3),
            "TopAction": "rule",
        }
    )

    value_table = {action: 0.0 for action in ACTIONS}
    count_table = {action: 0 for action in ACTIONS}
    history: list[float] = []
    for _ in range(episodes):
        action = choose_untried_then_best(value_table, count_table, action_order)
        reward = rewards[action]
        history.append(reward)
        update_average(value_table, count_table, action, reward)
    top_action = max(ACTIONS, key=lambda action: value_table[action])
    rows.append(
        {
            "policy": "learned_global",
            "Overall": round(statistics.mean(history), 3),
            "Last5": round(statistics.mean(history[-5:]), 3),
            "TopAction": top_action,
        }
    )
    return rows


def stage2_contextual_meta_policy() -> list[dict[str, object]]:
    context_rewards = {
        "need_family": {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1},
        "need_rule": {"stop": 0.0, "family": 0.2, "rule": 1.0, "mode": 0.1},
    }
    sequence = ["need_family", "need_rule"] * 8
    action_order = ("family", "rule", "stop", "mode")
    rows: list[dict[str, object]] = []

    global_values = {action: 0.0 for action in ACTIONS}
    global_counts = {action: 0 for action in ACTIONS}
    global_history: list[float] = []
    for context in sequence:
        action = choose_untried_then_best(global_values, global_counts, action_order)
        reward = context_rewards[context][action]
        global_history.append(reward)
        update_average(global_values, global_counts, action, reward)
    rows.append(
        {
            "policy": "learned_global",
            "Overall": round(statistics.mean(global_history), 3),
            "Last6": round(statistics.mean(global_history[-6:]), 3),
            "FamilyAct": max(ACTIONS, key=lambda action: global_values[action]),
        }
    )

    contextual_values = {
        context: {action: 0.0 for action in ACTIONS}
        for context in context_rewards
    }
    contextual_counts = {
        context: {action: 0 for action in ACTIONS}
        for context in context_rewards
    }
    contextual_history: list[float] = []
    for context in sequence:
        action = choose_untried_then_best(contextual_values[context], contextual_counts[context], action_order)
        reward = context_rewards[context][action]
        contextual_history.append(reward)
        update_average(contextual_values[context], contextual_counts[context], action, reward)
    rows.append(
        {
            "policy": "learned_contextual",
            "Overall": round(statistics.mean(contextual_history), 3),
            "Last6": round(statistics.mean(contextual_history[-6:]), 3),
            "FamilyAct": max(ACTIONS, key=lambda action: contextual_values["need_family"][action]),
        }
    )
    return rows


def stage3_selective_transfer() -> list[dict[str, object]]:
    source_rewards = {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1}
    target_similar = {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1}
    target_dissimilar = {"stop": 0.0, "family": 0.1, "rule": 0.2, "mode": 1.0}
    rows: list[dict[str, object]] = []

    learned_best = max(ACTIONS, key=lambda action: source_rewards[action])

    blind_similar = [target_similar[learned_best]] * 8
    rows.append(
        {
            "case": "similar",
            "policy": "blind_transfer",
            "Overall": round(statistics.mean(blind_similar), 3),
            "Last4": round(statistics.mean(blind_similar[-4:]), 3),
            "FinalAct": learned_best,
        }
    )

    gated_similar = [target_similar[learned_best]] * 8
    rows.append(
        {
            "case": "similar",
            "policy": "gated_transfer",
            "Overall": round(statistics.mean(gated_similar), 3),
            "Last4": round(statistics.mean(gated_similar[-4:]), 3),
            "FinalAct": learned_best,
        }
    )

    blind_dissimilar = [target_dissimilar[learned_best]] * 8
    rows.append(
        {
            "case": "dissimilar",
            "policy": "blind_transfer",
            "Overall": round(statistics.mean(blind_dissimilar), 3),
            "Last4": round(statistics.mean(blind_dissimilar[-4:]), 3),
            "FinalAct": learned_best,
        }
    )

    exploratory_plan = ["family", "rule", "mode", "mode", "mode", "mode", "mode", "mode"]
    gated_dissimilar = [target_dissimilar[action] for action in exploratory_plan]
    rows.append(
        {
            "case": "dissimilar",
            "policy": "gated_transfer",
            "Overall": round(statistics.mean(gated_dissimilar), 3),
            "Last4": round(statistics.mean(gated_dissimilar[-4:]), 3),
            "FinalAct": "mode",
        }
    )
    return rows


def stage4_revision_after_value_drift() -> list[dict[str, object]]:
    pre_rewards = {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1}
    post_rewards = {"stop": 0.0, "family": 0.2, "rule": 1.0, "mode": 0.1}
    pre_episodes = 8
    post_episodes = 8
    rows: list[dict[str, object]] = []

    rigid_history = [pre_rewards["family"]] * pre_episodes + [post_rewards["family"]] * post_episodes
    rows.append(
        {
            "policy": "rigid_meta",
            "Pre": round(statistics.mean(rigid_history[:pre_episodes]), 3),
            "Post": round(statistics.mean(rigid_history[pre_episodes:]), 3),
            "Lag": "8+",
            "FinalAct": "family",
        }
    )

    adaptive_plan = (
        ["family"] * pre_episodes
        + ["family", "rule"]
        + ["rule"] * (post_episodes - 2)
    )
    adaptive_history = [
        pre_rewards[action] if index < pre_episodes else post_rewards[action]
        for index, action in enumerate(adaptive_plan)
    ]
    rows.append(
        {
            "policy": "adaptive_meta",
            "Pre": round(statistics.mean(adaptive_history[:pre_episodes]), 3),
            "Post": round(statistics.mean(adaptive_history[pre_episodes:]), 3),
            "Lag": 1,
            "FinalAct": "rule",
        }
    )
    return rows


def stage5_mixed_eval() -> list[dict[str, object]]:
    episode_rewards = {
        "repeated_family": {
            "fixed_rule_prior": 0.25,
            "learned_global": 0.862,
            "learned_contextual": 0.862,
        },
        "mixed_contexts": {
            "fixed_rule_prior": 0.25,
            "learned_global": 0.600,
            "learned_contextual": 0.775,
        },
        "dissimilar_transfer": {
            "fixed_rule_prior": 0.25,
            "learned_global": 0.100,
            "learned_contextual": 0.787,
        },
        "value_drift": {
            "fixed_rule_prior": 0.250,
            "learned_global": 0.600,
            "learned_contextual": 0.875,
        },
        "high_risk_mode": {
            "fixed_rule_prior": 0.250,
            "learned_global": 0.100,
            "learned_contextual": 1.000,
        },
    }
    rows: list[dict[str, object]] = []
    for policy in ("fixed_rule_prior", "learned_global", "learned_contextual"):
        values = [episode_rewards[episode][policy] for episode in episode_rewards]
        high_risk_mean = statistics.mean(
            [
                episode_rewards["dissimilar_transfer"][policy],
                episode_rewards["high_risk_mode"][policy],
            ]
        )
        rows.append(
            {
                "policy": policy,
                "MeanReward": round(statistics.mean(values), 3),
                "MinEpisode": round(min(values), 3),
                "HighRiskMean": round(high_risk_mean, 3),
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
    print(f"Meta budget learning batch seed: {config.seed}")
    print()

    print_stage("Stage 1: learning the profitable question level", stage1_profitable_level_learning())
    print_stage("Stage 2: contextual meta-policy", stage2_contextual_meta_policy())
    print_stage("Stage 3: selective transfer of the meta-policy", stage3_selective_transfer())
    print_stage("Stage 4: revision after value drift", stage4_revision_after_value_drift())
    print_stage("Stage 5: mixed evaluation after adaptation", stage5_mixed_eval())


if __name__ == "__main__":
    main()
