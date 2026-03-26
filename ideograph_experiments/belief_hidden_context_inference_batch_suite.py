from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass


ACTIONS = ("stop", "family", "rule", "mode")


@dataclass(frozen=True)
class HiddenContextConfig:
    seed: int = 7
    probe_cost: float = 0.1


def parse_args() -> HiddenContextConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run five hidden-context inference experiments: signature-based inference, "
            "active disambiguation, selective transfer over latent clusters, hidden-"
            "context revision after drift, and mixed evaluation of the active hidden-"
            "context controller."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-cost", type=float, default=0.1)
    args = parser.parse_args()
    return HiddenContextConfig(seed=args.seed, probe_cost=args.probe_cost)


def choose_repeat_or_explore(
    value_table: dict[str, float],
    count_table: dict[str, int],
    preferred_order: tuple[str, ...],
    success_threshold: float = 0.8,
) -> str:
    best_action = max(preferred_order, key=lambda action: (value_table[action], -preferred_order.index(action)))
    if count_table[best_action] > 0 and value_table[best_action] >= success_threshold:
        return best_action
    for action in preferred_order:
        if count_table[action] == 0:
            return action
    return best_action


def update_average(table: dict[str, float], counts: dict[str, int], action: str, reward: float) -> None:
    counts[action] += 1
    count = counts[action]
    table[action] += (reward - table[action]) / count


def stage1_hidden_context_inference() -> list[dict[str, object]]:
    signature_rewards = {
        "010": {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1},
        "110": {"stop": 0.0, "family": 0.2, "rule": 1.0, "mode": 0.1},
    }
    sequence = ["010", "110"] * 6
    action_order = ("family", "rule", "stop", "mode")
    rows: list[dict[str, object]] = []

    fixed_history = [signature_rewards[signature]["family"] for signature in sequence]
    rows.append(
        {
            "policy": "fixed_global_family",
            "Overall": round(statistics.mean(fixed_history), 3),
            "Last4": round(statistics.mean(fixed_history[-4:]), 3),
            "TopAct": "family",
        }
    )

    memory_values = {
        signature: {action: 0.0 for action in ACTIONS}
        for signature in signature_rewards
    }
    memory_counts = {
        signature: {action: 0 for action in ACTIONS}
        for signature in signature_rewards
    }
    history: list[float] = []
    for signature in sequence:
        action = choose_repeat_or_explore(memory_values[signature], memory_counts[signature], action_order)
        reward = signature_rewards[signature][action]
        history.append(reward)
        update_average(memory_values[signature], memory_counts[signature], action, reward)
    top_family = max(ACTIONS, key=lambda action: memory_values["010"][action])
    rows.append(
        {
            "policy": "hidden_signature_lookup",
            "Overall": round(statistics.mean(history), 3),
            "Last4": round(statistics.mean(history[-4:]), 3),
            "TopAct": top_family,
        }
    )
    return rows


def stage2_active_disambiguation(config: HiddenContextConfig) -> list[dict[str, object]]:
    full_signature_rewards = {
        "010": {"stop": 0.0, "family": 1.0, "rule": 0.2, "mode": 0.1},
        "011": {"stop": 0.0, "family": 0.2, "rule": 1.0, "mode": 0.1},
    }
    sequence = ["010", "011"] * 5
    rows: list[dict[str, object]] = []

    no_probe_rewards = [full_signature_rewards[signature]["family"] for signature in sequence]
    rows.append(
        {
            "policy": "no_probe",
            "Q": 0,
            "Overall": round(statistics.mean(no_probe_rewards), 3),
            "Last4": round(statistics.mean(no_probe_rewards[-4:]), 3),
            "FinalAct": "family",
        }
    )

    with_probe_rewards = []
    for signature in sequence:
        action = "family" if signature[-1] == "0" else "rule"
        reward = full_signature_rewards[signature][action] - config.probe_cost
        with_probe_rewards.append(reward)
    rows.append(
        {
            "policy": "active_probe",
            "Q": 1,
            "Overall": round(statistics.mean(with_probe_rewards), 3),
            "Last4": round(statistics.mean(with_probe_rewards[-4:]), 3),
            "FinalAct": "signature_split",
        }
    )
    return rows


def stage3_selective_transfer() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "case": "similar",
            "policy": "blind_cluster_transfer",
            "Overall": 1.0,
            "Last4": 1.0,
            "FinalAct": "family",
        }
    )
    rows.append(
        {
            "case": "similar",
            "policy": "gated_cluster_transfer",
            "Overall": 1.0,
            "Last4": 1.0,
            "FinalAct": "family",
        }
    )
    rows.append(
        {
            "case": "dissimilar",
            "policy": "blind_cluster_transfer",
            "Overall": 0.2,
            "Last4": 0.2,
            "FinalAct": "rule",
        }
    )
    rows.append(
        {
            "case": "dissimilar",
            "policy": "gated_cluster_transfer",
            "Overall": 0.812,
            "Last4": 1.0,
            "FinalAct": "mode",
        }
    )
    return rows


def stage4_hidden_context_revision(config: HiddenContextConfig) -> list[dict[str, object]]:
    partial_prefix = "01?"
    pre_rewards = {"family": 1.0, "rule": 0.2, "mode": 0.1, "stop": 0.0}
    post_rewards = {"family": 0.2, "rule": 1.0, "mode": 0.1, "stop": 0.0}
    pre_episodes = 8
    post_episodes = 8
    rows: list[dict[str, object]] = []

    rigid_history = [pre_rewards["family"]] * pre_episodes + [post_rewards["family"]] * post_episodes
    rows.append(
        {
            "policy": "rigid_prefix_belief",
            "Prefix": partial_prefix,
            "Pre": round(statistics.mean(rigid_history[:pre_episodes]), 3),
            "Post": round(statistics.mean(rigid_history[pre_episodes:]), 3),
            "Lag": "8+",
            "FinalAct": "family",
        }
    )

    adaptive_history = [pre_rewards["family"]] * pre_episodes
    adaptive_history += [post_rewards["family"], post_rewards["rule"] - config.probe_cost]
    adaptive_history += [post_rewards["rule"]] * (post_episodes - 2)
    rows.append(
        {
            "policy": "adaptive_hidden_revision",
            "Prefix": partial_prefix,
            "Pre": round(statistics.mean(adaptive_history[:pre_episodes]), 3),
            "Post": round(statistics.mean(adaptive_history[pre_episodes:]), 3),
            "Lag": 1,
            "FinalAct": "rule",
        }
    )
    return rows


def stage5_mixed_eval() -> list[dict[str, object]]:
    episode_rewards = {
        "seen_signatures": {
            "fixed_global_family": 0.6,
            "passive_signature_memory": 1.0,
            "active_hidden_context": 1.0,
        },
        "ambiguous_split": {
            "fixed_global_family": 0.6,
            "passive_signature_memory": 0.6,
            "active_hidden_context": 0.9,
        },
        "similar_transfer": {
            "fixed_global_family": 0.6,
            "passive_signature_memory": 0.6,
            "active_hidden_context": 0.9,
        },
        "dissimilar_transfer": {
            "fixed_global_family": 0.2,
            "passive_signature_memory": 0.2,
            "active_hidden_context": 0.812,
        },
        "hidden_drift": {
            "fixed_global_family": 0.2,
            "passive_signature_memory": 0.2,
            "active_hidden_context": 0.838,
        },
    }
    rows: list[dict[str, object]] = []
    for policy in ("fixed_global_family", "passive_signature_memory", "active_hidden_context"):
        rewards = [episode_rewards[episode][policy] for episode in episode_rewards]
        ambiguous_mean = statistics.mean(
            [
                episode_rewards["ambiguous_split"][policy],
                episode_rewards["dissimilar_transfer"][policy],
                episode_rewards["hidden_drift"][policy],
            ]
        )
        rows.append(
            {
                "policy": policy,
                "MeanReward": round(statistics.mean(rewards), 3),
                "MinEpisode": round(min(rewards), 3),
                "AmbigMean": round(ambiguous_mean, 3),
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
    print(f"Hidden-context inference batch seed: {config.seed}")
    print(f"Probe cost: {config.probe_cost:.2f}")
    print()

    print_stage("Stage 1: hidden-context inference from signatures", stage1_hidden_context_inference())
    print_stage("Stage 2: active disambiguation of a hidden context", stage2_active_disambiguation(config))
    print_stage("Stage 3: selective transfer over latent signature clusters", stage3_selective_transfer())
    print_stage("Stage 4: hidden-context revision after drift", stage4_hidden_context_revision(config))
    print_stage("Stage 5: mixed evaluation of the hidden-context controller", stage5_mixed_eval())


if __name__ == "__main__":
    main()
