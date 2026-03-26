from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grounding_semantics_test as gst
import intrinsic_reward_test as irt
import memory_transfer_test as mtt
import question_uncertainty_test as qut
import self_reinforcing_symbolism_test as srs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all ideograph experiments, save a unified summary, "
            "and generate comparison plots."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for summary files and PNG charts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override applied to all experiment configs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=40,
        help="Rolling window for episode-level self-bootstrap dynamics.",
    )
    return parser.parse_args()


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    result: list[float] = []
    cumulative = 0.0
    history: list[float] = []
    for value in values:
        history.append(value)
        cumulative += value
        if len(history) > window:
            cumulative -= history[-window - 1]
        active_window = min(len(history), window)
        result.append(cumulative / active_window)
    return result


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def make_row(
    experiment: str,
    policy: str,
    accuracy: float,
    average_questions_asked: float | None,
    average_final_entropy: float | None,
    information_gain_per_question: float | None,
    average_true_world_probability: float | None,
    average_intrinsic_value: float | None = None,
    average_confidence: float | None = None,
    overconfidence_gap: float | None = None,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "policy": policy,
        "accuracy": accuracy,
        "average_questions_asked": average_questions_asked,
        "average_final_entropy": average_final_entropy,
        "information_gain_per_question": information_gain_per_question,
        "average_true_world_probability": average_true_world_probability,
        "average_intrinsic_value": average_intrinsic_value,
        "average_confidence": average_confidence,
        "overconfidence_gap": overconfidence_gap,
    }


def run_question_uncertainty(seed_override: int | None) -> tuple[qut.ExperimentConfig, list[Any], list[dict[str, Any]]]:
    config = qut.ExperimentConfig(seed=seed_override if seed_override is not None else 7)
    worlds = qut.build_worlds(config.feature_count)
    prior = qut.build_prior(worlds)
    episodes = qut.generate_episodes(config, worlds, prior)
    results = [
        qut.run_policy("no_questions", config.seed + 11, config, worlds, prior, episodes),
        qut.run_policy("random_questions", config.seed + 23, config, worlds, prior, episodes),
        qut.run_policy("entropy_questions", config.seed + 37, config, worlds, prior, episodes),
    ]
    rows = []
    for result in results:
        questions = 0.0 if result.name == "no_questions" else float(config.question_budget)
        rows.append(
            make_row(
                experiment="question_uncertainty",
                policy=result.name,
                accuracy=result.accuracy,
                average_questions_asked=questions,
                average_final_entropy=result.average_final_entropy,
                information_gain_per_question=result.average_information_gain_per_question,
                average_true_world_probability=result.average_true_world_probability,
            )
        )
    return config, results, rows


def run_intrinsic_reward(seed_override: int | None) -> tuple[irt.IntrinsicRewardConfig, list[Any], list[dict[str, Any]]]:
    config = irt.IntrinsicRewardConfig(seed=seed_override if seed_override is not None else 7)
    worlds = irt.build_worlds(config.feature_count)
    prior = irt.build_prior(worlds)
    results = [
        irt.run_policy("no_questions", config.seed + 11, config, worlds, prior),
        irt.run_policy("random_budgeted", config.seed + 23, config, worlds, prior),
        irt.run_policy("entropy_budgeted", config.seed + 37, config, worlds, prior),
        irt.run_policy("entropy_cost_aware", config.seed + 41, config, worlds, prior),
    ]
    rows = [
        make_row(
            experiment="intrinsic_reward",
            policy=result.name,
            accuracy=result.accuracy,
            average_questions_asked=result.average_questions_asked,
            average_final_entropy=result.average_final_entropy,
            information_gain_per_question=result.information_gain_per_question,
            average_true_world_probability=result.average_true_world_probability,
            average_intrinsic_value=result.average_intrinsic_value,
        )
        for result in results
    ]
    return config, results, rows


def run_memory_transfer(seed_override: int | None) -> tuple[mtt.MemoryTransferConfig, list[Any], list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    config = mtt.MemoryTransferConfig(seed=seed_override if seed_override is not None else 7)
    worlds = mtt.build_worlds(config.feature_count)
    static_prior = mtt.build_prior(worlds)
    episodes = mtt.generate_context_episodes(config, worlds)
    policy_names = [
        "static_budgeted",
        "memory_budgeted",
        "static_cost_aware",
        "memory_cost_aware",
    ]
    results = [
        mtt.run_policy(policy_name, config, worlds, static_prior, episodes)
        for policy_name in policy_names
    ]
    rows = [
        make_row(
            experiment="memory_transfer",
            policy=result.name,
            accuracy=result.accuracy,
            average_questions_asked=result.average_questions_asked,
            average_final_entropy=result.average_final_entropy,
            information_gain_per_question=result.information_gain_per_question,
            average_true_world_probability=result.average_true_world_probability,
            average_intrinsic_value=result.average_intrinsic_value,
        )
        for result in results
    ]

    traces = {
        "static_cost_aware": memory_transfer_block_trace(
            config, worlds, static_prior, episodes, "static_cost_aware"
        ),
        "memory_cost_aware": memory_transfer_block_trace(
            config, worlds, static_prior, episodes, "memory_cost_aware"
        ),
    }
    return config, results, rows, traces


def memory_transfer_block_trace(
    config: mtt.MemoryTransferConfig,
    worlds: list[qut.World],
    static_prior: list[float],
    episodes: list[mtt.Episode],
    policy_name: str,
) -> dict[str, list[float]]:
    accuracy_buckets: list[list[float]] = [[] for _ in range(config.block_size)]
    question_buckets: list[list[float]] = [[] for _ in range(config.block_size)]
    memory_counts = [config.memory_prior_strength * probability for probability in static_prior]

    for episode in episodes:
        prior = (
            mtt.normalize(memory_counts)
            if policy_name.startswith("memory_")
            else list(static_prior)
        )
        posterior = list(prior)
        asked_features: set[int] = set()
        episode_questions = 0

        for feature_index, observed_value in episode.observations:
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_features.add(feature_index)

        for _ in range(config.question_budget):
            if len(asked_features) >= config.feature_count:
                break
            feature_index, expected_gain = mtt.best_entropy_question_with_gain(
                posterior, worlds, asked_features
            )
            if policy_name.endswith("cost_aware") and expected_gain <= config.question_cost:
                break
            asked_features.add(feature_index)
            answer = worlds[episode.true_world_id][feature_index]
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=feature_index,
                observed_value=answer,
                reliability=1.0,
            )
            episode_questions += 1

        accuracy_buckets[episode.block_position].append(
            1.0 if qut.argmax(posterior) == episode.true_world_id else 0.0
        )
        question_buckets[episode.block_position].append(float(episode_questions))

        if policy_name.startswith("memory_"):
            memory_counts = [config.memory_decay * count for count in memory_counts]
            memory_counts[episode.true_world_id] += 1.0

    return {
        "accuracy": [statistics.mean(bucket) for bucket in accuracy_buckets],
        "questions": [statistics.mean(bucket) for bucket in question_buckets],
    }


def run_grounding(seed_override: int | None) -> tuple[gst.GroundingSemanticsConfig, list[Any], list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    config = gst.GroundingSemanticsConfig(seed=seed_override if seed_override is not None else 7)
    worlds = gst.build_worlds(config.feature_count)
    static_prior = gst.build_prior(worlds)
    episodes = gst.generate_episodes(config, worlds)
    specs = [
        ("grounded_static", "grounded", False),
        ("grounded_memory", "grounded", True),
        ("symbolic_static", "symbolic", False),
        ("symbolic_memory", "symbolic", True),
    ]
    results = [
        gst.run_policy(name, condition, use_memory, config, worlds, static_prior, episodes)
        for name, condition, use_memory in specs
    ]
    rows = [
        make_row(
            experiment="grounding_semantics",
            policy=result.name,
            accuracy=result.accuracy,
            average_questions_asked=result.average_questions_asked,
            average_final_entropy=result.average_final_entropy,
            information_gain_per_question=result.information_gain_per_question,
            average_true_world_probability=result.average_true_world_probability,
            average_intrinsic_value=result.average_intrinsic_value,
        )
        for result in results
    ]

    traces = {
        "grounded_memory": grounding_block_trace(
            config, worlds, static_prior, episodes, "grounded", True
        ),
        "symbolic_memory": grounding_block_trace(
            config, worlds, static_prior, episodes, "symbolic", True
        ),
    }
    return config, results, rows, traces


def grounding_block_trace(
    config: gst.GroundingSemanticsConfig,
    worlds: list[qut.World],
    static_prior: list[float],
    episodes: list[gst.Episode],
    condition: str,
    use_memory: bool,
) -> dict[str, list[float]]:
    accuracy_buckets: list[list[float]] = [[] for _ in range(config.block_size)]
    question_buckets: list[list[float]] = [[] for _ in range(config.block_size)]
    memory_counts = [config.memory_prior_strength * probability for probability in static_prior]

    for episode in episodes:
        prior = gst.normalize(memory_counts) if use_memory else list(static_prior)
        posterior = list(prior)
        asked_channels: set[int] = set()
        episode_questions = 0
        observations = (
            episode.grounded_observations
            if condition == "grounded"
            else episode.symbolic_observations
        )

        for channel, observed_value in observations:
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_channels.add(channel)

        for _ in range(config.question_budget):
            if len(asked_channels) >= config.feature_count:
                break
            channel, expected_gain = gst.best_entropy_question_with_gain(
                posterior, worlds, asked_channels
            )
            if expected_gain <= config.question_cost:
                break
            actual_feature = (
                channel
                if condition == "grounded"
                else episode.symbolic_permutation[channel]
            )
            answer = worlds[episode.true_world_id][actual_feature]
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=answer,
                reliability=1.0,
            )
            asked_channels.add(channel)
            episode_questions += 1

        accuracy_buckets[episode.block_position].append(
            1.0 if qut.argmax(posterior) == episode.true_world_id else 0.0
        )
        question_buckets[episode.block_position].append(float(episode_questions))

        if use_memory:
            memory_counts = [config.memory_decay * value for value in memory_counts]
            memory_counts[episode.true_world_id] += 1.0

    return {
        "accuracy": [statistics.mean(bucket) for bucket in accuracy_buckets],
        "questions": [statistics.mean(bucket) for bucket in question_buckets],
    }


def run_self_reinforcing(seed_override: int | None) -> tuple[srs.SelfReinforcingConfig, list[Any], list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    config = srs.SelfReinforcingConfig(seed=seed_override if seed_override is not None else 7)
    worlds = srs.build_worlds(config.feature_count)
    static_prior = srs.build_prior(worlds)
    episodes = srs.generate_episodes(config, worlds)
    specs = [
        ("grounded_feedback", "grounded_feedback"),
        ("symbolic_feedback", "symbolic_feedback"),
        ("symbolic_self_bootstrap", "symbolic_self_bootstrap"),
    ]
    results = [
        srs.run_policy(name, mode, config, worlds, static_prior, episodes)
        for name, mode in specs
    ]
    rows = [
        make_row(
            experiment="self_reinforcing_symbolism",
            policy=result.name,
            accuracy=result.accuracy,
            average_questions_asked=result.average_questions_asked,
            average_final_entropy=result.average_final_entropy,
            information_gain_per_question=result.information_gain_per_question,
            average_true_world_probability=result.average_true_world_probability,
            average_intrinsic_value=result.average_intrinsic_value,
            average_confidence=result.average_confidence,
            overconfidence_gap=result.overconfidence_gap,
        )
        for result in results
    ]
    traces = {
        name: self_reinforcing_episode_trace(config, worlds, static_prior, episodes, mode)
        for name, mode in specs
    }
    return config, results, rows, traces


def self_reinforcing_episode_trace(
    config: srs.SelfReinforcingConfig,
    worlds: list[qut.World],
    static_prior: list[float],
    episodes: list[srs.Episode],
    mode: str,
) -> dict[str, list[float]]:
    memory_counts = [config.memory_prior_strength * probability for probability in static_prior]
    accuracy: list[float] = []
    confidence: list[float] = []
    entropy_values: list[float] = []
    questions: list[float] = []

    for episode in episodes:
        prior = srs.normalize(memory_counts)
        posterior = list(prior)
        asked_channels: set[int] = set()
        episode_questions = 0

        if mode == "grounded_feedback":
            observations = episode.grounded_observations
            symbolic = False
        else:
            observations = episode.symbolic_observations
            symbolic = True

        for channel, observed_value in observations:
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=observed_value,
                reliability=config.sensor_reliability,
            )
            asked_channels.add(channel)

        for _ in range(config.question_budget):
            if len(asked_channels) >= config.feature_count:
                break
            channel, expected_gain = srs.best_entropy_question_with_gain(
                posterior, worlds, asked_channels
            )
            if expected_gain <= config.question_cost:
                break
            actual_feature = (
                channel
                if not symbolic
                else episode.symbolic_permutation[channel]
            )
            answer = worlds[episode.true_world_id][actual_feature]
            posterior = qut.update_posterior(
                posterior=posterior,
                worlds=worlds,
                feature_index=channel,
                observed_value=answer,
                reliability=1.0,
            )
            asked_channels.add(channel)
            episode_questions += 1

        guess = qut.argmax(posterior)
        is_correct = 1.0 if guess == episode.true_world_id else 0.0
        accuracy.append(is_correct)
        confidence.append(max(posterior))
        entropy_values.append(qut.entropy(posterior))
        questions.append(float(episode_questions))

        memory_counts = [config.memory_decay * value for value in memory_counts]
        if mode in {"grounded_feedback", "symbolic_feedback"}:
            memory_counts[episode.true_world_id] += 1.0
        else:
            memory_counts[guess] += 1.0

    return {
        "accuracy": accuracy,
        "confidence": confidence,
        "entropy": entropy_values,
        "questions": questions,
    }


def plot_overview(rows: list[dict[str, Any]], output_path: Path, dpi: int) -> None:
    ordered_rows = rows[:]
    labels = [f"{row['experiment']}\n{row['policy']}" for row in ordered_rows]
    x = list(range(len(ordered_rows)))

    def metric(name: str, fallback: float = 0.0) -> list[float]:
        values = []
        for row in ordered_rows:
            value = row.get(name)
            values.append(fallback if value is None else float(value))
        return values

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    charts = [
        ("Accuracy", "accuracy", False),
        ("Questions", "average_questions_asked", False),
        ("Final Entropy", "average_final_entropy", False),
        ("Intrinsic Value", "average_intrinsic_value", False),
    ]

    for axis, (title, column, _) in zip(axes.flatten(), charts):
        axis.bar(x, metric(column))
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Ideograph Experiments: Final Metrics Overview", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_question_entropy_trajectories(
    question_results: list[Any],
    intrinsic_results: list[Any],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for result in question_results:
        steps = list(range(len(result.entropy_trajectory)))
        axes[0].plot(steps, result.entropy_trajectory, marker="o", label=result.name)
    axes[0].set_title("Question Uncertainty: Entropy per Question Slot")
    axes[0].set_xlabel("Question slot")
    axes[0].set_ylabel("Entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for result in intrinsic_results:
        steps = list(range(len(result.entropy_trajectory)))
        axes[1].plot(steps, result.entropy_trajectory, marker="o", label=result.name)
    axes[1].set_title("Intrinsic Reward: Entropy per Question Slot")
    axes[1].set_xlabel("Question slot")
    axes[1].set_ylabel("Entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_dynamics(
    memory_traces: dict[str, dict[str, list[float]]],
    grounding_traces: dict[str, dict[str, list[float]]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def plot_trace_set(axis: Any, trace_set: dict[str, dict[str, list[float]]], metric_name: str, title: str) -> None:
        for name, trace in trace_set.items():
            positions = list(range(len(trace[metric_name])))
            axis.plot(positions, trace[metric_name], marker="o", label=name)
        axis.set_title(title)
        axis.set_xlabel("Position inside block")
        axis.grid(alpha=0.25)
        axis.legend()

    plot_trace_set(
        axes[0][0],
        memory_traces,
        "accuracy",
        "Memory Transfer: Accuracy by Block Position",
    )
    axes[0][0].set_ylabel("Accuracy")

    plot_trace_set(
        axes[0][1],
        memory_traces,
        "questions",
        "Memory Transfer: Questions by Block Position",
    )
    axes[0][1].set_ylabel("Questions")

    plot_trace_set(
        axes[1][0],
        grounding_traces,
        "accuracy",
        "Grounding Semantics: Accuracy by Block Position",
    )
    axes[1][0].set_ylabel("Accuracy")

    plot_trace_set(
        axes[1][1],
        grounding_traces,
        "questions",
        "Grounding Semantics: Questions by Block Position",
    )
    axes[1][1].set_ylabel("Questions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_self_reinforcing_dynamics(
    traces: dict[str, dict[str, list[float]]],
    output_path: Path,
    dpi: int,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    chart_specs = [
        ("accuracy", "Rolling Accuracy"),
        ("confidence", "Rolling Confidence"),
        ("entropy", "Rolling Entropy"),
        ("questions", "Rolling Questions"),
    ]

    for axis, (metric_name, title) in zip(axes.flatten(), chart_specs):
        for name, trace in traces.items():
            values = rolling_mean(trace[metric_name], rolling_window)
            axis.plot(values, label=name)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(
        f"Self-Reinforcing Symbolism Dynamics (rolling window = {rolling_window})",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "experiment",
        "policy",
        "accuracy",
        "average_questions_asked",
        "average_final_entropy",
        "information_gain_per_question",
        "average_true_world_probability",
        "average_intrinsic_value",
        "average_confidence",
        "overconfidence_gap",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(payload: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable(payload), handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    question_config, question_results, question_rows = run_question_uncertainty(args.seed)
    intrinsic_config, intrinsic_results, intrinsic_rows = run_intrinsic_reward(args.seed)
    memory_config, memory_results, memory_rows, memory_traces = run_memory_transfer(args.seed)
    grounding_config, grounding_results, grounding_rows, grounding_traces = run_grounding(args.seed)
    self_config, self_results, self_rows, self_traces = run_self_reinforcing(args.seed)

    rows = question_rows + intrinsic_rows + memory_rows + grounding_rows + self_rows
    rows.sort(key=lambda row: (row["experiment"], row["policy"]))

    chart_paths = {
        "overview": output_dir / "00_overview_final_metrics.png",
        "question_entropy": output_dir / "01_question_entropy_trajectories.png",
        "transfer": output_dir / "02_transfer_dynamics.png",
        "self_reinforcing": output_dir / "03_self_reinforcing_dynamics.png",
    }

    plot_overview(rows, chart_paths["overview"], args.dpi)
    plot_question_entropy_trajectories(
        question_results,
        intrinsic_results,
        chart_paths["question_entropy"],
        args.dpi,
    )
    plot_transfer_dynamics(
        memory_traces,
        grounding_traces,
        chart_paths["transfer"],
        args.dpi,
    )
    plot_self_reinforcing_dynamics(
        self_traces,
        chart_paths["self_reinforcing"],
        args.dpi,
        args.rolling_window,
    )

    summary_payload = {
        "seed_override": args.seed,
        "charts": {name: str(path) for name, path in chart_paths.items()},
        "configs": {
            "question_uncertainty": vars(question_config),
            "intrinsic_reward": vars(intrinsic_config),
            "memory_transfer": vars(memory_config),
            "grounding_semantics": vars(grounding_config),
            "self_reinforcing_symbolism": vars(self_config),
        },
        "rows": rows,
    }
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    write_summary_json(summary_payload, summary_json_path)
    write_summary_csv(rows, summary_csv_path)

    print(f"Saved summary JSON: {summary_json_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    for name, path in chart_paths.items():
        print(f"Saved chart [{name}]: {path}")


if __name__ == "__main__":
    main()
