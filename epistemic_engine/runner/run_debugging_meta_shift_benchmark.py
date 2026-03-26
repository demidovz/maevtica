from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.environments.debugging import (
    DebuggingAmbiguousShiftEnvironment,
    DebuggingQuestionValueShiftEnvironment,
)
from epistemic_engine.policies.hybrid_memory import HybridMemoryPolicy
from epistemic_engine.policies.switch_memory import (
    AdaptiveShiftMemoryPolicy,
    LatentAdaptiveShiftMemoryPolicy,
    PersistentShiftMemoryPolicy,
    SwitchAwareHybridMemoryPolicy,
)
from epistemic_engine.runner.run_debugging_benchmark import print_table, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark для смешанной shift-среды: часть эпизодов с ложными тревогами, часть с чистым shift."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--ambiguous-share", type=float, default=0.5)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    return parser.parse_args()


def select_environment(
    episode_no: int,
    ambiguous_share: float,
    false_alarm_length: int,
):
    bucket = int(round(ambiguous_share * 10))
    if episode_no % 10 < bucket:
        return (
            DebuggingAmbiguousShiftEnvironment,
            {
                "shift_after_step": 2,
                "shift_probability": 0.5,
                "false_alarm_length": false_alarm_length,
            },
            4,
            "ambiguous_shift",
        )

    return (
        DebuggingQuestionValueShiftEnvironment,
        {"shift_after_step": 2},
        3,
        "question_value_shift",
    )


def main() -> None:
    args = parse_args()
    policies = [
        HybridMemoryPolicy(),
        SwitchAwareHybridMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
        ),
        PersistentShiftMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
            recent_window=3,
            anomaly_scale=1.0,
            persistence_scale=0.85,
            rebound_penalty=0.60,
            anomaly_weight=0.30,
            persistence_weight=0.70,
            streak_bonus=0.35,
        ),
        AdaptiveShiftMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
            recent_window=3,
            anomaly_scale=1.0,
            persistence_scale=0.85,
            rebound_penalty=0.60,
            anomaly_weight=0.30,
            persistence_weight=0.70,
            streak_bonus=0.35,
            false_alarm_scale=1.20,
            neutral_gate=0.50,
            min_aggressive_gate=0.0,
        ),
        LatentAdaptiveShiftMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
            recent_window=3,
            anomaly_scale=1.0,
            persistence_scale=0.85,
            rebound_penalty=0.60,
            streak_bonus=0.35,
            false_alarm_scale=1.20,
            neutral_gate=0.50,
            min_aggressive_gate=0.0,
        ),
    ]
    rows: list[dict[str, object]] = []

    for policy in policies:
        episodes = []
        scenario_counts = {"ambiguous_shift": 0, "question_value_shift": 0}
        for episode_no in range(args.episodes):
            environment_factory, environment_kwargs, min_stop, scenario_label = select_environment(
                episode_no,
                args.ambiguous_share,
                args.false_alarm_length,
            )
            scenario_counts[scenario_label] += 1
            episodes.append(
                run_episode(
                    policy,
                    seed=args.seed + episode_no,
                    confidence_threshold=args.confidence_threshold,
                    max_cost=args.max_cost,
                    max_steps=args.max_steps,
                    environment_factory=environment_factory,
                    environment_kwargs=environment_kwargs,
                    min_steps_before_stop=min_stop,
                )
            )
        summary = summarize(policy.policy_name, episodes)
        rows.append(
            {
                "policy": summary.policy_name,
                "accuracy": round(summary.accuracy, 3),
                "mean_steps": round(summary.mean_steps, 3),
                "mean_cost": round(summary.mean_cost, 3),
                "mean_conf": round(summary.mean_final_confidence, 3),
                "mean_utility": round(summary.mean_utility, 3),
                "budget_stop": round(summary.budget_stop_rate, 3),
                "step_stop": round(summary.step_stop_rate, 3),
            }
        )

    print(f"Episodes: {args.episodes}")
    print(
        f"Environment mix: ambiguous_share={args.ambiguous_share}, "
        f"false_alarm={args.false_alarm_length} | "
        f"confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
