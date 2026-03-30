from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.environments.artifact_debugging import (
    ArtifactDebuggingQuestionValueShiftEnvironment,
)
from epistemic_engine.policies.hybrid_memory import HybridMemoryPolicy
from epistemic_engine.policies.switch_memory import (
    AdaptiveShiftMemoryPolicy,
    LatentAdaptiveShiftMemoryPolicy,
)
from epistemic_engine.questions.policy import InformationGainPolicy
from epistemic_engine.runner.run_debugging_benchmark import print_table, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark for artifact-level question-value shift environment."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [
        InformationGainPolicy(),
        HybridMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
        ),
        AdaptiveShiftMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
            recent_window=3,
            anomaly_scale=0.95,
            persistence_scale=0.80,
            rebound_penalty=0.55,
            anomaly_weight=0.30,
            persistence_weight=0.70,
            streak_bonus=0.35,
            false_alarm_scale=1.10,
            neutral_gate=0.50,
            min_aggressive_gate=0.0,
        ),
        LatentAdaptiveShiftMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.20,
            recent_window=3,
            anomaly_scale=0.95,
            persistence_scale=0.80,
            rebound_penalty=0.55,
            streak_bonus=0.35,
            false_alarm_scale=1.10,
            neutral_gate=0.50,
            min_aggressive_gate=0.0,
        ),
    ]
    rows: list[dict[str, object]] = []

    for policy in policies:
        episodes = [
            run_episode(
                policy,
                seed=args.seed + episode_no,
                confidence_threshold=args.confidence_threshold,
                max_cost=args.max_cost,
                max_steps=args.max_steps,
                environment_factory=ArtifactDebuggingQuestionValueShiftEnvironment,
                environment_kwargs={"shift_after_step": args.shift_after_step},
                min_steps_before_stop=3,
            )
            for episode_no in range(args.episodes)
        ]
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
        f"Environment: artifact_question_value_shift | "
        f"confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}, "
        f"shift_after_step={args.shift_after_step}, min_stop=3"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
