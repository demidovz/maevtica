from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.environments.debugging import DebuggingModeShiftEnvironment
from epistemic_engine.policies.baselines import CheapestQuestionPolicy, RandomQuestionPolicy
from epistemic_engine.policies.hybrid_memory import HybridMemoryPolicy
from epistemic_engine.policies.mode_memory import ModeMemoryPolicy
from epistemic_engine.policies.question_type_memory import QuestionTypeMemoryPolicy
from epistemic_engine.policies.switch_memory import SwitchAwareHybridMemoryPolicy
from epistemic_engine.questions.policy import InformationGainPolicy
from epistemic_engine.runner.run_debugging_benchmark import print_table, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark для debugging mode-shift среды."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [
        InformationGainPolicy(),
        QuestionTypeMemoryPolicy(),
        ModeMemoryPolicy(),
        HybridMemoryPolicy(),
        SwitchAwareHybridMemoryPolicy(
            mode_strength=0.02,
            type_bonus=0.05,
            switch_bonus=0.2,
        ),
        CheapestQuestionPolicy(),
        RandomQuestionPolicy(seed=args.seed + 2000),
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
                environment_factory=DebuggingModeShiftEnvironment,
                environment_kwargs={"shift_after_step": 2},
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
        f"Environment: mode_shift(step=2,min_stop=3) | "
        f"confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
