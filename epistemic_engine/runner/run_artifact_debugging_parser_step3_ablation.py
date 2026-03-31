from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.beliefs.state import top_probability
from epistemic_engine.environments.artifact_debugging import (
    ArtifactDebuggingAmbiguousShiftEnvironment,
)
from epistemic_engine.policies.switch_memory import LatentAdaptiveShiftMemoryPolicy
from epistemic_engine.runner.run_debugging_benchmark import print_table, run_episode


class ParserStep3LatentPolicy(LatentAdaptiveShiftMemoryPolicy):
    def __init__(
        self,
        *,
        policy_suffix: str,
        step3_action_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.policy_name = f"information_gain+latent_shift[{policy_suffix}]"
        self.step3_action_id = step3_action_id

    def select_action(self, state, environment):
        top_hypothesis_id, _confidence = top_probability(state.probabilities)

        if len(state.history) == 1 and top_hypothesis_id == "parser_bug":
            for action in environment.candidate_actions(state):
                if action.action_id == "ask_user_scope":
                    return action

        if (
            len(state.history) == 2
            and top_hypothesis_id == "parser_bug"
            and self.step3_action_id is not None
        ):
            for action in environment.candidate_actions(state):
                if action.action_id == self.step3_action_id:
                    return action

        return super().select_action(state, environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation runner for parser_bug second/third-step latent_shift variants."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    return parser.parse_args()


def policy_kwargs() -> dict[str, object]:
    return {
        "mode_strength": 0.02,
        "type_bonus": 0.05,
        "switch_bonus": 0.20,
        "recent_window": 3,
        "anomaly_scale": 0.95,
        "persistence_scale": 0.80,
        "rebound_penalty": 0.55,
        "streak_bonus": 0.35,
        "false_alarm_scale": 1.10,
        "neutral_gate": 0.50,
        "min_aggressive_gate": 0.0,
    }


def build_policies() -> list[LatentAdaptiveShiftMemoryPolicy]:
    kwargs = policy_kwargs()
    return [
        LatentAdaptiveShiftMemoryPolicy(**kwargs),
        ParserStep3LatentPolicy(
            policy_suffix="parser_scope",
            step3_action_id=None,
            **kwargs,
        ),
        ParserStep3LatentPolicy(
            policy_suffix="parser_scope+hot_path",
            step3_action_id="inspect_hot_path_code",
            **kwargs,
        ),
        ParserStep3LatentPolicy(
            policy_suffix="parser_scope+scope",
            step3_action_id="ask_user_scope",
            **kwargs,
        ),
        ParserStep3LatentPolicy(
            policy_suffix="parser_scope+regression",
            step3_action_id="run_targeted_regression",
            **kwargs,
        ),
        ParserStep3LatentPolicy(
            policy_suffix="parser_scope+error_log",
            step3_action_id="inspect_error_log",
            **kwargs,
        ),
    ]


def summarize_rows(
    *,
    episodes: int,
    seed: int,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    shift_after_step: int,
    false_alarm_length: int,
    shift_probability: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in build_policies():
        runs = [
            run_episode(
                policy,
                seed=seed + episode_no,
                confidence_threshold=confidence_threshold,
                max_cost=max_cost,
                max_steps=max_steps,
                environment_factory=ArtifactDebuggingAmbiguousShiftEnvironment,
                environment_kwargs={
                    "shift_after_step": shift_after_step,
                    "shift_probability": shift_probability,
                    "false_alarm_length": false_alarm_length,
                },
                min_steps_before_stop=4,
            )
            for episode_no in range(episodes)
        ]
        summary = summarize(policy.policy_name, runs)
        rows.append(
            {
                "policy": summary.policy_name,
                "accuracy": round(summary.accuracy, 3),
                "mean_steps": round(summary.mean_steps, 3),
                "mean_cost": round(summary.mean_cost, 3),
                "mean_conf": round(summary.mean_final_confidence, 3),
                "mean_utility": round(summary.mean_utility, 3),
                "step_stop": round(summary.step_stop_rate, 3),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    scenarios = [
        ("false_alarm", 0.0),
        ("mixed", 0.5),
        ("true_shift", 1.0),
    ]
    for scenario_name, shift_probability in scenarios:
        rows = summarize_rows(
            episodes=args.episodes,
            seed=args.seed,
            confidence_threshold=args.confidence_threshold,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            shift_after_step=args.shift_after_step,
            false_alarm_length=args.false_alarm_length,
            shift_probability=shift_probability,
        )
        print(
            f"Scenario: {scenario_name} | episodes={args.episodes}, "
            f"confidence>={args.confidence_threshold}, max_cost={args.max_cost}, "
            f"max_steps={args.max_steps}, shift_after_step={args.shift_after_step}, "
            f"false_alarm_length={args.false_alarm_length}"
        )
        print_table(rows)
        print()


if __name__ == "__main__":
    main()
