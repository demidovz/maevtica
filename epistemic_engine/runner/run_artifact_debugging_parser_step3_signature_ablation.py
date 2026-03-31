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
from epistemic_engine.runner.run_artifact_debugging_parser_step3_signature_report import (
    diff_category,
    dominant_risk_label,
    margin_bucket,
    scope_category,
)
from epistemic_engine.runner.run_debugging_benchmark import print_table, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation runner for parser step3 signature-conditioned overrides."
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


class ParserStep3SignaturePolicy(LatentAdaptiveShiftMemoryPolicy):
    def __init__(
        self,
        *,
        policy_suffix: str,
        parser_scope_enabled: bool = True,
        require_wide_margin: bool = True,
        require_false_alarm_risk: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            parser_scope_hypotheses=("parser_bug",) if parser_scope_enabled else (),
            parser_scope_action_id="ask_user_scope",
            **kwargs,
        )
        self.policy_name = f"information_gain+latent_shift[{policy_suffix}]"
        self.require_wide_margin = require_wide_margin
        self.require_false_alarm_risk = require_false_alarm_risk

    def select_action(self, state, environment):
        if len(state.history) == 2:
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature_matches = (
                    diff_category(state.history[0].outcome) == "ingest_path"
                    and scope_category(state.history[1].outcome) == "payload"
                )
                if signature_matches:
                    if self.require_wide_margin and margin_bucket(state.probabilities) != "wide":
                        return super().select_action(state, environment)
                    if self.require_false_alarm_risk and dominant_risk_label(state) != "false_alarm":
                        return super().select_action(state, environment)
                    for action in environment.candidate_actions(state):
                        if action.action_id == "inspect_error_log":
                            return action
        return super().select_action(state, environment)


class ParserScopeBaselinePolicy(LatentAdaptiveShiftMemoryPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            parser_scope_hypotheses=("parser_bug",),
            parser_scope_action_id="ask_user_scope",
            **kwargs,
        )
        self.policy_name = "information_gain+latent_shift[parser_scope]"


def build_policies() -> list[LatentAdaptiveShiftMemoryPolicy]:
    kwargs = policy_kwargs()
    return [
        LatentAdaptiveShiftMemoryPolicy(**kwargs),
        ParserScopeBaselinePolicy(**kwargs),
        ParserStep3SignaturePolicy(
            policy_suffix="parser_scope+payload_error_log_relaxed",
            require_wide_margin=False,
            require_false_alarm_risk=False,
            **kwargs,
        ),
        ParserStep3SignaturePolicy(
            policy_suffix="parser_scope+payload_error_log_false_alarm",
            require_wide_margin=False,
            require_false_alarm_risk=True,
            **kwargs,
        ),
        ParserStep3SignaturePolicy(
            policy_suffix="parser_scope+payload_error_log_narrow",
            require_wide_margin=True,
            require_false_alarm_risk=True,
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
