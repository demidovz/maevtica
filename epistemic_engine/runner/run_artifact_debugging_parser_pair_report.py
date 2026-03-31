from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.beliefs.state import top_probability, uniform_belief
from epistemic_engine.environments.artifact_debugging import (
    ArtifactDebuggingAmbiguousShiftEnvironment,
)
from epistemic_engine.models import Observation
from epistemic_engine.policies.switch_memory import LatentAdaptiveShiftMemoryPolicy
from epistemic_engine.revision.updater import apply_observation, refresh_shift_latent
from epistemic_engine.runner.run_artifact_debugging_parser_pair_ablation import (
    policy_kwargs,
)
from epistemic_engine.runner.run_debugging_benchmark import print_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report frequent parser diff+scope pairs before the third action."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    parser.add_argument("--shift-probability", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def build_policy() -> LatentAdaptiveShiftMemoryPolicy:
    return LatentAdaptiveShiftMemoryPolicy(
        parser_scope_hypotheses=("parser_bug",),
        parser_scope_action_id="ask_user_scope",
        **policy_kwargs(),
    )


def advance_one_step(state, environment, policy) -> None:
    action = policy.select_action(state, environment)
    observation = Observation(
        action_id=action.action_id,
        outcome=environment.sample_observation(action.action_id),
        cost=action.cost,
        action_type=action.action_type,
    )
    apply_observation(state, environment, observation, policy=policy)
    refresh_shift_latent(state, environment, policy)


def summarize_pairs(args: argparse.Namespace) -> list[dict[str, object]]:
    policy = build_policy()
    pair_counts: Counter[tuple[str, str]] = Counter()
    scenario_counts: dict[tuple[str, str], Counter[str]] = {}
    hypothesis_counts: dict[tuple[str, str], Counter[str]] = {}
    third_action_counts: dict[tuple[str, str], Counter[str]] = {}
    total_parser_scope_paths = 0

    for episode_no in range(args.episodes):
        environment = ArtifactDebuggingAmbiguousShiftEnvironment(
            seed=args.seed + episode_no,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            shift_after_step=args.shift_after_step,
            shift_probability=args.shift_probability,
            false_alarm_length=args.false_alarm_length,
        )
        state = uniform_belief(list(environment.hypotheses()))
        refresh_shift_latent(state, environment, policy)

        while len(state.history) < 2:
            advance_one_step(state, environment, policy)

        if state.history[1].action_id != "ask_user_scope":
            continue

        top_id, _confidence = top_probability(state.probabilities)
        if top_id != "parser_bug":
            continue

        total_parser_scope_paths += 1
        pair = (state.history[0].outcome, state.history[1].outcome)
        pair_counts[pair] += 1
        scenario_counts.setdefault(pair, Counter())[environment.scenario_label()] += 1
        hypothesis_counts.setdefault(pair, Counter())[environment.actual_hypothesis] += 1
        next_action = policy.select_action(state, environment)
        third_action_counts.setdefault(pair, Counter())[next_action.action_id] += 1

    rows: list[dict[str, object]] = []
    for pair, count in pair_counts.most_common(args.top_k):
        scenarios = scenario_counts[pair]
        hypotheses = hypothesis_counts[pair]
        next_actions = third_action_counts[pair]
        rows.append(
            {
                "count": count,
                "coverage": round(
                    count / max(total_parser_scope_paths, 1),
                    3,
                ),
                "true_shift_rate": round(
                    scenarios.get("true_shift", 0) / max(count, 1),
                    3,
                ),
                "dominant_hypothesis": hypotheses.most_common(1)[0][0],
                "default_step3": next_actions.most_common(1)[0][0],
                "diff_outcome": pair[0],
                "scope_outcome": pair[1],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = summarize_pairs(args)
    print(
        f"Parser diff+scope pair report | episodes={args.episodes}, "
        f"shift_probability={args.shift_probability}, shift_after_step={args.shift_after_step}, "
        f"false_alarm_length={args.false_alarm_length}"
    )
    if rows:
        print_table(rows)
    else:
        print("No parser_scope pairs observed.")


if __name__ == "__main__":
    main()
