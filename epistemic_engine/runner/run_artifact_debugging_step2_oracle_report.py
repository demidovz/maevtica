from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.beliefs.state import normalize, top_probability, uniform_belief
from epistemic_engine.environments.artifact_debugging import (
    ArtifactDebuggingAmbiguousShiftEnvironment,
)
from epistemic_engine.models import BeliefState, Observation
from epistemic_engine.policies.switch_memory import LatentAdaptiveShiftMemoryPolicy
from epistemic_engine.questions.policy import candidate_actions
from epistemic_engine.revision.updater import apply_observation, refresh_shift_latent
from epistemic_engine.runner.run_debugging_benchmark import print_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research report for oracle second action after the first diff step."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
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


def clone_with_observation(state, env, policy, action, outcome) -> BeliefState:
    posterior_scores = {}
    for hypothesis_id, prior in state.probabilities.items():
        likelihood = env.likelihood(action.action_id, outcome, hypothesis_id)
        posterior_scores[hypothesis_id] = prior * likelihood

    temp_state = BeliefState(
        probabilities=normalize(posterior_scores),
        asked_actions=[*state.asked_actions, action.action_id],
        total_cost=state.total_cost + action.cost,
        history=[
            *state.history,
            Observation(
                action_id=action.action_id,
                outcome=outcome,
                cost=action.cost,
                action_type=action.action_type,
            ),
        ],
        revisions=list(state.revisions),
        shift_latent=None,
        shift_latent_history=list(state.shift_latent_history),
    )
    refresh_shift_latent(temp_state, env, policy)
    return temp_state


def decision_confidence(state, env, policy) -> float:
    decision_probabilities = policy.decision_probabilities(state, env)
    _top_hypothesis_id, confidence = top_probability(decision_probabilities)
    return confidence


def collect_rows(
    *,
    shift_probability: float,
    episodes: int,
    seed: int,
    max_cost: float,
    max_steps: int,
    shift_after_step: int,
    false_alarm_length: int,
) -> list[dict[str, object]]:
    policy = LatentAdaptiveShiftMemoryPolicy(**policy_kwargs())
    score_buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    default_action_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)

    for episode_no in range(episodes):
        env = ArtifactDebuggingAmbiguousShiftEnvironment(
            seed=seed + episode_no,
            max_cost=max_cost,
            max_steps=max_steps,
            shift_after_step=shift_after_step,
            shift_probability=shift_probability,
            false_alarm_length=false_alarm_length,
        )
        state = uniform_belief(list(env.hypotheses()))
        refresh_shift_latent(state, env, policy)

        first_action = policy.select_action(state, env)
        observation = Observation(
            action_id=first_action.action_id,
            outcome=env.sample_observation(first_action.action_id),
            cost=first_action.cost,
            action_type=first_action.action_type,
        )
        apply_observation(state, env, observation, policy=policy)
        refresh_shift_latent(state, env, policy)

        top_hypothesis_id, _confidence = top_probability(state.probabilities)
        bucket_key = (top_hypothesis_id, env.scenario_label())
        default_action_buckets[bucket_key].append(policy.select_action(state, env).action_id)

        for action in candidate_actions(state, env):
            value = 0.0
            for outcome in env.outcomes_for(action.action_id):
                outcome_probability = sum(
                    state.probabilities[hypothesis_id]
                    * env.likelihood(action.action_id, outcome, hypothesis_id)
                    for hypothesis_id in state.probabilities
                )
                next_state = clone_with_observation(state, env, policy, action, outcome)
                value += outcome_probability * decision_confidence(next_state, env, policy)
            score_buckets[bucket_key][action.action_id].append(value)

        policy.record_episode(env, state)

    rows: list[dict[str, object]] = []
    for (top_hypothesis_id, scenario_label), action_scores in sorted(score_buckets.items()):
        ranked_actions = sorted(
            (
                (action_id, statistics.mean(values))
                for action_id, values in action_scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_action_id, best_score = ranked_actions[0]
        second_best_score = ranked_actions[1][1] if len(ranked_actions) > 1 else best_score
        default_action_id = statistics.mode(default_action_buckets[(top_hypothesis_id, scenario_label)])
        rows.append(
            {
                "top_hypothesis": top_hypothesis_id,
                "scenario": scenario_label,
                "current_step2": default_action_id,
                "oracle_step2": best_action_id,
                "oracle_conf": round(best_score, 3),
                "margin": round(best_score - second_best_score, 3),
                "n": len(default_action_buckets[(top_hypothesis_id, scenario_label)]),
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
    for label, shift_probability in scenarios:
        rows = collect_rows(
            shift_probability=shift_probability,
            episodes=args.episodes,
            seed=args.seed,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            shift_after_step=args.shift_after_step,
            false_alarm_length=args.false_alarm_length,
        )
        print(
            f"Scenario: {label} | episodes={args.episodes}, max_cost={args.max_cost}, "
            f"max_steps={args.max_steps}, shift_after_step={args.shift_after_step}, "
            f"false_alarm_length={args.false_alarm_length}"
        )
        print_table(rows)
        print()


if __name__ == "__main__":
    main()
