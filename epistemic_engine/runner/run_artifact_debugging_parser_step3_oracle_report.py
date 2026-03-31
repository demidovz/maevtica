from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
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
        description="Oracle report for the third action on parser_scope states."
    )
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    parser.add_argument("--shift-probability", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
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


def build_policy() -> LatentAdaptiveShiftMemoryPolicy:
    return LatentAdaptiveShiftMemoryPolicy(
        parser_scope_hypotheses=("parser_bug",),
        parser_scope_action_id="ask_user_scope",
        **policy_kwargs(),
    )


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


def dominant_risk_label(state) -> str:
    latent = state.shift_latent
    if latent is None:
        return "none"
    risk_items = [
        ("false_alarm", latent.false_alarm_risk),
        ("profile_shift", latent.profile_shift_risk),
        ("hypothesis_switch", latent.hypothesis_switch_risk),
        ("persistent_shift", latent.persistent_shift_risk),
    ]
    return max(risk_items, key=lambda item: item[1])[0]


def confidence_bucket(confidence: float) -> str:
    if confidence < 0.34:
        return "low"
    if confidence < 0.52:
        return "mid"
    return "high"


def advance_one_step(state, env, policy) -> None:
    action = policy.select_action(state, env)
    observation = Observation(
        action_id=action.action_id,
        outcome=env.sample_observation(action.action_id),
        cost=action.cost,
        action_type=action.action_type,
    )
    apply_observation(state, env, observation, policy=policy)
    refresh_shift_latent(state, env, policy)


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    policy = build_policy()
    score_buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    default_action_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    scenario_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    hypothesis_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)

    for episode_no in range(args.episodes):
        env = ArtifactDebuggingAmbiguousShiftEnvironment(
            seed=args.seed + episode_no,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            shift_after_step=args.shift_after_step,
            shift_probability=args.shift_probability,
            false_alarm_length=args.false_alarm_length,
        )
        state = uniform_belief(list(env.hypotheses()))
        refresh_shift_latent(state, env, policy)

        while len(state.history) < 2:
            advance_one_step(state, env, policy)

        if state.history[1].action_id != "ask_user_scope":
            continue

        top_hypothesis_id, confidence = top_probability(state.probabilities)
        if top_hypothesis_id != "parser_bug":
            continue

        bucket_key = (
            confidence_bucket(confidence),
            dominant_risk_label(state),
        )
        current_action_id = policy.select_action(state, env).action_id
        default_action_buckets[bucket_key].append(current_action_id)
        scenario_buckets[bucket_key].append(env.scenario_label())
        hypothesis_buckets[bucket_key].append(env.actual_hypothesis)

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

    rows: list[dict[str, object]] = []
    ranked_buckets = sorted(
        score_buckets.items(),
        key=lambda item: len(default_action_buckets[item[0]]),
        reverse=True,
    )
    for bucket_key, action_scores in ranked_buckets[: args.top_k]:
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
        default_action_id = Counter(default_action_buckets[bucket_key]).most_common(1)[0][0]
        scenarios = scenario_buckets[bucket_key]
        hypotheses = Counter(hypothesis_buckets[bucket_key]).most_common(1)[0][0]
        rows.append(
            {
                "conf_bucket": bucket_key[0],
                "dominant_risk": bucket_key[1],
                "current_step3": default_action_id,
                "oracle_step3": best_action_id,
                "oracle_conf": round(best_score, 3),
                "margin": round(best_score - second_best_score, 3),
                "true_shift_rate": round(
                    scenarios.count("true_shift") / max(len(scenarios), 1),
                    3,
                ),
                "actual_top": hypotheses,
                "n": len(default_action_buckets[bucket_key]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    print(
        f"Parser step3 oracle report | episodes={args.episodes}, "
        f"shift_probability={args.shift_probability}, shift_after_step={args.shift_after_step}, "
        f"false_alarm_length={args.false_alarm_length}"
    )
    if rows:
        print_table(rows)
    else:
        print("No parser_scope states observed.")


if __name__ == "__main__":
    main()
