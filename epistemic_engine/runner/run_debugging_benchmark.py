from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.beliefs.state import top_probability, uniform_belief
from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.environments.debugging import DebuggingToyEnvironment
from epistemic_engine.models import EpisodeMetrics, Observation
from epistemic_engine.policies.baselines import CheapestQuestionPolicy, RandomQuestionPolicy
from epistemic_engine.policies.hybrid_memory import HybridMemoryPolicy
from epistemic_engine.policies.mode_memory import ModeMemoryPolicy
from epistemic_engine.policies.memory import MemoryAugmentedInformationGainPolicy
from epistemic_engine.policies.question_type_memory import QuestionTypeMemoryPolicy
from epistemic_engine.policies.switch_memory import SwitchAwareHybridMemoryPolicy
from epistemic_engine.questions.policy import InformationGainPolicy
from epistemic_engine.revision.updater import apply_observation, refresh_shift_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark для debugging MVP."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser.parse_args()


def score_episode(
    *,
    correct: int,
    total_cost: float,
    final_confidence: float,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    steps: int,
    stop_reason: str,
) -> float:
    correct_reward = 1.0 if correct else -0.35
    cost_penalty = 0.30 * (total_cost / max(max_cost, 1e-9))
    step_penalty = 0.15 * (steps / max(max_steps, 1))
    uncertainty_penalty = 0.80 * max(0.0, confidence_threshold - final_confidence)
    forced_stop_penalty = 0.15 if stop_reason in {"budget_limit", "step_limit"} else 0.0
    return correct_reward - cost_penalty - step_penalty - uncertainty_penalty - forced_stop_penalty


def run_episode(
    policy,
    seed: int,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    environment_factory=DebuggingToyEnvironment,
    environment_kwargs: dict | None = None,
    min_steps_before_stop: int = 0,
) -> EpisodeMetrics:
    environment = environment_factory(
        seed=seed,
        max_cost=max_cost,
        max_steps=max_steps,
        **(environment_kwargs or {}),
    )
    state = uniform_belief(list(environment.hypotheses()))
    refresh_shift_latent(state, environment, policy)

    while True:
        policy_probabilities = policy.decision_probabilities(state, environment)
        top_id, confidence = top_probability(policy_probabilities)
        if (
            confidence >= confidence_threshold
            and len(state.history) >= min_steps_before_stop
        ):
            break
        candidates = environment.candidate_actions(state)
        if not candidates:
            break
        action = policy.select_action(state, environment)
        observation = Observation(
            action_id=action.action_id,
            outcome=environment.sample_observation(action.action_id),
            cost=action.cost,
            action_type=action.action_type,
        )
        apply_observation(state, environment, observation)
        refresh_shift_latent(state, environment, policy)

    final_probabilities = policy.decision_probabilities(state, environment)
    predicted_hypothesis, final_confidence = top_probability(final_probabilities)
    stop_reason = (
        "confidence"
        if (
            final_confidence >= confidence_threshold
            and len(state.history) >= min_steps_before_stop
        )
        else environment.stop_reason(state) or "open_ended"
    )
    policy.record_episode(environment, state)
    return EpisodeMetrics(
        actual_hypothesis=environment.actual_hypothesis,
        predicted_hypothesis=predicted_hypothesis,
        correct=int(predicted_hypothesis == environment.actual_hypothesis),
        steps=len(state.history),
        total_cost=state.total_cost,
        final_confidence=final_confidence,
        utility=score_episode(
            correct=int(predicted_hypothesis == environment.actual_hypothesis),
            total_cost=state.total_cost,
            final_confidence=final_confidence,
            confidence_threshold=confidence_threshold,
            max_cost=max_cost,
            max_steps=max_steps,
            steps=len(state.history),
            stop_reason=stop_reason,
        ),
        budget_stop=int(stop_reason == "budget_limit"),
        step_stop=int(stop_reason == "step_limit"),
    )


def print_table(rows: list[dict[str, object]]) -> None:
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = "".join(f"{column:<{widths[column] + 2}}" for column in columns)
    print(header.rstrip())
    print("-" * len(header.rstrip()))
    for row in rows:
        print(
            "".join(f"{str(row[column]):<{widths[column] + 2}}" for column in columns).rstrip()
        )


def main() -> None:
    args = parse_args()
    policies = [
        InformationGainPolicy(),
        MemoryAugmentedInformationGainPolicy(),
        QuestionTypeMemoryPolicy(),
        ModeMemoryPolicy(),
        HybridMemoryPolicy(),
        SwitchAwareHybridMemoryPolicy(),
        CheapestQuestionPolicy(),
        RandomQuestionPolicy(seed=args.seed + 2000),
    ]
    rows: list[dict[str, object]] = []

    for index, policy in enumerate(policies):
        episodes = [
            run_episode(
                policy,
                seed=args.seed + episode_no,
                confidence_threshold=args.confidence_threshold,
                max_cost=args.max_cost,
                max_steps=args.max_steps,
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
        f"Constraints: confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
