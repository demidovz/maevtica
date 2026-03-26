from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.beliefs.state import top_probability, uniform_belief
from epistemic_engine.environments.artifact_debugging import ArtifactDebuggingEnvironment
from epistemic_engine.models import Observation
from epistemic_engine.questions.policy import InformationGainPolicy
from epistemic_engine.revision.updater import apply_observation, refresh_shift_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-by-step demo for semi-real artifact debugging environment."
    )
    parser.add_argument(
        "--actual",
        choices=list(ArtifactDebuggingEnvironment().hypotheses()),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = ArtifactDebuggingEnvironment(
        actual_hypothesis=args.actual,
        seed=args.seed,
        max_cost=args.max_cost,
        max_steps=args.max_steps,
    )
    state = uniform_belief(list(environment.hypotheses()))
    policy = InformationGainPolicy()
    refresh_shift_latent(state, environment, policy)

    print("Epistemic Engine MVP: artifact debugging diagnosis")
    print(f"Hidden root cause: {environment.actual_hypothesis}")
    print(
        f"Constraints: confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}"
    )
    print()

    while True:
        policy_probabilities = policy.decision_probabilities(state, environment)
        top_id, confidence = top_probability(policy_probabilities)
        print(f"Current best hypothesis: {top_id} ({confidence:.3f})")
        if confidence >= args.confidence_threshold:
            break

        candidates = environment.candidate_actions(state)
        if not candidates:
            break

        action = policy.select_action(state, environment)
        outcome = environment.sample_observation(action.action_id)
        observation = Observation(
            action_id=action.action_id,
            outcome=outcome,
            cost=action.cost,
            action_type=action.action_type,
        )
        print(
            f"Next question/step: [{action.action_type}] {action.action_id} | "
            f"{action.description}"
        )
        print(f"Observation: {outcome}")
        apply_observation(state, environment, observation)
        refresh_shift_latent(state, environment, policy)
        remaining_budget = environment.remaining_budget(state)
        if remaining_budget is not None:
            print(f"Remaining budget: {remaining_budget:.2f}")
        print()

    print("Final")
    final_probabilities = policy.decision_probabilities(state, environment)
    final_id, final_confidence = top_probability(final_probabilities)
    print(f"Predicted cause: {final_id}")
    print(f"Confidence: {final_confidence:.3f}")
    print(f"Total cost: {state.total_cost:.2f}")
    print(f"Steps taken: {len(state.history)}")
    stop_reason = (
        "confidence"
        if final_confidence >= args.confidence_threshold
        else environment.stop_reason(state) or "open_ended"
    )
    print(f"Stop reason: {stop_reason}")
    if state.revisions:
        print("Revisions:")
        for revision in state.revisions:
            print(f"- {revision.previous_top} -> {revision.new_top}: {revision.reason}")


if __name__ == "__main__":
    main()
