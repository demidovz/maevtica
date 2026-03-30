from __future__ import annotations

import argparse
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-by-step demo for artifact-level ambiguous shift environment."
    )
    parser.add_argument(
        "--scenario",
        choices=("auto", "true_shift", "false_alarm"),
        default="auto",
    )
    parser.add_argument(
        "--actual",
        choices=list(ArtifactDebuggingAmbiguousShiftEnvironment().hypotheses()),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    return parser.parse_args()


def build_environment(args: argparse.Namespace) -> ArtifactDebuggingAmbiguousShiftEnvironment:
    shift_probability = 0.5
    if args.scenario == "true_shift":
        shift_probability = 1.0
    elif args.scenario == "false_alarm":
        shift_probability = 0.0

    return ArtifactDebuggingAmbiguousShiftEnvironment(
        actual_hypothesis=args.actual,
        seed=args.seed,
        max_cost=args.max_cost,
        max_steps=args.max_steps,
        shift_after_step=args.shift_after_step,
        shift_probability=shift_probability,
        false_alarm_length=args.false_alarm_length,
    )


def main() -> None:
    args = parse_args()
    environment = build_environment(args)
    state = uniform_belief(list(environment.hypotheses()))
    policy = LatentAdaptiveShiftMemoryPolicy(
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
    )
    min_stop = 4
    refresh_shift_latent(state, environment, policy)

    print("Epistemic Engine MVP: artifact ambiguous-shift diagnosis")
    print(f"Scenario: {environment.scenario_label()}")
    print(f"Hidden root cause: {environment.actual_hypothesis}")
    print(f"Initial profile: {environment.initial_profile}")
    print(f"Shifted profile: {environment.shifted_profile}")
    print(f"Shift starts after step: {environment.shift_after_step}")
    print(
        f"Constraints: confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}, "
        f"false_alarm_length={args.false_alarm_length}, min_stop={min_stop}"
    )
    print()

    while True:
        latent_state = state.shift_latent or policy.infer_latent_state(state, environment)
        policy_probabilities = policy.decision_probabilities(state, environment)
        top_id, confidence = top_probability(policy_probabilities)
        print(f"Current best hypothesis: {top_id} ({confidence:.3f})")
        print(
            "Latent state: "
            f"top_mode={latent_state.top_mode}, "
            f"candidate_mode={latent_state.candidate_mode}, "
            f"false_alarm_risk={latent_state.false_alarm_risk:.3f}, "
            f"persistent_shift_risk={latent_state.persistent_shift_risk:.3f}, "
            f"switch_pressure={latent_state.switch_pressure:.3f}, "
            f"aggressive_gate={latent_state.aggressive_gate:.3f}"
        )
        if confidence >= args.confidence_threshold and len(state.history) >= min_stop:
            break

        candidates = environment.candidate_actions(state)
        if not candidates:
            break

        action = policy.select_action(state, environment)
        profile_before = environment.active_profile()
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
        print(f"Active profile: {profile_before}")
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
        if final_confidence >= args.confidence_threshold and len(state.history) >= min_stop
        else environment.stop_reason(state) or "open_ended"
    )
    print(f"Stop reason: {stop_reason}")
    if state.revisions:
        print("Revisions:")
        for revision in state.revisions:
            print(f"- {revision.previous_top} -> {revision.new_top}: {revision.reason}")
    if state.shift_latent_history:
        print("Latent trace:")
        for step_no, latent in enumerate(state.shift_latent_history):
            print(
                f"- step={step_no}: "
                f"top_mode={latent.top_mode}, "
                f"candidate_mode={latent.candidate_mode}, "
                f"false_alarm_risk={latent.false_alarm_risk:.3f}, "
                f"persistent_shift_risk={latent.persistent_shift_risk:.3f}, "
                f"switch_pressure={latent.switch_pressure:.3f}, "
                f"aggressive_gate={latent.aggressive_gate:.3f}"
            )


if __name__ == "__main__":
    main()
