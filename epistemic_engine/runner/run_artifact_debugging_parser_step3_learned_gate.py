from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.benchmarks.metrics import summarize
from epistemic_engine.beliefs.state import top_probability, uniform_belief
from epistemic_engine.models import Observation
from epistemic_engine.policies.switch_memory import LatentAdaptiveShiftMemoryPolicy
from epistemic_engine.revision.updater import apply_observation, refresh_shift_latent
from epistemic_engine.runner.run_artifact_debugging_parser_step3_signature_report import (
    diff_category,
    dominant_risk_label,
    margin_bucket,
    scope_category,
)
from epistemic_engine.runner.run_debugging_benchmark import (
    print_table,
    score_episode,
)
from epistemic_engine.environments.artifact_debugging import (
    ArtifactDebuggingAmbiguousShiftEnvironment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a small learned step3 gate for parser_scope states."
    )
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--train-rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-after-step", type=int, default=2)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    parser.add_argument("--train-shift-probability", type=float, default=0.5)
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


def top_two_hypotheses(probabilities: dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    ordered = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_item = ordered[0]
    second_item = ordered[1] if len(ordered) > 1 else ordered[0]
    return top_item, second_item


def risk_bucket(value: float) -> str:
    if value < 0.25:
        return "low"
    if value < 0.55:
        return "mid"
    return "high"


def feature_signature(state) -> tuple[str, str, str, str, str, str, str]:
    (_top_id, _top_confidence), (second_id, _second_confidence) = top_two_hypotheses(
        state.probabilities
    )
    latent = state.shift_latent
    false_alarm_bucket = "none"
    hypothesis_switch_bucket = "none"
    if latent is not None:
        false_alarm_bucket = risk_bucket(latent.false_alarm_risk)
        hypothesis_switch_bucket = risk_bucket(latent.hypothesis_switch_risk)
    return (
        dominant_risk_label(state),
        false_alarm_bucket,
        hypothesis_switch_bucket,
        margin_bucket(state.probabilities),
        diff_category(state.history[0].outcome),
        scope_category(state.history[1].outcome),
        second_id,
    )


def backoff_keys(signature: tuple[str, str, str, str, str, str, str]) -> list[tuple[str, ...]]:
    dominant_risk, false_alarm_bucket, hypothesis_switch_bucket, margin, diff, scope, second_id = signature
    return [
        (
            dominant_risk,
            false_alarm_bucket,
            hypothesis_switch_bucket,
            margin,
            diff,
            scope,
            second_id,
        ),
        (
            dominant_risk,
            false_alarm_bucket,
            hypothesis_switch_bucket,
            margin,
            diff,
            scope,
        ),
        (
            dominant_risk,
            false_alarm_bucket,
            hypothesis_switch_bucket,
            diff,
            scope,
        ),
        (dominant_risk, diff, scope),
        (diff, scope),
        (diff,),
    ]


@dataclass
class Step3Sample:
    signature: tuple[str, str, str, str, str, str, str]
    scenario_label: str
    fallback_action_id: str
    action_values: dict[str, float]


class LearnedActionGate:
    def __init__(
        self,
        *,
        shrinkage: float = 3.0,
        min_advantage: float = 0.015,
        min_support: int = 2,
    ) -> None:
        self.shrinkage = shrinkage
        self.min_advantage = min_advantage
        self.min_support = min_support
        self.global_values: dict[str, float] = {}
        self.global_counts: dict[str, int] = {}
        self.scenario_tables: list[dict[tuple[str, ...], Counter[str]]] = [
            defaultdict(Counter)
            for _ in range(6)
        ]
        self.tables: list[dict[tuple[str, ...], dict[str, list[float]]]] = [
            defaultdict(lambda: defaultdict(list))
            for _ in range(6)
        ]

    def fit(self, samples: list[Step3Sample]) -> None:
        global_bucket: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            for action_id, value in sample.action_values.items():
                global_bucket[action_id].append(value)
            for level, key in enumerate(backoff_keys(sample.signature)):
                action_bucket = self.tables[level][key]
                self.scenario_tables[level][key][sample.scenario_label] += 1
                for action_id, value in sample.action_values.items():
                    action_bucket[action_id].append(value)
        self.global_values = {
            action_id: sum(values) / len(values)
            for action_id, values in global_bucket.items()
        }
        self.global_counts = {
            action_id: len(values)
            for action_id, values in global_bucket.items()
        }

    def predict_best(
        self,
        *,
        signature: tuple[str, str, str, str, str, str, str],
        candidate_action_ids: list[str],
    ) -> tuple[str, dict[str, float], dict[str, int]]:
        for level, key in enumerate(backoff_keys(signature)):
            if key not in self.tables[level]:
                continue
            best_scores, support_counts = self._scores_for_level(
                level=level,
                key=key,
                candidate_action_ids=candidate_action_ids,
            )
            if not best_scores:
                continue
            best_action_id = max(
                candidate_action_ids,
                key=lambda action_id: best_scores.get(
                    action_id,
                    self.global_values.get(action_id, float("-inf")),
                ),
            )
            return best_action_id, best_scores, support_counts
        return "", {}, {}

    def _scores_for_level(
        self,
        *,
        level: int,
        key: tuple[str, ...],
        candidate_action_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        action_bucket = self.tables[level][key]
        best_scores: dict[str, float] = {}
        support_counts: dict[str, int] = {}
        for action_id in candidate_action_ids:
            local_values = action_bucket.get(action_id)
            global_value = self.global_values.get(action_id, float("-inf"))
            if not local_values:
                best_scores[action_id] = global_value
                support_counts[action_id] = self.global_counts.get(action_id, 0)
                continue
            shrunk_mean = (
                sum(local_values) + self.shrinkage * global_value
            ) / (len(local_values) + self.shrinkage)
            best_scores[action_id] = shrunk_mean
            support_counts[action_id] = len(local_values)
        return best_scores, support_counts

    def predict(
        self,
        *,
        signature: tuple[str, str, str, str, str, str, str],
        candidate_action_ids: list[str],
        fallback_action_id: str,
    ) -> str:
        best_action_id, best_scores, support_counts = self.predict_best(
            signature=signature,
            candidate_action_ids=candidate_action_ids,
        )
        if not best_action_id:
            return fallback_action_id
        fallback_score = best_scores.get(
            fallback_action_id,
            self.global_values.get(fallback_action_id, float("-inf")),
        )
        best_score = best_scores.get(
            best_action_id,
            self.global_values.get(best_action_id, float("-inf")),
        )
        if (
            best_action_id != fallback_action_id
            and support_counts.get(best_action_id, 0) >= self.min_support
            and best_score - fallback_score >= self.min_advantage
        ):
            return best_action_id
        return fallback_action_id

    def regime_parameters(
        self,
        *,
        level: int,
        key: tuple[str, ...],
    ) -> tuple[float, int]:
        scenario_counts = self.scenario_tables[level].get(key)
        if not scenario_counts:
            return self.min_advantage, self.min_support
        total = sum(scenario_counts.values())
        true_shift_rate = scenario_counts.get("true_shift", 0) / max(total, 1)
        min_advantage = self.min_advantage
        min_support = self.min_support
        if total < 3:
            min_advantage += 0.008
            min_support = max(min_support, 3)
        if true_shift_rate <= 0.35:
            min_advantage += 0.010
            min_support = max(min_support, 3)
        elif true_shift_rate >= 0.65:
            min_advantage = max(0.005, min_advantage - 0.005)
        return min_advantage, min_support

    def predict_regime_aware(
        self,
        *,
        signature: tuple[str, str, str, str, str, str, str],
        candidate_action_ids: list[str],
        fallback_action_id: str,
    ) -> str:
        for level, key in enumerate(backoff_keys(signature)):
            if key not in self.tables[level]:
                continue
            best_scores, support_counts = self._scores_for_level(
                level=level,
                key=key,
                candidate_action_ids=candidate_action_ids,
            )
            if not best_scores:
                continue
            best_action_id = max(
                candidate_action_ids,
                key=lambda action_id: best_scores.get(
                    action_id,
                    self.global_values.get(action_id, float("-inf")),
                ),
            )
            min_advantage, min_support = self.regime_parameters(level=level, key=key)
            fallback_score = best_scores.get(
                fallback_action_id,
                self.global_values.get(fallback_action_id, float("-inf")),
            )
            best_score = best_scores.get(
                best_action_id,
                self.global_values.get(best_action_id, float("-inf")),
            )
            if (
                best_action_id != fallback_action_id
                and support_counts.get(best_action_id, 0) >= min_support
                and best_score - fallback_score >= min_advantage
            ):
                return best_action_id
            return fallback_action_id
        return fallback_action_id


class ParserScopeBaselinePolicy(LatentAdaptiveShiftMemoryPolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            parser_scope_hypotheses=("parser_bug",),
            parser_scope_action_id="ask_user_scope",
            **kwargs,
        )
        self.policy_name = "information_gain+latent_shift[parser_scope]"


class ParserScopeLearnedGatePolicy(ParserScopeBaselinePolicy):
    def __init__(self, *, gate: LearnedActionGate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.policy_name = "information_gain+latent_shift[parser_scope+learned_gate]"

    def select_action(self, state, environment):
        if len(state.history) == 2 and state.history[1].action_id == "ask_user_scope":
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature = feature_signature(state)
                candidate_action_ids = [
                    action.action_id for action in environment.candidate_actions(state)
                ]
                selected_action_id, _scores, _supports = self.gate.predict_best(
                    signature=signature,
                    candidate_action_ids=candidate_action_ids,
                )
                if not selected_action_id:
                    selected_action_id = super().select_action(state, environment).action_id
                for action in environment.candidate_actions(state):
                    if action.action_id == selected_action_id:
                        return action
        return super().select_action(state, environment)


class ParserScopeSelectiveLearnedGatePolicy(ParserScopeBaselinePolicy):
    def __init__(self, *, gate: LearnedActionGate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.policy_name = "information_gain+latent_shift[parser_scope+selective_learned_gate]"

    def select_action(self, state, environment):
        if len(state.history) == 2 and state.history[1].action_id == "ask_user_scope":
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature = feature_signature(state)
                candidate_action_ids = [
                    action.action_id for action in environment.candidate_actions(state)
                ]
                fallback_action_id = super().select_action(state, environment).action_id
                selected_action_id = self.gate.predict(
                    signature=signature,
                    candidate_action_ids=candidate_action_ids,
                    fallback_action_id=fallback_action_id,
                )
                for action in environment.candidate_actions(state):
                    if action.action_id == selected_action_id:
                        return action
        return super().select_action(state, environment)


class ParserScopeRegimeAwareGatePolicy(ParserScopeBaselinePolicy):
    def __init__(self, *, gate: LearnedActionGate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.policy_name = "information_gain+latent_shift[parser_scope+regime_aware_gate]"

    def select_action(self, state, environment):
        if len(state.history) == 2 and state.history[1].action_id == "ask_user_scope":
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature = feature_signature(state)
                candidate_action_ids = [
                    action.action_id for action in environment.candidate_actions(state)
                ]
                fallback_action_id = super().select_action(state, environment).action_id
                selected_action_id = self.gate.predict_regime_aware(
                    signature=signature,
                    candidate_action_ids=candidate_action_ids,
                    fallback_action_id=fallback_action_id,
                )
                for action in environment.candidate_actions(state):
                    if action.action_id == selected_action_id:
                        return action
        return super().select_action(state, environment)


class ParserScopeSplitRegimeGatePolicy(ParserScopeBaselinePolicy):
    def __init__(
        self,
        *,
        mixed_gate: LearnedActionGate,
        false_alarm_gate: LearnedActionGate,
        true_shift_gate: LearnedActionGate,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mixed_gate = mixed_gate
        self.false_alarm_gate = false_alarm_gate
        self.true_shift_gate = true_shift_gate
        self.policy_name = "information_gain+latent_shift[parser_scope+split_regime_gate]"

    def select_action(self, state, environment):
        if len(state.history) == 2 and state.history[1].action_id == "ask_user_scope":
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature = feature_signature(state)
                dominant_risk, false_alarm_bucket, hypothesis_switch_bucket, _margin, _diff, _scope, _second = signature
                candidate_action_ids = [
                    action.action_id for action in environment.candidate_actions(state)
                ]
                fallback_action_id = super().select_action(state, environment).action_id
                gate = self.mixed_gate
                if dominant_risk == "false_alarm" or false_alarm_bucket in {"mid", "high"}:
                    gate = self.false_alarm_gate
                elif dominant_risk == "hypothesis_switch" or hypothesis_switch_bucket in {"mid", "high"}:
                    gate = self.true_shift_gate
                selected_action_id = gate.predict(
                    signature=signature,
                    candidate_action_ids=candidate_action_ids,
                    fallback_action_id=fallback_action_id,
                )
                for action in environment.candidate_actions(state):
                    if action.action_id == selected_action_id:
                        return action
        return super().select_action(state, environment)


class ParserScopeHybridGatePolicy(ParserScopeBaselinePolicy):
    def __init__(self, *, gate: LearnedActionGate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.policy_name = "information_gain+latent_shift[parser_scope+hybrid_gate]"

    def select_action(self, state, environment):
        if len(state.history) == 2 and state.history[1].action_id == "ask_user_scope":
            top_hypothesis_id, _confidence = top_probability(state.probabilities)
            if top_hypothesis_id == "parser_bug":
                signature = feature_signature(state)
                dominant_risk, false_alarm_bucket, hypothesis_switch_bucket, _margin, _diff, _scope, _second = signature
                candidate_action_ids = [
                    action.action_id for action in environment.candidate_actions(state)
                ]
                fallback_action_id = super().select_action(state, environment).action_id
                if dominant_risk == "hypothesis_switch" or hypothesis_switch_bucket == "high":
                    selected_action_id, _scores, _supports = self.gate.predict_best(
                        signature=signature,
                        candidate_action_ids=candidate_action_ids,
                    )
                    if not selected_action_id:
                        selected_action_id = fallback_action_id
                elif dominant_risk == "false_alarm" or false_alarm_bucket in {"mid", "high"}:
                    selected_action_id = self.gate.predict(
                        signature=signature,
                        candidate_action_ids=candidate_action_ids,
                        fallback_action_id=fallback_action_id,
                    )
                else:
                    selected_action_id = self.gate.predict(
                        signature=signature,
                        candidate_action_ids=candidate_action_ids,
                        fallback_action_id=fallback_action_id,
                    )
                for action in environment.candidate_actions(state):
                    if action.action_id == selected_action_id:
                        return action
        return super().select_action(state, environment)


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


def continue_episode(
    *,
    state,
    environment,
    policy,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    min_steps_before_stop: int = 4,
) -> float:
    while True:
        decision_probabilities = policy.decision_probabilities(state, environment)
        predicted_hypothesis, confidence = top_probability(decision_probabilities)
        if confidence >= confidence_threshold and len(state.history) >= min_steps_before_stop:
            stop_reason = "confidence"
            break
        candidates = environment.candidate_actions(state)
        if not candidates:
            stop_reason = environment.stop_reason(state) or "open_ended"
            break
        action = policy.select_action(state, environment)
        observation = Observation(
            action_id=action.action_id,
            outcome=environment.sample_observation(action.action_id),
            cost=action.cost,
            action_type=action.action_type,
        )
        apply_observation(state, environment, observation, policy=policy)
        refresh_shift_latent(state, environment, policy)

    final_probabilities = policy.decision_probabilities(state, environment)
    predicted_hypothesis, final_confidence = top_probability(final_probabilities)
    if (
        final_confidence >= confidence_threshold
        and len(state.history) >= min_steps_before_stop
    ):
        stop_reason = "confidence"
    else:
        stop_reason = environment.stop_reason(state) or stop_reason
    return score_episode(
        correct=int(predicted_hypothesis == environment.actual_hypothesis),
        total_cost=state.total_cost,
        final_confidence=final_confidence,
        confidence_threshold=confidence_threshold,
        max_cost=max_cost,
        max_steps=max_steps,
        steps=len(state.history),
        stop_reason=stop_reason,
    )


def rollout_forced_step3(
    *,
    base_state,
    base_environment,
    policy,
    forced_action_id: str,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    rollout_seed: int,
) -> float:
    state = copy.deepcopy(base_state)
    environment = copy.deepcopy(base_environment)
    environment.rng.seed(rollout_seed)
    action = environment.action_by_id(forced_action_id)
    observation = Observation(
        action_id=action.action_id,
        outcome=environment.sample_observation(action.action_id),
        cost=action.cost,
        action_type=action.action_type,
    )
    apply_observation(state, environment, observation, policy=policy)
    refresh_shift_latent(state, environment, policy)
    return continue_episode(
        state=state,
        environment=environment,
        policy=policy,
        confidence_threshold=confidence_threshold,
        max_cost=max_cost,
        max_steps=max_steps,
    )


def collect_training_samples(args: argparse.Namespace) -> list[Step3Sample]:
    return collect_training_samples_for_shift_probability(
        args,
        shift_probability=args.train_shift_probability,
    )


def collect_training_samples_for_shift_probability(
    args: argparse.Namespace,
    *,
    shift_probability: float,
) -> list[Step3Sample]:
    policy = ParserScopeBaselinePolicy(**policy_kwargs())
    samples: list[Step3Sample] = []

    for episode_no in range(args.train_episodes):
        environment = ArtifactDebuggingAmbiguousShiftEnvironment(
            seed=args.seed + episode_no,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            shift_after_step=args.shift_after_step,
            shift_probability=shift_probability,
            false_alarm_length=args.false_alarm_length,
        )
        state = uniform_belief(list(environment.hypotheses()))
        refresh_shift_latent(state, environment, policy)

        while len(state.history) < 2:
            advance_one_step(state, environment, policy)

        if state.history[1].action_id != "ask_user_scope":
            continue
        top_hypothesis_id, _confidence = top_probability(state.probabilities)
        if top_hypothesis_id != "parser_bug":
            continue

        fallback_action_id = policy.select_action(state, environment).action_id
        action_values: dict[str, float] = {}
        for action in environment.candidate_actions(state):
            rollout_values = []
            for repeat in range(args.train_rollouts):
                rollout_seed = (
                    args.seed * 1_000_003
                    + episode_no * 10_007
                    + repeat * 101
                    + sum(ord(char) for char in action.action_id)
                )
                rollout_values.append(
                    rollout_forced_step3(
                        base_state=state,
                        base_environment=environment,
                        policy=policy,
                        forced_action_id=action.action_id,
                        confidence_threshold=args.confidence_threshold,
                        max_cost=args.max_cost,
                        max_steps=args.max_steps,
                        rollout_seed=rollout_seed,
                    )
                )
            action_values[action.action_id] = sum(rollout_values) / len(rollout_values)

        samples.append(
            Step3Sample(
                signature=feature_signature(state),
                scenario_label=environment.scenario_label(),
                fallback_action_id=fallback_action_id,
                action_values=action_values,
            )
        )

    return samples


def evaluate_policy(
    *,
    policy,
    episodes: int,
    seed: int,
    confidence_threshold: float,
    max_cost: float,
    max_steps: int,
    shift_after_step: int,
    false_alarm_length: int,
    shift_probability: float,
):
    from epistemic_engine.runner.run_debugging_benchmark import run_episode

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
    return {
        "policy": summary.policy_name,
        "accuracy": round(summary.accuracy, 3),
        "mean_steps": round(summary.mean_steps, 3),
        "mean_cost": round(summary.mean_cost, 3),
        "mean_conf": round(summary.mean_final_confidence, 3),
        "mean_utility": round(summary.mean_utility, 3),
        "step_stop": round(summary.step_stop_rate, 3),
    }


def print_training_summary(samples: list[Step3Sample]) -> None:
    signature_counts = Counter(sample.signature for sample in samples)
    best_action_counts = Counter(
        max(sample.action_values, key=sample.action_values.get)
        for sample in samples
    )
    override_opportunities = 0
    for sample in samples:
        best_action_id = max(sample.action_values, key=sample.action_values.get)
        fallback_value = sample.action_values.get(sample.fallback_action_id, float("-inf"))
        if (
            best_action_id != sample.fallback_action_id
            and sample.action_values[best_action_id] - fallback_value >= 0.015
        ):
            override_opportunities += 1
    print(f"Training parser_scope states: {len(samples)}")
    print("Best step3 action counts:")
    for action_id, count in best_action_counts.most_common():
        print(f"  {action_id}: {count}")
    print(f"Override opportunities >= 0.015 utility: {override_opportunities}")
    print("Top signatures:")
    for signature, count in signature_counts.most_common(6):
        print(f"  {count} x {signature}")
    print("Scenario mix:")
    print(
        "  true_shift=",
        sum(sample.scenario_label == "true_shift" for sample in samples),
        "false_alarm=",
        sum(sample.scenario_label == "false_alarm" for sample in samples),
    )
    print()


def main() -> None:
    args = parse_args()
    samples = collect_training_samples(args)
    false_alarm_samples = collect_training_samples_for_shift_probability(
        args,
        shift_probability=0.0,
    )
    true_shift_samples = collect_training_samples_for_shift_probability(
        args,
        shift_probability=1.0,
    )
    print_training_summary(samples)
    gate = LearnedActionGate()
    gate.fit(samples)
    false_alarm_gate = LearnedActionGate()
    false_alarm_gate.fit(false_alarm_samples)
    true_shift_gate = LearnedActionGate()
    true_shift_gate.fit(true_shift_samples)

    policies = [
        LatentAdaptiveShiftMemoryPolicy(**policy_kwargs()),
        ParserScopeBaselinePolicy(**policy_kwargs()),
        ParserScopeLearnedGatePolicy(gate=gate, **policy_kwargs()),
        ParserScopeSelectiveLearnedGatePolicy(gate=gate, **policy_kwargs()),
        ParserScopeRegimeAwareGatePolicy(gate=gate, **policy_kwargs()),
        ParserScopeSplitRegimeGatePolicy(
            mixed_gate=gate,
            false_alarm_gate=false_alarm_gate,
            true_shift_gate=true_shift_gate,
            **policy_kwargs(),
        ),
        ParserScopeHybridGatePolicy(gate=gate, **policy_kwargs()),
    ]
    scenarios = [
        ("false_alarm", 0.0),
        ("mixed", 0.5),
        ("true_shift", 1.0),
    ]
    for scenario_name, shift_probability in scenarios:
        rows = [
            evaluate_policy(
                policy=policy,
                episodes=args.eval_episodes,
                seed=args.seed + 20_000,
                confidence_threshold=args.confidence_threshold,
                max_cost=args.max_cost,
                max_steps=args.max_steps,
                shift_after_step=args.shift_after_step,
                false_alarm_length=args.false_alarm_length,
                shift_probability=shift_probability,
            )
            for policy in policies
        ]
        print(
            f"Scenario: {scenario_name} | train_episodes={args.train_episodes}, "
            f"eval_episodes={args.eval_episodes}, train_rollouts={args.train_rollouts}, "
            f"confidence>={args.confidence_threshold}, max_cost={args.max_cost}, "
            f"max_steps={args.max_steps}"
        )
        print_table(rows)
        print()


if __name__ == "__main__":
    main()
