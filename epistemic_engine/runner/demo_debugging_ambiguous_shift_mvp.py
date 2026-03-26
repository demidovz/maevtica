from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epistemic_engine.beliefs.state import top_probability, uniform_belief
from epistemic_engine.environments.debugging import DebuggingAmbiguousShiftEnvironment
from epistemic_engine.models import Observation
from epistemic_engine.policies.switch_memory import PersistentShiftMemoryPolicy
from epistemic_engine.revision.updater import apply_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пошаговая демонстрация debugging среды с ложными и настоящими shift-сигналами."
    )
    parser.add_argument(
        "--actual",
        choices=list(DebuggingAmbiguousShiftEnvironment().hypotheses()),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--shift-probability", type=float, default=0.5)
    parser.add_argument("--false-alarm-length", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = DebuggingAmbiguousShiftEnvironment(
        actual_hypothesis=args.actual,
        seed=args.seed,
        max_cost=args.max_cost,
        max_steps=args.max_steps,
        shift_after_step=2,
        shift_probability=args.shift_probability,
        false_alarm_length=args.false_alarm_length,
    )
    state = uniform_belief(list(environment.hypotheses()))
    policy = PersistentShiftMemoryPolicy(
        mode_strength=0.02,
        type_bonus=0.05,
        switch_bonus=0.20,
        recent_window=3,
        anomaly_scale=1.0,
        persistence_scale=0.85,
        rebound_penalty=0.60,
        anomaly_weight=0.30,
        persistence_weight=0.70,
        streak_bonus=0.35,
    )

    print("Epistemic Engine MVP: debugging ambiguous-shift diagnosis")
    print(f"Скрытая реальная причина: {environment.actual_hypothesis}")
    print(f"Сценарий среды: {environment.scenario_label()}")
    print(f"Начальный режим: {environment.initial_profile}")
    print(f"Ложный/новый режим: {environment.shifted_profile}")
    print(f"Сигнал сдвига начнётся после шага: {environment.shift_after_step}")
    print(
        f"Ограничения: confidence>={args.confidence_threshold}, "
        f"max_cost={args.max_cost}, max_steps={args.max_steps}"
    )
    print()

    while True:
        policy_probabilities = policy.decision_probabilities(state, environment)
        top_id, confidence = top_probability(policy_probabilities)
        print(f"Текущая лучшая гипотеза: {top_id} ({confidence:.3f})")
        if confidence >= args.confidence_threshold and len(state.history) >= 4:
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
            f"Следующий вопрос/шаг: [{action.action_type}] {action.action_id} | "
            f"{action.description}"
        )
        print(f"Активный режим мира на этом шаге: {profile_before}")
        print(f"Наблюдение: {outcome}")
        apply_observation(state, environment, observation)
        remaining_budget = environment.remaining_budget(state)
        if remaining_budget is not None:
            print(f"Осталось бюджета: {remaining_budget:.2f}")
        print()

    print("Финал")
    final_probabilities = policy.decision_probabilities(state, environment)
    final_id, final_confidence = top_probability(final_probabilities)
    print(f"Предсказанная причина: {final_id}")
    print(f"Уверенность: {final_confidence:.3f}")
    print(f"Суммарная стоимость: {state.total_cost:.2f}")
    print(f"Сделано шагов: {len(state.history)}")
    stop_reason = (
        "confidence"
        if final_confidence >= args.confidence_threshold and len(state.history) >= 4
        else environment.stop_reason(state) or "open_ended"
    )
    print(f"Причина остановки: {stop_reason}")
    if state.revisions:
        print("Пересмотры:")
        for revision in state.revisions:
            print(f"- {revision.previous_top} -> {revision.new_top}: {revision.reason}")


if __name__ == "__main__":
    main()
