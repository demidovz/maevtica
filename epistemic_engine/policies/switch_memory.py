from __future__ import annotations

import math

from epistemic_engine.beliefs.shift_latent import ShiftLatentUpdater
from epistemic_engine.beliefs.state import entropy, normalize
from epistemic_engine.memory.mode_memory import ModeMemory
from epistemic_engine.memory.question_type_memory import QuestionTypeMemory
from epistemic_engine.questions.policy import candidate_actions, InformationGainPolicy


class SwitchAwareHybridMemoryPolicy(InformationGainPolicy):
    policy_name = "information_gain+switch_memory"

    def __init__(
        self,
        *,
        mode_strength: float = 0.05,
        type_bonus: float = 0.05,
        switch_bonus: float = 0.10,
        surprise_floor: float = 0.95,
        surprise_scale: float = 1.3,
        mode_memory: ModeMemory | None = None,
        type_memory: QuestionTypeMemory | None = None,
    ) -> None:
        self.mode_strength = mode_strength
        self.type_bonus = type_bonus
        self.switch_bonus = switch_bonus
        self.surprise_floor = surprise_floor
        self.surprise_scale = surprise_scale
        self.mode_memory = mode_memory or ModeMemory()
        self.type_memory = type_memory or QuestionTypeMemory()

    def planning_probabilities(self, state, environment) -> dict[str, float]:
        mode_support, switch_pressure, _top_mode, _second_mode = self._mode_context(
            state,
            environment,
        )
        mode_hypotheses = environment.mode_support_to_hypotheses(mode_support)
        effective_mode_strength = self.mode_strength * (1.0 - switch_pressure)
        combined = {}
        for hypothesis_id, probability in state.probabilities.items():
            combined[hypothesis_id] = (
                (1.0 - effective_mode_strength) * probability
                + effective_mode_strength * mode_hypotheses[hypothesis_id]
            )
        return normalize(combined)

    def select_action(self, state, environment):
        candidates = candidate_actions(state, environment)
        mode_support, switch_pressure, top_mode, second_mode = self._mode_context(
            state,
            environment,
        )
        planning_probabilities = self.planning_probabilities(state, environment)
        current_entropy = entropy(planning_probabilities)
        type_support = self.type_memory.next_type_support(state.history, candidates)
        best_score = float("-inf")
        best_action = candidates[0]

        for action in candidates:
            expected_entropy = 0.0
            for outcome in environment.outcomes_for(action.action_id):
                outcome_probability = 0.0
                posterior_scores: dict[str, float] = {}
                for hypothesis_id, prior in planning_probabilities.items():
                    likelihood = environment.likelihood(
                        action.action_id,
                        outcome,
                        hypothesis_id,
                    )
                    posterior_scores[hypothesis_id] = prior * likelihood
                    outcome_probability += prior * likelihood
                posterior = normalize(posterior_scores)
                expected_entropy += outcome_probability * entropy(posterior)

            information_gain = current_entropy - expected_entropy
            score = information_gain / max(action.cost, 1e-9)
            score += self.type_bonus * type_support.get(action.action_type, 0.0)
            score += (
                self.switch_bonus
                * switch_pressure
                * self._mode_disagreement(environment, action.action_id, top_mode, second_mode)
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def record_episode(self, environment, state) -> None:
        self.mode_memory.remember(environment.actual_profile, state.history)
        self.type_memory.remember(state.history)

    def _mode_context(self, state, environment):
        mode_support = self.mode_memory.support(state.history, environment.mode_ids())
        ordered_modes = sorted(
            mode_support.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_mode, top_probability = ordered_modes[0]
        second_mode, second_probability = (
            ordered_modes[1]
            if len(ordered_modes) > 1
            else (top_mode, 0.0)
        )

        if not state.history:
            return mode_support, 0.0, top_mode, second_mode

        mean_surprise = 0.0
        for observation in state.history:
            likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                top_mode,
            )
            mean_surprise += -math.log(max(likelihood, 1e-6))
        mean_surprise /= len(state.history)

        surprise_pressure = min(
            1.0,
            max(0.0, (mean_surprise - self.surprise_floor) / self.surprise_scale),
        )
        mode_gap = top_probability - second_probability
        ambiguity_pressure = min(1.0, max(0.0, 1.0 - mode_gap))
        switch_pressure = min(
            1.0,
            max(0.0, 0.6 * surprise_pressure + 0.4 * ambiguity_pressure),
        )
        return mode_support, switch_pressure, top_mode, second_mode

    def _mode_disagreement(
        self,
        environment,
        action_id: str,
        top_mode: str,
        second_mode: str,
    ) -> float:
        if top_mode == second_mode:
            return 0.0
        distribution_a = environment.mode_distribution(action_id, top_mode)
        distribution_b = environment.mode_distribution(action_id, second_mode)
        return 0.5 * sum(
            abs(distribution_a[outcome] - distribution_b[outcome])
            for outcome in distribution_a
        )


class ReactivatingSwitchMemoryPolicy(SwitchAwareHybridMemoryPolicy):
    policy_name = "information_gain+switch_reactivation"

    def __init__(
        self,
        *,
        reactivation_bonus: float = 0.08,
        type_bonus_decay: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reactivation_bonus = reactivation_bonus
        self.type_bonus_decay = type_bonus_decay

    def select_action(self, state, environment):
        candidates = candidate_actions(state, environment)
        mode_support, switch_pressure, top_mode, second_mode = self._mode_context(
            state,
            environment,
        )
        planning_probabilities = self.planning_probabilities(state, environment)
        current_entropy = entropy(planning_probabilities)
        current_mode_entropy = entropy(mode_support)
        type_support = self.type_memory.next_type_support(state.history, candidates)
        effective_type_bonus = self.type_bonus * max(
            0.0,
            1.0 - self.type_bonus_decay * switch_pressure,
        )
        best_score = float("-inf")
        best_action = candidates[0]

        for action in candidates:
            expected_entropy = 0.0
            for outcome in environment.outcomes_for(action.action_id):
                outcome_probability = 0.0
                posterior_scores: dict[str, float] = {}
                for hypothesis_id, prior in planning_probabilities.items():
                    likelihood = environment.likelihood(
                        action.action_id,
                        outcome,
                        hypothesis_id,
                    )
                    posterior_scores[hypothesis_id] = prior * likelihood
                    outcome_probability += prior * likelihood
                posterior = normalize(posterior_scores)
                expected_entropy += outcome_probability * entropy(posterior)

            expected_mode_entropy = 0.0
            for outcome in environment.outcomes_for(action.action_id):
                outcome_probability = 0.0
                posterior_scores: dict[str, float] = {}
                for mode_id, prior in mode_support.items():
                    likelihood = environment.mode_likelihood(
                        action.action_id,
                        outcome,
                        mode_id,
                    )
                    posterior_scores[mode_id] = prior * likelihood
                    outcome_probability += prior * likelihood
                posterior = normalize(posterior_scores)
                expected_mode_entropy += outcome_probability * entropy(posterior)

            information_gain = current_entropy - expected_entropy
            mode_information_gain = current_mode_entropy - expected_mode_entropy
            score = information_gain / max(action.cost, 1e-9)
            score += effective_type_bonus * type_support.get(action.action_type, 0.0)
            score += (
                self.reactivation_bonus
                * switch_pressure
                * mode_information_gain
                / max(action.cost, 1e-9)
            )
            score += (
                self.switch_bonus
                * switch_pressure
                * self._mode_disagreement(environment, action.action_id, top_mode, second_mode)
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class ConfirmedSwitchMemoryPolicy(SwitchAwareHybridMemoryPolicy):
    policy_name = "information_gain+confirmed_switch"

    def __init__(
        self,
        *,
        confirmation_window: int = 2,
        required_confirmations: int = 2,
        fallback_scale: float = 0.30,
        confirmation_margin: float = 1.10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.confirmation_window = confirmation_window
        self.required_confirmations = required_confirmations
        self.fallback_scale = fallback_scale
        self.confirmation_margin = confirmation_margin

    def _mode_context(self, state, environment):
        mode_support, switch_pressure, top_mode, second_mode = super()._mode_context(
            state,
            environment,
        )
        confirmation_strength = self._confirmation_strength(
            state,
            environment,
            top_mode,
            second_mode,
        )
        return (
            mode_support,
            switch_pressure * confirmation_strength,
            top_mode,
            second_mode,
        )

    def _confirmation_strength(
        self,
        state,
        environment,
        top_mode: str,
        second_mode: str,
    ) -> float:
        if top_mode == second_mode or not state.history:
            return 0.0

        recent_history = state.history[-self.confirmation_window :]
        confirmations = 0
        total_margin = 0.0

        for observation in recent_history:
            top_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                top_mode,
            )
            second_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                second_mode,
            )
            if second_likelihood > top_likelihood * self.confirmation_margin:
                confirmations += 1
                total_margin += (
                    (second_likelihood - top_likelihood)
                    / max(second_likelihood, 1e-9)
                )

        if confirmations >= self.required_confirmations:
            margin_bonus = min(1.0, total_margin / max(confirmations, 1))
            return min(
                1.0,
                confirmations / max(len(recent_history), 1) + 0.35 * margin_bonus,
            )

        if not recent_history:
            return 0.0
        return self.fallback_scale * (confirmations / len(recent_history))


class PersistentShiftMemoryPolicy(SwitchAwareHybridMemoryPolicy):
    policy_name = "information_gain+persistent_shift"

    def __init__(
        self,
        *,
        recent_window: int = 3,
        anomaly_scale: float = 1.0,
        persistence_scale: float = 0.85,
        rebound_penalty: float = 0.60,
        anomaly_weight: float = 0.30,
        persistence_weight: float = 0.70,
        streak_bonus: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.recent_window = recent_window
        self.anomaly_scale = anomaly_scale
        self.persistence_scale = persistence_scale
        self.rebound_penalty = rebound_penalty
        self.anomaly_weight = anomaly_weight
        self.persistence_weight = persistence_weight
        self.streak_bonus = streak_bonus

    def _mode_context(self, state, environment):
        mode_support = self.mode_memory.support(state.history, environment.mode_ids())
        ordered_modes = sorted(
            mode_support.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_mode, top_probability = ordered_modes[0]
        default_second_mode, second_probability = (
            ordered_modes[1]
            if len(ordered_modes) > 1
            else (top_mode, 0.0)
        )

        if not state.history:
            return mode_support, 0.0, top_mode, default_second_mode

        mean_surprise = 0.0
        for observation in state.history:
            likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                top_mode,
            )
            mean_surprise += -math.log(max(likelihood, 1e-6))
        mean_surprise /= len(state.history)

        surprise_pressure = min(
            1.0,
            max(0.0, (mean_surprise - self.surprise_floor) / self.surprise_scale),
        )
        mode_gap = top_probability - second_probability
        ambiguity_pressure = min(1.0, max(0.0, 1.0 - mode_gap))
        base_switch_pressure = min(
            1.0,
            max(0.0, 0.6 * surprise_pressure + 0.4 * ambiguity_pressure),
        )

        candidate_mode = self._best_recent_alternative_mode(
            state,
            environment,
            top_mode,
            default_second_mode,
        )
        anomaly_pressure, persistence_pressure = self._recent_shift_signals(
            state,
            environment,
            top_mode,
            candidate_mode,
        )
        switch_evidence = min(
            1.0,
            self.anomaly_weight * anomaly_pressure
            + self.persistence_weight * persistence_pressure,
        )
        switch_pressure = min(1.0, base_switch_pressure * switch_evidence)
        return mode_support, switch_pressure, top_mode, candidate_mode

    def _best_recent_alternative_mode(
        self,
        state,
        environment,
        top_mode: str,
        default_second_mode: str,
    ) -> str:
        if len(environment.mode_ids()) <= 1 or not state.history:
            return default_second_mode

        recent_history = state.history[-self.recent_window :]
        best_mode = default_second_mode
        best_score = float("-inf")

        for mode_id in environment.mode_ids():
            if mode_id == top_mode:
                continue
            score = 0.0
            for index, observation in enumerate(recent_history, start=1):
                weight = index / len(recent_history)
                candidate_likelihood = environment.mode_likelihood(
                    observation.action_id,
                    observation.outcome,
                    mode_id,
                )
                top_likelihood = environment.mode_likelihood(
                    observation.action_id,
                    observation.outcome,
                    top_mode,
                )
                score += weight * (
                    math.log(max(candidate_likelihood, 1e-6))
                    - math.log(max(top_likelihood, 1e-6))
                )
            if score > best_score:
                best_score = score
                best_mode = mode_id

        return best_mode

    def _recent_shift_signals(
        self,
        state,
        environment,
        top_mode: str,
        candidate_mode: str,
    ) -> tuple[float, float]:
        if candidate_mode == top_mode or not state.history:
            return 0.0, 0.0

        recent_history = state.history[-self.recent_window :]
        weighted_positive = 0.0
        weighted_negative = 0.0
        total_weight = 0.0
        latest_advantage = 0.0
        positive_streak = 0

        for reverse_index, observation in enumerate(reversed(recent_history), start=1):
            forward_index = len(recent_history) - reverse_index + 1
            weight = forward_index / len(recent_history)
            candidate_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                candidate_mode,
            )
            top_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                top_mode,
            )
            advantage = (
                math.log(max(candidate_likelihood, 1e-6))
                - math.log(max(top_likelihood, 1e-6))
            )
            if reverse_index == 1:
                latest_advantage = advantage
            if advantage > 0.0 and positive_streak == reverse_index - 1:
                positive_streak += 1

            weighted_positive += weight * max(0.0, advantage)
            weighted_negative += weight * max(0.0, -advantage)
            total_weight += weight

        weighted_positive /= max(total_weight, 1e-9)
        weighted_negative /= max(total_weight, 1e-9)
        anomaly_pressure = min(
            1.0,
            max(0.0, latest_advantage / max(self.anomaly_scale, 1e-9)),
        )
        streak_pressure = positive_streak / max(len(recent_history), 1)
        persistence_score = (
            weighted_positive
            - self.rebound_penalty * weighted_negative
            + self.streak_bonus * streak_pressure
        )
        persistence_pressure = min(
            1.0,
            max(0.0, persistence_score / max(self.persistence_scale, 1e-9)),
        )
        return anomaly_pressure, persistence_pressure


class AdaptiveShiftMemoryPolicy(PersistentShiftMemoryPolicy):
    policy_name = "information_gain+adaptive_shift"

    def __init__(
        self,
        *,
        false_alarm_scale: float = 1.20,
        neutral_gate: float = 0.50,
        min_aggressive_gate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.false_alarm_scale = false_alarm_scale
        self.neutral_gate = neutral_gate
        self.min_aggressive_gate = min_aggressive_gate

    def _mode_context(self, state, environment):
        mode_support, aggressive_pressure, top_mode, aggressive_second_mode = (
            SwitchAwareHybridMemoryPolicy._mode_context(self, state, environment)
        )
        (
            _mode_support,
            cautious_pressure,
            _top_mode,
            cautious_second_mode,
        ) = super()._mode_context(state, environment)

        anomaly_pressure, persistence_pressure = self._recent_shift_signals(
            state,
            environment,
            top_mode,
            cautious_second_mode,
        )
        aggressive_gate = self._aggressive_gate(
            anomaly_pressure,
            persistence_pressure,
        )
        switch_pressure = (
            aggressive_gate * aggressive_pressure
            + (1.0 - aggressive_gate) * cautious_pressure
        )
        second_mode = (
            aggressive_second_mode
            if aggressive_gate >= 0.5
            else cautious_second_mode
        )
        return mode_support, switch_pressure, top_mode, second_mode

    def _aggressive_gate(
        self,
        anomaly_pressure: float,
        persistence_pressure: float,
    ) -> float:
        if anomaly_pressure <= 1e-9 and persistence_pressure <= 1e-9:
            return self.neutral_gate

        false_alarm_risk = max(0.0, anomaly_pressure - persistence_pressure)
        denominator = persistence_pressure + self.false_alarm_scale * false_alarm_risk
        if denominator <= 1e-9:
            return self.min_aggressive_gate

        gate = persistence_pressure / denominator
        return min(1.0, max(self.min_aggressive_gate, gate))


class LatentAdaptiveShiftMemoryPolicy(SwitchAwareHybridMemoryPolicy):
    policy_name = "information_gain+latent_shift"

    def __init__(
        self,
        *,
        recent_window: int = 3,
        anomaly_scale: float = 1.0,
        persistence_scale: float = 0.85,
        rebound_penalty: float = 0.60,
        streak_bonus: float = 0.35,
        false_alarm_scale: float = 1.20,
        neutral_gate: float = 0.50,
        min_aggressive_gate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.recent_window = recent_window
        self.anomaly_scale = anomaly_scale
        self.persistence_scale = persistence_scale
        self.rebound_penalty = rebound_penalty
        self.streak_bonus = streak_bonus
        self.false_alarm_scale = false_alarm_scale
        self.neutral_gate = neutral_gate
        self.min_aggressive_gate = min_aggressive_gate
        self.shift_latent_updater = ShiftLatentUpdater(
            recent_window=self.recent_window,
            surprise_floor=self.surprise_floor,
            surprise_scale=self.surprise_scale,
            anomaly_scale=self.anomaly_scale,
            persistence_scale=self.persistence_scale,
            rebound_penalty=self.rebound_penalty,
            streak_bonus=self.streak_bonus,
            false_alarm_scale=self.false_alarm_scale,
            neutral_gate=self.neutral_gate,
            min_aggressive_gate=self.min_aggressive_gate,
        )

    def infer_latent_state(self, state, environment):
        if state.shift_latent is not None:
            return state.shift_latent
        return self.shift_latent_updater.infer(
            state=state,
            environment=environment,
            mode_memory=self.mode_memory,
        )

    def planning_probabilities(self, state, environment) -> dict[str, float]:
        latent_state = self.infer_latent_state(state, environment)
        mode_hypotheses = environment.mode_support_to_hypotheses(latent_state.mode_support)
        effective_mode_strength = self.mode_strength * (1.0 - latent_state.switch_pressure)
        combined = {}
        for hypothesis_id, probability in state.probabilities.items():
            combined[hypothesis_id] = (
                (1.0 - effective_mode_strength) * probability
                + effective_mode_strength * mode_hypotheses[hypothesis_id]
            )
        return normalize(combined)

    def select_action(self, state, environment):
        candidates = candidate_actions(state, environment)
        latent_state = self.infer_latent_state(state, environment)
        planning_probabilities = self.planning_probabilities(state, environment)
        current_entropy = entropy(planning_probabilities)
        type_support = self.type_memory.next_type_support(state.history, candidates)
        best_score = float("-inf")
        best_action = candidates[0]

        for action in candidates:
            expected_entropy = 0.0
            for outcome in environment.outcomes_for(action.action_id):
                outcome_probability = 0.0
                posterior_scores: dict[str, float] = {}
                for hypothesis_id, prior in planning_probabilities.items():
                    likelihood = environment.likelihood(
                        action.action_id,
                        outcome,
                        hypothesis_id,
                    )
                    posterior_scores[hypothesis_id] = prior * likelihood
                    outcome_probability += prior * likelihood
                posterior = normalize(posterior_scores)
                expected_entropy += outcome_probability * entropy(posterior)

            information_gain = current_entropy - expected_entropy
            score = information_gain / max(action.cost, 1e-9)
            score += self.type_bonus * type_support.get(action.action_type, 0.0)
            score += (
                self.switch_bonus
                * latent_state.switch_pressure
                * self._mode_disagreement(
                    environment,
                    action.action_id,
                    latent_state.top_mode,
                    latent_state.default_second_mode
                    if latent_state.aggressive_gate >= 0.5
                    else latent_state.candidate_mode,
                )
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
