from __future__ import annotations

from abc import ABC, abstractmethod

from epistemic_engine.beliefs.state import entropy, normalize
from epistemic_engine.models import BeliefState, QuestionAction


def candidate_actions(state: BeliefState, environment) -> list[QuestionAction]:
    if hasattr(environment, "candidate_actions"):
        return environment.candidate_actions(state)
    return environment.available_actions(state.asked_actions)


class QuestionPolicy(ABC):
    policy_name = "base"

    def planning_probabilities(self, state: BeliefState, environment) -> dict[str, float]:
        return state.probabilities

    def decision_probabilities(self, state: BeliefState, environment) -> dict[str, float]:
        return state.probabilities

    def record_episode(self, environment, state: BeliefState) -> None:
        return None

    @abstractmethod
    def select_action(self, state: BeliefState, environment) -> QuestionAction:
        raise NotImplementedError


class InformationGainPolicy(QuestionPolicy):
    policy_name = "information_gain"

    def select_action(self, state: BeliefState, environment) -> QuestionAction:
        candidates = candidate_actions(state, environment)
        planning_probabilities = self.planning_probabilities(state, environment)
        current_entropy = entropy(planning_probabilities)
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
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
