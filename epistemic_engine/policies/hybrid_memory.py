from __future__ import annotations

from epistemic_engine.beliefs.state import entropy, normalize
from epistemic_engine.memory.mode_memory import ModeMemory
from epistemic_engine.memory.question_type_memory import QuestionTypeMemory
from epistemic_engine.questions.policy import candidate_actions, InformationGainPolicy


class HybridMemoryPolicy(InformationGainPolicy):
    policy_name = "information_gain+hybrid_memory"

    def __init__(
        self,
        *,
        mode_strength: float = 0.05,
        type_bonus: float = 0.05,
        mode_memory: ModeMemory | None = None,
        type_memory: QuestionTypeMemory | None = None,
    ) -> None:
        self.mode_strength = mode_strength
        self.type_bonus = type_bonus
        self.mode_memory = mode_memory or ModeMemory()
        self.type_memory = type_memory or QuestionTypeMemory()

    def planning_probabilities(self, state, environment) -> dict[str, float]:
        mode_support = self.mode_memory.support(state.history, environment.mode_ids())
        mode_hypotheses = environment.mode_support_to_hypotheses(mode_support)
        combined = {}
        for hypothesis_id, probability in state.probabilities.items():
            combined[hypothesis_id] = (
                (1.0 - self.mode_strength) * probability
                + self.mode_strength * mode_hypotheses[hypothesis_id]
            )
        return normalize(combined)

    def select_action(self, state, environment):
        candidates = candidate_actions(state, environment)
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
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def record_episode(self, environment, state) -> None:
        self.mode_memory.remember(environment.actual_profile, state.history)
        self.type_memory.remember(state.history)
