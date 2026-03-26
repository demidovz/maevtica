from __future__ import annotations

from epistemic_engine.beliefs.state import entropy, normalize
from epistemic_engine.memory.question_type_memory import QuestionTypeMemory
from epistemic_engine.questions.policy import candidate_actions, InformationGainPolicy


class QuestionTypeMemoryPolicy(InformationGainPolicy):
    policy_name = "information_gain+type_memory"

    def __init__(
        self,
        *,
        type_bonus: float = 0.05,
        memory: QuestionTypeMemory | None = None,
    ) -> None:
        self.type_bonus = type_bonus
        self.memory = memory or QuestionTypeMemory()

    def select_action(self, state, environment):
        candidates = candidate_actions(state, environment)
        planning_probabilities = self.planning_probabilities(state, environment)
        current_entropy = entropy(planning_probabilities)
        type_support = self.memory.next_type_support(state.history, candidates)
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
        self.memory.remember(state.history)
