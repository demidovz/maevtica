from __future__ import annotations

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.memory.case_memory import CaseMemory
from epistemic_engine.questions.policy import InformationGainPolicy


class MemoryAugmentedInformationGainPolicy(InformationGainPolicy):
    policy_name = "information_gain+memory"

    def __init__(
        self,
        *,
        memory_strength: float = 0.35,
        memory: CaseMemory | None = None,
    ) -> None:
        self.memory_strength = memory_strength
        self.memory = memory or CaseMemory()

    def planning_probabilities(self, state, environment) -> dict[str, float]:
        memory_support = self.memory.support(
            state.history,
            list(state.probabilities),
        )
        combined = {}
        for hypothesis_id, probability in state.probabilities.items():
            combined[hypothesis_id] = (
                (1.0 - self.memory_strength) * probability
                + self.memory_strength * memory_support[hypothesis_id]
            )
        return normalize(combined)

    def record_episode(self, environment, state) -> None:
        self.memory.remember(environment.actual_hypothesis, state.history)
