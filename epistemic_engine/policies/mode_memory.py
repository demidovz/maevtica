from __future__ import annotations

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.memory.mode_memory import ModeMemory
from epistemic_engine.questions.policy import InformationGainPolicy


class ModeMemoryPolicy(InformationGainPolicy):
    policy_name = "information_gain+mode_memory"

    def __init__(
        self,
        *,
        mode_strength: float = 0.22,
        memory: ModeMemory | None = None,
    ) -> None:
        self.mode_strength = mode_strength
        self.memory = memory or ModeMemory()

    def planning_probabilities(self, state, environment) -> dict[str, float]:
        mode_support = self.memory.support(state.history, environment.mode_ids())
        mode_hypotheses = environment.mode_support_to_hypotheses(mode_support)
        combined = {}
        for hypothesis_id, probability in state.probabilities.items():
            combined[hypothesis_id] = (
                (1.0 - self.mode_strength) * probability
                + self.mode_strength * mode_hypotheses[hypothesis_id]
            )
        return normalize(combined)

    def record_episode(self, environment, state) -> None:
        self.memory.remember(environment.actual_profile, state.history)
