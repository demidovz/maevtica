from __future__ import annotations

import random

from epistemic_engine.models import BeliefState, QuestionAction
from epistemic_engine.questions.policy import QuestionPolicy, candidate_actions


class CheapestQuestionPolicy(QuestionPolicy):
    policy_name = "cheapest"

    def select_action(self, state: BeliefState, environment) -> QuestionAction:
        return min(candidate_actions(state, environment), key=lambda action: action.cost)


class RandomQuestionPolicy(QuestionPolicy):
    policy_name = "random"

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def select_action(self, state: BeliefState, environment) -> QuestionAction:
        return self.rng.choice(candidate_actions(state, environment))
