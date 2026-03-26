from __future__ import annotations

from dataclasses import dataclass

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.models import Observation, QuestionAction


@dataclass(frozen=True)
class QuestionTypeCase:
    observations: tuple[tuple[str, str, str], ...]


class QuestionTypeMemory:
    def __init__(
        self,
        *,
        max_cases: int = 256,
        min_total_cases: int = 20,
        min_history_length: int = 2,
        match_bonus: float = 2.0,
        mismatch_penalty: float = 0.22,
        missing_penalty: float = 0.78,
    ) -> None:
        self.max_cases = max_cases
        self.min_total_cases = min_total_cases
        self.min_history_length = min_history_length
        self.match_bonus = match_bonus
        self.mismatch_penalty = mismatch_penalty
        self.missing_penalty = missing_penalty
        self.cases: list[QuestionTypeCase] = []

    def remember(self, history: list[Observation]) -> None:
        self.cases.append(
            QuestionTypeCase(
                observations=tuple(
                    (item.action_id, item.action_type, item.outcome)
                    for item in history
                )
            )
        )
        if len(self.cases) > self.max_cases:
            del self.cases[0]

    def next_type_support(
        self,
        history: list[Observation],
        available_actions: list[QuestionAction],
    ) -> dict[str, float]:
        action_types = sorted({action.action_type for action in available_actions})
        if (
            len(history) < self.min_history_length
            or len(self.cases) < self.min_total_cases
            or not action_types
        ):
            return {action_type: 0.0 for action_type in action_types}

        query = tuple((item.action_id, item.outcome) for item in history)
        asked_actions = {item.action_id for item in history}
        scores = {action_type: 1e-6 for action_type in action_types}

        for case in self.cases:
            case_score = self._match_score(query, case.observations)
            next_type = self._next_unasked_type(case.observations, asked_actions)
            if next_type is not None and next_type in scores:
                scores[next_type] += case_score

        normalized = normalize(scores)
        uniform = 1.0 / len(action_types)
        return {
            action_type: max(0.0, normalized[action_type] - uniform)
            for action_type in action_types
        }

    def _match_score(
        self,
        query: tuple[tuple[str, str], ...],
        observations: tuple[tuple[str, str, str], ...],
    ) -> float:
        case_map = {
            action_id: outcome
            for action_id, _action_type, outcome in observations
        }
        score = 1.0
        for action_id, outcome in query:
            remembered_outcome = case_map.get(action_id)
            if remembered_outcome is None:
                score *= self.missing_penalty
            elif remembered_outcome == outcome:
                score *= self.match_bonus
            else:
                score *= self.mismatch_penalty
        return score

    def _next_unasked_type(
        self,
        observations: tuple[tuple[str, str, str], ...],
        asked_actions: set[str],
    ) -> str | None:
        for action_id, action_type, _outcome in observations:
            if action_id not in asked_actions:
                return action_type
        return None
