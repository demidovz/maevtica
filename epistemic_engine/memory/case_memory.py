from __future__ import annotations

from dataclasses import dataclass

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.models import Observation


@dataclass(frozen=True)
class ResolvedCase:
    actual_hypothesis: str
    observations: tuple[tuple[str, str], ...]


class CaseMemory:
    def __init__(
        self,
        *,
        max_cases_per_hypothesis: int = 48,
        min_total_cases: int = 16,
        min_history_length: int = 2,
        match_bonus: float = 2.2,
        mismatch_penalty: float = 0.18,
        missing_penalty: float = 0.75,
    ) -> None:
        self.max_cases_per_hypothesis = max_cases_per_hypothesis
        self.min_total_cases = min_total_cases
        self.min_history_length = min_history_length
        self.match_bonus = match_bonus
        self.mismatch_penalty = mismatch_penalty
        self.missing_penalty = missing_penalty
        self.cases_by_hypothesis: dict[str, list[ResolvedCase]] = {}

    def remember(self, actual_hypothesis: str, history: list[Observation]) -> None:
        case = ResolvedCase(
            actual_hypothesis=actual_hypothesis,
            observations=tuple((item.action_id, item.outcome) for item in history),
        )
        bucket = self.cases_by_hypothesis.setdefault(actual_hypothesis, [])
        bucket.append(case)
        if len(bucket) > self.max_cases_per_hypothesis:
            del bucket[0]

    def support(
        self,
        history: list[Observation],
        hypothesis_ids: list[str],
    ) -> dict[str, float]:
        if (
            len(history) < self.min_history_length
            or self.total_cases() < self.min_total_cases
        ):
            uniform = 1.0 / len(hypothesis_ids)
            return {hypothesis_id: uniform for hypothesis_id in hypothesis_ids}

        query = tuple((item.action_id, item.outcome) for item in history)
        scores = {hypothesis_id: 1e-3 for hypothesis_id in hypothesis_ids}

        for hypothesis_id in hypothesis_ids:
            for case in self.cases_by_hypothesis.get(hypothesis_id, []):
                scores[hypothesis_id] += self._match_score(query, case.observations)

        return normalize(scores)

    def total_cases(self) -> int:
        return sum(len(cases) for cases in self.cases_by_hypothesis.values())

    def _match_score(
        self,
        query: tuple[tuple[str, str], ...],
        case_observations: tuple[tuple[str, str], ...],
    ) -> float:
        case_map = dict(case_observations)
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
