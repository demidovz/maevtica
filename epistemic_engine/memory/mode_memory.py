from __future__ import annotations

from dataclasses import dataclass

from epistemic_engine.beliefs.state import normalize
from epistemic_engine.models import Observation


@dataclass(frozen=True)
class ModeCase:
    mode_id: str
    observations: tuple[tuple[str, str], ...]


class ModeMemory:
    def __init__(
        self,
        *,
        max_cases_per_mode: int = 48,
        min_total_cases: int = 18,
        min_history_length: int = 2,
        match_bonus: float = 2.1,
        mismatch_penalty: float = 0.20,
        missing_penalty: float = 0.80,
    ) -> None:
        self.max_cases_per_mode = max_cases_per_mode
        self.min_total_cases = min_total_cases
        self.min_history_length = min_history_length
        self.match_bonus = match_bonus
        self.mismatch_penalty = mismatch_penalty
        self.missing_penalty = missing_penalty
        self.cases_by_mode: dict[str, list[ModeCase]] = {}

    def remember(self, mode_id: str, history: list[Observation]) -> None:
        case = ModeCase(
            mode_id=mode_id,
            observations=tuple((item.action_id, item.outcome) for item in history),
        )
        bucket = self.cases_by_mode.setdefault(mode_id, [])
        bucket.append(case)
        if len(bucket) > self.max_cases_per_mode:
            del bucket[0]

    def support(
        self,
        history: list[Observation],
        mode_ids: list[str],
    ) -> dict[str, float]:
        if (
            len(history) < self.min_history_length
            or self.total_cases() < self.min_total_cases
        ):
            uniform = 1.0 / len(mode_ids)
            return {mode_id: uniform for mode_id in mode_ids}

        query = tuple((item.action_id, item.outcome) for item in history)
        scores = {mode_id: 1e-3 for mode_id in mode_ids}

        for mode_id in mode_ids:
            for case in self.cases_by_mode.get(mode_id, []):
                scores[mode_id] += self._match_score(query, case.observations)

        return normalize(scores)

    def total_cases(self) -> int:
        return sum(len(cases) for cases in self.cases_by_mode.values())

    def _match_score(
        self,
        query: tuple[tuple[str, str], ...],
        observations: tuple[tuple[str, str], ...],
    ) -> float:
        case_map = dict(observations)
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
