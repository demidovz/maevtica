from __future__ import annotations

import math

from epistemic_engine.models import BeliefState


def normalize(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0.0:
        uniform = 1.0 / len(probabilities)
        return {key: uniform for key in probabilities}
    return {
        key: value / total
        for key, value in probabilities.items()
    }


def uniform_belief(hypothesis_ids: list[str]) -> BeliefState:
    uniform = 1.0 / len(hypothesis_ids)
    return BeliefState(
        probabilities={hypothesis_id: uniform for hypothesis_id in hypothesis_ids}
    )


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )


def top_hypothesis(state: BeliefState) -> tuple[str, float]:
    return top_probability(state.probabilities)


def top_probability(probabilities: dict[str, float]) -> tuple[str, float]:
    hypothesis_id = max(probabilities, key=probabilities.get)
    return hypothesis_id, probabilities[hypothesis_id]
