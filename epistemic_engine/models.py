from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_engine.beliefs.shift_latent import ShiftLatentState


@dataclass(frozen=True)
class QuestionAction:
    action_id: str
    description: str
    cost: float
    action_type: str = "probe"


@dataclass(frozen=True)
class Observation:
    action_id: str
    outcome: str
    cost: float
    action_type: str = "probe"


@dataclass(frozen=True)
class RevisionEvent:
    previous_top: str
    new_top: str
    reason: str


@dataclass
class BeliefState:
    probabilities: dict[str, float]
    asked_actions: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    history: list[Observation] = field(default_factory=list)
    revisions: list[RevisionEvent] = field(default_factory=list)
    shift_latent: ShiftLatentState | None = None
    shift_latent_history: list[ShiftLatentState] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeMetrics:
    actual_hypothesis: str
    predicted_hypothesis: str
    correct: int
    steps: int
    total_cost: float
    final_confidence: float
    utility: float
    budget_stop: int
    step_stop: int


@dataclass(frozen=True)
class BenchmarkResult:
    policy_name: str
    accuracy: float
    mean_steps: float
    mean_cost: float
    mean_final_confidence: float
    mean_utility: float
    budget_stop_rate: float
    step_stop_rate: float
