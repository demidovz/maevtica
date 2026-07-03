from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from epistemic_explorer.models import utc_now


@dataclass(frozen=True)
class StrategyGenome:
    criticality: float
    formalization_bias: float
    simulation_bias: float
    analogy_bias: float
    counterexample_bias: float
    merge_bias: float
    split_bias: float
    reasoning_depth: float
    quick_check_bias: float
    risk_tolerance: float
    collaboration_bias: float
    meta_bias: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ResearchStrategy:
    strategy_id: str
    genome: StrategyGenome
    age: int = 0
    parent_id: str | None = None
    mutation_history: list[str] = field(default_factory=list)
    existence_cost: float = 1.0
    energy: float = 10.0
    successes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    alive: bool = True
    species_hint: str = "unclassified"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["genome"] = self.genome.to_dict()
        return data


@dataclass(frozen=True)
class StrategyActionResult:
    strategy_id: str
    epoch: int
    domain: str
    cost: float
    energy_delta: float
    hypotheses_created: int
    hypotheses_falsified: int
    hypotheses_survived_strong_test: int
    discriminating_experiments: int
    valuable_questions: int
    bridges_created: int
    compression_delta: float
    dead_end_cost: float
    success_events: list[str]
    failure_events: list[str]

    def epistemic_value(self) -> float:
        return (
            2.0 * self.hypotheses_falsified
            + 2.5 * self.hypotheses_survived_strong_test
            + 3.0 * self.discriminating_experiments
            + 1.5 * self.valuable_questions
            + 2.0 * self.bridges_created
            + 2.0 * max(0.0, self.compression_delta)
            - self.dead_end_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    population_size: int
    alive_count: int
    births: int
    deaths: int
    selection_pressure: float
    experiment_yield: float
    question_value: float
    bridge_score: float
    compression_score: float
    research_diversity: float
    dead_end_rate: float
    mean_energy: float
    dominant_species_share: float
    meta_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EcosystemSummary:
    population_target: int
    epochs: int
    seed: int
    final_alive: int
    total_births: int
    total_deaths: int
    mean_selection_pressure: float
    mean_experiment_yield: float
    mean_question_value: float
    mean_bridge_score: float
    mean_compression_score: float
    mean_research_diversity: float
    mean_dead_end_rate: float
    final_species_counts: dict[str, int]
    total_epistemic_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

