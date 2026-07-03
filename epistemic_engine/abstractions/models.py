from __future__ import annotations

from dataclasses import dataclass, field


Atom = tuple[str, str]


@dataclass(frozen=True)
class ToyObject:
    object_id: str
    features: dict[str, str]
    effects: frozenset[str]

    @property
    def atoms(self) -> frozenset[Atom]:
        return frozenset(self.features.items())


@dataclass(frozen=True)
class ObjectObservation:
    step: int
    toy_object: ToyObject


@dataclass
class Abstraction:
    abstraction_id: str
    parents: tuple[str, ...]
    conditions: frozenset[Atom]
    expected_effects: frozenset[str]
    object_ids: set[str]
    created_at: int
    depth: int = 1
    use_count: int = 0
    survived_observations: int = 0
    destroyed_at: int | None = None
    children: set[str] = field(default_factory=set)
    composition_count: int = 0
    failure_count: int = 0

    @property
    def alive(self) -> bool:
        return self.destroyed_at is None

    @property
    def lifetime(self) -> int:
        end = self.destroyed_at if self.destroyed_at is not None else self.created_at + self.survived_observations
        return max(0, end - self.created_at)

    @property
    def life_reuse(self) -> int:
        return self.lifetime * self.use_count


@dataclass
class HyperedgeAbstraction:
    edge_id: str
    tail_atoms: frozenset[Atom]
    head_effects: frozenset[str]
    source_edges: frozenset[str]
    object_ids: set[str]
    created_at: int
    rank: int = 1
    use_count: int = 0
    survived_observations: int = 0
    destroyed_at: int | None = None
    failure_count: int = 0

    @property
    def alive(self) -> bool:
        return self.destroyed_at is None

    @property
    def lifetime(self) -> int:
        end = self.destroyed_at if self.destroyed_at is not None else self.created_at + self.survived_observations
        return max(0, end - self.created_at)

    @property
    def life_reuse(self) -> int:
        return self.lifetime * self.use_count


@dataclass(frozen=True)
class StepMetrics:
    step: int
    observations: int
    abstractions: int
    alive_abstractions: int
    max_depth: int
    mean_lifetime: float
    mean_reuse: float
    merges: int
    splits: int
    destroyed: int
    created: int
    compositions: int


@dataclass(frozen=True)
class ExperimentSummary:
    agent_name: str
    observations: int
    abstractions: int
    alive_abstractions: int
    max_depth: int
    mean_lifetime: float
    mean_reuse: float
    total_life_reuse: int
    transfer_reuse: int
    destroyed: int
    merges: int
    splits: int
    compositions: int
