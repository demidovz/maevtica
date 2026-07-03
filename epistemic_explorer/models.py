from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

NodeType = Literal["concept", "hypothesis", "question", "experiment", "observation", "claim", "critique"]
EdgeType = Literal["depends_on", "falsifies", "supports", "generalizes", "uses", "asks", "tests", "produces"]
HypothesisStatus = Literal["proposed", "formalized", "criticized", "tested", "survived", "falsified"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class KnowledgeNode:
    node_id: str
    node_type: NodeType
    title: str
    body: str
    status: str = "open"
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisCandidate:
    title: str
    body: str
    formal_statement: str
    testable: bool
    assumptions: list[str]


@dataclass(frozen=True)
class CritiqueResult:
    summary: str
    counterexamples: list[str]
    hidden_assumptions: list[str]
    severity: float


@dataclass(frozen=True)
class SimulationResult:
    experiment_title: str
    setup: dict[str, Any]
    observations: list[str]
    falsified: bool
    reusable_concepts: list[str]


@dataclass(frozen=True)
class ObjectiveScore:
    objective_name: str
    metrics: dict[str, float]
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CycleLog:
    cycle_index: int
    selected_question_id: str
    selected_question: str
    hypothesis_ids: list[str]
    critique_ids: list[str]
    experiment_ids: list[str]
    observation_ids: list[str]
    new_question_ids: list[str]
    objective_scores: list[ObjectiveScore]
    meta_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["objective_scores"] = [score.to_dict() for score in self.objective_scores]
        return data

