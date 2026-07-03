from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ArtifactKind(StrEnum):
    RESEARCH_QUESTION = "research_question"
    HYPOTHESIS = "hypothesis"
    ASSUMPTION = "assumption"
    EXPERIMENT = "experiment"
    COUNTEREXAMPLE = "counterexample"
    OBSERVATION = "observation"
    APPLICABILITY_ATLAS = "applicability_atlas"
    CONFIDENCE = "confidence"
    OPEN_QUESTION = "open_question"
    CONTRADICTION = "contradiction"
    RESEARCH_HISTORY = "research_history"
    DERIVED_CONCEPT = "derived_concept"
    INVARIANT_CANDIDATE = "invariant_candidate"
    UNKNOWN_REGION = "unknown_region"
    THEORY = "theory"
    WORLD_MODEL = "world_model"
    APPLICATION = "application"
    FRONTIER = "frontier"
    PROCESS_NOTE = "process_note"
    META_MEMORY = "meta_memory"
    HEALTH_REPORT = "health_report"


class EdgeKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    EQUIVALENT = "equivalent"
    UNKNOWN_RELATION = "unknown_relation"


@dataclass
class Provenance:
    agent_id: str
    agent_role: str
    cycle: int
    source: str


@dataclass
class Link:
    target_id: str
    relation: str


@dataclass
class Artifact:
    kind: str
    title: str
    body: str
    provenance: Provenance
    id: str = field(default_factory=lambda: new_id("art"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    status: str = "active"
    confidence: float = 0.5
    links: list[Link] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        item = dict(data)
        item["provenance"] = Provenance(**item["provenance"])
        item["links"] = [Link(**link) for link in item.get("links", [])]
        return cls(**item)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    kind: str
    provenance: Provenance
    id: str = field(default_factory=lambda: new_id("edge"))
    created_at: str = field(default_factory=now_iso)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        item = dict(data)
        item["provenance"] = Provenance(**item["provenance"])
        return cls(**item)


@dataclass
class AgentRecord:
    id: str
    role: str
    created_at: str = field(default_factory=now_iso)
    last_cycle: int = 0
    activity_count: int = 0
    status: str = "active"
    reputation: float = 1.0
    generation: int = 0
    parent_id: str | None = None
    strategy: str = "balanced"
    domain: str = "possible_worlds"
    attention_balance: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketBid:
    agent_id: str
    role: str
    target_id: str | None
    bid_type: str
    requested_attention: float
    expected_value: float
    rationale: str
    cycle: int
    id: str = field(default_factory=lambda: new_id("bid"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionAllocation:
    agent_id: str
    role: str
    target_id: str | None
    allocated: float
    requested: float
    expected_value: float
    reputation: float
    domain: str
    cycle: int
    id: str = field(default_factory=lambda: new_id("alloc"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrganizationAction:
    action: str
    agent_id: str
    reason: str
    cycle: int
    new_agent_id: str | None = None
    id: str = field(default_factory=lambda: new_id("org"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainRecord:
    name: str
    status: str = "active"
    priority: float = 1.0
    attention_received: float = 0.0
    cycles_since_active: int = 0
    last_cycle: int = 0
    stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CycleRecord:
    cycle: int
    started_at: str
    completed_at: str
    frontier_id: str | None
    artifact_ids: list[str]
    edge_ids: list[str]
    metrics: dict[str, Any]
    next_frontier_id: str | None
    attention_budget: float = 0.0
    attention_spent: float = 0.0
    bid_ids: list[str] = field(default_factory=list)
    allocation_ids: list[str] = field(default_factory=list)
    organization_action_ids: list[str] = field(default_factory=list)
    health_report_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchState:
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    cycles: list[CycleRecord] = field(default_factory=list)
    bids: dict[str, MarketBid] = field(default_factory=dict)
    allocations: dict[str, AttentionAllocation] = field(default_factory=dict)
    organization_actions: dict[str, OrganizationAction] = field(default_factory=dict)
    domains: dict[str, DomainRecord] = field(default_factory=dict)

    def artifacts_by_kind(self, kind: str) -> list[Artifact]:
        return [artifact for artifact in self.artifacts.values() if artifact.kind == kind]

    def active_artifacts_by_kind(self, kind: str) -> list[Artifact]:
        return [
            artifact
            for artifact in self.artifacts.values()
            if artifact.kind == kind and artifact.status == "active"
        ]
