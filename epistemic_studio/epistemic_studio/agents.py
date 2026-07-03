from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import compute_metrics
from .models import AgentRecord, Artifact, ArtifactKind, EdgeKind, GraphEdge, Provenance, ResearchState, now_iso


@dataclass
class AgentOutput:
    artifacts: list[Artifact] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    updates: list[Artifact] = field(default_factory=list)


class ResearchAgent:
    role = "agent"

    def __init__(self, agent_id: str):
        self.id = agent_id

    def record(self) -> AgentRecord:
        return AgentRecord(id=self.id, role=self.role)

    def provenance(self, cycle: int) -> Provenance:
        return Provenance(agent_id=self.id, agent_role=self.role, cycle=cycle, source="research_cycle")

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        raise NotImplementedError


class ExplorerAgent(ResearchAgent):
    role = "explorer"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        frontier = state.artifacts.get(frontier_id or "")
        title = "Unmapped research frontier" if frontier is None else f"Unexplored implication of {frontier.title}"
        hypothesis = Artifact(
            kind=ArtifactKind.HYPOTHESIS,
            title=f"H{cycle}: {title}",
            body="Candidate explanation proposed from an unknown region; requires Lab formalization and Engine attack.",
            provenance=provenance,
            confidence=0.42,
            metadata={"frontier_id": frontier_id, "priority": 0.6, "attention_spent": attention},
        )
        edges = []
        if frontier:
            edges.append(GraphEdge(hypothesis.id, frontier.id, EdgeKind.DERIVED_FROM, provenance, confidence=0.55))
        return AgentOutput(artifacts=[hypothesis], edges=edges)


class LabAgent(ResearchAgent):
    role = "lab"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        hypotheses = state.active_artifacts_by_kind(ArtifactKind.HYPOTHESIS)
        if not hypotheses:
            return AgentOutput()
        target = hypotheses[-1]
        concept = Artifact(
            kind=ArtifactKind.DERIVED_CONCEPT,
            title=f"Formal handle for {target.title}",
            body="A reusable formal handle that explains when this hypothesis should apply.",
            provenance=provenance,
            confidence=0.5,
            metadata={"derived_from": target.id, "attention_spent": attention * 0.55},
        )
        experiment = Artifact(
            kind=ArtifactKind.EXPERIMENT,
            title=f"Discrimination test for {target.title}",
            body="Compare predictions against a nearby counterexample region before increasing confidence.",
            provenance=provenance,
            confidence=0.5,
            metadata={"target_hypothesis": target.id, "priority": 0.7, "attention_spent": attention * 0.45},
        )
        edges = [
            GraphEdge(concept.id, target.id, EdgeKind.GENERALIZES, provenance, confidence=0.5),
            GraphEdge(experiment.id, target.id, EdgeKind.DEPENDS_ON, provenance, confidence=0.55),
        ]
        return AgentOutput(artifacts=[concept, experiment], edges=edges)


class EngineAgent(ResearchAgent):
    role = "engine"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        hypotheses = state.active_artifacts_by_kind(ArtifactKind.HYPOTHESIS)
        if not hypotheses:
            return AgentOutput()
        target = hypotheses[0]
        counterexample = Artifact(
            kind=ArtifactKind.COUNTEREXAMPLE,
            title=f"Stress case against {target.title}",
            body="A constructed case that should remain visible until the hypothesis survives or fails.",
            provenance=provenance,
            confidence=0.6,
            metadata={"target_hypothesis": target.id, "attention_spent": attention * 0.5},
        )
        contradiction = Artifact(
            kind=ArtifactKind.CONTRADICTION,
            title=f"Unresolved pressure on {target.title}",
            body="Current explanation and stress case cannot both be accepted without a condition split.",
            provenance=provenance,
            confidence=0.58,
            metadata={"target_hypothesis": target.id, "priority": 0.9, "attention_spent": attention * 0.5},
        )
        updated = Artifact.from_dict(target.to_dict())
        updated.confidence = max(0.0, round(updated.confidence - 0.08, 3))
        updated.updated_at = now_iso()
        updated.version += 1
        if updated.confidence < 0.25:
            updated.status = "falsified"
        edges = [
            GraphEdge(counterexample.id, target.id, EdgeKind.CONTRADICTS, provenance, confidence=0.65),
            GraphEdge(contradiction.id, target.id, EdgeKind.CONTRADICTS, provenance, confidence=0.6),
        ]
        return AgentOutput(artifacts=[counterexample, contradiction], edges=edges, updates=[updated])


class HistorianAgent(ResearchAgent):
    role = "historian"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        history = Artifact(
            kind=ArtifactKind.RESEARCH_HISTORY,
            title=f"Cycle {cycle} timeline entry",
            body="Recorded artifact-only work products and preserved the append-only state transition.",
            provenance=provenance,
            confidence=1.0,
            metadata={"frontier_id": frontier_id, "attention_spent": attention},
        )
        return AgentOutput(artifacts=[history])


class CartographerAgent(ResearchAgent):
    role = "cartographer"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        metrics = compute_metrics(state)
        atlas = Artifact(
            kind=ArtifactKind.APPLICABILITY_ATLAS,
            title=f"Applicability atlas update {cycle}",
            body="Map update tracking unresolved regions, contradictions, and compression pressure.",
            provenance=provenance,
            confidence=0.62,
            metadata={"metrics": metrics, "attention_spent": attention},
        )
        return AgentOutput(artifacts=[atlas])


class PlannerAgent(ResearchAgent):
    role = "planner"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        frontier = state.artifacts.get(frontier_id or "")
        title = "Next highest-value frontier" if frontier is None else f"Next test for {frontier.title}"
        plan = Artifact(
            kind=ArtifactKind.FRONTIER,
            title=title,
            body="Prioritize work that resolves contradictions, increases counterexample density, and improves compression.",
            provenance=provenance,
            confidence=0.7,
            metadata={"priority": 0.75, "selected_frontier_id": frontier_id, "attention_spent": attention},
        )
        edges = []
        if frontier:
            edges.append(GraphEdge(plan.id, frontier.id, EdgeKind.DEPENDS_ON, provenance, confidence=0.6))
        return AgentOutput(artifacts=[plan], edges=edges)


class MetaObserverAgent(ResearchAgent):
    role = "meta_observer"

    def run(
        self, state: ResearchState, frontier_id: str | None, cycle: int, attention: float = 0.0
    ) -> AgentOutput:
        provenance = self.provenance(cycle)
        metrics = compute_metrics(state)
        note = "No stagnation detected."
        if metrics["compression"] == 0 and cycle > 2:
            note = "Stagnation risk: graph growth has not yielded compression."
        process = Artifact(
            kind=ArtifactKind.PROCESS_NOTE,
            title=f"Process review {cycle}",
            body=note,
            provenance=provenance,
            confidence=0.65,
            metadata={"metrics": metrics, "attention_spent": attention},
        )
        return AgentOutput(artifacts=[process])


DEFAULT_AGENTS: list[ResearchAgent] = [
    ExplorerAgent("agent_explorer"),
    LabAgent("agent_lab"),
    EngineAgent("agent_engine"),
    HistorianAgent("agent_historian"),
    CartographerAgent("agent_cartographer"),
    PlannerAgent("agent_planner"),
    MetaObserverAgent("agent_meta_observer"),
]
