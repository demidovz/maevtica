from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .models import ArtifactKind, EdgeKind, ResearchState


def compute_metrics(state: ResearchState) -> dict[str, Any]:
    kinds = Counter(artifact.kind for artifact in state.artifacts.values())
    active_hypotheses = state.active_artifacts_by_kind(ArtifactKind.HYPOTHESIS)
    dead_hypotheses = [
        artifact
        for artifact in state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
        if artifact.status in {"dead", "falsified"}
    ]
    concepts = state.artifacts_by_kind(ArtifactKind.DERIVED_CONCEPT)
    contradictions = state.active_artifacts_by_kind(ArtifactKind.CONTRADICTION)
    counterexamples = state.artifacts_by_kind(ArtifactKind.COUNTEREXAMPLE)
    reusable = most_reusable_concepts(state)
    compression = compression_score(state)
    velocity = research_velocity(state)
    density = len(counterexamples) / max(1, len(active_hypotheses) + len(dead_hypotheses))
    duplicate_rate = estimate_duplicate_rate(state)
    attention_efficiency = compute_attention_efficiency(state)
    stagnation = stagnation_score(state)
    diversity = organizational_diversity(state)
    epistemic_value = compression + density + min(1.0, len(contradictions) / 5)
    return {
        "artifact_counts": dict(sorted(kinds.items())),
        "active_hypotheses": len(active_hypotheses),
        "dead_hypotheses": len(dead_hypotheses),
        "contradictions": len(contradictions),
        "counterexample_density": round(density, 3),
        "compression": round(compression, 3),
        "research_velocity": velocity,
        "most_reusable_concepts": reusable,
        "attention_efficiency": round(attention_efficiency, 3),
        "duplicate_rate": round(duplicate_rate, 3),
        "stagnation_score": round(stagnation, 3),
        "organizational_diversity": round(diversity, 3),
        "active_domains": [
            {"name": domain.name, "priority": domain.priority, "status": domain.status}
            for domain in sorted(state.domains.values(), key=lambda item: item.priority, reverse=True)
            if domain.status == "active"
        ],
        "epistemic_value": round(epistemic_value, 3),
    }


def compression_score(state: ResearchState) -> float:
    concepts = state.artifacts_by_kind(ArtifactKind.DERIVED_CONCEPT)
    hypotheses = state.artifacts_by_kind(ArtifactKind.HYPOTHESIS)
    if not hypotheses:
        return 0.0
    explained = {
        edge.target_id
        for edge in state.edges.values()
        if edge.kind in {EdgeKind.GENERALIZES, EdgeKind.SUPPORTS}
        and edge.source_id in {concept.id for concept in concepts}
    }
    return min(1.0, len(explained) / max(1, len(hypotheses)))


def research_velocity(state: ResearchState) -> dict[str, int]:
    if not state.cycles:
        return {"last_cycle_artifacts": 0, "last_cycle_edges": 0}
    last = state.cycles[-1]
    return {"last_cycle_artifacts": len(last.artifact_ids), "last_cycle_edges": len(last.edge_ids)}


def most_reusable_concepts(state: ResearchState, limit: int = 5) -> list[dict[str, Any]]:
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for edge in state.edges.values():
        incoming[edge.target_id] += 1
        outgoing[edge.source_id] += 1
    concepts = state.artifacts_by_kind(ArtifactKind.DERIVED_CONCEPT)
    ranked = sorted(
        concepts,
        key=lambda item: (incoming[item.id] + outgoing[item.id], item.confidence),
        reverse=True,
    )
    return [
        {
            "id": concept.id,
            "title": concept.title,
            "reuse": incoming[concept.id] + outgoing[concept.id],
            "confidence": round(concept.confidence, 3),
        }
        for concept in ranked[:limit]
    ]


def choose_frontier(state: ResearchState) -> str | None:
    candidates = [
        artifact
        for artifact in state.artifacts.values()
        if artifact.kind in {ArtifactKind.UNKNOWN_REGION, ArtifactKind.OPEN_QUESTION, ArtifactKind.CONTRADICTION}
        and artifact.status == "active"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda artifact: (artifact.metadata.get("priority", 0.5), artifact.created_at)).id


def estimate_duplicate_rate(state: ResearchState) -> float:
    titles = [artifact.title.lower().split(":", 1)[-1].strip() for artifact in state.artifacts.values()]
    if not titles:
        return 0.0
    duplicates = len(titles) - len(set(titles))
    return duplicates / len(titles)


def compute_attention_efficiency(state: ResearchState) -> float:
    spent = sum(allocation.allocated for allocation in state.allocations.values())
    if spent <= 0:
        return 0.0
    value = sum(cycle.metrics.get("epistemic_value", 0.0) for cycle in state.cycles)
    return value / spent


def stagnation_score(state: ResearchState, window: int = 10) -> float:
    if len(state.cycles) < 3:
        return 0.0
    recent = state.cycles[-window:]
    values = [cycle.metrics.get("compression", 0.0) for cycle in recent]
    if not values:
        return 0.0
    improvement = max(values) - min(values)
    return max(0.0, min(1.0, 1.0 - improvement))


def organizational_diversity(state: ResearchState) -> float:
    active = [agent for agent in state.agents.values() if agent.status == "active"]
    if not active:
        return 0.0
    roles = {agent.role for agent in active}
    generations = {agent.generation for agent in active}
    strategies = {agent.strategy for agent in active}
    return min(1.0, (len(roles) + len(generations) + len(strategies)) / (3 * len(active)))
